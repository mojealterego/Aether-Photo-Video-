"""One-neural-tile-at-a-time full-frame SR bridge for Galaxy S24 Ultra.

The phone never allocates a full-image torch tensor. It writes per-tile PNGs,
receives remote candidates and accumulates HIGH-PASS residuals in disk maps.
Output is uint16 RGB BigTIFF in sRGB; neural candidates are RGB8 hypotheses.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

try:
    import cv2
except ImportError:  # Termux can run without Python OpenCV
    cv2 = None
import numpy as np
from PIL import Image, UnidentifiedImageError
import tifffile

from .s24_render import Source, _fingerprint, _icc, _to_linear, _to_srgb, _focus_source_roi
from .neural_bridge import _json_atomic, _safe_name, _uint16


def prepare_sr_tiles(source: Path, out_dir: Path, *, tile: int = 512,
                     halo: int = 48, stage: str = 'macro',
                     max_source_pixels: int = 500_000_000) -> dict[str, Any]:
    if stage not in ('macro', 'meso', 'micro'):
        raise ValueError('Unknown SR stage')
    if (tile < 32 or tile > 1024 or tile % 16 or halo < 8 or
            halo > 128 or tile + 2 * halo > 1024):
        raise ValueError('Bounded neural tile must include its halo and fit 1024 px')
    source, out_dir = Path(source), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError('Prepared directory must be empty')
    s = Source(source, max_source_pixels, 900)
    try:
        if s.pixel_dtype not in (np.dtype('uint8'), np.dtype('uint16')):
            raise ValueError('SR source must be RGB8/RGB16 in sRGB')
        out_dir.mkdir(parents=True, exist_ok=True)
        H, W = s.height, s.width
        jobs = []
        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                ey0, ey1 = max(0, y0 - halo), min(H, y1 + halo)
                ex0, ex1 = max(0, x0 - halo), min(W, x1 + halo)
                suffix = f'{y0}_{y1}_{x0}_{x1}.png'
                ref, foc, restored = ('reference_' + suffix, 'focus_' + suffix,
                                      'restored_' + suffix)
                raw = s.crop((ex0, ey0, ex1, ey1))
                u8 = (raw if raw.dtype == np.uint8 else
                      np.rint(raw.astype(np.float32) / 257).astype(np.uint8))
                Image.fromarray(u8, 'RGB').save(out_dir / ref)
                jobs.append({'extent': [ey0, ey1, ex0, ex1],
                             'core': [y0, y1, x0, x1],
                             'reference': ref, 'focus': foc, 'restored': restored,
                             'sha256_reference': _fingerprint(out_dir / ref)})
        manifest = {'version': '2.0.0', 'purpose': 's24_neural_sr_full_frame',
                    'stage': stage, 'source_sha256': _fingerprint(source),
                    'source_size': [W, H], 'tile': tile, 'halo': halo,
                    'tiles': jobs,
                    'output': 'Synthetic neural residuals; not recovered historical data'}
        _json_atomic(out_dir / 'manifest.json', manifest)
        return {'selected_tiles': len(jobs), 'source_size': [W, H],
                'manifest': str(out_dir / 'manifest.json')}
    finally:
        s.close()


def _validate_candidate_png(path: Path, size: tuple[int, int]) -> None:
    if not path.is_file() or path.stat().st_size > 48 * 1024 * 1024:
        raise ValueError('Candidate missing or exceeds 48MiB limit')
    try:
        with Image.open(path) as im:
            if im.format != 'PNG' or im.mode != 'RGB' or im.size != size:
                raise ValueError('Candidate PNG format/dimensions differ from tile')
            im.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError('Corrupted candidate image') from exc


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur in linear light; OpenCV is only a fast optional backend."""
    if cv2 is not None:
        return cv2.GaussianBlur(image, (0, 0), sigma)
    radius = int(math.ceil(3. * sigma))
    taps = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-.5 * (taps / sigma) ** 2)
    kernel /= kernel.sum()
    out = np.asarray(image, dtype=np.float32)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode='reflect')
        result = np.zeros_like(out)
        for offset, weight in enumerate(kernel):
            view = [slice(None)] * padded.ndim
            view[axis] = slice(offset, offset + out.shape[axis])
            result += padded[tuple(view)] * weight
        out = result
    return out


def _gaussian_weight(h: int, w: int) -> np.ndarray:
    # strictly positive at image edges; sum-of-weights normalized after blending
    yy = (np.arange(h, dtype=np.float32) + .5 - h * .5) / max(1., h*.34)
    xx = (np.arange(w, dtype=np.float32) + .5 - w * .5) / max(1., w*.34)
    return np.maximum(np.exp(-.5 * (yy[:, None]**2 + xx[None, :]**2)), .002)


def _normalize_roi(src: Source, accum: np.memmap, weights: np.memmap,
                   y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    raw = _uint16(src.crop((x0, y0, x1, y1)))
    rgb = raw.astype(np.float32) / 65535.
    residual = np.asarray(accum[y0:y1, x0:x1], np.float32)
    w = np.asarray(weights[y0:y1, x0:x1], np.float32)
    result = _to_linear(rgb) + residual / np.maximum(w[..., None], 1e-6)
    final = np.rint(_to_srgb(result) * 65535).astype(np.uint16)
    final[w <= 0] = raw[w <= 0]
    return final


def stitch_sr_tiles(source: Path, manifest_file: Path, candidates: Path,
                    output: Path, *, stage: str, detail_strength: float = .65,
                    low_shift: float = .0, sigma: float = 4.,
                    final_tiled: bool = False, tiff_tile: int = 256,
                    reserve_free_gb: float = 0.,
                    max_source_pixels: int = 500_000_000,
                    focus_mask: Path | None = None) -> dict[str, Any]:
    """Blend remote candidates with strict reference LF lock and WOLA.

    Working set on CPU is one <=1024px expanded tile. Large accumulators are
    disk-backed and final is written blockwise. No PyTorch/Android CUDA.
    """
    if stage not in ('macro', 'meso', 'micro') or not all(
            math.isfinite(v) for v in (detail_strength, low_shift, sigma, reserve_free_gb)):
        raise ValueError('Bad stage/fusion parameters')
    if not (0 <= detail_strength <= 1 and 0 <= low_shift <= .3 and
            .5 <= sigma <= 32 and reserve_free_gb >= 0):
        raise ValueError('Bad bounded neural fusion settings')
    source, manifest_file, candidates, output = map(Path, (source, manifest_file, candidates, output))
    if output.exists():
        raise FileExistsError('Never overwrite committed output')
    rawdoc = manifest_file.read_bytes()
    if len(rawdoc) > 80*1024*1024:
        raise ValueError('Manifest unreasonably large')
    doc = json.loads(rawdoc)
    if (doc.get('version') != '2.0.0' or doc.get('purpose') != 's24_neural_sr_full_frame'
            or doc.get('stage') != stage or not isinstance(doc.get('tiles'), list)
            or not doc['tiles']):
        raise ValueError('SR manifest stage/purpose mismatch')
    if (candidates/'manifest.json').read_bytes() != rawdoc:
        raise ValueError('Candidate set not committed for this exact manifest')
    ledger = json.loads((candidates/'batch_state.json').read_text())
    if (ledger.get('manifest_sha256') != hashlib.sha256(rawdoc).hexdigest()
            or not isinstance(ledger.get('tiles'), list)
            or len(ledger['tiles']) != len(doc['tiles'])):
        raise ValueError('Candidate ledger is absent or invalid')
    if doc.get('source_sha256') != _fingerprint(source):
        raise ValueError('Source SHA256 mismatch')
    s = Source(source, max_source_pixels, 900)
    focus=None
    try:
        H, W = s.height, s.width
        if focus_mask is not None:
            focus=Image.open(focus_mask)
            if focus.mode != "L" or focus.width*focus.height > 16_000_000:
                raise ValueError("Focus confidence must be <=16MP grayscale L")
            focus.load()
        if [W, H] != doc['source_size']:
            raise ValueError('SR manifest dimensions mismatch')
        seen = set()
        for job, mark in zip(doc['tiles'], ledger['tiles']):
            ey0,ey1,ex0,ex1 = job['extent']
            y0,y1,x0,x1 = job['core']
            if not (0 <= ey0 <= y0 < y1 <= ey1 <= H and
                    0 <= ex0 <= x0 < x1 <= ex1 <= W):
                raise ValueError('Invalid tile geometry')
            identity = (y0,y1,x0,x1)
            if identity in seen:
                raise ValueError('Duplicate tile core')
            seen.add(identity)
            ref, candidate = _safe_name(job['reference']), _safe_name(job['restored'])
            if mark.get('status') != 'done' or mark.get('output') != candidate:
                raise ValueError('Incomplete neural candidate ledger')
            if not isinstance(mark.get('sha256'),str) or len(mark['sha256']) != 64:
                raise ValueError('Missing candidate checksum')
            if _fingerprint(manifest_file.parent/ref) != job['sha256_reference']:
                raise ValueError('Reference tile tampered')
            if _fingerprint(candidates/candidate) != mark['sha256']:
                raise ValueError('Neural result SHA256 changed')
            if max(ey1-ey0,ex1-ex0) > 1024:
                raise ValueError('Neural tile too large')
            _validate_candidate_png(candidates/candidate, (ex1-ex0, ey1-ey0))
        output.parent.mkdir(parents=True, exist_ok=True)
        bytes_need = H*W*(12+4+6) + 64*1024*1024 + int(reserve_free_gb*1024**3)
        if shutil.disk_usage(output.parent).free < bytes_need:
            raise OSError('Insufficient free disk for 400MP neural WOLA accumulators + output')
        with tempfile.TemporaryDirectory(prefix='photo400-sr-', dir=output.parent) as folder:
            scratch=Path(folder)
            a = np.memmap(scratch/'residual.float32',dtype=np.float32,mode='w+',shape=(H,W,3))
            weights = np.memmap(scratch/'weight.float32',dtype=np.float32,mode='w+',shape=(H,W))
            # Avoid giant Python temporary arrays when initializing 400MP.
            for yy in range(0,H,128):
                a[yy:yy+128]=0
                weights[yy:yy+128]=0
            for job in doc['tiles']:
                ey0,ey1,ex0,ex1 = job['extent']
                with Image.open(candidates/job['restored']) as im:
                    candidate=np.asarray(im,dtype=np.float32)/255.
                base=_uint16(s.crop((ex0,ey0,ex1,ey1))).astype(np.float32)/65535.
                base_lin=_to_linear(base)
                cand_lin=_to_linear(candidate)
                blur_base=_gaussian_blur(base_lin,sigma)
                blur_cand=_gaussian_blur(cand_lin,sigma)
                # Spatial base remains dominant; all remote details are explicitly hypotheses.
                detail=(cand_lin-blur_cand) - (base_lin-blur_base)
                delta=detail_strength*(detail+low_shift*(blur_cand-blur_base))
                if focus is not None:
                    focused=_focus_source_roi(focus,(ex0,ey0,ex1,ey1),(H,W))
                    delta*=np.clip(focused,0.,1.)[...,None]
                wmap=_gaussian_weight(ey1-ey0,ex1-ex0)
                a[ey0:ey1,ex0:ex1] += (delta*wmap[...,None]).astype(np.float32)
                weights[ey0:ey1,ex0:ex1] += wmap
            a.flush(); weights.flush()
            part = output.with_name(output.name+'.partial.tif')
            if part.exists():
                raise FileExistsError('Untracked partial SR BigTIFF; inspect before continuing')
            try:
                if final_tiled:
                    def iter_tiles():
                        for yy in range(0,H,tiff_tile):
                            for xx in range(0,W,tiff_tile):
                                y1=min(H,yy+tiff_tile);x1=min(W,xx+tiff_tile)
                                block=np.zeros((tiff_tile,tiff_tile,3),dtype=np.uint16)
                                block[:y1-yy,:x1-xx]=_normalize_roi(s,a,weights,yy,y1,xx,x1)
                                yield block
                    tifffile.imwrite(part,iter_tiles(),shape=(H,W,3),dtype=np.uint16,
                                     photometric='rgb',bigtiff=True,tile=(tiff_tile,tiff_tile),
                                     compression='deflate',metadata=None,iccprofile=_icc())
                else:
                    dst=tifffile.memmap(part,shape=(H,W,3),dtype=np.uint16,
                                       photometric='rgb',bigtiff=True,
                                       metadata=None,iccprofile=_icc())
                    for yy in range(0,H,64):
                        dst[yy:yy+64]=_normalize_roi(s,a,weights,yy,min(yy+64,H),0,W)
                    dst.flush();del dst
                with tifffile.TiffFile(part) as tf:
                    page=tf.pages[0]
                    if (not tf.is_bigtiff or page.shape!=(H,W,3) or
                            page.dtype!=np.dtype('uint16') or page.is_tiled!=final_tiled):
                        raise RuntimeError('Invalid emitted BigTIFF')
                os.replace(part,output)
            finally:
                part.unlink(missing_ok=True)
            del a,weights
        return {'selected_tiles':len(doc['tiles']),'output':str(output),
                'source_dimensions':[W,H],'final_tiled':final_tiled,
                'generative_hypothesis':True,'base_locked':True}
    finally:
        s.close()
        if focus is not None:
            focus.close()