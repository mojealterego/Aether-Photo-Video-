"""Disk-backed faithful upscaling for Android/Termux; no PyTorch/ComfyUI on device.

Contract: source RGB in sRGB primaries, encoded as sRGB or linear light;
16-bit BigTIFF in sRGB encoding. Gaussian high-pass is OPTIONAL and never
recovers information missing from source. External inpaint hypotheses may be merged only within an explicit mask; no on-device face synthesis.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageCms
import tifffile


class RenderInterrupted(RuntimeError):
    """Interrupted rendering; checkpoint can be resumed on next execution."""


def target_from_mp(width: int, height: int, megapixels: float) -> tuple[int, int]:
    if width < 1 or height < 1 or not math.isfinite(megapixels) or megapixels <= 0:
        raise ValueError('Invalid image size or megapixels')
    factor = math.sqrt(1_000_000.0 * megapixels / (width * height))
    return round(width * factor), round(height * factor)


@dataclass(frozen=True)
class MobileConfig:
    source: Path
    output: Path
    stage_dir: Path
    target_width: int | None = None
    target_height: int | None = None
    target_mp: float = 400.0
    tile_size: int = 256
    detail_gain: float = 0.0
    psf_sigma: float = 0.0
    rl_iterations: int = 0
    ca_red: float = 0.0
    ca_blue: float = 0.0
    focus_mask: Path | None = None
    inpainted_base: Path | None = None
    sr_candidate: Path | None = None
    wavelet_levels: int = 3
    wavelet_gain: float = 1.0
    inpaint_mask: Path | None = None
    protected_mask: Path | None = None
    inpaint_strength: float = 1.0
    lab_color_match: float = 0.0
    input_transfer: str = 'srgb'
    max_input_pixels: int = 225_000_000
    max_decoded_ram_mb: int = 900
    reserve_free_gb: float = 1.0
    keep_stage: bool = False
    contiguous_output: bool = False
    compression: str = "none"

    def __post_init__(self) -> None:
        for name in ('source', 'output', 'stage_dir', 'focus_mask', 'inpainted_base', 'sr_candidate', 'inpaint_mask', 'protected_mask'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser())
        if (self.inpainted_base is None) != (self.inpaint_mask is None):
            raise ValueError('Inpaint candidate requires an explicit inpaint mask')
        if self.sr_candidate is not None and self.focus_mask is None:
            raise ValueError('Wavelet candidate requires a focus mask for bokeh protection')
        if type(self.wavelet_levels) is not int or not 1 <= self.wavelet_levels <= 5:
            raise ValueError('wavelet_levels must be 1..5')
        if not math.isfinite(self.wavelet_gain) or not 0 <= self.wavelet_gain <= 2:
            raise ValueError('wavelet_gain must be 0..2')
        if self.protected_mask is not None and self.inpaint_mask is None:
            raise ValueError('Protected mask requires an inpaint mask')
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in
               (self.inpaint_strength, self.lab_color_match)):
            raise ValueError('Invalid inpaint strength / color match')
        if self.source.resolve() == self.output.resolve():
            raise ValueError('Cannot overwrite input')
        if self.stage_dir.resolve() in (self.source.resolve(), self.output.resolve()):
            raise ValueError('Stage must be a separate directory')
        if (self.target_width is None) != (self.target_height is None):
            raise ValueError('Supply both explicit target dimensions or neither')
        if (self.target_width is not None and
            (self.target_width <= 0 or self.target_height <= 0 or
             self.target_width * self.target_height > 1_000_000_000)):
            raise ValueError('Target geometry must be positive and <= 1 GP')
        if self.tile_size < 16 or self.tile_size > 1024 or self.tile_size % 16:
            raise ValueError('Tile size must be a multiple of 16, 16..1024')
        if not math.isfinite(self.detail_gain) or not (0 <= self.detail_gain <= 1):
            raise ValueError('detail_gain must be in [0,1]')
        if (not math.isfinite(self.psf_sigma) or self.psf_sigma < 0 or self.psf_sigma > 3
                or self.rl_iterations < 0 or self.rl_iterations > 12):
            raise ValueError('Optics: sigma 0..3 and iterations 0..12 required')
        if any(not math.isfinite(v) or abs(v)>0.002 for v in (self.ca_red, self.ca_blue)):
            raise ValueError('CA radial coefficients outside mobile safety limit +/-0.002')
        if self.max_input_pixels < 1 or self.max_input_pixels > 225_000_000 or self.max_decoded_ram_mb < 1:
            raise ValueError('Invalid source pixel or RAM budget')
        if not math.isfinite(self.reserve_free_gb) or self.reserve_free_gb < 0:
            raise ValueError('Invalid free space reserve')
        if self.compression not in ('none', 'deflate') or (self.contiguous_output and self.compression != 'none'):
            raise ValueError('Compressed intermediate TIFF cannot be memory mapped; use none')
        if self.input_transfer not in ('srgb', 'linear'):
            raise ValueError('Only sRGB primaries with sRGB or linear transfer are accepted')


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _to_linear(encoded: np.ndarray) -> np.ndarray:
    x = np.asarray(encoded, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _to_srgb(linear: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1/2.4) - 0.055).astype(np.float32)


def _check_srgb_profile(data: bytes | None) -> None:
    if not data:
        return
    try:
        profile = ImageCms.ImageCmsProfile(BytesIO(data))
        description = ImageCms.getProfileDescription(profile).lower()
    except (OSError, ValueError) as e:
        raise ValueError('Embedded ICC profile is not readable; normalize color on a workstation') from e
    if 'srgb' not in description and 's-rgb' not in description:
        raise ValueError('Input ICC is not sRGB; convert color to sRGB before processing')


class Source:
    """Read ROI only from mapped RGB TIFF or bounded PIL RGB decode."""
    def __init__(self, path: Path, max_pixels: int, max_ram_mb: int):
        self.path = path
        self.image: Image.Image | None = None
        self.mapped: np.ndarray | None = None
        self.width = 0
        self.height = 0
        if path.suffix.lower() in ('.tiff', '.tif'):
            with tifffile.TiffFile(path) as f:
                if len(f.pages) != 1:
                    raise ValueError('Only one-page RGB TIFF supported on device')
                page = f.pages[0]
                if len(page.shape) != 3 or page.shape[-1] != 3 or page.photometric != tifffile.PHOTOMETRIC.RGB:
                    raise ValueError('TIFF must be RGB H,W,3')
                orientation = page.tags.get('Orientation')
                if orientation is not None and orientation.value != 1:
                    raise ValueError('Normalize TIFF orientation first')
                if page.dtype not in (np.dtype('uint8'), np.dtype('uint16'), np.dtype('float32')):
                    raise ValueError('TIFF requires RGB uint8/uint16 or float32')
                icc = page.tags.get('InterColorProfile') or page.tags.get('ICCProfile')
                _check_srgb_profile(icc.value if icc is not None else None)
                self.height, self.width = page.shape[:2]
            if self.width * self.height > max_pixels:
                raise MemoryError('Input exceeds configured source pixel budget')
            try:
                self.mapped = tifffile.memmap(path)
            except (ValueError, OSError) as e:
                raise ValueError('On-device TIFF input must be uncompressed and memmappable') from e
            if self.mapped.shape != (self.height, self.width, 3):
                raise ValueError('Unexpected TIFF memory map shape')
        else:
            # Bounded decoding: Android process may be killed before Python raises MemoryError.
            Image.MAX_IMAGE_PIXELS = 225_000_000
            im = Image.open(path)
            try:
                self.width, self.height = im.size
                if self.width * self.height > max_pixels:
                    raise MemoryError('Input exceeds configured source pixel budget')
                if self.width * self.height * 4 > max_ram_mb * 1024 * 1024:
                    raise MemoryError('Source image would exceed configured decode memory')
                if im.getexif().get(274, 1) != 1:
                    raise ValueError('EXIF orientation must be normalized before reconstruction')
                if im.mode != 'RGB':
                    raise ValueError('On-device JPEG/PNG must decode as RGB8')
                _check_srgb_profile(im.info.get('icc_profile'))
                im.load()
            except BaseException:
                im.close()
                raise
            self.image = im

    @property
    def pixel_dtype(self) -> np.dtype:
        return self.mapped.dtype if self.mapped is not None else np.dtype('uint8')

    def crop(self, box: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = box
        if self.mapped is not None:
            return np.asarray(self.mapped[y0:y1, x0:x1, :])
        if self.image is None:
            raise RuntimeError('Closed image source')
        return np.asarray(self.image.crop(box))

    def close(self) -> None:
        if self.image is not None:
            self.image.close()
            self.image = None
        self.mapped = None


def _resample_linear(source: Source, ext: tuple[int, int, int, int],
                     target_hw: tuple[int, int], transfer: str, config: MobileConfig,
                     focus_image: Image.Image | None,
                     overlay: tuple[Source, Image.Image, Image.Image | None] | None = None) -> np.ndarray:
    """Exact coordinate mapping with Lanczos in linear light using only a local ROI."""
    x0, y0, x1, y1 = ext
    height, width = target_hw
    sx, sy = source.width / width, source.height / height
    l, t, r, b = x0 * sx, y0 * sy, x1 * sx, y1 * sy
    # 3-lobe Lanczos + margin for downsampling; source boundaries are clamped.
    optical_support = math.ceil(config.psf_sigma * 3) * config.rl_iterations
    max_ca_dist = max(abs(config.ca_red), abs(config.ca_blue)) * math.hypot(source.width, source.height)/2
    margin = max(8, math.ceil(4 * max(sx, sy)) + 2) + optical_support + math.ceil(max_ca_dist)
    bx0 = max(0, math.floor(l) - margin)
    by0 = max(0, math.floor(t) - margin)
    bx1 = min(source.width, math.ceil(r) + margin)
    by1 = min(source.height, math.ceil(b) + margin)
    patch = source.crop((bx0, by0, bx1, by1)).astype(np.float32)
    if np.issubdtype(source.pixel_dtype, np.integer):
        patch /= float(np.iinfo(source.pixel_dtype).max)
    if transfer == 'srgb':
        patch = _to_linear(patch)
    # Pre-SR optical stage in float32, with a global source-space coordinate system.
    if config.ca_red != 0.0 or config.ca_blue != 0.0:
        from aether.core.optical import correct_lca, LCAChannel
        patch = correct_lca(patch, origin_xy=(bx0,by0),
                            full_wh=(source.width, source.height),
                            red=LCAChannel(radial=(config.ca_red,0.,0.,0.)),
                            blue=LCAChannel(radial=(config.ca_blue,0.,0.,0.)))
    if config.rl_iterations and config.psf_sigma:
        from aether.core.optical import anisotropic_gaussian_psf, richardson_lucy_tv
        radius = max(1, math.ceil(config.psf_sigma*3))
        psf = anisotropic_gaussian_psf(config.psf_sigma, config.psf_sigma, 0., radius=radius)
        padding = radius * config.rl_iterations
        # Reflect-extended optical halo prevents zero-boundary RL from pulling
        # artificial edge intensity into the central working tile.
        extended = np.pad(patch, ((padding,padding),(padding,padding),(0,0)), mode='reflect')
        filtered = richardson_lucy_tv(extended, psf, iterations=config.rl_iterations,
                                      tv_weight=0.001)
        restored = filtered[padding:-padding, padding:-padding]
        if focus_image is not None:
            fm = _focus_source_roi(focus_image, (bx0,by0,bx1,by1), (source.height,source.width))
            patch = patch + fm[:,:,None]*(restored - patch)
        else:
            patch = restored
    # Masked source-space low-frequency base replacement after optical correction
    # and *before* spatial super-resolution. Candidate may be an AI hypothesis.
    if overlay is not None:
        from .face_occlusion import masked_linear_fusion
        candidate_source, mask_image, protected_image = overlay
        hypothesis = candidate_source.crop((bx0,by0,bx1,by1)).astype(np.float32)
        if np.issubdtype(candidate_source.pixel_dtype, np.integer):
            hypothesis /= float(np.iinfo(candidate_source.pixel_dtype).max)
        if transfer == 'srgb':
            hypothesis = _to_linear(hypothesis)
        mask = _focus_source_roi(mask_image,(bx0,by0,bx1,by1),(source.height,source.width))
        protected = (_focus_source_roi(protected_image,(bx0,by0,bx1,by1),
                                     (source.height,source.width))
                     if protected_image is not None else None)
        patch = masked_linear_fusion(patch, hypothesis, mask, protected=protected,
                                     strength=config.inpaint_strength,
                                     color_match_strength=config.lab_color_match)
    output = np.empty((y1 - y0, x1 - x0, 3), dtype=np.float32)
    box = (l - bx0, t - by0, r - bx0, b - by0)
    for c in range(3):
        layer = Image.fromarray(np.ascontiguousarray(patch[:, :, c]), mode='F')
        output[:, :, c] = np.asarray(layer.resize((x1-x0, y1-y0),
                                    Image.Resampling.LANCZOS, box=box), dtype=np.float32)
    return np.clip(output, 0, 1)



def _focus_source_roi(mask: Image.Image, source_box: tuple[int,int,int,int],
                      source_hw: tuple[int,int]) -> np.ndarray:
    bx0, by0, bx1, by1 = source_box
    h,w = source_hw
    mw,mh = mask.size
    result = mask.resize((bx1-bx0, by1-by0),Image.Resampling.BILINEAR,
                         box=(bx0*mw/w, by0*mh/h, bx1*mw/w, by1*mh/h))
    return np.asarray(result,dtype=np.float32)/255.0


def _radial_ca_source(patch: np.ndarray, x0: int, y0: int, width: int, height: int,
                      k_red: float, k_blue: float) -> np.ndarray:
    # Lateral CA estimate: radial (R,B) remap relative to G in global source coordinates.
    hp,wp = patch.shape[:2]
    gx = x0 + np.arange(wp,dtype=np.float32)[None,:]
    gy = y0 + np.arange(hp,dtype=np.float32)[:,None]
    cx,cy=(width-1)/2,(height-1)/2
    dx,dy=gx-cx,gy-cy
    r2 = (dx*dx+dy*dy)/max(cx*cx+cy*cy,1.0)
    result=patch.copy()
    for channel,k in ((0,k_red),(2,k_blue)):
        if k == 0:
            continue
        sample_x=np.clip(gx+dx*(k*r2)-x0,0,wp-1)
        sample_y=np.clip(gy+dy*(k*r2)-y0,0,hp-1)
        ax=np.floor(sample_x).astype(np.int32)
        ay=np.floor(sample_y).astype(np.int32)
        bx=np.minimum(ax+1,wp-1)
        by=np.minimum(ay+1,hp-1)
        fx=sample_x-ax
        fy=sample_y-ay
        a=patch[ay,ax,channel]
        b=patch[ay,bx,channel]
        c=patch[by,ax,channel]
        d=patch[by,bx,channel]
        result[:,:,channel]=(1-fy)*((1-fx)*a+fx*b)+fy*((1-fx)*c+fx*d)
    return np.clip(result,0,1)


def _rl_gaussian(observed: np.ndarray,sigma: float,iterations:int) -> np.ndarray:
    # Non-blind RL with symmetric *Gaussian approximation*, NOT a measured Airy PSF.
    estimate=np.clip(observed.copy(),1.e-6,1.)
    obs=np.clip(observed,0.,1.)
    for _ in range(iterations):
        blurred=_gaussian_blur(estimate,sigma)
        correction=_gaussian_blur(obs/np.maximum(blurred,1.e-5),sigma)
        np.clip(correction,0.5,2.,out=correction)
        estimate=np.clip(estimate*correction,0.,1.)
    return estimate


def _gaussian_blur(channel: np.ndarray, sigma: float) -> np.ndarray:
    radius = math.ceil(sigma * 3)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    data = channel
    for axis in (0, 1):
        pad = [(0, 0)] * data.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(data, pad, mode='reflect')
        accum = np.zeros_like(data)
        for i, coeff in enumerate(kernel):
            index = [slice(None)] * data.ndim
            index[axis] = slice(i, i + data.shape[axis])
            accum += float(coeff) * padded[tuple(index)]
        data = accum
    return data


def _focus_tile(image: Image.Image | None, ext: tuple[int, int, int, int],
                target_hw: tuple[int, int]) -> np.ndarray | None:
    if image is None:
        return None
    x0, y0, x1, y1 = ext
    th, tw = target_hw
    w, h = image.size
    # L format mask: bilinear prevents artificial ringing at transitions.
    mask = image.resize((x1-x0, y1-y0), Image.Resampling.BILINEAR,
                        box=(x0*w/tw, y0*h/th, x1*w/tw, y1*h/th))
    return np.asarray(mask, dtype=np.float32) / 255.0


def _render_output_tile(source: Source, config: MobileConfig,
                        rect: tuple[int, int, int, int], target_hw: tuple[int, int],
                        focus_image: Image.Image | None,
                        overlay: tuple[Source, Image.Image, Image.Image | None] | None = None,
                        sr_source: Source | None = None) -> np.ndarray:
    x0, y0, x1, y1 = rect
    th, tw = target_hw
    # Local working halo, so no border artifacts in optional acutance.
    halo = max(12, (1 << config.wavelet_levels) * 2) if sr_source is not None else 12
    # Align Haar window origins to global dyadic phase for inter-tile continuity.
    phase = (1 << config.wavelet_levels) if sr_source is not None else 1
    ext = (max(0, ((x0-halo)//phase)*phase), max(0, ((y0-halo)//phase)*phase),
           min(tw, ((x1+halo+phase-1)//phase)*phase),
           min(th, ((y1+halo+phase-1)//phase)*phase))
    linear = _resample_linear(source, ext, (th, tw), config.input_transfer, config, focus_image, overlay)
    if sr_source is not None:
        from aether.core.wavelet import wavelet_faithful_fusion
        # Candidate may be synthetic. No duplicate optics on candidate.
        neutral = replace(config, ca_red=0.0, ca_blue=0.0, psf_sigma=0.0, rl_iterations=0)
        sr_linear = _resample_linear(sr_source, ext, (th,tw), config.input_transfer,
                                     neutral, None, None)
        spatial_mask = _focus_tile(focus_image,ext,(th,tw))
        if spatial_mask is None:
            raise RuntimeError('Missing focus mask for wavelet fusion')
        protected_target = _focus_tile(overlay[2],ext,(th,tw)) if overlay is not None else None
        linear = wavelet_faithful_fusion(linear,sr_linear,spatial_mask,
                                        levels=config.wavelet_levels,high_gain=config.wavelet_gain,
                                        protected_mask=protected_target)
    if config.detail_gain > 0:
        blur = _gaussian_blur(linear, sigma=1.15)
        high = np.clip(linear - blur, -0.025, 0.025)
        mask = _focus_tile(focus_image, ext, (th, tw))
        if mask is not None:
            high *= mask[:, :, None]
        linear = np.clip(linear + config.detail_gain * high, 0, 1)
    crop = linear[y0-ext[1]:y1-ext[1], x0-ext[0]:x1-ext[0]]
    return np.round(np.clip(_to_srgb(crop), 0, 1) * 65535).astype(np.uint16)


def _save_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_name(path.name + '.partial')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, separators=(',', ':'))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    try:
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _file_signature(cfg: MobileConfig, tw: int, th: int) -> str:
    values = {'source_sha256': _fingerprint(cfg.source),
              'focus_sha256': _fingerprint(cfg.focus_mask) if cfg.focus_mask else None,
              'inpaint_sha256': _fingerprint(cfg.inpainted_base) if cfg.inpainted_base else None,
              'sr_sha256': _fingerprint(cfg.sr_candidate) if cfg.sr_candidate else None,
              'wavelet_levels': cfg.wavelet_levels, 'wavelet_gain': cfg.wavelet_gain,
              'inpaint_mask_sha256': _fingerprint(cfg.inpaint_mask) if cfg.inpaint_mask else None,
              'protected_sha256': _fingerprint(cfg.protected_mask) if cfg.protected_mask else None,
              'inpaint_strength': cfg.inpaint_strength, 'lab_color_match': cfg.lab_color_match,
              'width': tw, 'height': th, 'tile_size': cfg.tile_size,
              'detail_gain': cfg.detail_gain, 'transfer': cfg.input_transfer,
              'psf_sigma': cfg.psf_sigma, 'rl_iterations': cfg.rl_iterations,
              'ca_red': cfg.ca_red, 'ca_blue': cfg.ca_blue,
              'contiguous_output':cfg.contiguous_output, 'compression':cfg.compression,
              'program': 'photo400-s24-2.3-cascade'}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode('utf-8')).hexdigest()


def _icc() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()


def _write_tiff(stage: np.memmap, path: Path, tile: int, compression: str = "none") -> None:
    h, w = stage.shape[:2]
    part = path.with_name(path.name + '.partial')
    if part.exists():
        part.unlink()

    def blocks():
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                a = np.zeros((tile, tile, 3), dtype=np.uint16)
                ymin, xmin = min(y+tile, h), min(x+tile, w)
                a[:ymin-y, :xmin-x] = stage[y:ymin, x:xmin]
                yield a
    try:
        with tifffile.TiffWriter(part, bigtiff=True) as t:
            t.write(data=blocks(), shape=(h, w, 3), dtype=np.uint16,
                    photometric='rgb', tile=(tile,tile),
                    compression='deflate' if compression == 'deflate' else None,
                    metadata=None, iccprofile=_icc())
        with tifffile.TiffFile(part) as f:
            p = f.pages[0]
            if not f.is_bigtiff or p.shape != (h,w,3) or p.dtype != np.dtype('uint16') or not p.is_tiled:
                raise RuntimeError('Final BigTIFF validation failed')
        os.replace(part, path)
    finally:
        if part.exists():
            part.unlink()


def _write_tiff_contiguous(stage: np.memmap, path: Path) -> None:
    """Output memmappable intermediate RGB16 BigTIFF, without a full RAM array."""
    h,w=stage.shape[:2]
    part=path.with_name(path.name+'.partial')
    if part.exists():
        raise FileExistsError('Unexpected unfinished contiguous BigTIFF')
    dst=None
    try:
        dst=tifffile.memmap(part,shape=(h,w,3),dtype=np.uint16,
                            bigtiff=True,photometric='rgb',metadata=None,
                            iccprofile=_icc())
        for y0 in range(0,h,128):
            dst[y0:min(y0+128,h)]=stage[y0:min(y0+128,h)]
        dst.flush();del dst;dst=None
        with tifffile.TiffFile(part) as f:
            p=f.pages[0]
            if not f.is_bigtiff or p.is_tiled or p.shape!=(h,w,3) or p.dtype!=np.dtype('uint16'):
                raise RuntimeError('Invalid contiguous RGB16 BigTIFF')
        os.replace(part,path)
    finally:
        if dst is not None:
            dst.flush();del dst
        part.unlink(missing_ok=True)


def run(cfg: MobileConfig) -> dict[str, Any]:
    """Safely restartable CPU/SSD rendering; max working frame = tile+24 pixels."""
    if not cfg.source.is_file():
        raise FileNotFoundError(cfg.source)
    if cfg.output.is_file():
        raise FileExistsError('Refusing to overwrite existing result')
    t0 = time.monotonic()
    source: Source | None = None
    focus: Image.Image | None = None
    candidate: Source | None = None
    sr_source: Source | None = None
    inpaint: Image.Image | None = None
    protected: Image.Image | None = None
    data: np.memmap | None = None
    try:
        # Reserve the configured process decode RAM across *both* PNG/JPEG sources.
        decoded_inputs = sum(Path(p).suffix.lower() not in ('.tif','.tiff')
                             for p in (cfg.source, cfg.inpainted_base, cfg.sr_candidate) if p is not None)
        per_decode_budget = max(1, cfg.max_decoded_ram_mb // max(1, decoded_inputs))
        source = Source(cfg.source, cfg.max_input_pixels, per_decode_budget)
        if cfg.inpainted_base is not None:
            if not cfg.inpainted_base.is_file():
                raise FileNotFoundError(cfg.inpainted_base)
            candidate = Source(cfg.inpainted_base,cfg.max_input_pixels,per_decode_budget)
            if (source.width,source.height)!=(candidate.width,candidate.height):
                raise ValueError('Inpainted base must have the same source dimensions')
            if not cfg.inpaint_mask.is_file():
                raise FileNotFoundError(cfg.inpaint_mask)
            inpaint=Image.open(cfg.inpaint_mask)
            if inpaint.mode!='L' or inpaint.width*inpaint.height>16_000_000:
                raise ValueError('Inpaint mask must be grayscale L and <=16 MP')
            inpaint.load()
            if cfg.protected_mask is not None:
                if not cfg.protected_mask.is_file():
                    raise FileNotFoundError(cfg.protected_mask)
                protected=Image.open(cfg.protected_mask)
                if protected.mode!='L' or protected.width*protected.height>16_000_000:
                    raise ValueError('Protected mask must be grayscale L and <=16 MP')
                protected.load()
        tw, th = ((cfg.target_width, cfg.target_height)
                  if cfg.target_width is not None else target_from_mp(source.width, source.height, cfg.target_mp))
        if tw*th > 1_000_000_000 or tw*th < 1:
            raise ValueError('Invalid or oversized requested output')
        if cfg.sr_candidate is not None:
            if not cfg.sr_candidate.is_file():
                raise FileNotFoundError(cfg.sr_candidate)
            # Uncompressed/memmappable RGB TIFF is read by ROI: allow a 400MP
            # target-sized SR candidate while keeping compressed inputs capped.
            sr_limit = max(cfg.max_input_pixels, tw*th) if cfg.sr_candidate.suffix.lower() in ('.tif','.tiff') else cfg.max_input_pixels
            sr_source = Source(cfg.sr_candidate, sr_limit, per_decode_budget)
            if ((sr_source.width,sr_source.height) != (source.width,source.height) and
                (sr_source.width,sr_source.height) != (tw,th)):
                raise ValueError('SR candidate must match source or output dimensions')
        if cfg.focus_mask is not None:
            if not cfg.focus_mask.is_file():
                raise FileNotFoundError(cfg.focus_mask)
            focus = Image.open(cfg.focus_mask)
            if focus.mode != 'L' or focus.width*focus.height>16_000_000:
                raise ValueError('Focus mask must be grayscale L and <= 16 MP')
            focus.load()
        cfg.stage_dir.mkdir(parents=True, exist_ok=True)
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        buffer = cfg.stage_dir / 'frame.rgb16.raw'
        checkpoint = cfg.stage_dir / 'checkpoint.json'
        signature = _file_signature(cfg, tw, th)
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text(encoding='utf-8'))
            if state.get('signature') != signature or state.get('dimensions') != [tw,th]:
                raise ValueError('Resume signature does not match source or configuration')
            if not buffer.is_file() or buffer.stat().st_size != tw*th*6:
                raise ValueError('Incomplete or corrupted raw stage buffer')
            completed = state['next_row']
            if type(completed) is not int or not (0 <= completed <= math.ceil(th/cfg.tile_size)):
                raise ValueError('Invalid checkpoint progress')
            data = np.memmap(buffer, mode='r+', shape=(th,tw,3), dtype=np.uint16)
        else:
            if buffer.exists() or any(cfg.stage_dir.iterdir()):
                raise FileExistsError('Nonempty work directory without matching checkpoint; refusing to overwrite')
            completed = 0
        # Reserve enough space for remaining staging + final TIFF + unallocated filesystem headroom.
        new_stage_bytes = 0 if data is not None else tw*th*6
        padded_w = math.ceil(tw/cfg.tile_size)*cfg.tile_size
        padded_h = math.ceil(th/cfg.tile_size)*cfg.tile_size
        output_bytes = padded_w*padded_h*6 + 16*1024*1024
        reserve = int(cfg.reserve_free_gb*1024**3)
        stage_free = shutil.disk_usage(cfg.stage_dir).free
        output_free = shutil.disk_usage(cfg.output.parent).free
        if cfg.stage_dir.stat().st_dev == cfg.output.parent.stat().st_dev:
            required = new_stage_bytes + output_bytes + reserve
            if stage_free < required:
                raise OSError(f'Not enough shared filesystem storage: need {required/2**30:.2f} GiB, '
                              f'available {stage_free/2**30:.2f} GiB')
        elif stage_free < new_stage_bytes+reserve or output_free < output_bytes+reserve:
            raise OSError('Insufficient storage on stage or output filesystem')
        if data is None:
            data = np.memmap(buffer, mode='w+', shape=(th,tw,3), dtype=np.uint16)
            _save_json_atomic(checkpoint, {'signature':signature, 'dimensions':[tw,th], 'next_row':0})
        for row in range(completed, math.ceil(th / cfg.tile_size)):
            y0, y1 = row*cfg.tile_size, min(th,(row+1)*cfg.tile_size)
            for x0 in range(0,tw,cfg.tile_size):
                x1 = min(x0+cfg.tile_size,tw)
                overlay = ((candidate, inpaint, protected)
                           if candidate is not None and inpaint is not None else None)
                tile = _render_output_tile(source,cfg,(x0,y0,x1,y1),(th,tw),
                                           focus,overlay,sr_source)
                data[y0:y1,x0:x1] = tile
            data.flush()
            _save_json_atomic(checkpoint, {'signature':signature,'dimensions':[tw,th],'next_row':row+1})
            print(f'Photo400 S24: {min(y1,th)}/{th} rows', file=sys.stderr, flush=True)
        if cfg.contiguous_output:
            _write_tiff_contiguous(data, cfg.output)
        else:
            _write_tiff(data, cfg.output, cfg.tile_size, cfg.compression)
        report = {'source_dimensions':[source.width,source.height],
                  'target_dimensions':[tw,th], 'pixels':tw*th,
                  'backend':'cpu_numpy_pillow_tifffile',
                  'processing':'linear-light Lanczos + optional masked high-pass',
                  'generative_restoration':candidate is not None,
                  'inpaint_hypothesis':candidate is not None,
                  'wavelet_sr_candidate':sr_source is not None,
                  'wavelet_levels':cfg.wavelet_levels if sr_source is not None else None,
                  'identity_verified':False,
                  'inpaint_strength':cfg.inpaint_strength if candidate is not None else 0.,
                  'resumed_from_row':completed,
                  'output':str(cfg.output), 'contiguous_output':cfg.contiguous_output,
                  'compression':cfg.compression,
                  'size_bytes':cfg.output.stat().st_size,
                  'elapsed_seconds':round(time.monotonic()-t0,2)}
        if not cfg.keep_stage:
            del data
            data = None
            buffer.unlink(missing_ok=True)
            checkpoint.unlink(missing_ok=True)
        return report
    finally:
        if source is not None:
            source.close()
        if focus is not None:
            focus.close()
        if inpaint is not None:
            inpaint.close()
        if protected is not None:
            protected.close()
        if candidate is not None:
            candidate.close()
        if sr_source is not None:
            sr_source.close()
        if data is not None:
            data.flush()


def main() -> int:
    p=argparse.ArgumentParser(description='Photo400 / S24 Ultra: local CPU-only, disk-backed BigTIFF renderer')
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--workdir',required=True,type=Path)
    p.add_argument('--target-mp',type=float,default=400.0)
    p.add_argument('--width',type=int)
    p.add_argument('--height',type=int)
    p.add_argument('--tile',type=int,default=256)
    p.add_argument('--detail-gain',type=float,default=0.0)
    p.add_argument('--psf-sigma',type=float,default=0.0)
    p.add_argument('--rl-iterations',type=int,default=0)
    p.add_argument('--ca-red',type=float,default=0.0)
    p.add_argument('--ca-blue',type=float,default=0.0)
    p.add_argument('--focus-mask',type=Path)
    p.add_argument('--inpainted-base',type=Path,help='Registered hypothesis at original image size; not verifiable original detail')
    p.add_argument('--inpaint-mask',type=Path,help='Explicit grayscale L damage mask (<=16 MP)')
    p.add_argument('--sr-candidate',type=Path,help='Registered SR hypothesis as RGB TIFF/PNG')
    p.add_argument('--wavelet-levels',type=int,default=3)
    p.add_argument('--wavelet-gain',type=float,default=1.0)
    p.add_argument('--protected-mask',type=Path,help='Regions never replaced by the hypothesis')
    p.add_argument('--inpaint-strength',type=float,default=1.0)
    p.add_argument('--lab-color-match',type=float,default=0.0)
    p.add_argument('--input-transfer',choices=['srgb','linear'],default='srgb')
    p.add_argument('--max-source-pixels',type=int,default=225_000_000)
    p.add_argument('--max-decoded-mb',type=int,default=900)
    p.add_argument('--reserve-free-gb',type=float,default=1.0)
    p.add_argument('--keep-stage',action='store_true')
    p.add_argument('--compression',choices=['none','deflate'],default='none')
    p.add_argument('--contiguous-output',action='store_true',
                   help='Memmappable BigTIFF for an intermediate cascade stage')
    args=p.parse_args()
    cfg=MobileConfig(args.input,args.output,args.workdir,
                     target_width=args.width,target_height=args.height,
                     target_mp=args.target_mp,tile_size=args.tile,detail_gain=args.detail_gain,
                     psf_sigma=args.psf_sigma,rl_iterations=args.rl_iterations,
                     ca_red=args.ca_red,ca_blue=args.ca_blue,
                     focus_mask=args.focus_mask,inpainted_base=args.inpainted_base,
                     sr_candidate=args.sr_candidate,wavelet_levels=args.wavelet_levels,
                     wavelet_gain=args.wavelet_gain,
                     inpaint_mask=args.inpaint_mask,protected_mask=args.protected_mask,
                     inpaint_strength=args.inpaint_strength,lab_color_match=args.lab_color_match,
                     input_transfer=args.input_transfer,
                     max_input_pixels=args.max_source_pixels,
                     max_decoded_ram_mb=args.max_decoded_mb,
                     reserve_free_gb=args.reserve_free_gb,keep_stage=args.keep_stage,
                     contiguous_output=args.contiguous_output,compression=args.compression)
    print(json.dumps(run(cfg),indent=2,ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())