"""Android/S24 bridge: bounded ComfyUI tiles and loss-minimized source-space master.

The ComfyUI client already exists in `photo400.comfy` (server API). This module
only prepares its manifest and assembles *external hypotheses*, one at a time.
No face model or CUDA is run locally. Candidate PNGs are necessarily display RGB8.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
from PIL import Image
import tifffile

from .face_occlusion import masked_linear_fusion
from .s24_render import Source, _focus_source_roi, _fingerprint, _icc, _to_linear, _to_srgb


_NAME = re.compile(r'^(reference|focus|restored)_\d+_\d+_\d+_\d+\.png$')


def _mask_file(path: Path) -> Image.Image:
    img=Image.open(path)
    if img.mode!='L' or img.width*img.height>16_000_000:
        img.close()
        raise ValueError('Mask must be grayscale L, <=16 MP')
    img.load()
    return img


def _uint16(raw: np.ndarray) -> np.ndarray:
    if raw.dtype==np.uint8:
        return raw.astype(np.uint16)*np.uint16(257)
    if raw.dtype==np.uint16:
        return raw.copy()
    raise ValueError('Neural bridge supports only sRGB RGB8/RGB16 originals')


def _safe_name(name: Any) -> str:
    if not isinstance(name,str) or not _NAME.fullmatch(name):
        raise ValueError('Unsafe manifest filename')
    return name


def _json_atomic(path:Path,record:dict)->None:
    tmp=path.with_suffix(path.suffix+'.partial')
    try:
        tmp.write_text(json.dumps(record,indent=2,sort_keys=True),encoding='utf-8')
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def prepare_neural_tiles(source:Path,mask_file:Path,out_dir:Path,
                         *,tile:int=512,halo:int=48,max_source_pixels:int=225_000_000,
                         region_file:Path|None=None)->dict[str,Any]:
    """Create ComfyUI reference RGB and LoadImage alpha masks, only for masked cores."""
    source,mask_file,out_dir = Path(source),Path(mask_file),Path(out_dir)
    if tile<32 or tile>1024 or tile%16!=0 or halo<0 or halo>128:
        raise ValueError('Bad tile/halo bounds')
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError('Output tile dir must be empty')
    s=Source(source,max_source_pixels,900)
    m=_mask_file(mask_file)
    region=_mask_file(Path(region_file)) if region_file else None
    try:
        if s.pixel_dtype not in (np.dtype('uint8'),np.dtype('uint16')):
            raise ValueError('Neural bridge supports only sRGB RGB8/RGB16 originals')
        out_dir.mkdir(parents=True,exist_ok=True)
        jobs=[]
        H,W=s.height,s.width
        for y0 in range(0,H,tile):
            for x0 in range(0,W,tile):
                y1=min(H,y0+tile);x1=min(W,x0+tile)
                coremask=_focus_source_roi(m,(x0,y0,x1,y1),(H,W))
                if region is not None:
                    coremask*=(_focus_source_roi(region,(x0,y0,x1,y1),(H,W))>.5)
                if coremask.max()<.5/255.:continue
                ey0=max(0,y0-halo);ey1=min(H,y1+halo)
                ex0=max(0,x0-halo);ex1=min(W,x1+halo)
                # Filenames key by *core*, not by halo-expanded extent:
                # multiple cores can share the same extent for small inputs.
                suffix=f'{y0}_{y1}_{x0}_{x1}.png'
                ref='reference_'+suffix
                foc='focus_'+suffix
                ret='restored_'+suffix
                patch=s.crop((ex0,ey0,ex1,ey1))
                u8=(patch if patch.dtype==np.uint8 else
                    np.rint(patch.astype('float32')/257.).astype(np.uint8))
                Image.fromarray(u8,'RGB').save(out_dir/ref)
                imask=_focus_source_roi(m,(ex0,ey0,ex1,ey1),(H,W))
                if region is not None:
                    imask*=(_focus_source_roi(region,(ex0,ey0,ex1,ey1),(H,W))>.5)
                # Comfy LoadImage MASK = 1 - normalized alpha (RGB channels unused).
                rgba=np.zeros((ey1-ey0,ex1-ex0,4),dtype=np.uint8)
                rgba[:,:,:3]=255
                rgba[:,:,3]=255-np.rint(imask*255).astype(np.uint8)
                Image.fromarray(rgba,'RGBA').save(out_dir/foc)
                jobs.append({'extent':[ey0,ey1,ex0,ex1],
                             'core':[y0,y1,x0,x1],
                             'reference':ref,'focus':foc,
                             'protection':None,'restored':ret,
                             'sha256_reference':_fingerprint(out_dir/ref),
                             'sha256_focus':_fingerprint(out_dir/foc)})
        manifest={'version':'2.0.0','purpose':'s24_source_space_occlusion_inpaint',
                  'source_sha256':_fingerprint(source),
                  'mask_sha256':_fingerprint(mask_file),
                  'region_sha256':_fingerprint(Path(region_file)) if region_file else None,
                  'source_size':[W,H], 'tile':tile,'halo':halo,'tiles':jobs,
                  'output':'External hypotheses, not recovered historical pixels'}
        _json_atomic(out_dir/'manifest.json',manifest)
        return {'selected_tiles':len(jobs),'source_size':[W,H],
                'manifest':str(out_dir/'manifest.json')}
    finally:
        s.close();m.close()
        if region is not None:region.close()


def stitch_neural_tiles(source:Path,mask_file:Path,manifest_file:Path,
                        candidates:Path,output:Path,*,strength:float=1.,
                        color_match:float=0.,protected_file:Path|None=None,
                        max_source_pixels:int=225_000_000,
                        region_file:Path|None=None)->dict[str,Any]:
    """Assemble external hypotheses to contiguous mmap RGB16 BigTIFF (at *source* size).

    This output is the candidate `--inpainted-base` for s24_render. For accurate
    final masking, keep passing the same `--inpaint-mask` to the final renderer.
    """
    source,mask_file,manifest_file,candidates,output=map(Path,
        (source,mask_file,manifest_file,candidates,output))
    if output.exists():raise FileExistsError(output)
    if not (math.isfinite(strength) and 0<=strength<=1 and
            math.isfinite(color_match) and 0<=color_match<=1):
        raise ValueError('Invalid strengths')
    if manifest_file.stat().st_size>16*1024*1024:
        raise ValueError('Manifest too large')
    doc=json.loads(manifest_file.read_text(encoding='utf-8'))
    if doc.get('version')!='2.0.0' or doc.get('purpose')!='s24_source_space_occlusion_inpaint':
        raise ValueError('Not an S24 inpaint manifest')
    if doc.get('source_sha256')!=_fingerprint(source) or doc.get('mask_sha256')!=_fingerprint(mask_file):
        raise ValueError('Source/mask fingerprint mismatch')
    if bool(doc.get('region_sha256')) != bool(region_file):
        raise ValueError('Region mask required iff manifest records a region')
    if region_file is not None and _fingerprint(Path(region_file))!=doc['region_sha256']:
        raise ValueError('Region mask fingerprint mismatch')
    if protected_file is not None and not protected_file.is_file():
        raise FileNotFoundError(protected_file)
    marker=candidates/'manifest.json'
    if not marker.exists() or marker.read_bytes()!=manifest_file.read_bytes():
        raise ValueError('Candidates must contain exact completed manifest marker')
    s=Source(source,max_source_pixels,900)
    m=_mask_file(mask_file)
    p=_mask_file(protected_file) if protected_file else None
    region=_mask_file(Path(region_file)) if region_file else None
    temp=output.with_name(output.name+'.partial.tif')
    stage=None
    try:
        H,W=s.height,s.width
        if doc['source_size']!=[W,H] or s.pixel_dtype not in (np.dtype('uint8'),np.dtype('uint16')):
            raise ValueError('Source dimensions or dtype changed')
        jobs=doc['tiles']
        if not isinstance(jobs,list) or len(jobs)>math.ceil(H/32)*math.ceil(W/32):
            raise ValueError('Invalid tile count')
        seen=set()
        for job in jobs:
            ey0,ey1,ex0,ex1=job['extent'];y0,y1,x0,x1=job['core']
            if not(0<=ey0<=y0<y1<=ey1<=H and 0<=ex0<=x0<x1<=ex1<=W):
                raise ValueError('Invalid tile geometry')
            key=(y0,y1,x0,x1)
            if key in seen:raise ValueError('Duplicate tile core')
            seen.add(key)
            for field in ('reference','focus','restored'):_safe_name(job[field])
            ref=manifest_file.parent/job['reference']; foc=manifest_file.parent/job['focus']
            if _fingerprint(ref)!=job['sha256_reference'] or _fingerprint(foc)!=job['sha256_focus']:
                raise ValueError('Prepared tile fingerprint mismatch')
            file=candidates/job['restored']
            if file.stat().st_size>48*1024*1024:
                raise ValueError('Candidate tile PNG exceeds 48 MiB bound')
            with Image.open(file) as img:
                if img.format!='PNG' or img.mode!='RGB' or img.size!=(ex1-ex0,ey1-ey0):
                    raise ValueError('Wrong candidate dimensions/format')
                img.verify()
        output.parent.mkdir(parents=True,exist_ok=True)
        need=H*W*6+256*1024*1024
        if shutil.disk_usage(output.parent).free<need:
            raise OSError('Insufficient disk space for source-resolution candidate master')
        if temp.exists():raise FileExistsError(temp)
        stage=tifffile.memmap(temp,shape=(H,W,3),dtype=np.uint16,
                             photometric='rgb',bigtiff=True,metadata=None,iccprofile=_icc())
        strip=128
        for y0 in range(0,H,strip):
            y1=min(H,y0+strip)
            stage[y0:y1]=_uint16(s.crop((0,y0,W,y1)))
        stage.flush()
        for job in jobs:
            ey0,ey1,ex0,ex1=job['extent'];y0,y1,x0,x1=job['core']
            dy0,dy1=y0-ey0,y1-ey0;dx0,dx1=x0-ex0,x1-ex0
            with Image.open(candidates/job['restored']) as f:
                patch=np.asarray(f.crop((dx0,dy0,dx1,dy1)),dtype=np.uint8)
            raw=s.crop((x0,y0,x1,y1))
            origin=_uint16(raw).astype(np.float32)/65535.
            proposal=patch.astype(np.float32)/255.
            roi=_focus_source_roi(m,(x0,y0,x1,y1),(H,W))
            if region is not None:
                roi *= (_focus_source_roi(region,(x0,y0,x1,y1),(H,W))>.5)
            protect=_focus_source_roi(p,(x0,y0,x1,y1),(H,W)) if p is not None else None
            # LAB statistics are in linear light; untouched original pixels remain exactly 16-bit.
            linear=masked_linear_fusion(_to_linear(origin),_to_linear(proposal),roi,
                                        protected=protect,strength=strength,
                                        color_match_strength=color_match)
            blended=np.rint(_to_srgb(linear)*65535).astype(np.uint16)
            effective=roi if protect is None else roi*(1-protect)
            orig_u16=_uint16(raw)
            blended[effective==0]=orig_u16[effective==0]
            stage[y0:y1,x0:x1]=blended
        stage.flush()
        with tifffile.TiffFile(temp) as tf:
            pg=tf.pages[0]
            if not tf.is_bigtiff or pg.shape!=(H,W,3) or pg.dtype!=np.dtype('uint16'):
                raise RuntimeError('Bridge TIFF validation failed')
        del stage;stage=None
        os.replace(temp,output)
        return {'candidate_tiles':len(jobs),'output':str(output),
                'source_dimensions':[W,H],'identity_verified':False,
                'is_reconstruction_hypothesis':True}
    finally:
        if stage is not None:
            stage.flush()
            del stage
        temp.unlink(missing_ok=True)
        s.close();m.close()
        if p is not None:p.close()
        if region is not None:region.close()


def main()->int:
    parser=argparse.ArgumentParser(description='Photo400 S24: offline tile bridge to external ComfyUI')
    commands=parser.add_subparsers(dest='command',required=True)
    a=commands.add_parser('prepare')
    a.add_argument('--source',required=True,type=Path)
    a.add_argument('--mask',required=True,type=Path)
    a.add_argument('--out-dir',required=True,type=Path)
    a.add_argument('--tile',type=int,default=512)
    a.add_argument('--halo',type=int,default=48)
    a.add_argument('--region',type=Path)
    b=commands.add_parser('stitch')
    b.add_argument('--source',required=True,type=Path)
    b.add_argument('--mask',required=True,type=Path)
    b.add_argument('--manifest',required=True,type=Path)
    b.add_argument('--candidates',required=True,type=Path)
    b.add_argument('--output',required=True,type=Path)
    b.add_argument('--protected',type=Path)
    b.add_argument('--region',type=Path)
    b.add_argument('--strength',type=float,default=1.0)
    b.add_argument('--lab-color-match',type=float,default=0.)
    args=parser.parse_args()
    if args.command=='prepare':
        report=prepare_neural_tiles(args.source,args.mask,args.out_dir,tile=args.tile,
                                    halo=args.halo,region_file=args.region)
    else:
        report=stitch_neural_tiles(args.source,args.mask,args.manifest,args.candidates,
                                  args.output,strength=args.strength,
                                  color_match=args.lab_color_match,
                                  protected_file=args.protected,region_file=args.region)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())