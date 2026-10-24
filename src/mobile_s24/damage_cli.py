"""On-device <=2MP reviewable masks/Perona–Malik preview; does NOT invent face pixels."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .face_occlusion import anisotropic_deblock, propose_damage_mask
from .s24_render import Source


def _preview_from_source(source:Source,max_side:int)->np.ndarray:
    width,height=source.width,source.height
    factor=max(1,math.ceil(max(width,height)/max_side))
    if source.mapped is not None:
        # Sample without materializing full 200 MP input in RAM.
        sampled=np.asarray(source.mapped[::factor,::factor,:])
        if sampled.dtype==np.uint16:
            return np.rint(sampled.astype(np.float32)/257).astype(np.uint8)
        if sampled.dtype==np.float32:
            return np.rint(np.clip(sampled,0,1)*255).astype(np.uint8)
        return sampled.astype(np.uint8,copy=True)
    assert source.image is not None
    out=source.image.resize((math.ceil(width/factor),math.ceil(height/factor)),
                            Image.Resampling.BOX)
    return np.asarray(out,dtype=np.uint8)


def analyze_photo(source:Path,mask_out:Path,*,preview_out:Path|None=None,
                  deblocked_out:Path|None=None,region:Path|None=None,
                  max_side:int=1200,feather:float=2.0,iterations:int=10)->dict:
    source,mask_out=Path(source),Path(mask_out)
    if not (64<=max_side<=1400):
        raise ValueError('max_side must be 64..1400')
    targets=[p for p in (mask_out,preview_out,deblocked_out) if p is not None]
    if len({str(Path(p).resolve()) for p in targets})!=len(targets):
        raise ValueError('Output paths must be distinct')
    if any(Path(p).exists() for p in targets):
        raise FileExistsError('Refusing to overwrite prior mask or preview')
    if source.resolve() in {Path(p).resolve() for p in targets}:
        raise ValueError('Cannot overwrite source')
    s=Source(source,225_000_000,900)
    try:
        arr=_preview_from_source(s,max_side)
        roi=None
        if region is not None:
            with Image.open(region) as r:
                if r.mode!='L' or r.width*r.height>16_000_000:
                    raise ValueError('Region must be grayscale L <=16 MP')
                roi=np.asarray(r.resize((arr.shape[1],arr.shape[0]),
                                        Image.Resampling.BILINEAR),dtype=np.float32)/255
        mask=propose_damage_mask(arr,region=roi,feather=feather)
        m_u8=np.rint(mask*255).astype(np.uint8)
        planned=[(mask_out,Image.fromarray(m_u8,'L'))]
        if preview_out is not None:
            overlay=arr.copy()
            overlay[:,:,0]=np.clip(overlay[:,:,0].astype(np.float32)+mask*90,0,255).astype(np.uint8)
            overlay[:,:,1]=np.clip(overlay[:,:,1].astype(np.float32)*(1-mask*.25),0,255).astype(np.uint8)
            planned.append((Path(preview_out),Image.fromarray(overlay,'RGB')))
        if deblocked_out is not None:
            smooth=anisotropic_deblock(arr.astype(np.float32)/255.,mask,iterations=iterations)
            planned.append((Path(deblocked_out),Image.fromarray(
                np.rint(np.clip(smooth,0,1)*255).astype(np.uint8),'RGB')))
        created=[]
        try:
            for path,img in planned:
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix(path.suffix+'.partial')
                try:
                    img.save(tmp,format='PNG')
                    os.replace(tmp,path)
                    created.append(path)
                finally:
                    tmp.unlink(missing_ok=True)
        except BaseException:
            for path in created:path.unlink(missing_ok=True)
            raise
        return {'mask_size':[arr.shape[1],arr.shape[0]],
                'source_size':[s.width,s.height],
                'mask_path':str(mask_out),
                'suggested_fraction':round(float(np.mean(mask>.5)),5),
                'review_required':True,'authenticity_guarantee':False}
    finally:
        s.close()


def main()->int:
    p=argparse.ArgumentParser(description='S24: heuristic mask *proposal* (review before inpainting)')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--mask-out',type=Path,required=True)
    p.add_argument('--preview-out',type=Path)
    p.add_argument('--deblocked-out',type=Path)
    p.add_argument('--region',type=Path)
    p.add_argument('--max-side',type=int,default=1200)
    p.add_argument('--feather',type=float,default=2.)
    p.add_argument('--deblock-iterations',type=int,default=10)
    a=p.parse_args()
    report=analyze_photo(a.input,a.mask_out,preview_out=a.preview_out,
        deblocked_out=a.deblocked_out,region=a.region,max_side=a.max_side,
        feather=a.feather,iterations=a.deblock_iterations)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())