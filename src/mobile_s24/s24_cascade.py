"""Memory-bounded S24 Ultra spatial cascade (optional local 25/100/400 MP).

This is *not* diffusion. Intermediate BigTIFF pages are contiguous and can
be memory-mapped; the final 400 MP output is tiled. Because interpolating
repeatedly can degrade fidelity, direct 400 MP is normally preferred unless
external neural passes are injected between stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tifffile

from .s24_render import MobileConfig, Source, run, target_from_mp, _fingerprint


def plan_stages(width:int,height:int,target_mp:float=400.,
                *,levels:Sequence[float]=(25.,100.,400.))->list[float]:
    if width<=0 or height<=0 or not math.isfinite(target_mp) or target_mp<=0:
        raise ValueError('Invalid cascade geometry')
    if not levels or not all(math.isfinite(v) and v>0 for v in levels) or any(a>=b for a,b in zip(levels,levels[1:])):
        raise ValueError('Stages must be strictly increasing')
    src_mp=width*height/1_000_000.
    plan=[float(x) for x in levels if src_mp<x<=target_mp]
    if not plan or plan[-1]!=target_mp:
        plan.append(float(target_mp))
    return plan


def _atomic_json(file:Path,obj:dict)->None:
    partial=file.with_name(file.name+'.partial')
    try:
        with partial.open('w',encoding='utf-8') as f:
            json.dump(obj,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(partial,file)
    finally:
        partial.unlink(missing_ok=True)


def run_cascade(source:Path,output:Path,workdir:Path,
                *,levels:Sequence[float]=(25.,100.,400.),tile:int=256,
                reserve_free_gb:float=1., max_source_pixels:int=225_000_000,
                detail_gain:float=0.)->dict[str,Any]:
    """Execute or resume local spatial stages without ever loading 400 MP to RAM."""
    source,output,workdir=Path(source),Path(output),Path(workdir)
    if output.resolve()==source.resolve() or workdir.resolve()==source.resolve():
        raise ValueError('Output, staging and input must be different')
    src=Source(source,max_source_pixels,900)
    try:
        plan=plan_stages(src.width,src.height,float(levels[-1]),levels=levels)
    finally:src.close()
    workdir.mkdir(parents=True,exist_ok=True)
    executed=reused=0
    srcfile=source
    for i,mp in enumerate(plan):
        last=i==len(plan)-1
        dest=output if last else workdir/f'level_{i:02d}_{mp:g}mp.contiguous.tif'
        state=workdir/f'level_{i:02d}.json'
        sh=_fingerprint(srcfile)
        record={'source_sha256':sh,'source':str(srcfile.resolve()),
                'target_mp':mp,'tile':tile,'detail_gain':detail_gain,
                'contiguous_output':not last}
        if dest.exists():
            if not state.exists():
                raise FileExistsError(f'Untracked cascade output {dest}')
            old=json.loads(state.read_text())
            if not all(old.get(key)==value for key,value in record.items()):
                raise ValueError('Cascade checkpoint signature mismatch')
            if _fingerprint(dest)!=old.get('output_sha256'):
                raise ValueError('Cascade checkpoint output SHA256 mismatch')
            with tifffile.TiffFile(dest) as f:
                page=f.pages[0]
                if not f.is_bigtiff or page.dtype!=np.dtype('uint16') or page.is_tiled!=last:
                    raise ValueError('Corrupt or incompatible intermediate BigTIFF')
            reused+=1
        else:
            cfg=MobileConfig(srcfile,dest,workdir/f'stage_{i:02d}',
                             target_mp=mp,tile_size=tile,
                             reserve_free_gb=reserve_free_gb,
                             detail_gain=detail_gain,
                             max_input_pixels=max_source_pixels,
                             contiguous_output=not last)
            run(cfg)
            record['output_sha256']=_fingerprint(dest)
            _atomic_json(state,record)
            executed+=1
        srcfile=dest
    return {'output':str(output),'stages':plan,
            'stages_executed':executed,'stages_reused':reused,
            'generative_models':False}


def main()->int:
    p=argparse.ArgumentParser(description='Galaxy S24 Ultra out-of-core spatial 25/100/400 MP cascade')
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--tile',type=int,default=256)
    p.add_argument('--reserve-free-gb',type=float,default=1.)
    p.add_argument('--target-mp',type=float,default=400.)
    p.add_argument('--execute',action='store_true',help='Run renders; default only prints stage plan')
    a=p.parse_args()
    if a.execute:
        levels=tuple(x for x in (25.,100.) if x<a.target_mp)+(a.target_mp,)
        report=run_cascade(a.source,a.output,a.workdir,levels=levels,
                           tile=a.tile,reserve_free_gb=a.reserve_free_gb)
    else:
        src=Source(a.source,225_000_000,900)
        try:
            levels=tuple(x for x in (25.,100.) if x<a.target_mp)+(a.target_mp,)
            report={'stages':plan_stages(src.width,src.height,a.target_mp,levels=levels),
                    'execute':False, 'input_dimensions':[src.width,src.height]}
        finally:src.close()
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())