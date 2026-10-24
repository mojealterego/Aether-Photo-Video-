"""Strict RGB16 ICC ingest followed optionally by offline S24 BigTIFF rendering."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .icc16 import rgb16_tiff_to_srgb

def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(prog='aether-ingest',description='RGB16 ICC conversion; optional 400MP S24 CPU render')
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--normalized',type=Path,required=True,help='Mapped RGB16 sRGB master')
    p.add_argument('--assume-srgb',action='store_true',help='Explicit trust for untagged TIFF')
    p.add_argument('--stripe-rows',type=int,default=64)
    p.add_argument('--render',type=Path,help='Optional final RGB16 BigTIFF')
    p.add_argument('--workdir',type=Path)
    p.add_argument('--target-mp',type=float,default=400.)
    p.add_argument('--tile',type=int,default=256)
    p.add_argument('--reserve-free-gb',type=float,default=1.)
    args=p.parse_args(argv)
    if args.render is not None and args.workdir is None:
        p.error('--render requires --workdir')
    result=rgb16_tiff_to_srgb(args.source,args.normalized,rows=args.stripe_rows,
                               assume_srgb=args.assume_srgb)
    if args.render is not None:
        from mobile_s24.s24_render import MobileConfig,run
        cfg=MobileConfig(args.normalized,args.render,args.workdir,
                         target_mp=args.target_mp,tile_size=args.tile,
                         reserve_free_gb=args.reserve_free_gb)
        result['render']=run(cfg)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
