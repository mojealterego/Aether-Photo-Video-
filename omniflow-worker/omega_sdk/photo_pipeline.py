"""Executable first-stage native NEF -> linear processing -> ICC16 BigTIFF.

Output keeps input resolution; 400MP reconstruction, chart color accuracy,
GPU models and physical-camera acceptance are not asserted by this command.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import tempfile
import uuid
from typing import Any
import numpy as np
from .photo_faithful_400mp import I002_LoadNEF, DiskBackedStitcher
from .protocol import ExecutionContext


def _tile_starts(length:int,tile:int,overlap:int)->list[int]:
    if length<=tile:return [0]
    starts=list(range(0,length-tile+1,tile-overlap))
    if starts[-1]!=length-tile:starts.append(length-tile)
    return starts


def run_native_nef_to_bigtiff(
    source:Path,output:Path,*,tile_size:int=512,overlap:int=128,
    reserve_bytes:int=1<<30,max_source_pixels:int=40_000_000
)->dict[str,Any]:
    source,output=Path(source),Path(output)
    if output.exists() or output.resolve()==source.resolve():
        raise FileExistsError("Source must never be replaced and target must not exist")
    if tile_size<16 or tile_size%16 or not 0<overlap<tile_size:
        raise ValueError("Invalid tile overlap or TIFF tile size")
    output.parent.mkdir(parents=True,exist_ok=True)
    job_id="job_"+uuid.uuid4().hex
    with tempfile.TemporaryDirectory(dir=output.parent,prefix=".omniflow-") as temp:
        working=Path(temp)
        ctx=ExecutionContext(job_id,"local_photo",1,"DOC",0,working,working)
        decoded=I002_LoadNEF().execute(
            ctx,{"nef_path":str(source)},{"max_source_pixels":max_source_pixels})
        rgb=decoded["linear_image_f32"]
        h,w=rgb.shape[:2]
        with DiskBackedStitcher(
            h,w,working/"tiles",reserve_bytes=reserve_bytes
        ) as compositor:
            for y in _tile_starts(h,tile_size,overlap):
                for x in _tile_starts(w,tile_size,overlap):
                    compositor.add_tile(y,x,np.asarray(
                        rgb[y:min(y+tile_size,h),x:min(x+tile_size,w)],
                        dtype=np.float32))
            rendered=compositor.write_srgb16_bigtiff(
                output,tile_size=tile_size,
                source_sha256=decoded["source_nef_sha256"])
        del rgb
    return {**rendered,
            "source_nef_sha256":decoded["source_nef_sha256"],
            "status":"MASTER_WRITTEN_UNVERIFIED",
            "source_preserved":True,
            "neural_models_executed":False,
            "full_400mp_acceptance":False}


def main()->int:
    parser=argparse.ArgumentParser(
        description="AETHERIS native Nikon NEF -> RGB16 BigTIFF")
    parser.add_argument("--nef",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--tile",type=int,default=512)
    parser.add_argument("--overlap",type=int,default=128)
    args=parser.parse_args()
    import json
    print(json.dumps(run_native_nef_to_bigtiff(
        args.nef,args.output,tile_size=args.tile,
        overlap=args.overlap),indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
