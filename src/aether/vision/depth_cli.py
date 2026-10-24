"""Offline small-preview Depth Anything ONNX -> L focus mask CLI.

Only works with an externally provided compatible ONNX checkpoint. Input
must already be a small RGB8 preview, never the full 200/400 MP original.
Output L PNG may be supplied to `aether restore --focus-mask`.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from .depth_onnx import DepthAnyOnnxAdapter, relative_depth_focus_mask

def build_depth_mask(preview_rgb:np.ndarray,session,*,focus_depth:float=.5,
                     tolerance:float=.15,near_is_higher:bool=True)->np.ndarray:
    model=DepthAnyOnnxAdapter(session,near_is_higher=near_is_higher)
    relative=model.predict(preview_rgb)
    mask=relative_depth_focus_mask(preview_rgb,relative,
                                   focus_depth=focus_depth,tolerance=tolerance)
    return np.rint(mask*255.).astype(np.uint8)

def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description='Aether S24: relative-depth ONNX -> focus mask')
    p.add_argument('--preview',type=Path,required=True,help='RGB8 preview <= 1MP')
    p.add_argument('--model',type=Path,required=True,help='Compatible Depth Anything ONNX file')
    p.add_argument('--output',type=Path,required=True,help='Grayscale PNG for --focus-mask')
    p.add_argument('--focus-depth',type=float,default=.5)
    p.add_argument('--tolerance',type=float,default=.15)
    p.add_argument('--near-is-lower',action='store_true')
    args=p.parse_args(argv)
    if not args.model.is_file() or args.model.suffix.lower()!='.onnx':
        p.error('Provide an existing compatible .onnx file')
    with Image.open(args.preview) as im:
        if im.width*im.height>1_000_000 or im.mode!='RGB':
            p.error('Preview must be RGB8 and <= 1MP')
        rgb=np.asarray(im,dtype=np.float32)/255.
    try:
        import onnxruntime as ort
    except ImportError:
        p.error('onnxruntime unavailable: install an Android-compatible CPU backend')
    session=ort.InferenceSession(str(args.model),providers=['CPUExecutionProvider'])
    mask=build_depth_mask(rgb,session,focus_depth=args.focus_depth,
                          tolerance=args.tolerance,near_is_higher=not args.near_is_lower)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(mask,mode='L').save(args.output)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
