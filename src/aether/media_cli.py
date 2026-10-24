"""Mobile first DNG/RAW and video workflows. No GUI or CUDA required."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def main(argv=None)->int:
    parser=argparse.ArgumentParser(prog='aether-media',description='Photo RAW and lossless video frame workflows')
    sub=parser.add_subparsers(dest='command',required=True)
    dng=sub.add_parser('dng',help='Stream supported uncompressed CFA16 DNG')
    dng.add_argument('--source',type=Path,required=True)
    dng.add_argument('--output',type=Path,required=True)
    dng.add_argument('--tile',type=int,default=256)
    dng.add_argument('--max-pixels',type=int,default=250_000_000)
    lib=sub.add_parser('libraw',help='Full-frame optional LibRaw (bounded pixels/RAM)')
    lib.add_argument('--source',type=Path,required=True)
    lib.add_argument('--output',type=Path,required=True)
    lib.add_argument('--max-pixels',type=int,default=40_000_000)
    lib.add_argument('--max-decoded-mb',type=int,default=900)
    extr=sub.add_parser('video-extract',help='Stream frames as lossless RGB16 TIFF')
    extr.add_argument('--source',type=Path,required=True)
    extr.add_argument('--out-dir',type=Path,required=True)
    extr.add_argument('--max-frames',type=int,default=3600)
    proc=sub.add_parser('video-render',help='Resume framewise Aether RGB16 BigTIFF')
    proc.add_argument('--frames-dir',type=Path,required=True)
    proc.add_argument('--out-dir',type=Path,required=True)
    proc.add_argument('--width',type=int,required=True)
    proc.add_argument('--height',type=int,required=True)
    proc.add_argument('--tile',type=int,default=256)
    mux=sub.add_parser('video-mux',help='Create lossless MKV FFV1 with source audio')
    mux.add_argument('--frames-dir',type=Path,required=True)
    mux.add_argument('--source',type=Path,required=True)
    mux.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='dng':
        from .ingest.dng import decode_uncompressed_dng
        result=decode_uncompressed_dng(args.source,args.output,
            tile_size=args.tile,max_pixels=args.max_pixels)
    elif args.command=='libraw':
        from .ingest.libraw_adapter import decode_libraw_safe
        result=decode_libraw_safe(args.source,args.output,
            max_source_pixels=args.max_pixels,max_decoded_mb=args.max_decoded_mb)
    elif args.command=='video-extract':
        from .video.frame_pipeline import extract_frames_rgb16
        result=extract_frames_rgb16(args.source,args.out_dir,max_frames=args.max_frames)
    elif args.command=='video-render':
        from .video.frame_render import process_video_frames
        result=process_video_frames(args.frames_dir,args.out_dir,
            width=args.width,height=args.height,tile=args.tile)
    else:
        from .video.frame_pipeline import mux_frames_lossless
        result=mux_frames_lossless(args.frames_dir,args.source,args.output)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
