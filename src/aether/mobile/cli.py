"""Aether Photo & Video: planning and offline S24 BigTIFF rendering."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ..core.geometry import target_dimensions, iter_tiles, memory_estimate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='aether', description='Aether Photo & Video mobile-first tools')
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='Plan only; no image decoding or rendering')
    plan.add_argument('--width', type=int, required=True)
    plan.add_argument('--height', type=int, required=True)
    plan.add_argument('--target-mp', type=float, default=400.0)
    plan.add_argument('--tile', type=int, default=256)
    plan.add_argument('--halo', type=int, default=32)
    restore = sub.add_parser('restore', help='CPU-only tile renderer, RGB16 BigTIFF')
    restore.add_argument('--source', type=Path, required=True)
    restore.add_argument('--output', type=Path, required=True)
    restore.add_argument('--workdir', type=Path, required=True)
    restore.add_argument('--target-mp', type=float, default=400.0)
    restore.add_argument('--width', type=int)
    restore.add_argument('--height', type=int)
    restore.add_argument('--tile', type=int, default=256)
    restore.add_argument('--detail-gain', type=float, default=0.0)
    restore.add_argument('--psf-sigma', type=float, default=0.0)
    restore.add_argument('--rl-iterations', type=int, default=0)
    restore.add_argument('--ca-red', type=float, default=0.0)
    restore.add_argument('--ca-blue', type=float, default=0.0)
    restore.add_argument('--focus-mask', type=Path)
    restore.add_argument('--inpainted-base', type=Path)
    restore.add_argument('--inpaint-mask', type=Path)
    restore.add_argument('--protected-mask', type=Path)
    restore.add_argument('--reserve-free-gb', type=float, default=1.0)
    restore.add_argument('--keep-stage', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'restore':
        from mobile_s24.s24_render import MobileConfig, run
        try:
            cfg = MobileConfig(source=args.source, output=args.output, stage_dir=args.workdir,
                               target_width=args.width, target_height=args.height,
                               target_mp=args.target_mp, tile_size=args.tile,
                               detail_gain=args.detail_gain, psf_sigma=args.psf_sigma,
                               rl_iterations=args.rl_iterations, ca_red=args.ca_red,
                               ca_blue=args.ca_blue, focus_mask=args.focus_mask,
                               inpainted_base=args.inpainted_base, inpaint_mask=args.inpaint_mask,
                               protected_mask=args.protected_mask, reserve_free_gb=args.reserve_free_gb,
                               keep_stage=args.keep_stage)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(run(cfg), indent=2, ensure_ascii=False))
        return 0
    try:
        w, h = target_dimensions(args.width, args.height, args.target_mp)
        tiles = sum(1 for _ in iter_tiles(w, h, tile_size=args.tile, halo=args.halo))
    except ValueError as exc:
        parser.error(str(exc))
    payload = {
        'project': 'Aether Photo & Video',
        'device_profile': 'Samsung Galaxy S24 Ultra',
        'processing': 'plan_only',
        'source': {'width': args.width, 'height': args.height},
        'output': {'width': w, 'height': h},
        'tile_size': args.tile,
        'halo': args.halo,
        'tiles': tiles,
        'estimated_disk_buffers': memory_estimate(w, h),
        'warning': 'Disk estimate excludes source, intermediates, filesystem overhead and reserve.'
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
