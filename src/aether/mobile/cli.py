"""Zero-dependency planning CLI. Rendering is not implemented here."""
from __future__ import annotations
import argparse
import json
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
    args = parser.parse_args(argv)
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
