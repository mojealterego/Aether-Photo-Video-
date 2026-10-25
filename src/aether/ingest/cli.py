"""Fail-closed ICC16 TIFF and RAW/DNG import into an RGB16 sRGB BigTIFF master.

The strict DNG decoder is out-of-core for supported unpacked Bayer DNG files.
Complex RAW/DNG requires an explicit LibRaw backend, which allocates a full
frame and is therefore subject to conservative pixel and memory limits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dng import decode_uncompressed_dng
from .icc16 import rgb16_tiff_to_srgb
from .libraw_adapter import decode_libraw_safe

_TIFF_SUFFIXES = frozenset({".tif", ".tiff"})
_LIBRAW_SUFFIXES = frozenset({
    ".dng", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".raf",
    ".rw2", ".orf", ".pef", ".3fr", ".iiq",
})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aether-ingest",
        description="Normalize RGB TIFF or RAW/DNG to RGB16 BigTIFF; optional offline render",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, required=True,
                        help="New RGB16 sRGB BigTIFF master; never overwritten")
    parser.add_argument("--assume-srgb", action="store_true",
                        help="Only for untagged RGB TIFF; never camera RAW")
    parser.add_argument("--stripe-rows", type=int, default=64)
    parser.add_argument("--raw-backend", choices=("strict", "libraw"), default="strict",
                        help="Strict supports unpacked Bayer DNG; LibRaw is explicit opt-in")
    parser.add_argument("--raw-tile", type=int, default=256)
    parser.add_argument("--max-dng-pixels", type=int, default=225_000_000)
    parser.add_argument("--max-libraw-pixels", type=int, default=40_000_000)
    parser.add_argument("--max-decoded-mb", type=int, default=900)
    parser.add_argument("--render", type=Path,
                        help="Optional final RGB16 BigTIFF rendered from the normalized master")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--target-mp", type=float, default=400.0)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--reserve-free-gb", type=float, default=1.0)
    args = parser.parse_args(argv)

    if args.render is not None and args.workdir is None:
        parser.error("--render requires --workdir")
    suffix = args.source.suffix.lower()

    if suffix in _TIFF_SUFFIXES:
        if args.raw_backend != "strict":
            parser.error("--raw-backend libraw is for camera RAW/DNG, not RGB TIFF")
        result = rgb16_tiff_to_srgb(
            args.source, args.normalized, rows=args.stripe_rows,
            assume_srgb=args.assume_srgb,
        )
        result["backend"] = "lcms2_rgb16"
    elif suffix == ".dng" and args.raw_backend == "strict":
        if args.assume_srgb:
            parser.error("--assume-srgb is forbidden for camera RAW/DNG")
        result = decode_uncompressed_dng(
            args.source, args.normalized,
            tile_size=args.raw_tile, max_pixels=args.max_dng_pixels,
        )
        result["backend"] = "strict_bayer_dng"
    elif suffix in _LIBRAW_SUFFIXES and args.raw_backend == "libraw":
        if args.assume_srgb:
            parser.error("--assume-srgb is forbidden for camera RAW/DNG")
        result = decode_libraw_safe(
            args.source, args.normalized,
            max_source_pixels=args.max_libraw_pixels,
            max_decoded_mb=args.max_decoded_mb,
            reserve_free_gb=args.reserve_free_gb,
        )
        result["backend"] = "libraw_bounded_full_frame"
    else:
        parser.error("Unsupported input: use RGB TIFF, strict DNG, or explicit LibRaw")

    if args.render is not None:
        from mobile_s24.s24_render import MobileConfig, run

        config = MobileConfig(
            args.normalized, args.render, args.workdir,
            target_mp=args.target_mp,
            tile_size=args.tile,
            reserve_free_gb=args.reserve_free_gb,
        )
        result["render"] = run(config)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
