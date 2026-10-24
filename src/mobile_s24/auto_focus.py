"""Samsung S24 Ultra offline focus proposal and memory-bounded BigTIFF pipeline.

Focus is an acutance heuristic (optionally modulated by a supplied relative-depth
map), never a calibrated depth map or a neural inference result.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .s24_render import MobileConfig, Source, run


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


@dataclass(frozen=True)
class FocusConfig:
    source: Path
    workdir: Path
    depth_map: Path | None = None
    preview_limit: int = 768
    energy_threshold: float = .12
    focus_depth: float = .5
    depth_tolerance: float = .24
    max_source_pixels: int = 225_000_000
    max_decoded_mb: int = 900
    input_transfer: str = "srgb"

    def __post_init__(self):
        for key in ("source", "workdir", "depth_map"):
            val = getattr(self, key)
            if val is not None:
                object.__setattr__(self, key, Path(val).expanduser())
        if not 64 <= self.preview_limit <= 1536:
            raise ValueError("preview_limit must be 64..1536")
        if (not math.isfinite(self.energy_threshold)
            or not 0 <= self.energy_threshold < 1
            or not 0 <= self.focus_depth <= 1
            or not 0 < self.depth_tolerance <= 1):
            raise ValueError("Invalid mask thresholds")
        if self.input_transfer not in ("srgb", "linear"):
            raise ValueError("Unsupported transfer function")


def _preview(source: Source, limit: int, transfer: str) -> np.ndarray:
    """Read bounded image regions; never decode a 400MP RGB32 frame."""
    scale = min(1., limit / max(source.width, source.height))
    width = max(1, round(source.width * scale))
    height = max(1, round(source.height * scale))
    image = np.empty((height, width, 3), dtype=np.float32)
    step = 48
    for y in range(0, height, step):
        y1 = min(y + step, height)
        sy0 = max(0, math.floor(y * source.height / height))
        sy1 = min(source.height, math.ceil(y1 * source.height / height))
        for x in range(0, width, step):
            x1 = min(x + step, width)
            sx0 = max(0, math.floor(x * source.width / width))
            sx1 = min(source.width, math.ceil(x1 * source.width / width))
            if (sx1 - sx0) * (sy1 - sy0) > 8_000_000:
                raise MemoryError("Source crop exceeds 8MP preview cap")
            raw = source.crop((sx0, sy0, sx1, sy1))
            dtype = raw.dtype
            tile = raw.astype("f4")
            if np.issubdtype(dtype, np.integer):
                tile /= float(np.iinfo(dtype).max)
            if transfer == "srgb":
                tile = np.where(tile <= .04045, tile / 12.92,
                                ((tile + .055) / 1.055) ** 2.4).astype("f4")
            res = np.stack([
                np.asarray(Image.fromarray(tile[:, :, k], mode="F").resize(
                    (x1-x, y1-y), Image.Resampling.BOX), dtype="f4")
                for k in range(3)
            ], axis=2)
            image[y:y1, x:x1, :] = res
    return image


def _gaussian_blur(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Separable reflect-padded Gaussian for a bounded preview; no OpenCV."""
    radius = max(1, math.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-.5 * (x / sigma) ** 2).astype("f4")
    kernel /= kernel.sum()
    out = gray.astype("f4")
    for axis in (0, 1):
        pads = [(0, 0)] * out.ndim
        pads[axis] = (radius, radius)
        src = np.pad(out, pads, mode="reflect")
        dest = np.zeros_like(out)
        for n, weight in enumerate(kernel):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(n, n + out.shape[axis])
            dest += weight * src[tuple(sl)]
        out = dest
    return out


def _focus_from_preview(preview: np.ndarray, config: FocusConfig,
                        depth: np.ndarray | None) -> np.ndarray:
    lum = np.asarray(preview @ np.array([.2126, .7152, .0722], dtype="f4"), dtype="f4")
    edges = np.abs(lum - _gaussian_blur(lum, 1.))
    pad = np.pad(lum, 1, mode="reflect")
    lap = np.abs(4*lum - pad[:-2, 1:-1] - pad[2:, 1:-1]
                 - pad[1:-1, :-2] - pad[1:-1, 2:])
    energy = _gaussian_blur(edges + .15 * lap, 1.35)
    p98 = float(np.percentile(energy, 98))
    detail = np.clip(energy / max(p98, 1.e-5), 0., 1.)
    mask = np.clip((detail - config.energy_threshold) /
                   (1.-config.energy_threshold), 0., 1.)
    if depth is not None:
        if depth.shape != lum.shape or not np.isfinite(depth).all():
            raise ValueError("Invalid relative-depth map")
        plane = np.exp(-.5 * np.square((depth-config.focus_depth) / config.depth_tolerance))
        mask *= plane
    return np.clip(_gaussian_blur(mask, 1.1), 0., 1.).astype("f4")


def prepare_focus_mask(config: FocusConfig) -> Path:
    """Cache focus mask with SHA-256 provenance; refuse inconsistent resume."""
    if not config.source.is_file():
        raise FileNotFoundError(config.source)
    if config.depth_map is not None and not config.depth_map.is_file():
        raise FileNotFoundError(config.depth_map)
    config.workdir.mkdir(parents=True, exist_ok=True)
    out = config.workdir / "focus_mask.png"
    manifest = config.workdir / "focus_manifest.json"
    sig = {
        "source_sha256": _sha256(config.source),
        "depth_sha256": _sha256(config.depth_map) if config.depth_map else None,
        "preview_limit": config.preview_limit,
        "energy_threshold": config.energy_threshold,
        "focus_depth": config.focus_depth,
        "depth_tolerance": config.depth_tolerance,
        "input_transfer": config.input_transfer,
        "method": "global_edge_energy_preview_v1",
    }
    if out.exists() or manifest.exists():
        if not (out.exists() and manifest.exists()):
            raise ValueError("Incomplete focus mask cache")
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if any(old.get(k) != v for k, v in sig.items()) or old.get("mask_sha256") != _sha256(out):
            raise ValueError("Focus mask signature mismatch; use separate workdir")
        return out

    reader = Source(config.source, config.max_source_pixels, config.max_decoded_mb)
    try:
        preview = _preview(reader, config.preview_limit, config.input_transfer)
    finally:
        reader.close()
    depth = None
    if config.depth_map:
        with Image.open(config.depth_map) as img:
            if img.mode != "L" or img.width * img.height > 16_000_000:
                raise ValueError("External depth map must be grayscale L and <=16MP")
            img.load()
            depth = np.asarray(img.resize((preview.shape[1], preview.shape[0]),
                                          Image.Resampling.BILINEAR), dtype="f4") / 255.
    mask = _focus_from_preview(preview, config, depth)
    tmp = out.with_name(out.stem + ".partial.png")
    try:
        Image.fromarray(np.rint(mask * 255).astype("u1"), "L").save(tmp)
        os.replace(tmp, out)
        sig["mask_sha256"] = _sha256(out)
        mtmp = manifest.with_suffix(".partial")
        with mtmp.open("w", encoding="utf-8") as handle:
            json.dump(sig, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(mtmp, manifest)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def run_offline_auto(config: FocusConfig, *, output: Path,
                     target_width: int | None = None, target_height: int | None = None,
                     target_mp: float = 400., tile: int = 256, detail_gain: float = .0,
                     psf_sigma: float = 0., rl_iterations: int = 0,
                     ca_red: float = 0., ca_blue: float = 0.,
                     reserve_gb: float = 1.) -> dict:
    """Apply global focus to both optical correction and output high-pass."""
    mask_path = prepare_focus_mask(config)
    job = MobileConfig(config.source, Path(output), config.workdir / "render",
                       target_width=target_width, target_height=target_height,
                       target_mp=target_mp, tile_size=tile, detail_gain=detail_gain,
                       focus_mask=mask_path, psf_sigma=psf_sigma,
                       rl_iterations=rl_iterations, ca_red=ca_red, ca_blue=ca_blue,
                       input_transfer=config.input_transfer,
                       max_input_pixels=config.max_source_pixels,
                       max_decoded_ram_mb=config.max_decoded_mb,
                       reserve_free_gb=reserve_gb)
    report = run(job)
    report["focus_method"] = "global_edge_energy_preview"
    report["focus_mask_path"] = str(mask_path)
    report["depth_inferred_on_device"] = False
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Samsung S24 Ultra: offline BigTIFF with automatic focus proposal")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--depth-map", type=Path)
    parser.add_argument("--target-mp", type=float, default=400.)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--detail-gain", type=float, default=0.)
    parser.add_argument("--psf-sigma", type=float, default=0.)
    parser.add_argument("--rl-iterations", type=int, default=0)
    parser.add_argument("--ca-red", type=float, default=0.)
    parser.add_argument("--ca-blue", type=float, default=0.)
    parser.add_argument("--reserve-free-gb", type=float, default=1.)
    args = parser.parse_args(argv)
    cfg = FocusConfig(args.input, args.workdir, depth_map=args.depth_map)
    report = run_offline_auto(
        cfg, output=args.output, target_width=args.width, target_height=args.height,
        target_mp=args.target_mp, tile=args.tile, detail_gain=args.detail_gain,
        psf_sigma=args.psf_sigma, rl_iterations=args.rl_iterations,
        ca_red=args.ca_red, ca_blue=args.ca_blue, reserve_gb=args.reserve_free_gb)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
