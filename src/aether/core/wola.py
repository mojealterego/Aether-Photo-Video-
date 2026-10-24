"""Disk-backed Gaussian WOLA compositing for Android-sized RAM budgets.

Maintains a float32 RGB sum and scalar float32 weight map on disk, never
materializes the complete image in Python heap. Normalization happens exactly
once, in bounded row stripes, after ALL tiles have been accumulated.

This module is intentionally independent of ComfyUI/CUDA/PyTorch. A client
should consume iter_uint16_rows with a streaming TIFF writer; it should not
collect the iterator for full-size images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator
import numpy as np


class WolaMemmap:
    """Single-writer, new-job-only accumulator; coordinates are half-open."""

    def __init__(self, workdir: str | Path, *, width: int, height: int) -> None:
        if type(width) is not int or type(height) is not int or min(width, height) < 1:
            raise ValueError("width and height must be positive integers")
        self.width, self.height = width, height
        root = Path(workdir)
        root.mkdir(parents=True, exist_ok=True)
        self.image_path = root / "wola_sum.f32"
        self.weight_path = root / "wola_weight.f32"
        if self.image_path.exists() or self.weight_path.exists():
            raise FileExistsError("WOLA work directory already contains accumulator files")
        self._image = np.memmap(self.image_path, dtype=np.float32, mode="w+", shape=(height, width, 3))
        self._weights = np.memmap(self.weight_path, dtype=np.float32, mode="w+", shape=(height, width))
        self._closed = False
        self._added = 0

    @staticmethod
    def gaussian_weights(height: int, width: int) -> np.ndarray:
        """Positive separable Gaussian window, including border pixels."""
        if min(height, width) < 1:
            raise ValueError("Invalid tile dimensions")
        y = np.arange(height, dtype=np.float32) - (height - 1) * .5
        x = np.arange(width, dtype=np.float32) - (width - 1) * .5
        wy = np.exp(-.5 * np.square(y / max(1, height * .25)))
        wx = np.exp(-.5 * np.square(x / max(1, width * .25)))
        return (wy[:, None] * wx[None, :]).astype(np.float32)

    def add(self, tile: np.ndarray, *, x0: int, y0: int, weight: np.ndarray | None = None) -> None:
        if self._closed:
            raise RuntimeError("WOLA accumulator is closed")
        arr = np.asarray(tile)
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValueError("Tile must have shape (height,width,3)")
        if type(x0) is not int or type(y0) is not int or x0 < 0 or y0 < 0:
            raise ValueError("Negative or non-integer tile origin")
        th, tw = arr.shape[:2]
        if th > 2048 or tw > 2048:
            raise ValueError("Tile exceeds configured safety limit of 2048 pixels per side")
        if x0 + tw > self.width or y0 + th > self.height:
            raise ValueError("Tile extends outside output image")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Tile contains NaN/Inf")
        pixels = np.asarray(arr, dtype=np.float32)
        weights = self.gaussian_weights(th, tw) if weight is None else np.asarray(weight, dtype=np.float32)
        if weights.shape != (th, tw) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("WOLA weights must be finite, positive and match tile size")
        ys, xs = slice(y0, y0 + th), slice(x0, x0 + tw)
        self._image[ys, xs, :] += pixels * weights[:, :, None]
        self._weights[ys, xs] += weights
        self._added += 1

    def iter_normalized_rows(self, *, stripe_rows: int = 128) -> Iterator[np.ndarray]:
        if self._closed:
            raise RuntimeError("WOLA accumulator is closed")
        if type(stripe_rows) is not int or stripe_rows < 1 or stripe_rows > 512:
            raise ValueError("stripe_rows must be 1..512")
        if self._added == 0:
            raise ValueError("Cannot normalize before tiles have been added")
        for y0 in range(0, self.height, stripe_rows):
            y1 = min(y0 + stripe_rows, self.height)
            w = np.asarray(self._weights[y0:y1, :])
            if np.any(w <= 0):
                raise ValueError("WOLA coverage incomplete: zero-weight output pixels")
            value = np.asarray(self._image[y0:y1, :, :]) / w[:, :, None]
            yield np.clip(value, 0., 1.).astype(np.float32)

    def iter_uint16_rows(self, *, stripe_rows: int = 128) -> Iterator[np.ndarray]:
        for rows in self.iter_normalized_rows(stripe_rows=stripe_rows):
            yield np.rint(rows * 65535.).astype(np.uint16)

    def flush(self) -> None:
        if not self._closed:
            self._image.flush()
            self._weights.flush()

    def close(self) -> None:
        if not self._closed:
            self.flush()
            del self._image
            del self._weights
            self._closed = True

    def __enter__(self) -> "WolaMemmap":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
