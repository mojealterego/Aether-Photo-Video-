"""Pure, Android-compatible tiling and 400 MP geometry contracts.

No image buffers are allocated by this module. Coordinates are half-open.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Iterator

@dataclass(frozen=True, slots=True)
class Tile:
    index: int
    x0: int
    y0: int
    x1: int
    y1: int
    read_x0: int
    read_y0: int
    read_x1: int
    read_y1: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def target_dimensions(source_width: int, source_height: int, target_mp: float = 400.0) -> tuple[int, int]:
    """Preserve source aspect ratio; target area is approximately target_mp.

    Exact aspect ratio cannot always be represented by integer dimensions.
    """
    if (type(source_width) is not int or type(source_height) is not int
            or source_width < 2 or source_height < 2):
        raise ValueError('Source dimensions must be integers >= 2')
    if not isfinite(target_mp) or target_mp <= 0 or target_mp > 1_000:
        raise ValueError('Invalid target megapixels')
    ratio = source_width / source_height
    target_pixels = target_mp * 1_000_000
    return max(2, round(sqrt(target_pixels * ratio))), max(2, round(sqrt(target_pixels / ratio)))


def iter_tiles(width: int, height: int, *, tile_size: int = 256, halo: int = 32) -> Iterator[Tile]:
    """Partition the exact output pixels, read with a clipped halo.

    This is a partition plan, not WOLA overlap itself; WOLA is a separate
    compositing concern of the image backend.
    """
    if (type(width) is not int or type(height) is not int or width < 1 or height < 1):
        raise ValueError('Invalid image dimensions')
    if (type(tile_size) is not int or not 64 <= tile_size <= 1024):
        raise ValueError('tile_size outside 64..1024')
    if type(halo) is not int or halo < 0 or halo > tile_size:
        raise ValueError('halo must be between 0 and tile_size')
    index = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            x1, y1 = min(x + tile_size, width), min(y + tile_size, height)
            yield Tile(index, x, y, x1, y1,
                       max(0, x - halo), max(0, y - halo),
                       min(width, x1 + halo), min(height, y1 + halo))
            index += 1


def memory_estimate(width: int, height: int) -> dict[str, int]:
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError('Invalid dimensions')
    pixels = width * height
    return {
        'pixels': pixels,
        'rgb16_bytes': pixels * 3 * 2,
        'rgb_float32_bytes': pixels * 3 * 4,
        'wola_disk_bytes': pixels * (3 * 4 + 4),
        'min_final_plus_wola_bytes': pixels * (3 * 2 + 3 * 4 + 4),
    }
