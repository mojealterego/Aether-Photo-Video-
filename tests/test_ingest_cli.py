"""Integration contracts for the unified TIFF/RAW/DNG ingest entrypoint."""
from __future__ import annotations

import contextlib
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

from aether.ingest.cli import main
from aether.ingest.dng import XYZ_TO_SRGB


def _synthetic_dng(path: Path) -> None:
    # A deterministic 2x2 RGGB, one-plane, unpacked CFA with DNG calibration.
    scene = np.full((34, 46, 3), (0.25, 0.5, 0.75), dtype=np.float32)
    pattern = (0, 1, 1, 2)
    raw = np.empty(scene.shape[:2], dtype=np.uint16)
    for y in range(raw.shape[0]):
        for x in range(raw.shape[1]):
            raw[y, x] = round(64 + 4031 * float(scene[y, x, pattern[2 * (y % 2) + x % 2]]))
    tags = [
        (33421, "H", 2, (2, 2), False),
        (33422, "B", 4, pattern, False),
        (50714, "H", 1, 64, False),
        (50717, "H", 1, 4095, False),
        (50728, "d", 3, (1.0, 1.0, 1.0), False),
        (50721, "d", 9, tuple(float(v) for v in XYZ_TO_SRGB.ravel()), False),
    ]
    tifffile.imwrite(path, raw, photometric="cfa", extratags=tags, metadata=None)


class TestUnifiedIngest(unittest.TestCase):
    def test_real_strict_dng_to_rgb16_bigtiff(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "source.dng"
            output = root / "normalized.tif"
            _synthetic_dng(raw)
            with contextlib.redirect_stdout(StringIO()):
                self.assertEqual(main([
                    "--source", str(raw), "--normalized", str(output),
                    "--raw-tile", "16",
                ]), 0)
            with tifffile.TiffFile(output) as result:
                self.assertTrue(result.is_bigtiff)
                self.assertEqual(result.pages[0].dtype, np.dtype("uint16"))
                self.assertEqual(result.pages[0].shape, (34, 46, 3))
                self.assertTrue(np.any(result.asarray() % 257 != 0))

    def test_explicit_libraw_routes_raw_without_auto_fallback(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, normalized = root / "camera.nef", root / "output.tif"
            with patch("aether.ingest.cli.decode_libraw_safe",
                       return_value={"output": str(normalized)}) as decoder:
                with contextlib.redirect_stdout(StringIO()):
                    self.assertEqual(main([
                        "--source", str(source), "--normalized", str(normalized),
                        "--raw-backend", "libraw", "--max-libraw-pixels", "1000",
                    ]), 0)
                self.assertEqual(decoder.call_args.kwargs["max_source_pixels"], 1000)
            with contextlib.redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["--source", str(source), "--normalized", str(normalized)])
            self.assertEqual(error.exception.code, 2)

    def test_never_assume_srgb_for_raw(self):
        with contextlib.redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                main([
                    "--source", "camera.dng", "--normalized", "result.tif",
                    "--assume-srgb",
                ])
        self.assertEqual(error.exception.code, 2)

    def test_render_requires_workdir(self):
        with contextlib.redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                main([
                    "--source", "input.tif", "--normalized", "master.tif",
                    "--render", "rendered.tif",
                ])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
