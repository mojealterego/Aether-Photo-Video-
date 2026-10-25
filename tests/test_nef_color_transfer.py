"""NEF -> RGB16: verify LibRaw gamma and ICC color contract without a camera fixture."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import sys

import numpy as np
import tifffile

from aether.ingest.libraw_adapter import decode_libraw_safe


class FakeDecoder:
    def __init__(self):
        self.sizes = SimpleNamespace(width=8, height=6)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def postprocess(self, **kwargs):
        self.calls.append(kwargs)
        return np.arange(8 * 6 * 3, dtype=np.uint16).reshape(6, 8, 3)


class TestNefColorContract(TestCase):
    def test_nef_to_rgb16_uses_srgb_transfer_curve_and_embeds_icc(self):
        decoder = FakeDecoder()
        fake_rawpy = SimpleNamespace(
            imread=lambda _: decoder,
            ColorSpace=SimpleNamespace(sRGB="sRGB"),
        )
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "camera.nef"
            target = Path(tmp) / "master.tif"
            source.write_bytes(b"camera RAW test double")
            with patch.dict(sys.modules, {"rawpy": fake_rawpy}):
                result = decode_libraw_safe(
                    source, target, max_source_pixels=1000,
                    max_decoded_mb=100, reserve_free_gb=0,
                )
            self.assertEqual(len(decoder.calls), 1)
            settings = decoder.calls[0]
            self.assertEqual(settings["output_bps"], 16)
            self.assertEqual(settings["output_color"], "sRGB")
            self.assertTrue(settings["use_camera_wb"])
            self.assertTrue(settings["no_auto_bright"])
            # The embedded ICC is sRGB, so the encoded RGB transfer curve
            # must also be sRGB, not the distinct Rec.709-style curve.
            self.assertEqual(settings["gamma"], (2.4, 12.92))
            with tifffile.TiffFile(target) as tif:
                self.assertTrue(tif.is_bigtiff)
                self.assertEqual(tif.pages[0].dtype, np.dtype("uint16"))
                self.assertEqual(tif.pages[0].shape, (6, 8, 3))
                profile = tif.pages[0].tags["InterColorProfile"].value
                self.assertEqual(profile[36:40], b"acsp")
            self.assertEqual(result["bits_per_channel"], 16)
            self.assertFalse((Path(tmp) / "master.tif.partial.tif").exists())
