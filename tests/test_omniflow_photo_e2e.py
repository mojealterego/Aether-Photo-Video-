"""Native Nikon NEF -> BigTIFF CLI pipeline contract with fake LibRaw decoder."""
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tifffile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"omniflow-worker"))
from omega_sdk.photo_pipeline import run_native_nef_to_bigtiff

class IntegratedNefPipelineTests(unittest.TestCase):
    def test_native_worker_path_nef_to_icc16_bigtiff_odd_dimensions(self):
        class FakeRaw:
            sizes=SimpleNamespace(width=57,height=65)
            black_level_per_channel=[0,0,0,0]
            white_level=16383
            camera_whitebalance=[2,1,1,1]
            def __enter__(self):return self
            def __exit__(self,*_):return False
            def postprocess(self,**kwargs):
                self.kwargs=kwargs
                return np.ones((65,57,3),dtype=np.uint16)*13000
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"camera.nef"
            source.write_bytes(b"II*\x00"+b"\x00"*2000)
            output=Path(tmp)/"finished.tiff"
            fake=FakeRaw()
            module=SimpleNamespace(imread=lambda _:fake,
                ColorSpace=SimpleNamespace(sRGB="sRGB"))
            with patch.dict(sys.modules,{"rawpy":module}):
                report=run_native_nef_to_bigtiff(
                    source,output,tile_size=16,overlap=4,reserve_bytes=0)
            self.assertEqual(source.read_bytes()[:4],b"II*\x00")
            self.assertEqual(report["dimensions"],[65,57,3])
            self.assertEqual(report["status"],"MASTER_WRITTEN_UNVERIFIED")
            with tifffile.TiffFile(output) as tif:
                self.assertTrue(tif.is_bigtiff)
                self.assertEqual(tif.asarray().shape,(65,57,3))
                self.assertTrue(np.all(tif.asarray()==tif.asarray()[0,0,:]))
            self.assertTrue(fake.kwargs["no_auto_bright"])

    def test_no_overwrite_even_for_valid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"camera.nef"
            source.write_bytes(b"II*\x00"+b"\x00"*2000)
            output=Path(tmp)/"finished.tiff"
            output.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                run_native_nef_to_bigtiff(
                    source,output,tile_size=16,overlap=4,reserve_bytes=0)
            self.assertEqual(output.read_bytes(),b"keep")
