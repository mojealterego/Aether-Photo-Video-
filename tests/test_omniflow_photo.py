"""Native photo pipeline acceptance from SYS-AETHERIS-OMNIFLOW-016."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "omniflow-worker"))
from omega_sdk.photo_faithful_400mp import DiskBackedStitcher, linear_to_srgb_u16, I002_LoadNEF
from omega_sdk.protocol import ExecutionContext


class NativePhotoTests(TestCase):
    def test_oetf_monotonic_and_reference_points(self):
        x=np.array([0.,0.0031308,0.18,1.],dtype=np.float32).reshape(1,4,1).repeat(3,axis=2)
        result=linear_to_srgb_u16(x)
        self.assertEqual(result.dtype,np.uint16)
        self.assertEqual(int(result[0,0,0]),0)
        self.assertEqual(int(result[0,-1,0]),65535)
        self.assertLess(abs(int(result[0,1,0])-int(0.04045*65535)),3)
        self.assertTrue(np.all(np.diff(result[0,:,0].astype(np.int64))>=0))

    def test_oetf_rejects_nan(self):
        with self.assertRaises(ValueError):
            linear_to_srgb_u16(np.array([[[float("nan")]*3]],dtype="f4"))

    def test_tiles_reconstruct_constant_with_overlap_and_icc(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp); output=work/"out.tif"
            with DiskBackedStitcher(43,39,work/"scratch",reserve_bytes=0) as st:
                st.add_tile(0,0,np.ones((32,32,3),dtype="f4")*0.18)
                st.add_tile(0,16,np.ones((32,23,3),dtype="f4")*0.18)
                st.add_tile(16,0,np.ones((27,32,3),dtype="f4")*0.18)
                st.add_tile(16,16,np.ones((27,23,3),dtype="f4")*0.18)
                st.write_srgb16_bigtiff(output,tile_size=16,source_sha256="sha256:"+"a"*64)
            with tifffile.TiffFile(output) as tif:
                self.assertTrue(tif.is_bigtiff)
                self.assertEqual(tif.pages[0].shape,(43,39,3))
                self.assertEqual(tif.asarray().dtype,np.uint16)
                self.assertIn("InterColorProfile",tif.pages[0].tags)
                self.assertLess(np.ptp(tif.asarray()),2)

    def test_zero_coverage_rejected_before_committing(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp); output=work/"fail.tif"
            with DiskBackedStitcher(32,32,work/"scratch",reserve_bytes=0) as st:
                st.add_tile(0,0,np.ones((16,16,3),dtype="f4"))
                with self.assertRaisesRegex(ValueError,"ERR_TILE_COVERAGE_GAP"):
                    st.write_srgb16_bigtiff(output,tile_size=16)
            self.assertFalse(output.exists())

    def test_existing_target_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);out=work/"existing.tif";out.write_bytes(b"keep")
            with DiskBackedStitcher(16,16,work/"scratch",reserve_bytes=0) as st:
                st.add_tile(0,0,np.ones((16,16,3),dtype="f4"))
                with self.assertRaises(FileExistsError):
                    st.write_srgb16_bigtiff(out,tile_size=16)
            self.assertEqual(out.read_bytes(),b"keep")

    def test_failing_real_raw_header_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"not.nef";p.write_bytes(b"not raw"*300)
            ctx=ExecutionContext("job","tenant",1,"DOC",1,Path(tmp),Path(tmp))
            with self.assertRaises(ValueError): I002_LoadNEF().execute(ctx,{"nef_path":str(p)}, {})

    def test_raw_decoder_forces_linear_transfer(self):
        class FakeRaw:
            sizes=SimpleNamespace(width=4,height=3)
            black_level_per_channel=[0,0,0,0]
            white_level=16383
            camera_whitebalance=[2.,1.,1.,1.]
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def postprocess(self,**kwargs):
                self.settings=kwargs
                return np.ones((3,4,3),dtype="u2")*30000
        fake=FakeRaw()
        fake_rawpy=SimpleNamespace(imread=lambda p:fake,ColorSpace=SimpleNamespace(sRGB="srgb"))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"real.nef";p.write_bytes(b"II*\x00"+b"\x00"*2000)
            ctx=ExecutionContext("job","tenant",1,"DOC",1,Path(tmp),Path(tmp))
            with patch.dict(sys.modules,{"rawpy":fake_rawpy}):
                got=I002_LoadNEF().execute(ctx,{"nef_path":str(p)}, {})
            self.assertEqual(fake.settings["gamma"],(1,1))
            self.assertEqual(fake.settings["output_bps"],16)
            self.assertTrue(fake.settings["no_auto_bright"])
            self.assertEqual(got["color_space"],"SceneLinear_sRGB")
            self.assertEqual(got["linear_image_f32"].dtype,np.float32)
