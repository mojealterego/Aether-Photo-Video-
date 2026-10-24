"""LibRaw optional backend safety contracts; no real RAW weights/decoder installed."""
import sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,tifffile
from aether.ingest.libraw_adapter import decode_libraw_safe,UnsupportedRawMemory
class Raw:
    sizes=types.SimpleNamespace(width=10,height=8)
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def postprocess(self,**kwargs):return np.full((8,10,3),20301,'u2')
class TestRAWAdapter(unittest.TestCase):
    def test_guarded_rgb16_output(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);src=root/'f.dng';out=root/'out.tif'
            src.write_bytes(b'synthetic')
            fake=types.SimpleNamespace(imread=lambda x:Raw(),ColorSpace=types.SimpleNamespace(sRGB=1))
            with patch.dict(sys.modules,{'rawpy':fake}):
                result=decode_libraw_safe(src,out,max_source_pixels=1000,
                                         max_decoded_mb=100,reserve_free_gb=0)
            self.assertFalse(result['oom_free_guaranteed'])
            self.assertEqual(int(tifffile.imread(out)[0,0,0]),20301)
    def test_pixel_preflight_blocks_decoder(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);src=root/'f.dng';src.write_bytes(b'synthetic')
            fake=types.SimpleNamespace(imread=lambda x:Raw(),ColorSpace=types.SimpleNamespace(sRGB=1))
            with patch.dict(sys.modules,{'rawpy':fake}):
                with self.assertRaises(UnsupportedRawMemory):
                    decode_libraw_safe(src,root/'out.tif',max_source_pixels=20)
    def test_fail_implicit_8bit(self):
        class Broken(Raw):
            def postprocess(self,**kwargs):return np.zeros((8,10,3),'u1')
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);src=root/'f.dng';src.write_bytes(b'synthetic')
            fake=types.SimpleNamespace(imread=lambda x:Broken(),ColorSpace=types.SimpleNamespace(sRGB=1))
            with patch.dict(sys.modules,{'rawpy':fake}):
                with self.assertRaises(ValueError):
                    decode_libraw_safe(src,root/'out.tif',reserve_free_gb=0)
if __name__=='__main__':unittest.main()
