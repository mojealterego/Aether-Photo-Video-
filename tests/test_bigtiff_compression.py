"""Verify lossless tiled Deflate BigTIFF from the S24 CPU renderer."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
import tifffile
from mobile_s24.s24_render import MobileConfig,run

class TestBigTiffDeflate(unittest.TestCase):
    def test_rgb16_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            Image.fromarray(np.full((23,31,3),140,np.uint8)).save(p/'src.png')
            cfg=MobileConfig(p/'src.png',p/'out.tif',p/'scratch',
                             target_width=31,target_height=23,tile_size=32,
                             reserve_free_gb=0,compression='deflate')
            report=run(cfg)
            with tifffile.TiffFile(p/'out.tif') as t:
                page=t.pages[0]
                self.assertTrue(t.is_bigtiff)
                self.assertEqual(page.compression,tifffile.COMPRESSION.DEFLATE)
                result=page.asarray()
                self.assertEqual(result.shape,(23,31,3))
                self.assertEqual(result.dtype,np.dtype('uint16'))
            self.assertEqual(report['compression'],'deflate')

    def test_memmapped_intermediate_cannot_be_compressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with self.assertRaises(ValueError):
                MobileConfig(p/'s.png',p/'o.tif',p/'w',compression='deflate',contiguous_output=True)

if __name__=='__main__':
    unittest.main()
