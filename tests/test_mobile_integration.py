"""Minimal real RGB16 BigTIFF smoke test of the migrated mobile renderer."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image
from mobile_s24.s24_render import MobileConfig, run

class TestMobileEndToEnd(unittest.TestCase):
    def test_rgb16_bigtiff_and_tiling(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/"input.png"; output=root/"out.tif"
            x=np.zeros((15,21,3),dtype=np.uint8)
            x[:,:,:]=[60,120,200]
            Image.fromarray(x,"RGB").save(source)
            cfg=MobileConfig(source,output,root/"stage",target_width=53,target_height=37,
                             tile_size=32,detail_gain=0,reserve_free_gb=0)
            report=run(cfg)
            self.assertEqual(report["target_dimensions"],[53,37])
            with tifffile.TiffFile(output) as f:
                self.assertTrue(f.is_bigtiff)
                self.assertEqual(f.pages[0].dtype,np.dtype("uint16"))
                self.assertEqual(f.pages[0].shape,(37,53,3))
                a=f.pages[0].asarray()
                self.assertLessEqual(int(np.ptp(a.astype("int32"),axis=(0,1)).max()),2)

if __name__=="__main__":
    unittest.main()
