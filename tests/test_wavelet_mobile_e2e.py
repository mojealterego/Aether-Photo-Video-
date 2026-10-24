"""End-to-end mobile Haar/DWT frequency fusion and BigTIFF regression tests."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import tifffile
from mobile_s24.s24_render import MobileConfig, run

class TestNeuralFrequencyRenderer(unittest.TestCase):
    def test_masked_wavelet_sr_in_actual_rgb16_bigtiff(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            base=np.full((24,36,3),128,np.uint8)
            candidate=base.copy()
            candidate[::2,::2]=255
            candidate[1::2,1::2]=0
            Image.fromarray(base).save(d/'base.png')
            tifffile.imwrite(d/'sr.tif',candidate,photometric='rgb',metadata=None)
            Image.fromarray(np.full((24,36),255,np.uint8)).save(d/'focus.png')
            cfg=MobileConfig(d/'base.png',d/'out.tif',d/'scratch',target_width=72,target_height=48,
                             tile_size=32,reserve_free_gb=0,sr_candidate=d/'sr.tif',
                             focus_mask=d/'focus.png',wavelet_levels=2,wavelet_gain=1.)
            report=run(cfg)
            with tifffile.TiffFile(d/'out.tif') as f:
                result=f.pages[0].asarray()
                self.assertTrue(f.is_bigtiff)
                self.assertEqual(result.shape,(48,72,3))
                self.assertGreater(result.std(),800)
            self.assertTrue(report['wavelet_sr_candidate'])

    def test_target_size_sr_can_exceed_source_budget_if_memmapped(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            Image.fromarray(np.full((2,3,3),120,np.uint8)).save(d/'source.png')
            tifffile.imwrite(d/'target.tif',np.full((8,10,3),150,np.uint16),
                             photometric='rgb',metadata=None)
            Image.fromarray(np.full((2,3),255,np.uint8)).save(d/'focus.png')
            cfg=MobileConfig(d/'source.png',d/'result.tif',d/'working',
                             target_width=10,target_height=8,tile_size=16,
                             reserve_free_gb=0,max_input_pixels=6,
                             sr_candidate=d/'target.tif',focus_mask=d/'focus.png')
            report=run(cfg)
            self.assertEqual(report['target_dimensions'],[10,8])

    def test_wavelet_requires_explicit_focus_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            with self.assertRaises(ValueError):
                MobileConfig(d/'a.png',d/'out.tif',d/'scratch',sr_candidate=d/'sr.tif')

if __name__=='__main__':
    unittest.main()
