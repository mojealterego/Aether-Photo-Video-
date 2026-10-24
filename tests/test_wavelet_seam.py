"""Wavelet global dyadic alignment must be independent of output tile size."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image
from mobile_s24.s24_render import MobileConfig,run

class TestWaveletSeam(unittest.TestCase):
    def test_tile_size_16_and_32_produce_identical_rgb16(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            yy,xx=np.indices((35,39))
            base=np.stack([(xx*3+yy*5)%255,(xx*7+yy*3)%255,(yy*8)%255],axis=2).astype(np.uint8)
            hypothesis=np.clip(base.astype(np.int16)+((xx%2)*2-1)[:,:,None]*24,0,255).astype(np.uint8)
            Image.fromarray(base).save(p/'src.png')
            tifffile.imwrite(p/'candidate.tif',hypothesis,photometric='rgb',metadata=None)
            Image.fromarray(np.full((35,39),255,np.uint8)).save(p/'focus.png')
            results=[]
            for tile in (16,32):
                cfg=MobileConfig(p/'src.png',p/f'out-{tile}.tif',p/f'w-{tile}',
                                 target_width=78,target_height=70,tile_size=tile,
                                 reserve_free_gb=0,focus_mask=p/'focus.png',
                                 sr_candidate=p/'candidate.tif',wavelet_levels=3,wavelet_gain=1.)
                run(cfg)
                results.append(tifffile.imread(p/f'out-{tile}.tif'))
            np.testing.assert_array_equal(results[0],results[1])

if __name__=='__main__':
    unittest.main()
