"""Measured PSF -> RL-TV -> RGB16 BigTIFF regression contracts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import tifffile
from mobile_s24.s24_render import MobileConfig, run


class TestMeasuredPSF(unittest.TestCase):
    def test_kernel_integration_and_rgb16(self):
        with tempfile.TemporaryDirectory() as p:
            d=Path(p)
            source=d/'src.png'; output=d/'out.tif'; kernel=d/'psf.npy'
            Image.fromarray(np.full((30,30,3),120,dtype='uint8'),'RGB').save(source)
            np.save(kernel,np.ones((3,3),dtype='f4')/9)
            cfg=MobileConfig(source,output,d/'work',target_width=48,target_height=48,
                             tile_size=32,psf_kernel=kernel,rl_iterations=2,
                             reserve_free_gb=0)
            import aether.core.optical as optical
            with patch.object(optical,'richardson_lucy_tv',
                              wraps=optical.richardson_lucy_tv) as runner:
                run(cfg)
                self.assertGreater(runner.call_count,0)
            with tifffile.TiffFile(output) as tif:
                self.assertTrue(tif.is_bigtiff)
                self.assertEqual(tif.pages[0].shape,(48,48,3))
                self.assertEqual(tif.pages[0].dtype,np.dtype('uint16'))

    def test_psf_hash_is_part_of_resume_signature(self):
        with tempfile.TemporaryDirectory() as p:
            d=Path(p)
            source=d/'src.png'; output=d/'out.tif'; kernel=d/'psf.npy'
            Image.fromarray(np.full((10,10,3),120,dtype='uint8'),'RGB').save(source)
            np.save(kernel,np.ones((3,3),dtype='f4')/9)
            cfg=MobileConfig(source,output,d/'work',target_width=32,target_height=32,
                             tile_size=32,psf_kernel=kernel,rl_iterations=1,
                             reserve_free_gb=0,keep_stage=True)
            run(cfg)
            output.unlink()
            np.save(kernel,np.eye(3,dtype='f4')/3)
            with self.assertRaisesRegex(ValueError,'signature'):
                run(cfg)

    def test_reject_ambiguous_psf_sources(self):
        with tempfile.TemporaryDirectory() as p:
            d=Path(p)
            with self.assertRaises(ValueError):
                MobileConfig(d/'a.png',d/'b.tif',d/'work',psf_sigma=.7,
                             psf_kernel=d/'calibrated.npy',rl_iterations=4)


if __name__=='__main__':
    unittest.main()
