import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import tifffile
from mobile_s24.s24_render import MobileConfig,run

class TestMobileOpticalIntegration(unittest.TestCase):
    def test_renderer_uses_aether_optical_core(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);source=d/'source.png';output=d/'optical.tif'
            Image.fromarray(np.full((24,24,3),[120,120,120],dtype=np.uint8),'RGB').save(source)
            cfg=MobileConfig(source,output,d/'work',target_width=48,target_height=48,
                             tile_size=32,psf_sigma=.8,rl_iterations=2,ca_red=.0001,
                             reserve_free_gb=0)
            from aether.core import optical
            old_rl=optical.richardson_lucy_tv
            old_ca=optical.correct_lca
            with patch.object(optical,'richardson_lucy_tv',wraps=old_rl) as rl, \
                 patch.object(optical,'correct_lca',wraps=old_ca) as ca:
                run(cfg)
                self.assertGreater(rl.call_count,0)
                self.assertGreater(ca.call_count,0)
            with tifffile.TiffFile(output) as f:
                self.assertTrue(f.is_bigtiff)
                out=f.pages[0].asarray()
                self.assertEqual(out.shape,(48,48,3))
                self.assertEqual(out.dtype,np.dtype('uint16'))
                self.assertLess(int(np.ptp(out.astype('int32'))),8)

if __name__=='__main__':unittest.main()
