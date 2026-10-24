import unittest
import numpy as np
from aether.core.focus_frequency import guided_filter_gray,focus_bokeh_mask,adaptive_master_base,faithful_frequency_fusion

class TestFocusFrequency(unittest.TestCase):
    def test_master_base_mask_and_protected(self):
        a=np.full((16,16,3),.2,dtype="f4");b=np.full_like(a,.8)
        m=np.zeros((16,16),dtype="f4");m[3:12,3:12]=1
        p=np.zeros_like(m);p[5:8,5:8]=1
        z=adaptive_master_base(a,b,m,p)
        self.assertAlmostEqual(float(z[4,4,0]),.8,places=6)
        self.assertAlmostEqual(float(z[6,6,0]),.2,places=6)
        self.assertAlmostEqual(float(z[0,0,0]),.2,places=6)
    def test_zero_focus_preserves_master(self):
        rng=np.random.default_rng(2);a=rng.random((32,35,3),dtype="f4")
        z=faithful_frequency_fusion(a,1-a,np.zeros(a.shape[:2],dtype="f4"),sigma=3)
        np.testing.assert_allclose(z,a,atol=3e-7)
    def test_focus_one_uses_sr_detail(self):
        rng=np.random.default_rng(3);a=np.full((30,30,3),.4,dtype="f4")
        b=a+rng.normal(0,.1,a.shape).astype("f4")
        z=faithful_frequency_fusion(a,b,np.ones(a.shape[:2],dtype="f4"),sigma=2)
        self.assertGreater(float(z.std()),.03)
    def test_bokeh_mask_registered(self):
        a=np.zeros((48,48,3),dtype="f4");a[10:36,10:36]=.3
        d=np.zeros((48,48),dtype="f4");d[10:36,10:36]=.5
        m=focus_bokeh_mask(a,depth=d,focus_depth=.5,edge_weight=.3,guide_radius=3)
        self.assertEqual(m.shape,(48,48))
        self.assertTrue(np.isfinite(m).all())
        self.assertGreater(float(m[24,24]),float(m[0,0]))
    def test_guided_filter_invalid_shape(self):
        with self.assertRaises(ValueError):guided_filter_gray(np.ones((4,4)),np.ones((5,5)))
if __name__=="__main__":unittest.main()
