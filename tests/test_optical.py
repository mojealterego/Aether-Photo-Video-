import unittest
import numpy as np
from aether.core.optical import (
    srgb_to_linear, linear_to_srgb, LCAChannel,
    correct_lca, anisotropic_gaussian_psf, richardson_lucy_tv
)

class TestOptical(unittest.TestCase):
    def test_srgb_eotf_roundtrip(self):
        x=np.linspace(0,1,1024,dtype="f4").reshape((1,1024,1)).repeat(3,axis=2)
        np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)),x,atol=2e-7)

    def test_srgb_reference(self):
        x=np.full((1,1,3),.04045,dtype="f4")
        self.assertAlmostEqual(float(srgb_to_linear(x)[0,0,0]),.04045/12.92,places=7)

    def test_lca_identity_and_green_unchanged(self):
        x=np.random.default_rng(4).random((31,45,3),dtype="f4")
        np.testing.assert_array_equal(correct_lca(x,origin_xy=(5,9),full_wh=(100,80)),x)
        y=correct_lca(x,origin_xy=(5,9),full_wh=(100,80),red=LCAChannel(radial=(.0002,0,0,0)))
        np.testing.assert_array_equal(y[:,:,1],x[:,:,1])
        self.assertTrue(np.isfinite(y).all())

    def test_anisotropic_psf(self):
        p=anisotropic_gaussian_psf(1.,2.,.5,7)
        self.assertEqual(p.shape,(15,15))
        self.assertAlmostEqual(float(p.sum()),1.,places=6)
        self.assertTrue((p>=0).all())

    def test_rl_focus_zero_preserves_input(self):
        x=np.random.default_rng(5).random((15,15,3),dtype="f4")
        psf=anisotropic_gaussian_psf(1,1,0,3)
        np.testing.assert_array_equal(richardson_lucy_tv(x,psf,focus=np.zeros(x.shape[:2])),x)

    def test_rl_constant_interior(self):
        x=np.full((48,40,3),.4,dtype="f4")
        p=anisotropic_gaussian_psf(1.3,1.2,.2,5)
        r=richardson_lucy_tv(x,p,iterations=4,tv_weight=0)
        self.assertTrue(np.isfinite(r).all())
        self.assertLess(float(np.max(abs(r[22:26,18:22]-.4))),.01)

    def test_invalid_kernel(self):
        with self.assertRaises(ValueError):
            richardson_lucy_tv(np.ones((5,5,3),dtype="f4"),np.ones((2,2),dtype="f4"))

if __name__=="__main__":
    unittest.main()
