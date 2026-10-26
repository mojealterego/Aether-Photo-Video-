"""HDR radiometric invariants for AETHERIS Ω-VAE."""
import unittest
import numpy as np
from omega_sdk.vae.color_hdr import (srgb_linear_to_acescg,acescg_to_srgb_linear,
     signed_log1p,inverse_signed_log1p,premultiply_alpha,unpremultiply_alpha)
from omega_sdk.vae.contracts import ColorContract,ContractError

class HdrColorTests(unittest.TestCase):
    def test_negative_and_hdr_lights_round_trip_signed_log(self):
        x=np.array([-0.25,0.0,.18,12.0],dtype='f4').reshape(1,1,1,4).repeat(3,axis=1)
        recovered=inverse_signed_log1p(signed_log1p(x,ColorContract()),ColorContract())
        np.testing.assert_allclose(recovered,x,rtol=1e-6,atol=1e-6)
    def test_scene_linear_aces_round_trip_negative_values(self):
        x=np.random.default_rng(3).random((1,3,4,5),dtype=np.float32)*4-.5
        y=acescg_to_srgb_linear(srgb_linear_to_acescg(x))
        np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-6)
    def test_zero_alpha_has_zero_not_nan(self):
        x=np.ones((1,3,5,6),dtype='f4')
        alpha=np.zeros((1,1,5,6),dtype='f4')
        z=unpremultiply_alpha(premultiply_alpha(x,alpha),alpha)
        self.assertFalse(np.isnan(z).any())
        np.testing.assert_array_equal(z,np.zeros_like(x))
    def test_bad_alpha_refused_not_silently_clipped(self):
        with self.assertRaises(ContractError):premultiply_alpha(
            np.ones((1,3,2,2),dtype='f4'),np.ones((1,1,2,2),dtype='f4')*2)
    def test_bad_hdr_inverse_denied(self):
        with self.assertRaises(ContractError):inverse_signed_log1p(
            np.ones((1,3,2,2),dtype='f4')*80,ColorContract())
if __name__=='__main__':unittest.main()
