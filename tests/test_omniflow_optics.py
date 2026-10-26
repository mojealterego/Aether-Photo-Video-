"""Regression tests for P028 Richardson-Lucy and P029 lateral chromatic correction."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"omniflow-worker"))
from omega_sdk.optics import chromatic_aberration_correct_f32, richardson_lucy_regularized_f32

class OpticsTests(unittest.TestCase):
    def test_zero_ca_is_identity(self):
        rgb=np.random.default_rng(1).random((17,19,3),dtype=np.float32)
        result=chromatic_aberration_correct_f32(rgb,(0,0,0),(0,0,0))
        np.testing.assert_allclose(result,rgb,atol=1e-5)
    def test_ca_keeps_green_channel(self):
        rgb=np.random.default_rng(3).random((17,19,3),dtype=np.float32)
        result=chromatic_aberration_correct_f32(rgb,(.05,0,0),(-.05,0,0))
        np.testing.assert_array_equal(result[:,:,1],rgb[:,:,1])
    def test_identity_psf_preserves_image(self):
        rgb=np.random.default_rng(4).random((15,19,3),dtype=np.float32)*0.7+0.1
        result=richardson_lucy_regularized_f32(
            rgb,np.ones((1,1),dtype="f4"),iterations=2,tv_lambda=0)
        np.testing.assert_allclose(result,rgb,atol=1e-4)
    def test_invalid_psf_fails(self):
        rgb=np.zeros((10,10,3),dtype="f4")
        with self.assertRaises(ValueError):
            richardson_lucy_regularized_f32(rgb,np.zeros((3,3),dtype="f4"))
        with self.assertRaises(ValueError):
            richardson_lucy_regularized_f32(rgb,np.ones((3,3)),iterations=0)
    def test_nans_fail_before_processing(self):
        rgb=np.zeros((10,10,3),dtype="f4")
        rgb[0,0,0]=float("nan")
        with self.assertRaises(ValueError):
            chromatic_aberration_correct_f32(rgb,(0,0,0),(0,0,0))
