"""Contract tests for bounded ONNX depth-map generation on S24 previews."""
import unittest
import numpy as np
from aether.vision.depth_onnx import DepthAnyOnnxAdapter

class _Tensor:
    name='input'
    shape=[1,3,64,64]

class FakeSession:
    def get_inputs(self):
        return [_Tensor()]
    def run(self, outputs, inputs):
        x=inputs['input']
        assert x.shape==(1,3,64,64)
        depth=np.tile(np.linspace(0,1,64,dtype=np.float32),(64,1))
        return [depth[None,None,:,:]]

class TestDepthOnnx(unittest.TestCase):
    def test_preserve_preview_shape_and_range(self):
        image=np.zeros((31,73,3),np.float32)
        depth=DepthAnyOnnxAdapter(FakeSession()).predict(image)
        self.assertEqual(depth.shape,(31,73))
        self.assertTrue(np.isfinite(depth).all())
        self.assertGreaterEqual(float(depth.min()),0)
        self.assertLessEqual(float(depth.max()),1)
        self.assertLess(float(depth[15,0]),float(depth[15,-1]))
    def test_near_far_orientation(self):
        image=np.zeros((31,73,3),np.float32)
        a=DepthAnyOnnxAdapter(FakeSession(),near_is_higher=True).predict(image)
        b=DepthAnyOnnxAdapter(FakeSession(),near_is_higher=False).predict(image)
        np.testing.assert_allclose(a+b,1,atol=1e-5)
    def test_fail_closed_invalid_input(self):
        adapter=DepthAnyOnnxAdapter(FakeSession())
        with self.assertRaises(ValueError):
            adapter.predict(np.zeros((2000,2000,3),np.float32))
        with self.assertRaises(ValueError):
            adapter.predict(np.full((8,8,3),np.nan,np.float32))

if __name__=='__main__':
    unittest.main()
