"""End-to-end contract for ONNX depth to PNG-compatible focus mask."""
import unittest
import numpy as np
from aether.vision.depth_cli import build_depth_mask

class _Tensor:
    name='input'
    shape=[1,3,64,64]

class FakeSession:
    def get_inputs(self): return [_Tensor()]
    def run(self, outputs, inputs):
        assert inputs['input'].shape==(1,3,64,64)
        depth=np.tile(np.linspace(0,1,64,dtype=np.float32),(64,1))
        return [depth[None,None,:,:]]

class TestDepthMaskCLI(unittest.TestCase):
    def test_mask_from_rgb_preview(self):
        rgb=np.full((24,40,3),.4,np.float32)
        mask=build_depth_mask(rgb,FakeSession(),focus_depth=.5,tolerance=.22)
        self.assertEqual(mask.shape,(24,40))
        self.assertEqual(mask.dtype,np.uint8)
        self.assertGreater(int(mask.max()),0)
    def test_reject_oversized_preview(self):
        with self.assertRaises(ValueError):
            build_depth_mask(np.zeros((1200,1200,3),np.float32),FakeSession())

class TestRelativeFocusMask(unittest.TestCase):
    def test_subject_region_dominates_background(self):
        from aether.vision.depth_onnx import relative_depth_focus_mask
        preview=np.full((48,48,3),.4,np.float32)
        depth=np.zeros((48,48),np.float32);depth[12:36,12:36]=.72
        mask=relative_depth_focus_mask(preview,depth,focus_depth=.72,tolerance=.15,texture_weight=0)
        self.assertGreater(float(mask[24,24]),float(mask[0,0])+.7)
    def test_invalid_relative_depth_fails_closed(self):
        from aether.vision.depth_onnx import relative_depth_focus_mask
        with self.assertRaises(ValueError):
            relative_depth_focus_mask(np.full((8,8,3),.4,np.float32),np.full((8,8),3,np.float32))

if __name__=='__main__':
    unittest.main()
