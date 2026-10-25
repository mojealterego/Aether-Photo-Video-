"""CodeFormer inference adapter contracts; no real checkpoint bundled."""
from __future__ import annotations

import unittest

import numpy as np

from aether.vision.codeformer_onnx import CodeFormerOnnxAdapter


class Descriptor:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeCodeFormer:
    def __init__(self, shape=(1, 3, 512, 512), value=0.0):
        self.shape = shape
        self.value = value
        self.last_input = None

    def get_inputs(self):
        return [Descriptor("face", [1, 3, 512, 512])]

    def get_outputs(self):
        return [Descriptor("result", list(self.shape))]

    def run(self, names, inputs):
        self.last_input = inputs["face"]
        return [np.full(self.shape, self.value, dtype=np.float32)]


class TestCodeFormerOnnx(unittest.TestCase):
    def test_preprocess_postprocess_and_shape(self):
        session = FakeCodeFormer()
        adapter = CodeFormerOnnxAdapter(session)
        white = np.full((512, 512, 3), 255, dtype=np.uint8)
        result = adapter.restore_crop(white)
        self.assertEqual(result.shape, white.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(int(result[0, 0, 0]), 128)
        self.assertEqual(session.last_input.shape, (1, 3, 512, 512))
        self.assertAlmostEqual(float(session.last_input[0, 0, 0, 0]), 1.0)

    def test_refuse_wrong_crop(self):
        adapter = CodeFormerOnnxAdapter(FakeCodeFormer())
        for crop in (np.zeros((256, 256, 3), "u1"),
                     np.zeros((512, 512, 3), "f4")):
            with self.assertRaises(ValueError):
                adapter.restore_crop(crop)

    def test_refuse_export_geometry_mismatch(self):
        with self.assertRaises(ValueError):
            CodeFormerOnnxAdapter(FakeCodeFormer(shape=(1, 3, 256, 256)))

    def test_refuse_nonfinite_and_unsupported_output_scale(self):
        white = np.zeros((512, 512, 3), "u1")
        for value in (float("nan"), 1.25, -1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CodeFormerOnnxAdapter(FakeCodeFormer(value=value)).restore_crop(white)


if __name__ == "__main__":
    unittest.main()
