"""Server-authoritative ComfyUI model availability: no implicit checkpoint claims."""
from __future__ import annotations

import unittest

from aether.remote.preflight import verify_remote_graph


class FakeComfy:
    choices = {
        "DepthAnythingLoader": ("depth-anything-v2.onnx",),
        "ArcFaceLoader": ("arcface-r100.onnx",),
        "CodeFormerLoader": ("codeformer.onnx",),
    }

    def json(self, endpoint: str):
        name = endpoint.rsplit("/", 1)[-1]
        if name not in self.choices:
            return {}
        return {
            name: {
                "input": {
                    "required": {"model_name": [list(self.choices[name]), {}]},
                },
                "output": ["MODEL"],
            }
        }


def graph():
    return {
        "1": {"class_type": "DepthAnythingLoader",
              "inputs": {"model_name": "depth-anything-v2.onnx"}},
        "2": {"class_type": "ArcFaceLoader",
              "inputs": {"model_name": "arcface-r100.onnx"}},
        "3": {"class_type": "CodeFormerLoader",
              "inputs": {"model_name": "codeformer.onnx"}},
    }


class TestComfyModels(unittest.TestCase):
    def test_each_neural_loader_verified_against_remote_inventory(self):
        report = verify_remote_graph(FakeComfy(), graph())
        self.assertEqual(report["checked_classes"], 3)
        self.assertTrue(report["server_verified"])

    def test_missing_codeformer_checkpoint_is_rejected(self):
        request = graph()
        request["3"]["inputs"]["model_name"] = "missing-codeformer.onnx"
        with self.assertRaisesRegex(RuntimeError, "CodeFormerLoader.model_name"):
            verify_remote_graph(FakeComfy(), request)

    def test_missing_depth_checkpoint_is_rejected(self):
        request = graph()
        request["1"]["inputs"]["model_name"] = "missing-depth.onnx"
        with self.assertRaisesRegex(RuntimeError, "DepthAnythingLoader.model_name"):
            verify_remote_graph(FakeComfy(), request)

    def test_unavailable_server_node_fails_closed(self):
        request = {"1": {"class_type": "UninstalledLoader", "inputs": {}}}
        with self.assertRaisesRegex(RuntimeError, "Missing ComfyUI node"):
            verify_remote_graph(FakeComfy(), request)

    def test_malformed_node_ids_rejected_before_requests(self):
        request = {"../1": {"class_type": "CodeFormerLoader", "inputs": {}}}
        with self.assertRaises(ValueError):
            verify_remote_graph(FakeComfy(), request)


if __name__ == "__main__":
    unittest.main()
