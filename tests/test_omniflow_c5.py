"""ComfyUI native dependency gate C5 regression tests."""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/"ci/verify_zero_comfy_c5.py"
spec=importlib.util.spec_from_file_location("omniflow_c5",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class NoComfyGate(unittest.TestCase):
    def test_detects_direct_comfy_import(self):
        self.assertTrue(module.scan_python("import comfy\n"))
    def test_detects_dynamic_comfy_import(self):
        self.assertTrue(module.scan_python(
            'import importlib\nimportlib.import_module("comfy.model_management")\n'))
    def test_detects_legacy_plugin_import(self):
        self.assertTrue(module.scan_python("from custom_nodes.foo import bar\n"))
    def test_allows_native_only(self):
        self.assertEqual(module.scan_python(
            "import hashlib\nfrom omega_sdk.protocol import OmegaNode\n"),[])
    def test_allows_documentation_string(self):
        self.assertEqual(module.scan_python(
            '"""does not depend on ComfyUI"""\nimport json\n'),[])
