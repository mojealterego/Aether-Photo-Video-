"""Native DA3 and CodeFormer worker integration contract tests with injected fakes."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"omniflow-worker"))
from omega_sdk.protocol import ContentAddressedStorage,ExecutionContext
from omega_sdk.vision_native import DepthOmega,CodeFormerOmega

class FakePrediction:
    depth=np.ones((1,8,8),dtype="f4")*.3
    conf=np.ones((1,8,8),dtype="f4")*.75
    processed_images=np.zeros((1,8,8,3),dtype="u1")
class FakeDepth:
    def inference(self,frames,process_res=504):return FakePrediction()
class FakeCodeFormer:
    def __init__(self):self.weight=None
    def restore_crop(self,rgb512,w):
        self.weight=w
        return np.minimum(rgb512+1,255).astype("u1")

class VisionAcceptance(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.cas=ContentAddressedStorage(root/"cas")
        self.ctx=ExecutionContext("job","ten_a",1,"DOC",42,
                                  self.cas.root_dir,self.cas.tmp_dir)
    def store_img(self,arr):
        path=self.cas.tmp_dir/"input.png"
        Image.fromarray(arr,"RGB").save(path)
        return self.cas.commit_file_atomically(
            path,"ten_a","image/png","display-sRGB")
    def test_depth_creates_relative_float32_artifact(self):
        image=self.store_img(np.zeros((8,8,3),dtype="u1"))
        out=DepthOmega(FakeDepth(),"depth-anything/DA3-SMALL").execute(
            self.ctx,{"image_artifact":vars(image)}, {})
        self.assertEqual(out["units"],"RELATIVE_NORMALIZED")
        self.assertEqual(out["verification_status"],"UNVERIFIED")
        self.assertEqual(out["dimensions"],[8,8])
        with np.load(out["artifact_handle"]["cas_path"]) as data:
            self.assertEqual(data["depth"].dtype,np.float32)
            self.assertEqual(float(data["depth"][0,0]),float(np.float32(.3)))
    def test_depth_rejects_nan_confidence(self):
        class Bad:
            def inference(self,frames,process_res=504):
                p=FakePrediction()
                p.conf=p.conf.copy()
                p.conf[0,0,0]=float("nan")
                return p
        image=self.store_img(np.zeros((8,8,3),dtype="u1"))
        with self.assertRaises(ValueError):
            DepthOmega(Bad(),"depth-anything/DA3-SMALL").execute(
                self.ctx,{"image_artifact":vars(image)}, {})
    def test_depth_refuses_cross_tenant_artifact(self):
        image=self.store_img(np.zeros((8,8,3),dtype="u1"))
        meta=vars(image)
        meta["tenant_id"]="ten_b"
        with self.assertRaises(PermissionError):
            DepthOmega(FakeDepth(),"depth-anything/DA3-SMALL").execute(
                self.ctx,{"image_artifact":meta}, {})
    def test_codeformer_candidate_cannot_be_auto_accepted(self):
        image=self.store_img(np.zeros((512,512,3),dtype="u1"))
        model=FakeCodeFormer()
        out=CodeFormerOmega(model).execute(self.ctx,
            {"image_artifact":vars(image)},{"fidelity_weight":.9})
        self.assertEqual(model.weight,.9)
        self.assertEqual(out["verification_status"],"NEEDS_REVIEW")
        self.assertFalse(out["identity_verified"])
        self.assertTrue(Path(out["artifact_handle"]["cas_path"]).is_file())
    def test_codeformer_rejects_invalid_fidelity_and_shape(self):
        image=self.store_img(np.zeros((8,8,3),dtype="u1"))
        with self.assertRaises(ValueError):
            CodeFormerOmega(FakeCodeFormer()).execute(
                self.ctx,{"image_artifact":vars(image)},{"fidelity_weight":.5})
        with self.assertRaises(ValueError):
            CodeFormerOmega(FakeCodeFormer()).execute(
                self.ctx,{"image_artifact":vars(image)},{"fidelity_weight":2})
