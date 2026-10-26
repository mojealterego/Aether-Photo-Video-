"""Face evidence evaluation: no neural weight invocation in these contract tests."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"omniflow-worker"))
from omega_sdk.protocol import ExecutionContext
from omega_sdk.identity_guard import F007_ArcFaceIdentityGuard

class IdentityNodeTests(unittest.TestCase):
    def setUp(self):
        self.node=F007_ArcFaceIdentityGuard()
        self.ctx=ExecutionContext("job","tenant",1,"DOC",0,Path("."),Path("."))
        self.ref=np.zeros(512,dtype="f4");self.ref[0]=1
        self.points=[[20,20],[60,20],[40,35],[25,50],[55,50]]
        self.inputs=dict(reference_embedding=self.ref.tolist(),
            candidate_embedding=self.ref.tolist(),
            reference_landmarks_5pt=self.points,candidate_landmarks_5pt=self.points,
            source_eye_visibility_confidence=1.0,reference_model_id="arcface-r100",
            candidate_model_id="arcface-r100")
    def run_node(self,**updates):
        x=dict(self.inputs);x.update(updates)
        return self.node.execute(self.ctx,x,dict(
            min_cosine_similarity=.8,max_landmark_l2_deviation=.08))
    def test_valid_result_is_review_not_auto_pass(self):
        self.assertEqual(self.run_node()["decision"],"NEEDS_REVIEW")
    def test_matching_human_approval(self):
        self.assertEqual(self.run_node(human_approved=True)["decision"],"APPROVED")
    def test_missing_eye_visibility_abstains(self):
        x=dict(self.inputs);x.pop("source_eye_visibility_confidence")
        self.assertEqual(self.node.execute(self.ctx,x,{})["decision"],"ABSTAIN_UNVERIFIABLE")
    def test_eyes_occluded_abstains_despite_approval(self):
        self.assertEqual(self.run_node(source_eye_visibility_confidence=.1,
            human_approved=True)["decision"],"ABSTAIN_UNVERIFIABLE")
    def test_model_id_mismatch_rejected(self):
        self.assertEqual(self.run_node(candidate_model_id="different",
            human_approved=True)["decision"],"REJECT_IDENTITY_DRIFT")
    def test_biometric_similarity_veto(self):
        other=np.zeros(512,dtype="f4");other[1]=1
        self.assertEqual(self.run_node(candidate_embedding=other.tolist(),
            human_approved=True)["decision"],"REJECT_IDENTITY_DRIFT")
    def test_corrupt_embedding_abstains(self):
        other=self.ref.copy();other[1]=float("nan")
        self.assertEqual(self.run_node(candidate_embedding=other.tolist())["decision"],
                         "ABSTAIN_UNVERIFIABLE")
    def test_landmark_drift_rejected(self):
        points=[[x+12,y] for x,y in self.points]
        self.assertEqual(self.run_node(candidate_landmarks_5pt=points)["decision"],
                         "REJECT_IDENTITY_DRIFT")
if __name__=="__main__":
    unittest.main()
