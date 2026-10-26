"""OmniFlow F007: fail-closed ArcFace observation evaluation.

The node evaluates independently supplied 512D features and landmarks.
No weights are loaded here and operator approval is not authenticated by
this library; a production control plane must validate the actor.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from .protocol import (
    OmegaNode, NodeManifest, NodeExecutionMode, TrustClass,
    OMEGA_ABI_VERSION, ExecutionContext
)


class F007_ArcFaceIdentityGuard(OmegaNode):
    @classmethod
    def manifest(cls)->NodeManifest:
        return NodeManifest(
            "F007_ArcFaceIdentityGuard","1.1.0","FACE_AND_IDENTITY",
            OMEGA_ABI_VERSION,NodeExecutionMode.PURE_DETERMINISTIC,
            TrustClass.T0_CORE_TRUSTED,("arcface-r100",),False)

    @staticmethod
    def _decision(label:str,reason:str,*,cosine:float|None=None,
                  landmark_drift:float|None=None)->dict[str,Any]:
        return dict(decision=label,reason=reason,cosine_similarity=cosine,
                    landmark_l2_deviation=landmark_drift,identity_proven=False,
                    may_apply=(label=="APPROVED"))

    def execute(self,ctx:ExecutionContext,inputs:dict[str,Any],
                config:dict[str,Any])->dict[str,Any]:
        threshold=float(config.get("min_cosine_similarity",.8))
        max_drift=float(config.get("max_landmark_l2_deviation",.08))
        min_eye=float(config.get("min_eye_visibility_confidence",.65))
        if (not all(math.isfinite(v) for v in (threshold,max_drift,min_eye))
                or not 0<=threshold<=1 or not 0<=max_drift<=2 or not 0<=min_eye<=1):
            raise ValueError("Uncalibrated or invalid Identity Guard thresholds")
        try:
            ref=np.asarray(inputs["reference_embedding"],dtype=np.float32).reshape(-1)
            cand=np.asarray(inputs["candidate_embedding"],dtype=np.float32).reshape(-1)
            ref_pts=np.asarray(inputs["reference_landmarks_5pt"],dtype=np.float32)
            cand_pts=np.asarray(inputs["candidate_landmarks_5pt"],dtype=np.float32)
            eye=float(inputs["source_eye_visibility_confidence"])
        except (KeyError,TypeError,ValueError,OverflowError):
            return self._decision("ABSTAIN_UNVERIFIABLE","Missing or malformed source evidence")
        if (ref.shape!=(512,) or cand.shape!=(512,)
                or ref_pts.shape!=(5,2) or cand_pts.shape!=(5,2)
                or not np.isfinite(ref).all() or not np.isfinite(cand).all()
                or not np.isfinite(ref_pts).all() or not np.isfinite(cand_pts).all()
                or not math.isfinite(eye) or not 0<=eye<=1):
            return self._decision("ABSTAIN_UNVERIFIABLE","Nonfinite or invalid embedding/landmarks/eyes")
        ref_norm=float(np.linalg.norm(ref.astype(np.float64)))
        cand_norm=float(np.linalg.norm(cand.astype(np.float64)))
        eye_distance=float(np.linalg.norm(ref_pts[0]-ref_pts[1]))
        if min(ref_norm,cand_norm)<1e-8 or eye_distance<2:
            return self._decision("ABSTAIN_UNVERIFIABLE","Degenerate embeddings or eye landmarks")
        if eye<min_eye:
            return self._decision("ABSTAIN_UNVERIFIABLE","Source eyes not reliably observable")
        model_ref=inputs.get("reference_model_id")
        model_candidate=inputs.get("candidate_model_id")
        if not model_ref or not model_candidate or model_ref!=model_candidate:
            return self._decision("REJECT_IDENTITY_DRIFT","Incompatible ArcFace embedding spaces")
        similarity=float(np.clip(np.dot(ref.astype(np.float64),cand.astype(np.float64))
                                 /(ref_norm*cand_norm),-1,1))
        drift=float(np.mean(np.linalg.norm(ref_pts-cand_pts,axis=1))/eye_distance)
        if similarity<threshold or drift>max_drift:
            return self._decision("REJECT_IDENTITY_DRIFT","Embedding or landmark consistency veto",
                                  cosine=similarity,landmark_drift=drift)
        if type(inputs.get("human_approved",False)) is not bool:
            return self._decision("ABSTAIN_UNVERIFIABLE","Malformed operator approval")
        if not inputs.get("human_approved",False):
            return self._decision("NEEDS_REVIEW","Explicit human approval missing",
                                  cosine=similarity,landmark_drift=drift)
        return self._decision("APPROVED","Similarity veto passed and operator approval asserted",
                              cosine=similarity,landmark_drift=drift)
