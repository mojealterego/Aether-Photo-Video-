"""Fail-closed ArcFace similarity veto and mandatory human approval.

A matching embedding is not proof of identity. The caller must supply a
calibrated threshold and independently detected five-point landmarks. This
module neither changes image pixels nor persists a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Protocol

import numpy as np


class FaceEncoder(Protocol):
    def encode(self, rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class IdentityGuardDecision:
    status: Literal["rejected", "needs_review", "approved"]
    cosine_similarity: float
    threshold: float
    reason: str

    @property
    def may_apply(self) -> bool:
        return self.status == "approved"


def _validate_image_and_landmarks(image: np.ndarray, landmarks: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 16:
        raise ValueError(f"{label}: RGB uint8 image with valid dimensions required")
    if landmarks is None:
        raise ValueError(f"{label}: five facial landmarks required")
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError(f"{label}: five finite facial landmarks required")
    if (np.any(points < 0) or np.any(points[:, 0] >= rgb.shape[1]) or
            np.any(points[:, 1] >= rgb.shape[0])):
        raise ValueError(f"{label}: landmarks must lie inside the image")
    if np.linalg.norm(points[0] - points[1]) < 2.0:
        raise ValueError(f"{label}: degenerate eye landmarks")
    return rgb, points


def evaluate_identity_guard(
    encoder: FaceEncoder,
    reference_rgb: np.ndarray,
    reference_landmarks: np.ndarray,
    candidate_rgb: np.ndarray,
    candidate_landmarks: np.ndarray,
    *,
    min_cosine: float,
    human_approved: bool = False,
) -> IdentityGuardDecision:
    """Evaluate a candidate, never commit it automatically.

    Approval means only that an operator approved the candidate after an
    ArcFace similarity veto. It does not certify historical identity.
    """
    if not isinstance(min_cosine, (float, int)) or not math.isfinite(min_cosine) or not 0 <= min_cosine <= 1:
        raise ValueError("An explicit finite calibrated cosine threshold in [0,1] is required")
    if type(human_approved) is not bool:
        raise ValueError("human_approved must be a boolean")
    source, source_points = _validate_image_and_landmarks(reference_rgb, reference_landmarks, "reference")
    target, target_points = _validate_image_and_landmarks(candidate_rgb, candidate_landmarks, "candidate")
    ref = np.asarray(encoder.encode(source, source_points), dtype=np.float32).reshape(-1)
    cand = np.asarray(encoder.encode(target, target_points), dtype=np.float32).reshape(-1)
    if ref.shape != (512,) or cand.shape != (512,) or not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("ArcFace must return finite 512-d embeddings")
    den = float(np.linalg.norm(ref) * np.linalg.norm(cand))
    if den < 1e-8 or not math.isfinite(den):
        raise ValueError("Invalid ArcFace embedding norms")
    similarity = float(np.clip(np.dot(ref, cand) / den, -1.0, 1.0))
    if similarity < min_cosine:
        return IdentityGuardDecision("rejected", similarity, float(min_cosine), "ArcFace similarity below threshold")
    if not human_approved:
        return IdentityGuardDecision("needs_review", similarity, float(min_cosine), "Explicit operator approval missing")
    return IdentityGuardDecision("approved", similarity, float(min_cosine), "Similarity veto passed; operator approved")
