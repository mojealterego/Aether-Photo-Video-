"""Identity Guard: fail-closed acceptance policy (synthetic embeddings only)."""
from __future__ import annotations

import unittest
import numpy as np

from aether.vision.identity_guard import evaluate_identity_guard


class Encoder:
    def __init__(self, a, b):
        self.vectors = iter((a, b))
        self.calls = 0

    def encode(self, image, landmarks):
        self.calls += 1
        return next(self.vectors)


def vector(index):
    v = np.zeros(512, dtype=np.float32)
    v[index] = 1.0
    return v


class IdentityGuardTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((112, 112, 3), dtype=np.uint8)
        self.landmarks = np.array(
            [[30, 40], [70, 40], [50, 55], [35, 75], [65, 75]],
            dtype=np.float32,
        )

    def check(self, encoder, *, human_approved=False, threshold=0.80):
        return evaluate_identity_guard(
            encoder, self.image, self.landmarks, self.image, self.landmarks,
            min_cosine=threshold, human_approved=human_approved,
        )

    def test_matching_embeddings_never_auto_apply_without_human_approval(self):
        result = self.check(Encoder(vector(0), vector(0)))
        self.assertEqual(result.status, "needs_review")
        self.assertFalse(result.may_apply)

    def test_human_approval_and_matching_embeddings_are_both_required(self):
        result = self.check(Encoder(vector(0), vector(0)), human_approved=True)
        self.assertEqual(result.status, "approved")
        self.assertTrue(result.may_apply)

    def test_mismatched_embeddings_are_rejected_even_with_human_approval(self):
        result = self.check(Encoder(vector(0), vector(1)), human_approved=True)
        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.may_apply)

    def test_missing_landmarks_fail_closed(self):
        with self.assertRaises(ValueError):
            evaluate_identity_guard(
                Encoder(vector(0), vector(0)),
                self.image, None, self.image, self.landmarks,
                min_cosine=0.8, human_approved=True,
            )

    def test_threshold_must_be_explicit_and_calibrated(self):
        with self.assertRaises(ValueError):
            self.check(Encoder(vector(0), vector(0)), threshold=float("nan"))


if __name__ == "__main__":
    unittest.main()
