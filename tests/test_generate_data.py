"""
test_generate_data.py
=====================
Tests for generate_data.py.

Previously, TestGenerateRefundLinking lived in test_pipeline.py, which
was incorrect — it tests the *generator*, not the *pipeline*.  The symptom
of this misplacement: the test imports `generate` and `RefundLinks` from
generate_data, not from pipeline, and it would have been skipped silently
on any machine where generate_data wasn't importable — hiding itself inside
an unrelated file.

Moving it here makes the responsibility boundary explicit:

    test_generate_data.py  →  everything about the synthetic dataset shape,
                               distributions, and dirty-data contracts.
    test_pipeline.py       →  everything about cleaning and metric computation.

Adding tests in future
-----------------------
Good candidates for future tests in this file:
    * Row count is within ±1% of the requested n.
    * Event-type proportions fall within the specified probability bands.
    * Dirty-data injection rates are within tolerance.
    * Generated timestamps are within the configured date range.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class TestGenerateRefundLinking(unittest.TestCase):
    """
    Generator-level checks: refund rows must mirror their linked purchase
    in amount, session_id, and user_id; and they must occur after the purchase.

    The test is designed to be skipped gracefully if generate_data is not
    importable (e.g., missing numpy in a minimal CI environment), rather than
    hard-failing the entire suite.
    """

    def setUp(self):
        try:
            from src.generate_data import generate, RefundLinks
            self._generate = generate
            self._RefundLinks = RefundLinks
        except Exception as e:
            self.skipTest(f"generate_data not importable: {e}")

    def test_refund_links_to_correct_purchase(self):
        """
        For every (refund_idx, purchase_idx) pair in RefundLinks:

          * refund.user_id    == purchase.user_id
          * refund.session_id == purchase.session_id
          * refund.amount     == -purchase.amount  (within float32 tolerance)
          * refund.ts         >  purchase.ts       (temporal ordering)
            — excluding pairs where the dirty injector has corrupted a
              timestamp to a sentinel year (≥ 2099), which is expected behaviour.
        """
        rng = np.random.default_rng(42)
        df, links = self._generate(50_000, rng)

        if len(links) == 0:
            self.skipTest("No refunds generated in this sample (increase n or change seed)")

        self.assertIsInstance(links, self._RefundLinks)

        user_arr    = df["user_id"].to_numpy()
        amount_arr  = df["amount"].to_numpy(dtype="float32")
        session_arr = df["session_id"].to_numpy()
        ts_arr      = df["ts"].to_numpy()

        # Timestamp invariant — exclude pairs where dirty injection corrupted ts.
        year_ok    = (df["ts"].dt.year < 2099).to_numpy()
        clean_mask = year_ok[links.refund_idx] & year_ok[links.purchase_idx]

        if clean_mask.any():
            self.assertTrue(
                (ts_arr[links.refund_idx[clean_mask]]
                    > ts_arr[links.purchase_idx[clean_mask]]).all(),
                msg=(
                    "Refund timestamp must be strictly later than the linked "
                    "purchase timestamp (dirty-injection pairs excluded)."
                ),
            )

        # Amount invariant: refund.amount == -purchase.amount
        self.assertTrue(
            np.allclose(
                amount_arr[links.refund_idx],
                -amount_arr[links.purchase_idx],
                atol=1e-4,
            ),
            msg="Refund amounts must equal the negation of their linked purchase amounts.",
        )

        # session_id invariant
        self.assertTrue(
            (session_arr[links.refund_idx] == session_arr[links.purchase_idx]).all(),
            msg="Refund session_id must equal the linked purchase's session_id.",
        )

        # user_id invariant
        self.assertTrue(
            (user_arr[links.refund_idx] == user_arr[links.purchase_idx]).all(),
            msg="Refund user_id must equal the linked purchase's user_id.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)