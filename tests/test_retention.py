"""
test_retention.py
=================
Tests for _metric_retention_d1() in pipeline.py.

Why a dedicated file?
----------------------
D1 retention has the most complex behavioural contracts of any metric:

  1. Cohort  = a user's *first-seen* date (not all active dates).
  2. Retained = active on exactly cohort + 1 day (not D+2, not D+3).
  3. Only D+1 counts — later activity does not backfill D1 credit.
  4. The last date in the dataset is excluded from cohort output (no
     observable D+1 window exists for those users).
  5. A user active multiple times on a single day is counted once.

Each contract is a separate test group.  Retention also has more edge-case
tests than any other metric, which would dominate test_metrics.py if colocated.

All tests build the *smallest* possible dataset to exercise the contract,
and comments explicitly state what max_date will be and why it matters.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.conftest import make_clean_df, _date
from src.pipeline import _metric_retention_d1


# ===========================================================================
# Contracts 1 + 2 — Cohort assignment and basic D+1 retention
# ===========================================================================

class TestRetentionCohortMechanics(unittest.TestCase):
    """
    Contract 1: A user's cohort date is their *first-seen* date.
    Contract 2: Retained means active on exactly cohort_date + 1 day.
    """

    def test_all_users_retained_100_percent(self):
        """
        All cohort users return on D+1 → rate = 1.0.

        Dataset:
            user 1: 2023-01-01 and 2023-01-02
            user 2: 2023-01-01 and 2023-01-02
            max_date = 2023-01-02 → only cohort 2023-01-01 in output
        """
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 2, "date": _date("2023-01-01")},
            {"event_id": 3, "user_id": 1, "date": _date("2023-01-02")},
            {"event_id": 4, "user_id": 2, "date": _date("2023-01-02")},
        ])
        result = _metric_retention_d1(df)
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["cohort_date"], "2023-01-01")
        self.assertEqual(row["users"], 2)
        self.assertEqual(row["retained"], 2)
        self.assertAlmostEqual(row["rate"], 1.0, places=4)

    def test_zero_retention(self):
        """
        No cohort user returns on D+1 → retained = 0, rate = 0.0.

        Dataset:
            user 1 and 2: active 2023-01-01 only
            user 3: active 2023-01-03 ← sets max_date so cohort 01 is included
        """
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 2, "date": _date("2023-01-01")},
            {"event_id": 3, "user_id": 3, "date": _date("2023-01-03")},
        ])
        result = _metric_retention_d1(df)
        cohort = next(r for r in result if r["cohort_date"] == "2023-01-01")
        self.assertEqual(cohort["retained"], 0)
        self.assertAlmostEqual(cohort["rate"], 0.0, places=4)

    def test_partial_retention_50_percent(self):
        """
        Half the cohort returns → rate = 0.5.

        Dataset:
            user 1: 2023-01-01 and 2023-01-02  → retained
            user 2: 2023-01-01 only             → not retained
            user 3: 2023-01-03                  ← sets max_date
        """
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 2, "date": _date("2023-01-01")},
            {"event_id": 3, "user_id": 1, "date": _date("2023-01-02")},
            {"event_id": 4, "user_id": 3, "date": _date("2023-01-03")},
        ])
        result = _metric_retention_d1(df)
        cohort = next(r for r in result if r["cohort_date"] == "2023-01-01")
        self.assertEqual(cohort["users"], 2)
        self.assertEqual(cohort["retained"], 1)
        self.assertAlmostEqual(cohort["rate"], 0.5, places=4)


# ===========================================================================
# Contract 3 — Only D+1 counts
# ===========================================================================

class TestRetentionOnlyD1Counts(unittest.TestCase):
    """
    Contract 3: Activity on D+2, D+3, … does NOT count as D1 retention.
    A user must be active on exactly cohort_date + 1 day.
    """

    def test_d2_activity_does_not_count(self):
        """
        User active on D+2 but NOT D+1 must not be counted as retained.

        Dataset:
            user 1: 2023-01-01 (cohort) and 2023-01-03 (D+2, not D+1)
            user 2: 2023-01-04 ← sets max_date so cohort 01 is included
        """
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 1, "date": _date("2023-01-03")},
            {"event_id": 3, "user_id": 2, "date": _date("2023-01-04")},
        ])
        result = _metric_retention_d1(df)
        cohort = next(r for r in result if r["cohort_date"] == "2023-01-01")
        self.assertEqual(cohort["retained"], 0)
        self.assertAlmostEqual(cohort["rate"], 0.0, places=4)

    def test_d1_and_d2_activity_counted_once(self):
        """
        A user active on D+1 AND D+2 is retained once, not twice.

        Dataset:
            user 1: active 2023-01-01, 2023-01-02, 2023-01-03
            user 2: 2023-01-04 ← sets max_date
        """
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 1, "date": _date("2023-01-02")},
            {"event_id": 3, "user_id": 1, "date": _date("2023-01-03")},
            {"event_id": 4, "user_id": 2, "date": _date("2023-01-04")},
        ])
        result = _metric_retention_d1(df)
        cohort = next(r for r in result if r["cohort_date"] == "2023-01-01")
        self.assertEqual(cohort["users"], 1)
        self.assertEqual(cohort["retained"], 1)


# ===========================================================================
# Contract 4 — Last date excluded from cohort output
# ===========================================================================

class TestRetentionLastDateExclusion(unittest.TestCase):
    """
    Contract 4: The max date in the dataset must not appear as a cohort_date.

    Rationale: users who first appear on the last day cannot have a
    measurable D+1 window — including them would artificially deflate rates.
    """

    def test_last_date_not_in_output(self):
        """max_date must be absent from cohort_date values."""
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 2, "date": _date("2023-01-02")},
        ])
        cohort_dates = {r["cohort_date"] for r in _metric_retention_d1(df)}
        self.assertNotIn("2023-01-02", cohort_dates)

    def test_single_date_dataset_returns_empty(self):
        """
        Every event on the same date → that date is both min and max.
        It is excluded (it's the max), so result is empty.
        """
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 2, "date": _date("2023-01-01")},
        ])
        self.assertEqual(_metric_retention_d1(df), [])


# ===========================================================================
# Contract 5 — Deduplication within a day
# ===========================================================================

class TestRetentionDeduplication(unittest.TestCase):
    """
    Contract 5: A user active multiple times on a single day is counted once —
    whether on the cohort day or the return day.
    """

    def test_multiple_events_same_user_same_day_counted_once(self):
        """
        User 1 fires 5 events on D+1. Retained count must be 1, not 5.

        Dataset:
            user 1: 2023-01-01 (cohort) + 5 events on 2023-01-02
            user 2: 2023-01-03 ← sets max_date so cohort 01 is included
        """
        df = make_clean_df(
            [{"event_id": 1, "user_id": 1, "date": _date("2023-01-01")}]
            + [{"event_id": 1 + i, "user_id": 1, "date": _date("2023-01-02")}
               for i in range(1, 6)]
            + [{"event_id": 10, "user_id": 2, "date": _date("2023-01-03")}]
        )
        result = _metric_retention_d1(df)
        cohort = next(r for r in result if r["cohort_date"] == "2023-01-01")
        self.assertEqual(cohort["retained"], 1)


# ===========================================================================
# Edge cases and output schema
# ===========================================================================

class TestRetentionEdgeCases(unittest.TestCase):
    """
    Edge cases and output-schema validation.

    Kept separate so CI output distinguishes "wrong business logic" failures
    from "wrong output format" failures.
    """

    def test_empty_dataframe_returns_empty_list(self):
        """_metric_retention_d1 must handle an empty DataFrame gracefully."""
        df = pd.DataFrame(columns=[
            "user_id", "date", "event_id", "event_type",
            "amount", "country", "device", "session_id", "ts",
        ])
        self.assertEqual(_metric_retention_d1(df), [])

    def test_multiple_cohort_dates_computed_independently(self):
        """
        Users with different first dates form separate cohort buckets.

        Dataset:
            cohort 2023-01-01: user 1, user 2 (neither retained)
            cohort 2023-01-02: user 3 (retained on 2023-01-03)
            max_date = 2023-01-03 → both cohorts must appear in output
        """
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 2, "date": _date("2023-01-01")},
            {"event_id": 3, "user_id": 3, "date": _date("2023-01-02")},
            {"event_id": 4, "user_id": 3, "date": _date("2023-01-03")},
        ])
        result = _metric_retention_d1(df)
        by_date = {r["cohort_date"]: r for r in result}

        self.assertIn("2023-01-01", by_date)
        self.assertIn("2023-01-02", by_date)

        self.assertEqual(by_date["2023-01-01"]["users"],    2)
        self.assertEqual(by_date["2023-01-01"]["retained"], 0)

        self.assertEqual(by_date["2023-01-02"]["users"],    1)
        self.assertEqual(by_date["2023-01-02"]["retained"], 1)
        self.assertAlmostEqual(by_date["2023-01-02"]["rate"], 1.0, places=4)

    def test_output_keys_match_spec(self):
        """Each result dict must have exactly the four specified keys."""
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 2, "date": _date("2023-01-02")},
        ])
        result = _metric_retention_d1(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            set(result[0].keys()),
            {"cohort_date", "users", "retained", "rate"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)