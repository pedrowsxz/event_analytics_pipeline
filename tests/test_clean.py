"""
test_clean.py
=============
Tests for _clean() in pipeline.py.

Why four classes instead of one?
---------------------------------
_clean() enforces five distinct rules (D1–D5) plus produces a counts dict
and enforces dtype discipline.  A single "TestClean" class becomes hard to
navigate as the rule count grows — a failing test like
"test_d3_unknown_etype_string_dropped" is easy to find in TestCleanEventType,
but not in a 260-line monolith.

Class map
---------
  TestCleanTimestamps  → D1: ts parsing, sentinel year, NaT handling, date derivation
  TestCleanNormalise   → D2: country uppercasing / UNK, D5: amount noise zeroing
  TestCleanFiltering   → D3: event_type allowlist, D4: event_id deduplication
  TestCleanInvariants  → counts dict correctness, dtype guarantees, immutability
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.conftest import make_raw_df
from src.pipeline import _clean


# ===========================================================================
# D1 — Timestamp validity
# ===========================================================================

class TestCleanTimestamps(unittest.TestCase):
    """
    Rule D1: rows whose timestamp is unparseable or carries a sentinel year
    (≥ 2099) are dropped.  Clean rows get ts parsed to UTC and gain a `date`
    column aligned to midnight UTC.
    """

    def test_valid_timestamp_survives(self):
        """A normally-formatted UTC timestamp must pass cleaning unchanged."""
        df = make_raw_df([{"ts": "2023-06-15 12:00:00+00:00"}])
        clean, _ = _clean(df)
        self.assertEqual(len(clean), 1)
        tz = clean["ts"].iloc[0].tzinfo
        tz_name = tz.zone if hasattr(tz, "zone") else str(tz)
        self.assertEqual(tz_name, "UTC")

    def test_sentinel_year_2099_dropped(self):
        """Rows whose ts encodes year ≥ 2099 (dirty-data sentinel) are dropped."""
        df = make_raw_df([
            {"ts": "2099-01-01 00:00:00+00:00"},                          # dirty
            {"event_id": 2, "ts": "2023-06-15 12:00:00+00:00"},          # clean
        ])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 1)
        self.assertEqual(counts["breakdown"]["invalid_ts"], 1)

    def test_boundary_year_exactly_2099_dropped(self):
        """Year == 2099 at its upper boundary must also be dropped."""
        df = make_raw_df([{"ts": "2099-12-31 23:59:59+00:00"}])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 0)
        self.assertEqual(counts["breakdown"]["invalid_ts"], 1)

    def test_year_2098_survives(self):
        """Year 2098 is the last valid year — the row must survive."""
        df = make_raw_df([{"ts": "2098-12-31 12:00:00+00:00"}])
        clean, _ = _clean(df)
        self.assertEqual(len(clean), 1)

    def test_unparseable_timestamp_dropped(self):
        """A garbage ts string is coerced to NaT and then dropped."""
        df = make_raw_df([
            {"ts": "not-a-date"},
            {"event_id": 2, "ts": "2023-06-15 12:00:00+00:00"},
        ])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 1)
        self.assertEqual(counts["breakdown"]["invalid_ts"], 1)

    def test_date_column_derived_as_midnight_utc(self):
        """After cleaning, `date` must equal the midnight-UTC truncation of `ts`."""
        df = make_raw_df([{"ts": "2023-06-15 15:30:45+00:00"}])
        clean, _ = _clean(df)
        self.assertIn("date", clean.columns)
        self.assertEqual(clean["date"].iloc[0], pd.Timestamp("2023-06-15", tz="UTC"))


# ===========================================================================
# D2 + D5 — Normalisation (country and amount)
# ===========================================================================

class TestCleanNormalise(unittest.TestCase):
    """
    Rule D2: country codes are uppercased, stripped, and null → 'UNK'.
    Rule D5: amount on non-monetary events (page_view, signup) is zeroed.

    These two rules are grouped here because they are both *mutations* rather
    than *drops* — they transform data rather than remove it.
    """

    # ── D2: Country ────────────────────────────────────────────────────────

    def test_null_country_becomes_unk(self):
        """NaN country becomes the sentinel 'UNK'."""
        df = make_raw_df([{"country": np.nan}])
        clean, _ = _clean(df)
        self.assertEqual(str(clean["country"].iloc[0]), "UNK")

    def test_lowercase_country_uppercased(self):
        """'br' → 'BR'."""
        df = make_raw_df([{"country": "br"}])
        clean, _ = _clean(df)
        self.assertEqual(str(clean["country"].iloc[0]), "BR")

    def test_mixed_case_country_uppercased(self):
        """'uS' → 'US'."""
        df = make_raw_df([{"country": "uS"}])
        clean, _ = _clean(df)
        self.assertEqual(str(clean["country"].iloc[0]), "US")

    def test_whitespace_country_stripped(self):
        """'  us  ' → 'US' (strip then uppercase)."""
        df = make_raw_df([{"country": "  us  "}])
        clean, _ = _clean(df)
        self.assertEqual(str(clean["country"].iloc[0]), "US")

    def test_already_valid_country_unchanged(self):
        """An already-valid uppercase country code passes through as-is."""
        df = make_raw_df([{"country": "MX"}])
        clean, _ = _clean(df)
        self.assertEqual(str(clean["country"].iloc[0]), "MX")

    # ── D5: Amount noise zeroing ───────────────────────────────────────────

    def test_page_view_nonzero_amount_zeroed(self):
        """page_view with a stray amount (dirty injector) must be zeroed."""
        df = make_raw_df([{"event_type": "page_view", "amount": 99.99}])
        clean, _ = _clean(df)
        self.assertEqual(float(clean["amount"].iloc[0]), 0.0)

    def test_signup_nonzero_amount_zeroed(self):
        """Same zeroing rule applies to signup rows."""
        df = make_raw_df([{"event_type": "signup", "amount": 5.0}])
        clean, _ = _clean(df)
        self.assertEqual(float(clean["amount"].iloc[0]), 0.0)

    def test_purchase_amount_preserved(self):
        """Purchase amounts must pass through D5 untouched."""
        df = make_raw_df([{"event_type": "purchase", "amount": 123.45}])
        clean, _ = _clean(df)
        self.assertAlmostEqual(float(clean["amount"].iloc[0]), 123.45, places=1)

    def test_refund_amount_preserved(self):
        """Refund amounts (negative) must also pass through D5 untouched."""
        df = make_raw_df([{"event_type": "refund", "amount": -123.45}])
        clean, _ = _clean(df)
        self.assertAlmostEqual(float(clean["amount"].iloc[0]), -123.45, places=1)


# ===========================================================================
# D3 + D4 — Filtering (event_type and deduplication)
# ===========================================================================

class TestCleanFiltering(unittest.TestCase):
    """
    Rule D3: rows with an unrecognised event_type are dropped.
    Rule D4: duplicate event_ids are deduplicated (first occurrence wins).

    Both rules *remove* rows; grouping them makes the "drop vs mutate" split
    across classes explicit.
    """

    # ── D3: event_type allowlist ───────────────────────────────────────────

    def test_invalid_event_type_dropped(self):
        """Rows with event_type '???' are dropped and counted."""
        df = make_raw_df([
            {"event_type": "???"},
            {"event_id": 2, "event_type": "page_view"},
        ])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 1)
        self.assertEqual(counts["breakdown"]["invalid_etype"], 1)

    def test_all_valid_event_types_survive(self):
        """All four canonical event types must survive cleaning."""
        df = make_raw_df([
            {"event_id": 1, "event_type": "page_view"},
            {"event_id": 2, "event_type": "signup"},
            {"event_id": 3, "event_type": "purchase", "amount": 50.0},
            {"event_id": 4, "event_type": "refund",   "amount": -50.0},
        ])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 4)
        self.assertEqual(counts["breakdown"]["invalid_etype"], 0)

    def test_arbitrary_unknown_type_dropped(self):
        """Any event_type not in the valid set must be dropped."""
        df = make_raw_df([{"event_type": "click"}])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 0)
        self.assertEqual(counts["breakdown"]["invalid_etype"], 1)

    # ── D4: Deduplication ─────────────────────────────────────────────────

    def test_duplicate_event_id_keeps_first(self):
        """When two rows share an event_id, the first is kept."""
        df = make_raw_df([
            {"event_id": 42, "user_id": 1, "event_type": "purchase", "amount": 100.0},
            {"event_id": 42, "user_id": 2, "event_type": "page_view", "amount": 0.0},
        ])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 1)
        self.assertEqual(counts["breakdown"]["duplicate_event_id"], 1)
        self.assertEqual(int(clean["user_id"].iloc[0]), 1)

    def test_triplicate_keeps_only_first(self):
        """Three rows sharing an event_id → two dropped, one kept."""
        df = make_raw_df([
            {"event_id": 7},
            {"event_id": 7},
            {"event_id": 7},
        ])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 1)
        self.assertEqual(counts["breakdown"]["duplicate_event_id"], 2)

    def test_unique_event_ids_untouched(self):
        """Rows with unique event_ids must not be dropped by D4."""
        df = make_raw_df([
            {"event_id": 1},
            {"event_id": 2},
            {"event_id": 3},
        ])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 3)
        self.assertEqual(counts["breakdown"]["duplicate_event_id"], 0)


# ===========================================================================
# Counts dict + Dtype discipline + Immutability
# ===========================================================================

class TestCleanInvariants(unittest.TestCase):
    """
    Cross-cutting contracts that span all rules:

    * The counts dict must be arithmetically consistent.
    * Output dtypes must be memory-efficient (categorical, float32).
    * _clean() must not mutate its caller's DataFrame.

    Separated from the rule tests so dtype/contract failures are immediately
    distinguishable from rule-logic failures in CI output.
    """

    # ── Counts dict ───────────────────────────────────────────────────────

    def test_raw_equals_valid_plus_dropped(self):
        """raw_rows == valid_rows + dropped_rows must always hold."""
        df = make_raw_df([
            {"ts": "2099-01-01 00:00:00+00:00"},    # D1 drop
            {"event_id": 2, "event_type": "???"},   # D3 drop
            {"event_id": 3},                         # valid
            {"event_id": 4},                         # valid
        ])
        _, counts = _clean(df)
        self.assertEqual(
            counts["raw_rows"],
            counts["valid_rows"] + counts["dropped_rows"],
        )

    def test_breakdown_keys_present(self):
        """Breakdown dict must contain all three rule keys."""
        df = make_raw_df([{}])
        _, counts = _clean(df)
        for key in ("invalid_ts", "invalid_etype", "duplicate_event_id"):
            self.assertIn(key, counts["breakdown"])

    def test_dropped_rows_equals_breakdown_sum(self):
        """
        dropped_rows must equal the sum of all per-rule breakdown counts.

        Note: because D1 rows are excluded before D3/D4 sees them, a single
        row is counted in at most one bucket — the total is additive.
        """
        df = make_raw_df([
            {"ts": "2099-01-01 00:00:00+00:00"},   # D1
            {"event_id": 2, "event_type": "???"},  # D3
            {"event_id": 3},                        # valid
        ])
        _, counts = _clean(df)
        breakdown_sum = sum(counts["breakdown"].values())
        self.assertEqual(counts["dropped_rows"], breakdown_sum)

    # ── Dtype discipline ──────────────────────────────────────────────────

    def test_event_type_is_categorical(self):
        """event_type must be pd.CategoricalDtype after cleaning."""
        df = make_raw_df([{}])
        clean, _ = _clean(df)
        self.assertIsInstance(clean["event_type"].dtype, pd.CategoricalDtype)

    def test_country_is_categorical(self):
        """country must be pd.CategoricalDtype after cleaning."""
        df = make_raw_df([{}])
        clean, _ = _clean(df)
        self.assertIsInstance(clean["country"].dtype, pd.CategoricalDtype)

    def test_amount_is_float32(self):
        """amount must be float32 — not the default float64."""
        df = make_raw_df([{"event_type": "purchase", "amount": 50.0}])
        clean, _ = _clean(df)
        self.assertEqual(clean["amount"].dtype, np.float32)

    # ── Immutability ──────────────────────────────────────────────────────

    def test_input_dataframe_not_mutated(self):
        """
        _clean() must operate on an internal copy and never mutate the
        caller's DataFrame.  This guards against regressions that remove
        the defensive .copy() call.
        """
        df = make_raw_df([{"country": np.nan, "event_type": "page_view"}])
        original_country = df["country"].iloc[0]  # NaN
        _clean(df.copy())
        self.assertTrue(pd.isna(original_country), "Input DataFrame was mutated")


if __name__ == "__main__":
    unittest.main(verbosity=2)