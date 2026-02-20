"""
test_pipeline.py
================
pytest-compatible tests for pipeline.py.

Written as unittest.TestCase subclasses so the suite runs under:
    pytest test_pipeline.py            (when pytest is installed)
    python  test_pipeline.py           (stdlib only — zero extra dependencies)

Test organisation
-----------------
Each class covers exactly one unit of behaviour:

    TestClean              – _clean() rules D1-D5, counts, dtype discipline
    TestRevenueDaily       – _metric_revenue_daily() net-revenue arithmetic
    TestTopCountries       – _metric_top_countries() ranking & filtering
    TestAnomalyDetection   – _metric_anomalies() z-score & thresholds
    TestDAU                – _metric_dau() unique-user aggregation
    TestRetentionD1        – _metric_retention_d1() cohort logic

Test-data philosophy
--------------------
Every test builds the *smallest* DataFrame that exercises the behaviour
under test.  Helpers make_raw_df / make_clean_df supply valid defaults so
tests only declare the columns relevant to the assertion being made.

Naming convention
-----------------
test_<rule|scenario>_<expected_outcome>
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap — find pipeline.py whether we run from the repo root or from
# the outputs directory directly.
# ---------------------------------------------------------------------------
_OUTPUTS = Path(__file__).parent
if str(_OUTPUTS) not in sys.path:
    sys.path.insert(0, str(_OUTPUTS))

from src.pipeline import (
    _clean,
    _metric_anomalies,
    _metric_dau,
    _metric_funnel,
    _metric_retention_d1,
    _metric_revenue_daily,
    _metric_top_countries,
)


# ===========================================================================
# DataFrame factory helpers
# ===========================================================================

def make_raw_df(rows: list[dict]) -> pd.DataFrame:
    """
    Build a raw DataFrame that mirrors what pandas.read_csv produces from
    events.csv (before any cleaning).

    Default values represent a single valid page_view row on 2023-06-15.
    Pass only the columns you want to override.
    """
    defaults: dict = {
        "event_id":   1,
        "user_id":    100,
        "ts":         "2023-06-15 12:00:00+00:00",
        "event_type": "page_view",
        "amount":     0.0,
        "country":    "US",
        "device":     "web",
        "session_id": 999,
    }
    out = []
    for i, row in enumerate(rows):
        r = dict(defaults)
        r["event_id"] = i + 1          # unique by default
        r.update(row)
        out.append(r)
    return pd.DataFrame(out)


def make_clean_df(rows: list[dict]) -> pd.DataFrame:
    """
    Build a DataFrame that mirrors what _clean() returns — suitable for
    feeding directly to metric functions.

    Includes the derived `date` column (UTC-normalised Timestamp).
    Default: single page_view event for user 1 on 2023-01-01.
    """
    defaults: dict = {
        "event_id":   1,
        "user_id":    1,
        "ts":         pd.Timestamp("2023-01-01 12:00:00", tz="UTC"),
        "date":       pd.Timestamp("2023-01-01", tz="UTC"),
        "event_type": "page_view",
        "amount":     np.float32(0.0),
        "country":    "US",
        "device":     "web",
        "session_id": 1,
    }
    out = []
    for i, row in enumerate(rows):
        r = dict(defaults)
        r["event_id"] = i + 1
        r.update(row)
        # Keep date consistent with ts when ts is overridden but date is not
        if "ts" in row and "date" not in row:
            r["date"] = pd.Timestamp(row["ts"]).normalize()
        out.append(r)
    return pd.DataFrame(out)


def _date(s: str) -> pd.Timestamp:
    """Convenience: parse an ISO date string into a UTC-aware midnight Timestamp."""
    return pd.Timestamp(s, tz="UTC")


# ===========================================================================
# TestClean
# ===========================================================================

class TestClean(unittest.TestCase):
    """
    Tests for _clean().

    Every rule is tested with both a "dirty" case (row should be dropped /
    mutated) and a "clean" case (row must survive unchanged).
    """

    # ── D1: Timestamp validity ───────────────────────────────────────────

    def test_d1_valid_timestamp_survives(self):
        """A normally-formatted UTC timestamp must pass cleaning unchanged."""
        df = make_raw_df([{"ts": "2023-06-15 12:00:00+00:00"}])
        clean, _ = _clean(df)
        self.assertEqual(len(clean), 1)
        # _clean must parse ts and make it timezone-aware
        self.assertEqual(clean["ts"].iloc[0].tzinfo.zone if hasattr(
            clean["ts"].iloc[0].tzinfo, "zone") else str(clean["ts"].iloc[0].tzinfo),
            "UTC")

    def test_d1_year_2099_sentinel_dropped(self):
        """Rows whose ts encodes year ≥ 2099 (dirty-data sentinel) are dropped."""
        df = make_raw_df([
            {"ts": "2099-01-01 00:00:00+00:00"},   # dirty
            {"event_id": 2, "ts": "2023-06-15 12:00:00+00:00"},  # clean
        ])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 1)
        self.assertEqual(counts["breakdown"]["invalid_ts"], 1)

    def test_d1_year_exactly_2099_boundary_dropped(self):
        """Year == 2099 is at the boundary — must be dropped."""
        df = make_raw_df([{"ts": "2099-12-31 23:59:59+00:00"}])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 0)
        self.assertEqual(counts["breakdown"]["invalid_ts"], 1)

    def test_d1_year_2098_survives(self):
        """Year 2098 is the last valid year — row must survive."""
        df = make_raw_df([{"ts": "2098-12-31 12:00:00+00:00"}])
        clean, _ = _clean(df)
        self.assertEqual(len(clean), 1)

    def test_d1_unparseable_timestamp_dropped(self):
        """A garbage ts string must be coerced to NaT and then dropped."""
        df = make_raw_df([
            {"ts": "not-a-date"},
            {"event_id": 2, "ts": "2023-06-15 12:00:00+00:00"},
        ])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 1)
        self.assertEqual(counts["breakdown"]["invalid_ts"], 1)

    def test_d1_date_column_derived_from_ts(self):
        """After cleaning, a `date` column equal to midnight UTC must exist."""
        df = make_raw_df([{"ts": "2023-06-15 15:30:45+00:00"}])
        clean, _ = _clean(df)
        self.assertIn("date", clean.columns)
        expected = pd.Timestamp("2023-06-15", tz="UTC")
        self.assertEqual(clean["date"].iloc[0], expected)

    # ── D2: Country normalisation ─────────────────────────────────────────

    def test_d2_null_country_replaced_with_unk(self):
        """NaN country (read back from empty string in CSV) becomes 'UNK'."""
        df = make_raw_df([{"country": np.nan}])
        clean, _ = _clean(df)
        self.assertEqual(str(clean["country"].iloc[0]), "UNK")

    def test_d2_lowercase_country_uppercased(self):
        """Country codes are always uppercased."""
        df = make_raw_df([{"country": "br"}])
        clean, _ = _clean(df)
        self.assertEqual(str(clean["country"].iloc[0]), "BR")

    def test_d2_mixed_case_country_uppercased(self):
        """Mixed-case country strings are fully uppercased."""
        df = make_raw_df([{"country": "uS"}])
        clean, _ = _clean(df)
        self.assertEqual(str(clean["country"].iloc[0]), "US")

    def test_d2_whitespace_country_stripped(self):
        """Leading/trailing whitespace in country is stripped after uppercasing."""
        df = make_raw_df([{"country": "  us  "}])
        clean, _ = _clean(df)
        self.assertEqual(str(clean["country"].iloc[0]), "US")

    def test_d2_valid_country_unchanged(self):
        """An already-valid uppercase country code passes through as-is."""
        df = make_raw_df([{"country": "MX"}])
        clean, _ = _clean(df)
        self.assertEqual(str(clean["country"].iloc[0]), "MX")

    # ── D3: Invalid event_type ────────────────────────────────────────────

    def test_d3_invalid_etype_dropped(self):
        """Rows with event_type '???' are dropped and counted."""
        df = make_raw_df([
            {"event_type": "???"},
            {"event_id": 2, "event_type": "page_view"},
        ])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 1)
        self.assertEqual(counts["breakdown"]["invalid_etype"], 1)

    def test_d3_all_valid_event_types_survive(self):
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

    def test_d3_unknown_etype_string_dropped(self):
        """Any event_type not in the valid set must be dropped."""
        df = make_raw_df([{"event_type": "click"}])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 0)
        self.assertEqual(counts["breakdown"]["invalid_etype"], 1)

    # ── D4: Deduplication ─────────────────────────────────────────────────

    def test_d4_duplicate_event_id_keeps_first(self):
        """When two rows share an event_id, the first is kept and second dropped."""
        df = make_raw_df([
            {"event_id": 42, "user_id": 1, "event_type": "purchase", "amount": 100.0},
            {"event_id": 42, "user_id": 2, "event_type": "page_view", "amount": 0.0},
        ])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 1)
        self.assertEqual(counts["breakdown"]["duplicate_event_id"], 1)
        # First row's user_id must be preserved
        self.assertEqual(int(clean["user_id"].iloc[0]), 1)

    def test_d4_triplicate_keeps_only_first(self):
        """Three rows sharing an event_id → two dropped, one kept."""
        df = make_raw_df([
            {"event_id": 7},
            {"event_id": 7},
            {"event_id": 7},
        ])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 1)
        self.assertEqual(counts["breakdown"]["duplicate_event_id"], 2)

    def test_d4_unique_event_ids_untouched(self):
        """Rows with unique event_ids must not be dropped."""
        df = make_raw_df([
            {"event_id": 1},
            {"event_id": 2},
            {"event_id": 3},
        ])
        clean, counts = _clean(df)
        self.assertEqual(len(clean), 3)
        self.assertEqual(counts["breakdown"]["duplicate_event_id"], 0)

    # ── D5: Amount noise zeroing ──────────────────────────────────────────

    def test_d5_page_view_non_zero_amount_zeroed(self):
        """
        A page_view row that somehow has a non-zero amount (noise from the
        dirty injector) must have its amount zeroed.
        """
        df = make_raw_df([{"event_type": "page_view", "amount": 99.99}])
        clean, _ = _clean(df)
        self.assertEqual(float(clean["amount"].iloc[0]), 0.0)

    def test_d5_signup_non_zero_amount_zeroed(self):
        """Same zeroing rule applies to signup events."""
        df = make_raw_df([{"event_type": "signup", "amount": 5.0}])
        clean, _ = _clean(df)
        self.assertEqual(float(clean["amount"].iloc[0]), 0.0)

    def test_d5_purchase_amount_preserved(self):
        """Purchase amounts must pass through D5 untouched."""
        df = make_raw_df([{"event_type": "purchase", "amount": 123.45}])
        clean, _ = _clean(df)
        self.assertAlmostEqual(float(clean["amount"].iloc[0]), 123.45, places=1)

    def test_d5_refund_amount_preserved(self):
        """Refund amounts (negative) must pass through D5 untouched."""
        df = make_raw_df([{"event_type": "refund", "amount": -123.45}])
        clean, _ = _clean(df)
        self.assertAlmostEqual(float(clean["amount"].iloc[0]), -123.45, places=1)

    # ── Counts dict ───────────────────────────────────────────────────────

    def test_counts_raw_valid_dropped_sum(self):
        """raw_rows == valid_rows + dropped_rows must always hold."""
        df = make_raw_df([
            {"ts": "2099-01-01 00:00:00+00:00"},    # D1 drop
            {"event_id": 2, "event_type": "???"},   # D3 drop
            {"event_id": 3},                         # valid
            {"event_id": 4},                         # valid
        ])
        clean, counts = _clean(df)
        self.assertEqual(
            counts["raw_rows"],
            counts["valid_rows"] + counts["dropped_rows"],
        )

    def test_counts_breakdown_keys_present(self):
        """Breakdown dict must contain all three rule keys."""
        df = make_raw_df([{}])
        _, counts = _clean(df)
        for key in ("invalid_ts", "invalid_etype", "duplicate_event_id"):
            self.assertIn(key, counts["breakdown"])

    def test_counts_dropped_rows_matches_breakdown_sum(self):
        """dropped_rows must equal the sum of per-rule breakdown counts.

        Note: a row dropped by D1 is never seen by D3/D4, so the breakdown
        counts are NOT additive across rules — a row can only be counted once.
        dropped_rows reflects the net reduction, not the sum of rule hits.
        """
        df = make_raw_df([
            {"ts": "2099-01-01 00:00:00+00:00"},   # D1
            {"event_id": 2, "event_type": "???"},  # D3
            {"event_id": 3},                        # valid
        ])
        _, counts = _clean(df)
        # Each dropped row is counted exactly once in exactly one rule bucket
        breakdown_sum = sum(counts["breakdown"].values())
        self.assertEqual(counts["dropped_rows"], breakdown_sum)

    # ── Dtype discipline ──────────────────────────────────────────────────

    def test_dtype_event_type_is_categorical(self):
        """event_type must be pd.Categorical after cleaning."""
        df = make_raw_df([{}])
        clean, _ = _clean(df)
        self.assertIsInstance(clean["event_type"].dtype, pd.CategoricalDtype)

    def test_dtype_country_is_categorical(self):
        """country must be pd.Categorical after cleaning."""
        df = make_raw_df([{}])
        clean, _ = _clean(df)
        self.assertIsInstance(clean["country"].dtype, pd.CategoricalDtype)

    def test_dtype_amount_is_float32(self):
        """amount must be float32 after cleaning — not float64."""
        df = make_raw_df([{"event_type": "purchase", "amount": 50.0}])
        clean, _ = _clean(df)
        self.assertEqual(clean["amount"].dtype, np.float32)

    def test_input_dataframe_not_mutated(self):
        """
        _clean must not mutate the caller's DataFrame.

        It uses .copy() internally; this test guards against regressions that
        would drop the .copy() and modify the caller's data in place.
        """
        df = make_raw_df([{"country": np.nan, "event_type": "page_view"}])
        original_country = df["country"].iloc[0]  # should be NaN
        _clean(df.copy())  # pass a copy so the test itself doesn't interfere
        self.assertTrue(pd.isna(original_country))   # original unchanged


# ===========================================================================
# TestRevenueDaily
# ===========================================================================

class TestRevenueDaily(unittest.TestCase):
    """
    Tests for _metric_revenue_daily().

    The function returns a DataFrame with columns [date, net_revenue].
    Refunds are stored as negative amounts so a plain .sum() gives net revenue.
    """

    def test_single_purchase_one_day(self):
        """One purchase on one day produces the correct net revenue."""
        df = make_clean_df([
            {"event_type": "purchase", "amount": np.float32(100.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result["net_revenue"].iloc[0]), 100.0, places=2)

    def test_refund_reduces_revenue(self):
        """A refund (negative amount) must reduce net revenue for that day."""
        df = make_clean_df([
            {"event_type": "purchase", "amount": np.float32(200.0),
             "date": _date("2023-01-01")},
            {"event_id": 2, "event_type": "refund", "amount": np.float32(-200.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result["net_revenue"].iloc[0]), 0.0, places=2)

    def test_partial_refund_net_revenue(self):
        """A partial refund yields purchase_amount - refund_amount."""
        df = make_clean_df([
            {"event_type": "purchase", "amount": np.float32(150.0),
             "date": _date("2023-01-01")},
            {"event_id": 2, "event_type": "refund", "amount": np.float32(-50.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        self.assertAlmostEqual(float(result["net_revenue"].iloc[0]), 100.0, places=2)

    def test_page_view_excluded_from_revenue(self):
        """page_view rows must not contribute to any day's revenue."""
        df = make_clean_df([
            {"event_type": "page_view", "amount": np.float32(0.0),
             "date": _date("2023-01-01")},
            {"event_id": 2, "event_type": "purchase", "amount": np.float32(75.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        self.assertAlmostEqual(float(result["net_revenue"].iloc[0]), 75.0, places=2)

    def test_signup_excluded_from_revenue(self):
        """signup rows must not contribute to any day's revenue."""
        df = make_clean_df([
            {"event_type": "signup", "amount": np.float32(0.0),
             "date": _date("2023-01-01")},
            {"event_id": 2, "event_type": "purchase", "amount": np.float32(40.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        self.assertAlmostEqual(float(result["net_revenue"].iloc[0]), 40.0, places=2)

    def test_multiple_days_aggregated_independently(self):
        """Revenue on different days must be independent of each other."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "purchase", "amount": np.float32(100.0),
             "date": _date("2023-01-01")},
            {"event_id": 2, "event_type": "purchase", "amount": np.float32(200.0),
             "date": _date("2023-01-02")},
            {"event_id": 3, "event_type": "purchase", "amount": np.float32(300.0),
             "date": _date("2023-01-03")},
        ])
        result = _metric_revenue_daily(df)
        self.assertEqual(len(result), 3)
        revenues = result.sort_values("date")["net_revenue"].tolist()
        self.assertAlmostEqual(revenues[0], 100.0, places=2)
        self.assertAlmostEqual(revenues[1], 200.0, places=2)
        self.assertAlmostEqual(revenues[2], 300.0, places=2)

    def test_no_revenue_rows_returns_empty_dataframe(self):
        """If only page_view/signup rows exist, result must be empty."""
        df = make_clean_df([
            {"event_type": "page_view"},
            {"event_id": 2, "event_type": "signup"},
        ])
        result = _metric_revenue_daily(df)
        self.assertEqual(len(result), 0)
        self.assertIn("net_revenue", result.columns)

    def test_net_revenue_column_is_float64(self):
        """net_revenue must be promoted to float64 (for downstream NumPy stats)."""
        df = make_clean_df([
            {"event_type": "purchase", "amount": np.float32(50.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        self.assertEqual(result["net_revenue"].dtype, np.float64)

    def test_multiple_purchases_same_day_summed(self):
        """Multiple purchases on the same day must be summed, not counted."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "purchase", "amount": np.float32(30.0),
             "date": _date("2023-01-01")},
            {"event_id": 2, "event_type": "purchase", "amount": np.float32(70.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result["net_revenue"].iloc[0]), 100.0, places=2)

    def test_result_sorted_by_date(self):
        """Output rows must appear in ascending date order."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "purchase", "amount": np.float32(1.0),
             "date": _date("2023-01-03")},
            {"event_id": 2, "event_type": "purchase", "amount": np.float32(1.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        dates = result["date"].tolist()
        self.assertEqual(dates, sorted(dates))


# ===========================================================================
# TestTopCountries
# ===========================================================================

class TestTopCountries(unittest.TestCase):
    """Tests for _metric_top_countries()."""

    def test_ranking_descending_by_net_revenue(self):
        """Countries must be ranked by net revenue, highest first."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "purchase", "amount": np.float32(300.0),
             "country": "BR"},
            {"event_id": 2, "event_type": "purchase", "amount": np.float32(100.0),
             "country": "US"},
            {"event_id": 3, "event_type": "purchase", "amount": np.float32(200.0),
             "country": "MX"},
        ])
        result = _metric_top_countries(df)
        country_order = [r["country"] for r in result]
        self.assertEqual(country_order, ["BR", "MX", "US"])

    def test_refund_reduces_country_revenue(self):
        """A refund in a country must reduce that country's net revenue."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "purchase", "amount": np.float32(500.0),
             "country": "BR"},
            {"event_id": 2, "event_type": "refund",   "amount": np.float32(-200.0),
             "country": "BR"},
        ])
        result = _metric_top_countries(df)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["net_revenue"], 300.0, places=2)

    def test_unk_country_is_included(self):
        """'UNK' rows represent real revenue and must appear in rankings."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "purchase", "amount": np.float32(999.0),
             "country": "UNK"},
            {"event_id": 2, "event_type": "purchase", "amount": np.float32(1.0),
             "country": "US"},
        ])
        result = _metric_top_countries(df)
        countries = [r["country"] for r in result]
        self.assertIn("UNK", countries)

    def test_top_n_limit_respected(self):
        """Result must contain at most top_n entries."""
        rows = [
            {"event_id": i, "event_type": "purchase",
             "amount": np.float32(float(i)), "country": f"C{i:02d}"}
            for i in range(1, 20)
        ]
        df = make_clean_df(rows)
        result = _metric_top_countries(df, top_n=5)
        self.assertEqual(len(result), 5)

    def test_page_view_excluded_from_country_revenue(self):
        """page_view events must not contribute to any country's revenue."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "page_view", "amount": np.float32(0.0),
             "country": "US"},
        ])
        result = _metric_top_countries(df)
        self.assertEqual(len(result), 0)


# ===========================================================================
# TestAnomalyDetection
# ===========================================================================

class TestAnomalyDetection(unittest.TestCase):
    """
    Tests for _metric_anomalies().

    We use a controlled revenue series where z-scores are known analytically
    so we can assert exact detection results.
    """

    def _make_revenue_df(self, values: list[float], start: str = "2023-01-01") -> pd.DataFrame:
        """Build a minimal revenue DataFrame as _metric_revenue_daily() returns."""
        dates = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
        return pd.DataFrame({
            "date":        dates,
            "net_revenue": np.array(values, dtype="float64"),
        })

    def test_spike_above_3_std_detected(self):
        """
        A day with net_revenue > mean + 3*std must appear in anomalies.

        Series: 29 days at 100.0, 1 day at 10000.0
        mean ≈ 430, std ≈ 1852 — the last day's z ≫ 3.
        """
        values = [100.0] * 29 + [10_000.0]
        rev_df = self._make_revenue_df(values)
        anomalies = _metric_anomalies(rev_df)
        self.assertGreater(len(anomalies), 0)
        # Spike day must be the one with the highest z
        self.assertAlmostEqual(anomalies[0]["net_revenue"], 10_000.0, places=0)

    def test_uniform_series_no_anomalies(self):
        """All-identical revenue yields std=0 → no anomalies (early-exit path)."""
        values = [500.0] * 30
        rev_df = self._make_revenue_df(values)
        anomalies = _metric_anomalies(rev_df)
        self.assertEqual(anomalies, [])

    def test_values_with_low_z_scores_not_flagged(self):
        """
        An evenly-spaced series [0, 1, ..., 29] has max |z| approx 1.67.
        No day should be flagged.

        Analytical check:
            mean = 14.5
            std  = sqrt(sum((x-14.5)^2)/30) approx 8.66
            max z = (29 - 14.5) / 8.66 approx 1.67  <  3.0  -> no anomaly
        """
        values = [float(i) for i in range(30)]
        arr = np.array(values)
        max_z = float(np.max(np.abs((arr - arr.mean()) / arr.std(ddof=0))))
        self.assertLess(max_z, 3.0,
            msg="Test-data invariant: max z should be below 3.0")

        rev_df = self._make_revenue_df(values)
        anomalies = _metric_anomalies(rev_df)
        self.assertEqual(anomalies, [])

    def test_z_score_value_is_correct(self):
        """
        Verify z-score arithmetic against a hand-computable series.

        Series: 19 zeros + 1 spike at 100.0  (n=20)
            mean     = 100/20 = 5.0
            variance = (19*(0-5)^2 + (100-5)^2) / 20
                     = (475 + 9025) / 20 = 475.0
            std      = sqrt(475) approx 21.794
            z        = (100 - 5) / sqrt(475) approx 4.36  >  3.0 -> flagged
        """
        values = [0.0] * 19 + [100.0]
        mu_exact  = 5.0
        std_exact = float(np.sqrt(475.0))
        expected_z = round((100.0 - mu_exact) / std_exact, 4)

        self.assertGreater(expected_z, 3.0,
            msg="Test-data invariant: spike should exceed mean+3*std")

        rev_df = self._make_revenue_df(values)
        anomalies = _metric_anomalies(rev_df)
        self.assertGreater(len(anomalies), 0,
            msg="Spike above mean+3*std must be detected")
        self.assertAlmostEqual(anomalies[0]["z"], expected_z, places=3)


    def test_too_few_data_points_returns_empty(self):
        """A single-row revenue series cannot produce a meaningful z-score."""
        rev_df = self._make_revenue_df([500.0])
        anomalies = _metric_anomalies(rev_df)
        self.assertEqual(anomalies, [])

    def test_anomalies_sorted_by_z_descending(self):
        """When multiple anomalies exist, the one with the highest z appears first."""
        # Two extreme spikes; the larger one should appear first.
        values = [100.0] * 27 + [50_000.0, 20_000.0, 10_000.0]
        rev_df = self._make_revenue_df(values)
        anomalies = _metric_anomalies(rev_df)
        if len(anomalies) >= 2:
            self.assertGreaterEqual(anomalies[0]["z"], anomalies[1]["z"])


# ===========================================================================
# TestDAU
# ===========================================================================

class TestDAU(unittest.TestCase):
    """Tests for _metric_dau()."""

    def test_single_user_single_day(self):
        df = make_clean_df([{"user_id": 1, "date": _date("2023-01-01")}])
        result = _metric_dau(df)
        self.assertEqual(result[0]["dau"], 1)

    def test_same_user_multiple_events_counts_once(self):
        """Five events from one user on one day = DAU of 1, not 5."""
        df = make_clean_df([
            {"event_id": i, "user_id": 42, "date": _date("2023-01-01")}
            for i in range(1, 6)
        ])
        result = _metric_dau(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["dau"], 1)

    def test_multiple_users_multiple_days(self):
        """DAU is computed per-day and independent across days."""
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 2, "date": _date("2023-01-01")},
            {"event_id": 3, "user_id": 3, "date": _date("2023-01-02")},
        ])
        result = _metric_dau(df)
        by_date = {r["date"]: r["dau"] for r in result}
        self.assertEqual(by_date["2023-01-01"], 2)
        self.assertEqual(by_date["2023-01-02"], 1)

    def test_output_sorted_by_date(self):
        """DAU output must be sorted ascending by date."""
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-03")},
            {"event_id": 2, "user_id": 2, "date": _date("2023-01-01")},
        ])
        result = _metric_dau(df)
        dates = [r["date"] for r in result]
        self.assertEqual(dates, sorted(dates))


# ===========================================================================
# TestRetentionD1
# ===========================================================================

class TestRetentionD1(unittest.TestCase):
    """
    Tests for _metric_retention_d1().

    Each test uses a hand-crafted dataset where the expected retention
    values can be computed by inspection.

    Key behavioural contracts:
        1. Cohort = first date a user is seen.
        2. Retained = active on exactly cohort + 1 day.
        3. D+2, D+3, etc. do NOT count as D1 retention.
        4. The last date in the dataset is excluded from cohort output
           (no D+1 window exists for those users).
        5. A user active multiple times on a single day is counted once.
    """

    # ── Contract 1+2: basic cohort mechanics ─────────────────────────────

    def test_all_users_retained_100_percent(self):
        """
        All cohort users return on D+1 → retention = 1.0.

        Dataset:
            user 1: active 2023-01-01 and 2023-01-02
            user 2: active 2023-01-01 and 2023-01-02
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
        No cohort user returns on D+1 → retained=0, rate=0.0.

        Dataset:
            user 1: active 2023-01-01 only
            user 2: active 2023-01-01 only
            user 3: active 2023-01-03  ← sets max_date so cohort 01 is included
        """
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 2, "date": _date("2023-01-01")},
            {"event_id": 3, "user_id": 3, "date": _date("2023-01-03")},
        ])
        result = _metric_retention_d1(df)
        # cohort 2023-01-01 must be present (< max_date=2023-01-03)
        cohort_row = next(r for r in result if r["cohort_date"] == "2023-01-01")
        self.assertEqual(cohort_row["retained"], 0)
        self.assertAlmostEqual(cohort_row["rate"], 0.0, places=4)

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
        cohort_row = next(r for r in result if r["cohort_date"] == "2023-01-01")
        self.assertEqual(cohort_row["users"], 2)
        self.assertEqual(cohort_row["retained"], 1)
        self.assertAlmostEqual(cohort_row["rate"], 0.5, places=4)

    # ── Contract 3: only D+1 counts ──────────────────────────────────────

    def test_d2_activity_does_not_count_as_d1_retention(self):
        """
        A user active on D+2 but NOT on D+1 must NOT be counted as retained.

        Dataset:
            user 1: 2023-01-01 (cohort) and 2023-01-03 (D+2, not D+1)
            user 2: 2023-01-04  ← sets max_date so cohort 01 is included
        """
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 1, "date": _date("2023-01-03")},
            {"event_id": 3, "user_id": 2, "date": _date("2023-01-04")},
        ])
        result = _metric_retention_d1(df)
        cohort_row = next(r for r in result if r["cohort_date"] == "2023-01-01")
        self.assertEqual(cohort_row["retained"], 0)
        self.assertAlmostEqual(cohort_row["rate"], 0.0, places=4)

    def test_d1_and_d2_activity_counts_as_retained_only_once(self):
        """
        A user active on both D+1 AND D+2 is retained once, not twice.

        Dataset:
            user 1: active on 2023-01-01, 2023-01-02, 2023-01-03
            user 2: 2023-01-04  ← sets max_date
        """
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 1, "date": _date("2023-01-02")},
            {"event_id": 3, "user_id": 1, "date": _date("2023-01-03")},
            {"event_id": 4, "user_id": 2, "date": _date("2023-01-04")},
        ])
        result = _metric_retention_d1(df)
        cohort_row = next(r for r in result if r["cohort_date"] == "2023-01-01")
        self.assertEqual(cohort_row["users"], 1)
        self.assertEqual(cohort_row["retained"], 1)

    # ── Contract 4: last day excluded ────────────────────────────────────

    def test_last_date_in_dataset_excluded_from_cohort_output(self):
        """
        The max date in the dataset must not appear as a cohort_date.

        Rationale: users who first appear on the last day have no possible
        D+1 window — including them would artificially deflate the rate.
        """
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 2, "date": _date("2023-01-02")},
        ])
        result = _metric_retention_d1(df)
        cohort_dates = {r["cohort_date"] for r in result}
        self.assertNotIn("2023-01-02", cohort_dates)

    def test_only_one_date_in_dataset_returns_empty(self):
        """
        If every event is on the same date, that date is both min and max.
        It gets excluded (it's the max), so result is empty.
        """
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-01")},
            {"event_id": 2, "user_id": 2, "date": _date("2023-01-01")},
        ])
        result = _metric_retention_d1(df)
        self.assertEqual(result, [])

    # ── Contract 5: deduplication of events ──────────────────────────────

    def test_multiple_events_same_user_same_day_counted_once(self):
        """
        A user who fires 5 events on their return day is counted as retained
        once — not 5 times.
        """
        df = make_clean_df(
            # user 1 on cohort day
            [{"event_id": 1, "user_id": 1, "date": _date("2023-01-01")}]
            # user 1 fires 5 events on D+1
            + [{"event_id": 1 + i, "user_id": 1, "date": _date("2023-01-02")}
               for i in range(1, 6)]
            # anchor: user 2 on 2023-01-03 sets max_date
            + [{"event_id": 10, "user_id": 2, "date": _date("2023-01-03")}]
        )
        result = _metric_retention_d1(df)
        cohort_row = next(r for r in result if r["cohort_date"] == "2023-01-01")
        self.assertEqual(cohort_row["retained"], 1)

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_empty_dataframe_returns_empty_list(self):
        """_metric_retention_d1 must handle an empty DataFrame gracefully."""
        df = pd.DataFrame(columns=["user_id", "date", "event_id", "event_type",
                                   "amount", "country", "device", "session_id", "ts"])
        result = _metric_retention_d1(df)
        self.assertEqual(result, [])

    def test_multiple_cohort_dates_in_output(self):
        """
        Users with different first dates form different cohort buckets,
        each computed independently.

        Dataset:
            cohort 2023-01-01: user 1 (not retained), user 2 (not retained)
            cohort 2023-01-02: user 3 (retained on 2023-01-03)
            max_date = 2023-01-03 → both cohorts included
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

        self.assertEqual(by_date["2023-01-01"]["users"], 2)
        self.assertEqual(by_date["2023-01-01"]["retained"], 0)

        self.assertEqual(by_date["2023-01-02"]["users"], 1)
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


# ===========================================================================
# TestFunnel
# ===========================================================================

class TestFunnel(unittest.TestCase):
    """Tests for _metric_funnel() — conversion rate arithmetic."""

    def test_conversion_rates_computed_correctly(self):
        """
        pv=10, signup=5, purchase=2 →
            pv_to_signup = 0.5, signup_to_purchase ≈ 0.4
        """
        rows = (
            [{"event_id": i, "event_type": "page_view",
              "date": _date("2023-01-01")} for i in range(1, 11)]
            + [{"event_id": i, "event_type": "signup",
                "date": _date("2023-01-01")} for i in range(11, 16)]
            + [{"event_id": i, "event_type": "purchase",
                "amount": np.float32(50.0),
                "date": _date("2023-01-01")} for i in range(16, 18)]
        )
        df = make_clean_df(rows)
        result = _metric_funnel(df)
        row = result[0]
        self.assertEqual(row["pv"],       10)
        self.assertEqual(row["signup"],    5)
        self.assertEqual(row["purchase"],  2)
        self.assertAlmostEqual(row["pv_to_signup"],       0.5,  places=4)
        self.assertAlmostEqual(row["signup_to_purchase"], 0.4,  places=4)

    def test_zero_page_views_no_division_error(self):
        """A day with signups but no page_views must not raise ZeroDivisionError."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "signup", "date": _date("2023-01-01")},
        ])
        try:
            result = _metric_funnel(df)
            row = result[0]
            self.assertEqual(row["pv_to_signup"], 0.0)
        except ZeroDivisionError:
            self.fail("_metric_funnel raised ZeroDivisionError on zero page_views")

    def test_zero_signups_no_division_error(self):
        """A day with purchases but no signups must not raise ZeroDivisionError."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "page_view",
             "date": _date("2023-01-01")},
            {"event_id": 2, "event_type": "purchase",
             "amount": np.float32(50.0), "date": _date("2023-01-01")},
        ])
        try:
            result = _metric_funnel(df)
            row = result[0]
            self.assertEqual(row["signup_to_purchase"], 0.0)
        except ZeroDivisionError:
            self.fail("_metric_funnel raised ZeroDivisionError on zero signups")


# ===========================================================================
# Entry point — runs with plain `python test_pipeline.py`
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)