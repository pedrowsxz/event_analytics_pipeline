"""
test_metrics.py
===============
Tests for the five *stateless metric functions* in pipeline.py:

    _metric_revenue_daily   – daily net revenue aggregation
    _metric_top_countries   – country revenue ranking
    _metric_anomalies       – z-score anomaly detection
    _metric_dau             – daily active users
    _metric_funnel          – page_view → signup → purchase conversion rates

Why are these five classes in one file?
---------------------------------------
Each of these functions shares the same shape:
  * Input  : a cleaned DataFrame (or a revenue DataFrame for anomalies)
  * Output : a list[dict] or a small DataFrame
  * Tests  : small, fast, independent of each other

Keeping them together avoids a proliferation of tiny files while still
letting test discovery (`pytest test_metrics.py::TestAnomalyDetection`)
target a single class.

Retention (`_metric_retention_d1`) lives in test_retention.py because its
five named behavioural contracts, edge cases, and test density justify its
own file.
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

from tests.conftest import make_clean_df, _date
from src.pipeline import (
    _metric_anomalies,
    _metric_dau,
    _metric_funnel,
    _metric_revenue_daily,
    _metric_top_countries,
)


# ===========================================================================
# Net Revenue
# ===========================================================================

class TestRevenueDaily(unittest.TestCase):
    """
    Tests for _metric_revenue_daily().

    The function sums `amount` per day; refunds carry negative amounts so a
    plain .sum() gives net revenue without any conditional logic.
    """

    def test_single_purchase_one_day(self):
        """One purchase → correct net revenue for that day."""
        df = make_clean_df([
            {"event_type": "purchase", "amount": np.float32(100.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result["net_revenue"].iloc[0]), 100.0, places=2)

    def test_full_refund_yields_zero_net(self):
        """A refund equal to the purchase nets to zero."""
        df = make_clean_df([
            {"event_type": "purchase", "amount": np.float32(200.0),
             "date": _date("2023-01-01")},
            {"event_id": 2, "event_type": "refund", "amount": np.float32(-200.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        self.assertAlmostEqual(float(result["net_revenue"].iloc[0]), 0.0, places=2)

    def test_partial_refund_net_revenue(self):
        """A partial refund yields purchase_amount + refund_amount (refund is negative)."""
        df = make_clean_df([
            {"event_type": "purchase", "amount": np.float32(150.0),
             "date": _date("2023-01-01")},
            {"event_id": 2, "event_type": "refund", "amount": np.float32(-50.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        self.assertAlmostEqual(float(result["net_revenue"].iloc[0]), 100.0, places=2)

    def test_page_view_excluded(self):
        """page_view rows must not contribute to revenue."""
        df = make_clean_df([
            {"event_type": "page_view", "amount": np.float32(0.0),
             "date": _date("2023-01-01")},
            {"event_id": 2, "event_type": "purchase", "amount": np.float32(75.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        self.assertAlmostEqual(float(result["net_revenue"].iloc[0]), 75.0, places=2)

    def test_signup_excluded(self):
        """signup rows must not contribute to revenue."""
        df = make_clean_df([
            {"event_type": "signup", "amount": np.float32(0.0),
             "date": _date("2023-01-01")},
            {"event_id": 2, "event_type": "purchase", "amount": np.float32(40.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        self.assertAlmostEqual(float(result["net_revenue"].iloc[0]), 40.0, places=2)

    def test_multiple_purchases_same_day_summed(self):
        """Multiple purchases on the same day are summed, not counted."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "purchase", "amount": np.float32(30.0),
             "date": _date("2023-01-01")},
            {"event_id": 2, "event_type": "purchase", "amount": np.float32(70.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result["net_revenue"].iloc[0]), 100.0, places=2)

    def test_multiple_days_independent(self):
        """Revenue on different days must be computed independently."""
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
        for expected, actual in zip([100.0, 200.0, 300.0], revenues):
            self.assertAlmostEqual(actual, expected, places=2)

    def test_no_revenue_rows_returns_empty_dataframe(self):
        """If only page_view/signup rows exist, result must be an empty DataFrame."""
        df = make_clean_df([
            {"event_type": "page_view"},
            {"event_id": 2, "event_type": "signup"},
        ])
        result = _metric_revenue_daily(df)
        self.assertEqual(len(result), 0)
        self.assertIn("net_revenue", result.columns)

    def test_net_revenue_dtype_is_float64(self):
        """net_revenue must be float64 (for downstream NumPy statistics)."""
        df = make_clean_df([
            {"event_type": "purchase", "amount": np.float32(50.0),
             "date": _date("2023-01-01")},
        ])
        result = _metric_revenue_daily(df)
        self.assertEqual(result["net_revenue"].dtype, np.float64)

    def test_output_sorted_ascending_by_date(self):
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
# Top Countries
# ===========================================================================

class TestTopCountries(unittest.TestCase):
    """Tests for _metric_top_countries()."""

    def test_ranking_descending_by_net_revenue(self):
        """Countries must be ranked highest-first."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "purchase", "amount": np.float32(300.0), "country": "BR"},
            {"event_id": 2, "event_type": "purchase", "amount": np.float32(100.0), "country": "US"},
            {"event_id": 3, "event_type": "purchase", "amount": np.float32(200.0), "country": "MX"},
        ])
        result = _metric_top_countries(df)
        self.assertEqual([r["country"] for r in result], ["BR", "MX", "US"])

    def test_refund_reduces_country_revenue(self):
        """A refund must reduce that country's net revenue."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "purchase", "amount": np.float32(500.0), "country": "BR"},
            {"event_id": 2, "event_type": "refund",   "amount": np.float32(-200.0), "country": "BR"},
        ])
        result = _metric_top_countries(df)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["net_revenue"], 300.0, places=2)

    def test_unk_country_included_in_ranking(self):
        """'UNK' rows represent real revenue and must appear in rankings."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "purchase", "amount": np.float32(999.0), "country": "UNK"},
            {"event_id": 2, "event_type": "purchase", "amount": np.float32(1.0),   "country": "US"},
        ])
        result = _metric_top_countries(df)
        self.assertIn("UNK", [r["country"] for r in result])

    def test_top_n_limit_respected(self):
        """Result must contain at most top_n entries."""
        rows = [
            {"event_id": i, "event_type": "purchase",
             "amount": np.float32(float(i)), "country": f"C{i:02d}"}
            for i in range(1, 20)
        ]
        result = _metric_top_countries(make_clean_df(rows), top_n=5)
        self.assertEqual(len(result), 5)

    def test_non_monetary_events_excluded(self):
        """page_view events must not contribute to any country's revenue."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "page_view", "amount": np.float32(0.0), "country": "US"},
        ])
        result = _metric_top_countries(df)
        self.assertEqual(len(result), 0)


# ===========================================================================
# Anomaly Detection
# ===========================================================================

class TestAnomalyDetection(unittest.TestCase):
    """
    Tests for _metric_anomalies().

    We use controlled revenue series where z-scores are known analytically,
    so we can assert exact detection results without relying on large datasets.
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
        Series: 29 days at 100.0, 1 day at 10 000.0.
        """
        values = [100.0] * 29 + [10_000.0]
        anomalies = _metric_anomalies(self._make_revenue_df(values))
        self.assertGreater(len(anomalies), 0)
        self.assertAlmostEqual(anomalies[0]["net_revenue"], 10_000.0, places=0)

    def test_uniform_series_no_anomalies(self):
        """All-identical revenue → std = 0 → no anomalies (early-exit path)."""
        anomalies = _metric_anomalies(self._make_revenue_df([500.0] * 30))
        self.assertEqual(anomalies, [])

    def test_gradual_ramp_no_anomalies(self):
        """
        An evenly-spaced series [0, 1, …, 29] has max |z| ≈ 1.67 < 3.0.
        No day should be flagged.
        """
        values = [float(i) for i in range(30)]
        arr = np.array(values)
        max_z = float(np.max(np.abs((arr - arr.mean()) / arr.std(ddof=0))))
        self.assertLess(max_z, 3.0, msg="Test-data invariant: max z must be below 3.0")

        anomalies = _metric_anomalies(self._make_revenue_df(values))
        self.assertEqual(anomalies, [])

    def test_z_score_value_correct(self):
        """
        Verify z-score arithmetic against a hand-computable series.

        Series: 19 zeros + 1 spike at 100.0 (n=20, population std):
            mean     =  5.0
            variance = (19*(0−5)² + (100−5)²) / 20 = (475 + 9025) / 20 = 475.0
            std      = √475 ≈ 21.794
            z        = (100 − 5) / √475 ≈ 4.36  >  3.0 → flagged
        """
        values = [0.0] * 19 + [100.0]
        mu_exact  = 5.0
        std_exact = float(np.sqrt(475.0))
        expected_z = round((100.0 - mu_exact) / std_exact, 4)

        self.assertGreater(expected_z, 3.0, msg="Test-data invariant: spike must exceed mean+3*std")

        anomalies = _metric_anomalies(self._make_revenue_df(values))
        self.assertGreater(len(anomalies), 0, msg="Spike above mean+3*std must be detected")
        self.assertAlmostEqual(anomalies[0]["z"], expected_z, places=3)

    def test_single_row_returns_empty(self):
        """A single-row series cannot produce a meaningful z-score."""
        anomalies = _metric_anomalies(self._make_revenue_df([500.0]))
        self.assertEqual(anomalies, [])

    def test_multiple_anomalies_sorted_by_z_descending(self):
        """When multiple anomalies exist, the highest z appears first."""
        values = [100.0] * 27 + [50_000.0, 20_000.0, 10_000.0]
        anomalies = _metric_anomalies(self._make_revenue_df(values))
        if len(anomalies) >= 2:
            self.assertGreaterEqual(anomalies[0]["z"], anomalies[1]["z"])


# ===========================================================================
# Daily Active Users
# ===========================================================================

class TestDAU(unittest.TestCase):
    """Tests for _metric_dau()."""

    def test_single_user_single_day(self):
        df = make_clean_df([{"user_id": 1, "date": _date("2023-01-01")}])
        self.assertEqual(_metric_dau(df)[0]["dau"], 1)

    def test_same_user_multiple_events_counted_once(self):
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
        by_date = {r["date"]: r["dau"] for r in _metric_dau(df)}
        self.assertEqual(by_date["2023-01-01"], 2)
        self.assertEqual(by_date["2023-01-02"], 1)

    def test_output_sorted_ascending_by_date(self):
        """DAU output must be sorted ascending by date."""
        df = make_clean_df([
            {"event_id": 1, "user_id": 1, "date": _date("2023-01-03")},
            {"event_id": 2, "user_id": 2, "date": _date("2023-01-01")},
        ])
        dates = [r["date"] for r in _metric_dau(df)]
        self.assertEqual(dates, sorted(dates))


# ===========================================================================
# Funnel
# ===========================================================================

class TestFunnel(unittest.TestCase):
    """Tests for _metric_funnel() — conversion rate arithmetic."""

    def test_conversion_rates_computed_correctly(self):
        """
        pv=10, signup=5, purchase=2 →
            pv_to_signup = 0.5, signup_to_purchase = 0.4
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
        row = _metric_funnel(make_clean_df(rows))[0]
        self.assertEqual(row["pv"],      10)
        self.assertEqual(row["signup"],   5)
        self.assertEqual(row["purchase"], 2)
        self.assertAlmostEqual(row["pv_to_signup"],       0.5, places=4)
        self.assertAlmostEqual(row["signup_to_purchase"], 0.4, places=4)

    def test_zero_page_views_no_division_error(self):
        """Signups but no page_views must not raise ZeroDivisionError."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "signup", "date": _date("2023-01-01")},
        ])
        try:
            row = _metric_funnel(df)[0]
            self.assertEqual(row["pv_to_signup"], 0.0)
        except ZeroDivisionError:
            self.fail("_metric_funnel raised ZeroDivisionError on zero page_views")

    def test_zero_signups_no_division_error(self):
        """Purchases but no signups must not raise ZeroDivisionError."""
        df = make_clean_df([
            {"event_id": 1, "event_type": "page_view", "date": _date("2023-01-01")},
            {"event_id": 2, "event_type": "purchase",
             "amount": np.float32(50.0), "date": _date("2023-01-01")},
        ])
        try:
            row = _metric_funnel(df)[0]
            self.assertEqual(row["signup_to_purchase"], 0.0)
        except ZeroDivisionError:
            self.fail("_metric_funnel raised ZeroDivisionError on zero signups")


if __name__ == "__main__":
    unittest.main(verbosity=2)