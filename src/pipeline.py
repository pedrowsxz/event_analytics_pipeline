"""
pipeline.py
===========
Transforms the raw events.csv produced by generate_data.py into an analytics
report (report.json).

Public API
----------
    from pipeline import build_report
    report = build_report("events.csv")           # returns dict
    report = build_report("events.csv", chunksize=500_000)  # chunked load

CLI
---
    python pipeline.py events.csv                 # → report.json
    python pipeline.py events.csv --out my.json
    python pipeline.py events.csv --chunksize 500000

Cleaning decisions (documented once, applied in _clean())
---------------------------------------------------------
D1  Timestamp validity  — Rows whose parsed timestamp has year ≥ 2099 are
    artefacts from the dirty-data injector.  They are dropped rather than
    corrected because there is no recoverable "intended" timestamp.

D2  Country normalisation — country is uppercased unconditionally.  Empty /
    NaN values (injected as empty strings, read back as NaN by pandas) are
    replaced with "UNK" so aggregations remain stable.

D3  Invalid event_type — Only {"page_view","signup","purchase","refund"} are
    meaningful for any downstream metric.  Rows with "???" are dropped.

D4  Deduplication — event_id is the logical primary key.  Among duplicates
    the first-seen row is kept (stable sort order preserves the original
    generation sequence).

D5  Amount noise — generate_data.py can leave a small non-zero amount on rows
    whose event_type was originally purchase/refund but whose amount column
    was not zeroed before the event_type was corrupted to "???".  After D3
    those rows are gone.  For any surviving row that is neither purchase nor
    refund, amount is zeroed to prevent revenue metrics from being polluted.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_EVENT_TYPES: frozenset[str] = frozenset(
    {"page_view", "signup", "purchase", "refund"}
)

# Dtype map for read_csv — keeps memory low on initial load.
# event_type / country / device are low-cardinality strings; we convert them
# to pd.Categorical after cleaning, not before, because dirty values would
# pollute the category index.
_CSV_DTYPES: dict[str, str] = {
    "event_id":   "int64",
    "user_id":    "int32",
    "amount":     "float32",
    "session_id": "int32",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_csv_full(path: str, chunksize: Optional[int]) -> pd.DataFrame:
    """
    Load events CSV.

    If chunksize is given, reads in chunks and concatenates.  Chunking is
    useful when the file does not fit in RAM; for a 200 MB file on a modern
    machine a single read is faster.

    [PERF] We specify dtype for numeric columns upfront to avoid pandas
    defaulting everything to int64/float64 and then re-casting later.
    The string columns (ts, event_type, country, device) are left as object
    here because dtype=str at read time is equivalent but more explicit.
    """
    read_kwargs = dict(
        dtype=_CSV_DTYPES,
        low_memory=False,   # prevent mixed-type inference warnings
    )
    if chunksize:
        log.info("Loading %s in chunks of %d …", path, chunksize)
        chunks = pd.read_csv(path, chunksize=chunksize, **read_kwargs)
        df = pd.concat(chunks, ignore_index=True, copy=False)
    else:
        log.info("Loading %s …", path)
        df = pd.read_csv(path, **read_kwargs)

    log.info("Loaded %d raw rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Apply all cleaning rules (D1–D5) and return the cleaned DataFrame plus a
    counts dict for the report header.

    All operations are vectorised — no Python-level row loops.

    Returns
    -------
    clean_df   : pd.DataFrame  – ready for metric computation
    counts     : dict          – {"raw_rows", "valid_rows", "dropped_rows"}
    """
    raw_rows = len(df)
    dropped: dict[str, int] = {}

    # ── D1: Parse timestamps, drop year ≥ 2099 ───────────────────────────
    # errors="coerce" turns un-parseable strings into NaT (no exception).
    # utc=True ensures all timestamps are timezone-aware (UTC) so arithmetic
    # is unambiguous.
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    # Flag: NaT (unparseable) OR far-future sentinel (year ≥ 2099).
    invalid_ts_mask = df["ts"].isna() | (df["ts"].dt.year >= 2099)
    dropped["invalid_ts"] = int(invalid_ts_mask.sum())
    df = df[~invalid_ts_mask].copy()

    # Extract a date column once (reused by every metric) — date objects are
    # lighter than full Timestamps for groupby keys.
    df["date"] = df["ts"].dt.normalize()   # midnight-anchored Timestamp (UTC)

    # ── D2: Country normalisation ─────────────────────────────────────────
    # .fillna first, then .str.upper() — both are vectorised Series methods.
    df["country"] = df["country"].fillna("UNK").str.upper().str.strip()

    # ── D3: Drop invalid event_type rows ─────────────────────────────────
    valid_etype_mask = df["event_type"].isin(VALID_EVENT_TYPES)
    dropped["invalid_etype"] = int((~valid_etype_mask).sum())
    df = df[valid_etype_mask].copy()

    # ── D4: Deduplicate on event_id — keep first occurrence ───────────────
    # [DESIGN] ~duplicated(keep="first") is a single O(n) hash pass.
    # Sorting first would be O(n log n); we don't need sorted order here.
    dup_mask = df.duplicated(subset=["event_id"], keep="first")
    dropped["duplicate_event_id"] = int(dup_mask.sum())
    df = df[~dup_mask].copy()

    # ── D5: Zero out amount on non-revenue event types ─────────────────────
    # After D3 only valid types remain; zero any residual amount noise on
    # non-monetary rows using a boolean mask assignment (vectorised).
    non_revenue_mask = ~df["event_type"].isin({"purchase", "refund"})
    df.loc[non_revenue_mask, "amount"] = np.float32(0.0)

    # ── Dtype discipline ──────────────────────────────────────────────────
    # Converting low-cardinality string columns to Categorical after cleaning
    # means the category index only contains valid values.
    # [PERF] Reduces memory from ~50 B/row (object) to ~1 B/row (int8 code).
    for col in ("event_type", "country", "device"):
        df[col] = df[col].astype("category")

    # Downcast amount — float32 is sufficient for currency values ≤ $10 000.
    df["amount"] = df["amount"].astype("float32")

    valid_rows = len(df)
    total_dropped = raw_rows - valid_rows
    counts = {
        "raw_rows":     raw_rows,
        "valid_rows":   valid_rows,
        "dropped_rows": total_dropped,
        "breakdown":    dropped,  # per-rule counts for observability
    }

    log.info(
        "Clean: %d valid, %d dropped  (ts=%d, etype=%d, dups=%d)",
        valid_rows, total_dropped,
        dropped["invalid_ts"], dropped["invalid_etype"], dropped["duplicate_event_id"],
    )
    return df, counts


# ---------------------------------------------------------------------------
# Metric 1 – Daily Active Users
# ---------------------------------------------------------------------------

def _metric_dau(df: pd.DataFrame) -> list[dict]:
    """
    Unique user_ids per calendar day.

    [DESIGN] groupby + nunique is a single hash-aggregation pass.
    We sort by date so the output is deterministic and easy to plot.
    """
    dau = (
        df.groupby("date", observed=True)["user_id"]
        .nunique()
        .rename("dau")
        .sort_index()
        .reset_index()
    )
    dau["date"] = dau["date"].dt.strftime("%Y-%m-%d")
    return dau.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Metric 2 – Conversion Funnel Per Day
# ---------------------------------------------------------------------------

def _metric_funnel(df: pd.DataFrame) -> list[dict]:
    """
    Per-day counts of page_view, signup, purchase and the two conversion rates:

        pv_to_signup       = signup / page_view
        signup_to_purchase = purchase / signup

    [DESIGN] We filter to the three funnel event types, then use a groupby +
    pivot_table to get a (date × event_type) count matrix in one pass.
    np.where handles the zero-division case without a Python loop.
    """
    funnel_types = {"page_view", "signup", "purchase"}
    fdf = df[df["event_type"].isin(funnel_types)]

    pivot = (
        fdf.groupby(["date", "event_type"], observed=True)
        .size()
        .unstack(fill_value=0)
        .rename(columns={"page_view": "pv", "signup": "signup", "purchase": "purchase"})
    )

    # Ensure all three columns exist even if a day has no signups/purchases.
    for col in ("pv", "signup", "purchase"):
        if col not in pivot.columns:
            pivot[col] = 0

    pv       = pivot["pv"].to_numpy(dtype="float64")
    signup   = pivot["signup"].to_numpy(dtype="float64")
    purchase = pivot["purchase"].to_numpy(dtype="float64")

    # np.where for vectorised zero-safe division — no Python loop.
    pv_to_signup       = np.where(pv > 0,      signup   / pv,      0.0)
    signup_to_purchase = np.where(signup > 0,  purchase / signup,  0.0)

    result = pivot.reset_index().copy()
    result["pv_to_signup"]       = np.round(pv_to_signup,       4)
    result["signup_to_purchase"] = np.round(signup_to_purchase, 4)
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")

    return result[["date", "pv", "signup", "purchase",
                   "pv_to_signup", "signup_to_purchase"]].to_dict(orient="records")


# ---------------------------------------------------------------------------
# Metric 3 – Daily Net Revenue
# ---------------------------------------------------------------------------

def _metric_revenue_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Net revenue per day = sum(purchase amounts) + sum(refund amounts).

    Refund amounts are stored as negative floats (see generate_data.py), so
    a single .sum() on the filtered rows gives the correct net figure.

    Returns a DataFrame (not a list) because Metric 5 (anomaly detection)
    needs the numeric Series for NumPy statistics.
    """
    rev_df = (
        df[df["event_type"].isin({"purchase", "refund"})]
        .groupby("date", observed=True)["amount"]
        .sum()
        .rename("net_revenue")
        .sort_index()
        .reset_index()
    )
    rev_df["net_revenue"] = rev_df["net_revenue"].astype("float64")  # promote for stats
    return rev_df


# ---------------------------------------------------------------------------
# Metric 4 – Top 10 Countries by Net Revenue
# ---------------------------------------------------------------------------

def _metric_top_countries(df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """
    Sum net revenue (purchases + refunds, refunds already negative) per
    country, return top N descending.

    [DESIGN] groupby + sum + nlargest is fully vectorised.
    "UNK" rows are included — they represent real revenue whose origin is
    unknown; excluding them would silently distort totals.
    """
    rev = (
        df[df["event_type"].isin({"purchase", "refund"})]
        .groupby("country", observed=True)["amount"]
        .sum()
        .rename("net_revenue")
        .nlargest(top_n)
        .reset_index()
    )
    rev["net_revenue"] = rev["net_revenue"].round(2).astype("float64")
    return rev.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Metric 5 – Revenue Anomaly Detection
# ---------------------------------------------------------------------------

def _metric_anomalies(rev_df: pd.DataFrame) -> list[dict]:
    """
    Flag days where net_revenue > mean + 3 * std (upper-tail spike detection).

    Statistical approach
    --------------------
    We use population statistics (ddof=0) because we treat the observed
    window as the full population of interest — not a sample from a larger
    distribution.  For a monitoring use-case where we want to detect outliers
    within the visible dataset, population std is appropriate.

    [DESIGN] np.mean / np.std operate on a 1-D float64 array — single-pass
    BLAS-backed routines, not pandas rolling.  z-score is computed as a
    vectorised array division.  np.where selects anomalies without a loop.

    Returns days sorted by z-score descending so the worst outliers appear
    first.
    """
    values: np.ndarray = rev_df["net_revenue"].to_numpy(dtype="float64")

    if values.size < 2:
        log.warning("Too few data points for anomaly detection.")
        return []

    mu:  float = float(np.mean(values))
    std: float = float(np.std(values, ddof=0))

    if std == 0.0:
        log.warning("Revenue std is zero — all days have identical revenue.")
        return []

    z_scores: np.ndarray = (values - mu) / std            # vectorised
    anomaly_mask: np.ndarray = z_scores > 3.0             # upper-tail only

    anomaly_df = rev_df[anomaly_mask].copy()
    anomaly_df["z"] = np.round(z_scores[anomaly_mask], 4)
    anomaly_df["net_revenue"] = anomaly_df["net_revenue"].round(2)
    anomaly_df["date"] = anomaly_df["date"].dt.strftime("%Y-%m-%d")

    return (
        anomaly_df[["date", "net_revenue", "z"]]
        .sort_values("z", ascending=False)
        .to_dict(orient="records")
    )


# ---------------------------------------------------------------------------
# Metric 6 – D1 Retention
# ---------------------------------------------------------------------------

def _metric_retention_d1(df: pd.DataFrame) -> list[dict]:
    """
    D1 Retention: for each cohort (a user's first-seen date), what fraction
    of users returned the very next day?

    Algorithm (fully vectorised, no Python loops)
    ---------------------------------------------
    Step 1  — first_day: groupby user_id, take min(date).
              This is an O(n) single-pass aggregation.

    Step 2  — pairs: deduplicated (user_id, date) set.
              drop_duplicates is a hash-based O(n) operation.

    Step 3  — Merge pairs with first_day on user_id to get
              (user_id, date, cohort) for every day each user was active.

    Step 4  — Boolean mask: date == cohort + 1 day.
              Timedelta arithmetic is vectorised on DatetimeIndex.

    Step 5  — groupby cohort:
              • total users = nunique(user_id) on first_day
              • retained   = nunique(user_id) where mask is True

    Step 6  — align on cohort index, compute rate = retained / users.

    [DESIGN] We drop the last date in the dataset from the cohort list
    because users who first appeared on the final day have no "day 2" to
    return to — including them would artificially deflate retention.
    """
    if df.empty:
        return []

    # Step 1 – cohort date per user
    first_day = (
        df.groupby("user_id", sort=False)["date"]
        .min()
        .rename("cohort")
        .reset_index()
    )

    max_date: pd.Timestamp = df["date"].max()

    # Step 2 – unique (user_id, date) activity pairs
    pairs = df[["user_id", "date"]].drop_duplicates()

    # Step 3 – attach cohort date to every activity row
    merged = pairs.merge(first_day, on="user_id", how="left")

    # Step 4 – retained = active exactly 1 day after cohort
    day_delta: pd.Series = (merged["date"] - merged["cohort"]).dt.days
    retained_mask = day_delta == 1

    # Step 5 – aggregate
    cohort_sizes = (
        first_day.groupby("cohort")["user_id"]
        .nunique()
        .rename("users")
    )
    retained_counts = (
        merged[retained_mask]
        .groupby("cohort")["user_id"]
        .nunique()
        .rename("retained")
    )

    # Step 6 – align + compute rate
    result = pd.concat([cohort_sizes, retained_counts], axis=1)
    result["retained"] = result["retained"].fillna(0).astype("int32")

    # Drop the last day (no D+1 window available)
    result = result[result.index < max_date].copy()

    result["rate"] = np.round(
        result["retained"].to_numpy(dtype="float64") /
        result["users"].to_numpy(dtype="float64"),
        4,
    )
    result.index.name = "cohort_date"
    result = result.reset_index()
    result["cohort_date"] = result["cohort_date"].dt.strftime("%Y-%m-%d")

    return result[["cohort_date", "users", "retained", "rate"]].to_dict(orient="records")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_report(
    path_csv: str,
    chunksize: Optional[int] = None,
) -> dict:
    """
    Load, clean, and aggregate events.csv into a structured analytics report.

    Parameters
    ----------
    path_csv  : str            – Path to the events CSV file.
    chunksize : int, optional  – If set, reads the CSV in chunks of this many
                                 rows to cap peak memory usage.  Useful when
                                 the file is larger than available RAM.
                                 Typical sweet spot: 500_000.

    Returns
    -------
    dict with keys:
        range           – {"start": ISO date, "end": ISO date}
        counts          – raw/valid/dropped row counts
        dau             – list[{date, dau}]
        funnel          – list[{date, pv, signup, purchase, pv_to_signup,
                                signup_to_purchase}]
        revenue_daily   – list[{date, net_revenue}]
        top_countries   – list[{country, net_revenue}]
        anomalies       – list[{date, net_revenue, z}]
        retention_d1    – list[{cohort_date, users, retained, rate}]
    """
    t0 = time.perf_counter()

    # ── Load ─────────────────────────────────────────────────────────────
    df_raw = _read_csv_full(path_csv, chunksize)

    # ── Clean ─────────────────────────────────────────────────────────────
    df, counts = _clean(df_raw)
    del df_raw  # release raw memory as early as possible

    if df.empty:
        raise ValueError("No valid rows remain after cleaning — cannot build report.")

    date_range = {
        "start": df["date"].min().strftime("%Y-%m-%d"),
        "end":   df["date"].max().strftime("%Y-%m-%d"),
    }
    log.info("Date range: %s → %s", date_range["start"], date_range["end"])

    # ── Metrics ──────────────────────────────────────────────────────────
    log.info("Computing DAU …")
    dau = _metric_dau(df)

    log.info("Computing funnel …")
    funnel = _metric_funnel(df)

    log.info("Computing daily revenue …")
    rev_df = _metric_revenue_daily(df)
    revenue_daily_out = [
        {
            "date":        row["date"].strftime("%Y-%m-%d"),
            "net_revenue": round(float(row["net_revenue"]), 2),
        }
        for row in rev_df.to_dict(orient="records")
    ]

    log.info("Computing top countries …")
    top_countries = _metric_top_countries(df)

    log.info("Running anomaly detection …")
    anomalies = _metric_anomalies(rev_df)

    log.info("Computing D1 retention …")
    retention_d1 = _metric_retention_d1(df)

    elapsed = time.perf_counter() - t0
    log.info("Report built in %.2fs.", elapsed)

    return {
        "range":          date_range,
        "counts":         counts,
        "dau":            dau,
        "funnel":         funnel,
        "revenue_daily":  revenue_daily_out,
        "top_countries":  top_countries,
        "anomalies":      anomalies,
        "retention_d1":   retention_d1,
    }


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------

def save_report(report: dict, out_path: str) -> None:
    """Serialise report dict to JSON with readable formatting."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    log.info("Report written → %s  (%.1f KB)", path, path.stat().st_size / 1024)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Build analytics report from events.csv",
    )
    p.add_argument("csv",    help="Path to events CSV")
    p.add_argument("--out",  default="report.json", help="Output JSON path  [report.json]")
    p.add_argument(
        "--chunksize",
        type=int,
        default=None,
        metavar="N",
        help="Read CSV in chunks of N rows (trades speed for lower peak RAM)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity  [INFO]",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(args.log_level)

    try:
        report = build_report(args.csv, chunksize=args.chunksize)
        save_report(report, args.out)
    except FileNotFoundError:
        log.error("CSV not found: %s", args.csv)
        sys.exit(1)
    except ValueError as exc:
        log.error("Pipeline error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()