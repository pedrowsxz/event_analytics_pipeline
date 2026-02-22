"""
conftest.py
===========
Shared test fixtures and helpers for the pipeline test suite.

Imported implicitly by pytest (no explicit import needed in test files).
For plain `python -m unittest`, each test file imports directly:

    from tests.conftest import make_raw_df, make_clean_df, _date

Design note
-----------
All factory helpers live here so that:
  * There is one authoritative definition of what a "valid raw row" looks like.
  * Test files declare only the columns relevant to their assertion.
  * Changes to the CSV schema (e.g. a new column) are made in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DataFrame factories
# ---------------------------------------------------------------------------

def make_raw_df(rows: list[dict]) -> pd.DataFrame:
    """
    Build a raw DataFrame that mirrors what pandas.read_csv produces from
    events.csv (before any cleaning).

    Default values represent a single valid page_view row on 2023-06-15.
    Pass only the columns you want to override; event_id is auto-assigned
    unless explicitly set, so duplicate ids are opt-in.
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
        r["event_id"] = i + 1          # unique by default; override to test dedup
        r.update(row)
        out.append(r)
    return pd.DataFrame(out)


def make_clean_df(rows: list[dict]) -> pd.DataFrame:
    """
    Build a DataFrame that mirrors what _clean() returns — suitable for
    feeding directly to any _metric_*() function.

    Includes the derived `date` column (UTC-normalised midnight Timestamp).
    Default: a single page_view event for user 1 on 2023-01-01.

    When `ts` is overridden but `date` is not, `date` is derived from `ts`
    automatically so callers never have to keep them in sync manually.
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
    """Parse an ISO date string into a UTC-aware midnight Timestamp.

    Example
    -------
    >>> _date("2023-01-15")
    Timestamp('2023-01-15 00:00:00+0000', tz='UTC')
    """
    return pd.Timestamp(s, tz="UTC")