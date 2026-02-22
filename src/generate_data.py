"""
generate_data.py
================
Generates a large synthetic events CSV (1–3 million rows) using NumPy for
all data generation. No Python-level row loops are used anywhere in the hot
path.

Design decisions are explained in-line as "# [DESIGN]" comments so every
non-obvious choice is traceable.

Usage
-----
    python generate_data.py                  # 3 000 000 rows  → events.csv
    python generate_data.py 1000000          # custom row count
    python generate_data.py 3000000 out.csv  # custom output path

Requirements
------------
    pip install numpy pandas pyarrow  # pyarrow makes to_csv 2-3× faster
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RNG_SEED = 42

# Event-type probabilities (must sum ≤ 1; remainder = page_view)
# [DESIGN] We model the rare types first so their shares are exact; page_view
# absorbs all remaining probability mass.
PURCHASE_RATE  = 0.035   # 3.5 % → satisfies "2–5 %"
REFUND_RATE    = 0.005   # 0.5 % → satisfies "0.1–0.5 %"
SIGNUP_RATE    = 0.05    # 5 %
# page_view = 1 - PURCHASE_RATE - REFUND_RATE - SIGNUP_RATE ≈ 91.2 %

# Dirty-data injection rates (each independently 0.5 %)
DIRTY_RATE = 0.005

# Lognormal parameters for purchase amounts
# E[X] = exp(μ + σ²/2).  With μ=3.5, σ=1.2 → mean ≈ $60, long tail up to
# several thousand dollars — realistic for an e-commerce platform.
LOGNORMAL_MU    = 3.5
LOGNORMAL_SIGMA = 1.2

# Timestamp range: 2023-01-01 → 2024-12-31
TS_START = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp())
TS_END   = int(pd.Timestamp("2024-12-31 23:59:59", tz="UTC").timestamp())

COUNTRIES = ["BR", "US", "MX", "DE", "FR", "GB", "IN", "CA", "AU", "JP"]
DEVICES   = ["ios", "android", "web"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(msg: str) -> None:
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")


def elapsed(t0: float) -> str:
    return f"{time.perf_counter() - t0:.2f}s"


# ---------------------------------------------------------------------------
# Step 1 – Event-type generation (vectorised with np.searchsorted)
# ---------------------------------------------------------------------------
# [DESIGN] How to vectorise event_type generation efficiently
# -----------------------------------------------------------
# We draw a single uniform random array of shape (N,) and partition it into
# buckets using a cumulative-probability boundary array.  np.searchsorted
# maps every float to a bucket index in O(N log k) where k=4 — no Python
# loop, no repeated np.where chains.
#
#   0                0.003     0.038    0.088            1.0
#   |── refund ────|── pur ──|─ sig ─|────── page_view ──|
#
# The ordering (rarest first) keeps the boundary array tidy but has no
# correctness impact.

EVENT_LABELS   = np.array(["refund", "purchase", "signup", "page_view"], dtype="U9")
EVENT_BOUNDS   = np.array([REFUND_RATE,
                            REFUND_RATE + PURCHASE_RATE,
                            REFUND_RATE + PURCHASE_RATE + SIGNUP_RATE])


def generate_event_types(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return an array of event-type strings, shape (n,), no Python loops."""
    u = rng.uniform(0, 1, size=n)
    # searchsorted returns 0,1,2,3 → index into EVENT_LABELS
    indices = np.searchsorted(EVENT_BOUNDS, u, side="left")
    return EVENT_LABELS[indices]          # fancy-indexing: fully vectorised


# ---------------------------------------------------------------------------
# Step 2 – Enforce refund → purchase dependency without loops
# ---------------------------------------------------------------------------
# [DESIGN] Two-phase approach: plan then apply.
#
#   Phase 1 — _build_refund_links():
#     Pure function.  Decides which refund maps to which purchase using
#     np.searchsorted on cumulative probabilities.  Returns a RefundLinks
#     dataclass (named index arrays) — no column data is modified here
#     except for the edge-case demotion when no purchases exist.
#
#   Phase 2 — _apply_refund_links():
#     Applies every referential constraint that flows from the mapping:
#     session_id, user_id, timestamp, amount.  All four live here so the
#     invariant is maintained and extended in exactly one place.
#
# Why a dataclass instead of a raw tuple?
#   links.purchase_idx is unambiguous.
#   The second element of a 3-tuple (session_ids, refund_idx, purchase_idx)
#   is not — a reader must count positions and read the docstring to know
#   which array is which.  Naming eliminates that cognitive load and prevents
#   positional swap bugs when the signature is refactored.
#
# Protected-purchase mask:
#   generate() builds it from links.purchase_idx immediately after calling
#   _build_refund_links, before _apply_refund_links runs.  This shields
#   linked purchase rows from event_type corruption in inject_dirty_data,
#   preserving the refund→purchase invariant post-injection.

@dataclass(frozen=True)
class RefundLinks:
    """
    Resolved mapping from each refund row to its source purchase row.

    Both arrays are parallel, length == n_refunds.
    frozen=True prevents accidental mutation after construction.
    """
    refund_idx:   np.ndarray   # row positions of refund events
    purchase_idx: np.ndarray   # row positions of their linked purchases

    def __len__(self) -> int:
        return len(self.refund_idx)


def _build_refund_links(
    event_types: np.ndarray,
    rng: np.random.Generator,
) -> RefundLinks:
    """
    PLAN phase: decide which refund maps to which purchase.

    Pure function — only reads event_types.  The one exception is the
    edge-case demotion (no purchases exist), which changes a row's identity,
    not its derived data, so it belongs here.

    Parameters
    ----------
    event_types : np.ndarray  – mutable string array from generate_event_types()
    rng         : Generator   – passed in so the caller controls the RNG state

    Returns
    -------
    RefundLinks with parallel refund_idx / purchase_idx arrays.
    """
    purchase_mask = event_types == "purchase"
    refund_mask   = event_types == "refund"
    empty = RefundLinks(np.empty(0, np.intp), np.empty(0, np.intp))

    if not purchase_mask.any():
        # Edge case: no purchases — demote refunds so no dangling links exist.
        event_types[refund_mask] = "page_view"
        return empty

    refund_idx   = np.where(refund_mask)[0]      # shape: (n_refunds,)
    purchase_idx = np.where(purchase_mask)[0]    # shape: (n_purchases,)

    # Pick a random purchase for each refund (with replacement — realistic).
    # rng.choice is O(n_refunds) vectorised — no Python loop.
    linked = rng.choice(purchase_idx, size=len(refund_idx), replace=True)
    return RefundLinks(refund_idx, linked)


def _apply_refund_links(
    links:       RefundLinks,
    user_ids:    np.ndarray,
    session_ids: np.ndarray,
    timestamps:  np.ndarray,
    amounts:     np.ndarray,
    rng:         np.random.Generator,
) -> None:
    """
    APPLY phase: enforce all referential constraints that flow from the link.

    All four constraints live here together so the invariant is defined and
    extended in exactly one place.  Mutates arrays in-place; caller owns them.

    Constraints applied
    -------------------
    session_id : refund shares the purchase's session (CSV-level annotation)
    user_id    : refund must be by the same user who made the purchase
    timestamp  : refund occurs 1 hour–30 days after its purchase
    amount     : refund.amount == -purchase.amount  (the core invariant)
    """
    if not len(links):
        return

    ri = links.refund_idx
    pi = links.purchase_idx

    # Same session as the linked purchase (decorative — real link is the index)
    session_ids[ri] = session_ids[pi]

    # Same user — a refund must be issued by the purchaser
    user_ids[ri] = user_ids[pi]

    # Refund happens 1 hour–30 days after the purchase, capped at TS_END
    offsets = rng.integers(3_600, 30 * 86_400, size=len(ri), dtype=np.int64)
    timestamps[ri] = np.minimum(timestamps[pi] + offsets, TS_END)

    # Exact amount mirror — the invariant enforced throughout this codebase
    amounts[ri] = -amounts[pi]


# ---------------------------------------------------------------------------
# Step 3 – Inject dirty data without loops
# ---------------------------------------------------------------------------

INVALID_TS = int(pd.Timestamp("2099-01-01", tz="UTC").timestamp())  # year 2099


def inject_dirty_data(
    rng:                    np.random.Generator,
    n:                      int,
    timestamps:             np.ndarray,      # int64
    countries:              np.ndarray,      # object/str
    event_types:            np.ndarray,      # str
    event_ids:              np.ndarray,      # int64
    protected_purchase_mask: np.ndarray,     # bool, shape (n,) — [FIX 2]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mutate arrays in-place and return them."""

    # Independent masks — each ~0.5 % of rows
    r = rng.random((4, n))                         # shape (4, N), one draw

    ts_mask      = r[0] < DIRTY_RATE
    country_mask = r[1] < DIRTY_RATE
    # Exclude protected purchase rows from event_type corruption.
    etype_mask   = (r[2] < DIRTY_RATE) & ~protected_purchase_mask
    dup_mask     = r[3] < DIRTY_RATE

    # 1. Invalid timestamps
    timestamps[ts_mask] = INVALID_TS

    # 2. Null countries (empty string → treat as null in downstream pipelines)
    countries[country_mask] = ""

    # 3. Invalid event_type
    event_types[etype_mask] = "???"

    # 4. Duplicate event_ids — fix
    n_dups = dup_mask.sum()
    if n_dups > 0:
        dup_indices = np.where(dup_mask)[0]          # exact target positions

        # Snapshot before ANY mutation so chain-duplicates are impossible.
        ids_snapshot = event_ids.copy()

        # Draw source positions; ensure no self-copies in one vectorised step.
        src = rng.integers(0, n, size=n_dups)
        self_copy = src == dup_indices               # bool mask, shape (n_dups,)
        src[self_copy] = (src[self_copy] + 1) % n   # shift by 1, wrap around

        # Copy from the original snapshot — never from a row already mutated.
        event_ids[dup_indices] = ids_snapshot[src]

    return timestamps, countries, event_types, event_ids



# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate(n: int, rng: np.random.Generator) -> pd.DataFrame:
    section(f"Generating {n:,} rows")
    t0 = time.perf_counter()

    # ── IDs ──────────────────────────────────────────────────────────────
    # [DESIGN] Integer event_ids are generated as a shuffle of [0, n) so
    # they start as unique; dirty-data injection will intentionally create
    # ~0.5 % collisions afterward.
    event_ids  = rng.permutation(n).astype(np.int64)
    user_ids   = rng.integers(1, 500_001, size=n, dtype=np.int32)
    session_ids = rng.integers(1, 2_000_001, size=n, dtype=np.int32)

    print(f"  IDs generated          {elapsed(t0)}")

    # ── Timestamps ───────────────────────────────────────────────────────
    # [DESIGN] Draw uniform integers in [TS_START, TS_END] — second-level
    # resolution, no Python datetime objects until the final conversion.
    timestamps = rng.integers(TS_START, TS_END + 1, size=n, dtype=np.int64)

    print(f"  Timestamps generated   {elapsed(t0)}")

    # ── Event types (vectorised bucket method) ────────────────────────────
    event_types = generate_event_types(rng, n)
    print(f"  Event types generated  {elapsed(t0)}")

    # ── Refund → purchase linkage — Phase 1: plan ────────────────────────
    # _build_refund_links() decides which refund maps to which purchase.
    # It returns a RefundLinks dataclass (named index arrays) without touching
    # any column data beyond the edge-case demotion.
    links = _build_refund_links(event_types, rng)

    # Build the protected-purchase mask before dirty injection so that linked
    # purchase rows can never receive the "???" event_type sentinel (FIX 2).
    # np.zeros + fancy-index assignment is O(n_linked) — fully vectorised.
    protected_purchase_mask = np.zeros(n, dtype=bool)
    if len(links):
        protected_purchase_mask[links.purchase_idx] = True

    print(f"  Refund links resolved  {elapsed(t0)}")

    # ── Amounts (lognormal, purchase rows only) ───────────────────────────
    # [DESIGN] Draw lognormal for all rows — cheap — then zero non-purchase
    # rows with a boolean mask.  Avoids conditional loops.
    # Refund amounts are set to -purchase.amount inside _apply_refund_links
    # (Phase 2 below) so they don't need special treatment here.
    raw_amounts = rng.lognormal(mean=LOGNORMAL_MU, sigma=LOGNORMAL_SIGMA, size=n)
    amounts = raw_amounts.astype(np.float32)
    amounts[event_types != "purchase"] = 0.0

    print(f"  Amounts generated      {elapsed(t0)}")

    # ── Refund → purchase linkage — Phase 2: apply ───────────────────────
    # _apply_refund_links() enforces all four referential constraints
    # (session_id, user_id, timestamp, amount) in one place.
    _apply_refund_links(links, user_ids, session_ids, timestamps, amounts, rng)
    print(f"  Refund constraints applied  {elapsed(t0)}")

    # ── Countries & devices ───────────────────────────────────────────────
    # [DESIGN] rng.integers picks random indices into the lookup lists;
    # fancy indexing maps them to strings — fully vectorised.
    country_arr = np.array(COUNTRIES)
    device_arr  = np.array(DEVICES)

    country_idx = rng.integers(0, len(country_arr), size=n)
    device_idx  = rng.integers(0, len(device_arr), size=n)

    countries   = country_arr[country_idx]    # object array of strings
    devices     = device_arr[device_idx]

    print(f"  Countries/devices gen  {elapsed(t0)}")

    # ── Dirty data injection ──────────────────────────────────────────────
    timestamps, countries, event_types, event_ids = inject_dirty_data(
        rng, n, timestamps, countries, event_types, event_ids,
        protected_purchase_mask,                               # [FIX 2]
    )
    print(f"  Dirty data injected    {elapsed(t0)}")

    # ── Assemble DataFrame with dtype discipline ──────────────────────────
    df = pd.DataFrame(
        {
            "event_id":   pd.array(event_ids,  dtype="int64"),
            "user_id":    pd.array(user_ids,   dtype="int32"),
            "ts":         pd.to_datetime(timestamps, unit="s", utc=True),
            "event_type": pd.Categorical(event_types),
            "amount":     pd.array(amounts, dtype="float32"),
            "country":    pd.Categorical(countries),
            "device":     pd.Categorical(devices),
            "session_id": pd.array(session_ids, dtype="int32"),
        }
    )

    print(f"  DataFrame assembled    {elapsed(t0)}")
    return df, links


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(
    df: pd.DataFrame,
    links: RefundLinks,
) -> None:
    section("Dataset Summary")
    n = len(df)

    counts = df["event_type"].value_counts(dropna=False)
    print(f"  Total rows         : {n:>12,}")
    print(f"  Unique event_ids   : {df['event_id'].nunique():>12,}")
    print()
    print("  Event-type breakdown:")
    for etype, cnt in counts.items():
        print(f"    {str(etype):<12} {cnt:>10,}  ({cnt/n*100:5.2f} %)")

    print()
    print("  Dirty-data checks:")
    invalid_ts    = (df["ts"].dt.year >= 2099).sum()
    null_country  = (df["country"].astype(str) == "").sum()
    invalid_etype = (df["event_type"].astype(str) == "???").sum()
    dup_ids       = n - df["event_id"].nunique()
    print(f"    Invalid timestamps : {invalid_ts:>10,}  ({invalid_ts/n*100:.2f} %)")
    print(f"    Null countries     : {null_country:>10,}  ({null_country/n*100:.2f} %)")
    print(f"    Invalid event_type : {invalid_etype:>10,}  ({invalid_etype/n*100:.2f} %)")
    print(f"    Duplicate event_id : {dup_ids:>10,}  ({dup_ids/n*100:.2f} %)")

    print()
    purchase_amounts = df.loc[df["event_type"] == "purchase", "amount"]
    if len(purchase_amounts):
        print("  Purchase amount stats:")
        print(f"    Min    : ${purchase_amounts.min():>10.2f}")
        print(f"    Median : ${purchase_amounts.median():>10.2f}")
        print(f"    Mean   : ${purchase_amounts.mean():>10.2f}")
        print(f"    Max    : ${purchase_amounts.max():>10.2f}")

    # ── Invariant verification ────────────────────────────────────────────
    # [DESIGN] All three checks use the exact row-index arrays returned by
    # generate() — no session_id join, no ambiguity from shared session_ids.
    #
    #   Inv 1: event_types[linked_purchase_indices] are all "purchase".
    #          One boolean fancy-index + .all().
    #
    #   Inv 2: amounts[refund_indices] == -amounts[linked_purchase_indices].
    #          Two fancy-index reads + np.isclose, fully vectorised.
    #
    #   Inv 3: Duplicate of Inv 1, confirming the condition still holds
    #          post dirty injection (same check, run after injection).
    #
    # Using direct integer index arrays avoids the ambiguity of matching on
    # session_id (multiple purchases can share a session_id).
    print()
    print("  Invariant checks (must all PASS):")

    if len(links) == 0:
        print("    [SKIP] No refund rows — invariants vacuously satisfied.")
    else:
        amounts_arr    = df["amount"].to_numpy(dtype="float32")
        etypes_arr     = df["event_type"].astype(str).to_numpy()

        # Inv 1 & 3 — linked row still has event_type == "purchase"
        # (same check; run after injection, so this confirms post-injection state)
        linked_etypes  = etypes_arr[links.purchase_idx]               # fancy index
        inv1_ok        = bool((linked_etypes == "purchase").all())
        n_not_purchase = int((linked_etypes != "purchase").sum())
        print(f"    [{'PASS' if inv1_ok else 'FAIL'}] "
              f"Every linked row has event_type=='purchase' "
              f"(violations: {n_not_purchase})")

        # Inv 2 — refund.amount == -purchase.amount (exact row match)
        refund_amounts   = amounts_arr[links.refund_idx]              # fancy index
        purchase_amounts_linked = amounts_arr[links.purchase_idx]       # fancy index
        match            = np.isclose(refund_amounts, -purchase_amounts_linked, atol=1e-4)
        inv2_ok          = bool(match.all())
        n_mismatch       = int((~match).sum())
        print(f"    [{'PASS' if inv2_ok else 'FAIL'}] "
              f"refund.amount == -purchase.amount (mismatches: {n_mismatch})")

    print()
    print("  Memory usage (DataFrame):")
    mem = df.memory_usage(deep=True)
    for col, b in mem.items():
        if col == "Index":
            continue
        print(f"    {col:<12} {b/1024**2:>7.1f} MB")
    print(f"    {'TOTAL':<12} {mem.sum()/1024**2:>7.1f} MB")


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: Path) -> None:
    section(f"Writing → {path}")
    t0 = time.perf_counter()

    # [DESIGN] pyarrow's CSV writer is significantly faster and streams data
    # without materialising the entire string representation in RAM.
    # We fall back to pandas if pyarrow is unavailable.
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv

        # pyarrow does not natively understand pd.Categorical or
        # timezone-aware timestamps — convert those columns first.
        df_out = df.copy()
        df_out["ts"]         = df_out["ts"].astype(str)
        df_out["event_type"] = df_out["event_type"].astype(str)
        df_out["country"]    = df_out["country"].astype(str)
        df_out["device"]     = df_out["device"].astype(str)

        table = pa.Table.from_pandas(df_out, preserve_index=False)
        pacsv.write_csv(table, str(path))
        print(f"  Written via pyarrow    {elapsed(t0)}")

    except ImportError:
        # Fallback: pandas chunked write keeps peak RSS in check
        CHUNK = 500_000
        first = True
        for start in range(0, len(df), CHUNK):
            chunk = df.iloc[start : start + CHUNK]
            chunk.to_csv(path, mode="w" if first else "a",
                         index=False, header=first)
            first = False
            print(f"  Written rows {start:>9,} – {min(start+CHUNK, len(df)):>9,}  {elapsed(t0)}")

    size_mb = path.stat().st_size / 1024**2
    print(f"  File size: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    N_ROWS: int = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
    OUTPUT_PATH: Path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("events.csv")
    
    print(f"""
╔══════════════════════════════════════════════════╗
║          events.csv Generator                    ║
║  rows   : {N_ROWS:>12,}                          ║
║  output : {str(OUTPUT_PATH):<38}  ║
╚══════════════════════════════════════════════════╝""")

    rng = np.random.default_rng(RNG_SEED)
    t_total = time.perf_counter()

    df, links = generate(N_ROWS, rng)
    report(df, links)
    write_csv(df, OUTPUT_PATH)

    section("Done")
    print(f"  Total wall time: {elapsed(t_total)}")


if __name__ == "__main__":
    main()