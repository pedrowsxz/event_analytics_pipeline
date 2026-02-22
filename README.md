
# Events Analytics Pipeline

A high-performance, production-grade Python data engineering project that transforms raw event logs into structured analytics reports. Processes millions of rows efficiently using NumPy vectorisation and Pandas, with HTTP API layer and comprehensive test coverage.

## Quick Start

### Prerequisites
```bash
pip install numpy pandas pyarrow fastapi uvicorn
```

### Generate Sample Data
```bash
python src/generate_data.py              # 3M rows → events.csv
python src/generate_data.py 1000000      # Custom row count
```

### Build Analytics Report
```bash
python src/pipeline.py events.csv                          # → report.json
python src/pipeline.py events.csv --out custom.json        # Custom output
python src/pipeline.py events.csv --chunksize 500000       # Chunked load
```

### Run API Server
```bash
uvicorn src/api:app --reload
# Open http://127.0.0.1:8000/docs for interactive API docs

# Or CLI:
python src/api.py --host 0.0.0.0 --port 8000
```

### Run Tests
```bash
pytest tests/test_pipeline.py -v
# Or without pytest:
python tests/test_pipeline.py
```

## Architecture

### Data Flow
```
events.csv → [Load] → [Clean D1–D5] → [Metrics] → report.json
                            ↓
                    [6 aggregations]
                  DAU, Funnel, Revenue,
                Top Countries, Anomalies, D1 Retention
```

### Core Modules

**`generate_data.py`** — Synthetic data generator
- 3M rows of realistic event data with configurable dirty injection
- All vectorised (no Python loops) using NumPy
- Enforces refund→purchase dependency invariant
- Memory-efficient: ~93 MB for 3M rows with dtype discipline

**`pipeline.py`** — ETL & analytics engine
- Modular cleaning rules (D1–D5) with detailed documentation
- 6 metrics computed efficiently with grouped aggregations
- Automatic cache invalidation & resumption
- Produces structured JSON reports

**`api.py`** — FastAPI HTTP layer
- In-memory cache with mtime-based invalidation
- Thread pool for CPU-bound report building
- Two-level asyncio locking to prevent stampede
- Path security: traversal prevention, extension validation

**`test_pipeline.py`** — Comprehensive test suite
- 50+ unit tests covering all cleaning rules & metrics
- Minimal test data factories (no heavy fixtures)
- Runs standalone with `unittest` (zero extra dependencies)

## Design Decisions

### Cleaning Rules (D1–D5)

| Rule | Issue | Solution |
|------|-------|----------|
| **D1** | Invalid/far-future timestamps | Drop year ≥ 2099 sentinel rows |
| **D2** | Null/lowercase countries | Uppercase & replace NaN with "UNK" |
| **D3** | Junk event types | Drop "???" rows; keep only canonical 4 types |
| **D4** | Duplicate event IDs | Deduplicate on event_id, keep first occurrence |
| **D5** | Amount noise on non-revenue events | Zero amounts for page_view/signup |

All vectorised with boolean masks — no Python row loops.

### Performance Optimisations

**Vectorisation**
- All data transformations use NumPy/Pandas operations
- `np.searchsorted` for event-type bucketing (~O(N log k))
- Boolean fancy indexing for conditional updates

**Memory Efficiency**
- `int32` for user_id, session_id (4 B vs 8 B)
- `float32` for amounts (sufficient for currency)
- `pd.Categorical` for low-cardinality strings (~1 B/row vs 50 B in Pythn string)
- Result: ~200 MB for 3M rows (vs ~1.8 GB naive approach)

**I/O**
- Optional chunked CSV reads for files > RAM
- PyArrow CSV writer (2–3× faster than pandas)
- Lazy metric computation: only built when needed

### API Caching Strategy

**Two-level locking** prevents thundering herd:
1. Global `_meta_lock` protects the per-file lock dictionary
2. Per-file `asyncio.Lock` serialises builds for the same file
3. On cache hit, requests bypass locks entirely

**Invariants**
- mtime-based invalidation (no manual TTL)
- Different chunksize values hit the same cache
- Unique per (file_path, mtime) pair

**Thread safety**
- `build_report` runs in `ThreadPoolExecutor` (CPU-bound, ~16 s)
- Event loop stays responsive for other requests
- Single-worker deployment (process-local cache)

### Refund→Purchase Invariant

Critical design constraint: every refund must link to an existing purchase with matching amount and user.
Refund linkage is now explicitly split into:

1. Phase 1 — Planning: `_build_refund_links()`
Pure function. Determines which refund maps to which purchase.

2. Phase 2 — Application: `_apply_refund_links()`
Enforces all referential constraints in one place.


## Metrics

### 1. Daily Active Users (DAU)
Unique users per calendar day, sorted chronologically.

### 2. Conversion Funnel
Daily page_view → signup → purchase counts with conversion rates:
- pv_to_signup = signup / page_view
- signup_to_purchase = purchase / signup

### 3. Daily Net Revenue
Sum of purchase amounts + refund amounts (refunds are negative).
Used as input for anomaly detection.

### 4. Top 10 Countries by Revenue
Country-level net revenue ranking. Includes "UNK" (unknown origin).

### 5. Anomaly Detection
Z-score anomalies: days where net_revenue > mean + 3σ.
Uses population statistics (ddof=0) for dataset-level outliers.

### 6. D1 Retention
Fraction of users who return the day after their first-seen date.
Excludes final date in dataset (no D+1 window).

## Configuration

### Environment Variables (API)
```bash
DATA_DIR=./data          # Base directory for CSV files
MAX_WORKERS=2            # Thread pool size
LOG_LEVEL=INFO           # Verbosity (DEBUG, INFO, WARNING, ERROR)
```

### Logging
All components log to stdout with ISO8601 timestamps:
```
2024-01-15 14:23:45  INFO     Loaded 3000000 raw rows.
2024-01-15 14:23:47  INFO     Clean: 2987451 valid, 12549 dropped
```

## Testing Philosophy

- **Minimal fixtures** — Test factories generate only required columns
- **Isolation** — Each test is independent; no shared state
- **Behaviour-driven** — Names like `test_d1_year_2099_sentinel_dropped` document rules
- **Zero dependencies** — Runs with Python stdlib alone

Example:
```python
def test_d1_year_2099_sentinel_dropped(self):
    df = make_raw_df([{"ts": "2099-01-01 00:00:00+00:00"}])
    clean, counts = _clean(df)
    self.assertEqual(len(clean), 0)
    self.assertEqual(counts["breakdown"]["invalid_ts"], 1)
```

## Monitoring

### API Endpoints

**GET `/health`**
```json
{
  "status": "ok",
  "uptime_s": 42.3,
  "cache": {"entries": 2, "files": [...]}
}
```

**GET `/report?file=events.csv&chunksize=500000`**
```json
{
  "range": {"start": "2023-01-01", "end": "2024-12-31"},
  "counts": {"raw_rows": 3000000, "valid_rows": 2987451, ...},
  "dau": [...],
  "funnel": [...],
  "_meta": {"cache": "HIT", "built_at": "2024-01-15T14:23:48", ...}
}
```

Response headers:
- `X-Cache: HIT | MISS`
- `X-Response-Time: 0.0234s`

## Common Tasks

### Process a Large File
```bash
# Chunked read (500k rows at a time) to cap peak RAM:
python src/pipeline.py bigfile.csv --chunksize 500000 --out report.json
```

### Reload Data
```bash
# Cached reports auto-invalidate on file mtime change:
touch events.csv        # Updates mtime
curl http://localhost:8000/report?file=events.csv  # Cache MISS, rebuilds
```

### Debug Cleaning
```bash
python src/pipeline.py events.csv --log-level DEBUG
# See per-rule dropout counts and dtype conversions
```

## License & Attribution

Production data engineering template. All code is documented with in-line design decisions and rationales for learning purposes.
