"""
api.py
======
FastAPI layer that exposes pipeline.build_report over HTTP.

Endpoints
---------
    GET /health
        Service liveness: uptime, Python version, cache summary.

    GET /report?file=<path>
        Full analytics report for the given CSV file.
        Responses are cached in-memory; cache is auto-invalidated when the
        file's mtime changes.

        Optional query params:
            chunksize=<int>  – forwarded to build_report (default: None)

Cache design
------------
Two-level locking prevents the "thundering herd" / cache stampede problem:

    ┌─────────────────────────────────────────────────┐
    │  _meta_lock  (one global asyncio.Lock)          │
    │    protects _per_file_locks dict itself          │
    └──────────┬──────────────────────────────────────┘
               │ get or create
    ┌──────────▼──────────────────────────────────────┐
    │  _per_file_locks[path]  (one asyncio.Lock/file) │
    │    held while build_report() is running          │
    │    second request waits here, then hits cache    │
    └─────────────────────────────────────────────────┘

If two requests for the same uncached file arrive simultaneously, the first
acquires the per-file lock, builds the report, and stores it.  The second
waits, then on wakeup finds the cache warm and returns immediately — no
double-build.

CPU / event-loop safety
-----------------------
build_report is CPU and IO bound (~16 s for 3M rows).  It must not run on
the event loop.  All calls go through asyncio.get_event_loop().run_in_executor
with a bounded ThreadPoolExecutor (MAX_WORKERS, default 2).

Path security
-------------
The `file` parameter is resolved relative to DATA_DIR (env var, default ".").
Absolute paths and traversal sequences (../../) are rejected with 403.
Only .csv files are accepted.

Environment variables
---------------------
    DATA_DIR     Base directory for CSV files          [.]
    MAX_WORKERS  Thread pool size for build_report     [2]
    LOG_LEVEL    Logging verbosity                     [INFO]

Usage
-----
    uvicorn api:app --reload
    DATA_DIR=/data uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

    # Or:
    python api.py [--host 0.0.0.0] [--port 8000]

Note: use --workers 1 with uvicorn; the in-memory cache is process-local and
would be ineffective (and inconsistent) with multiple workers.  For
multi-process deployments, replace the cache with Redis.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Our pipeline — lives alongside this file.
from pipeline import build_report

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR: Path = Path(os.getenv("DATA_DIR", ".")).resolve()
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "2"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# [DESIGN] We configure a named logger rather than calling basicConfig so we
# don't interfere with uvicorn's own log setup.  pipeline.py calls
# basicConfig at import time, but that's a no-op once uvicorn has already
# installed its handlers.

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")

# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """
    One cached report.

    file_mtime is stored so the cache auto-invalidates when the source file
    is modified — no explicit TTL needed.
    """
    report:           dict
    file_path:        str          # resolved absolute path (for display)
    file_mtime:       float        # os.stat().st_mtime at build time
    built_at:         datetime
    build_duration_s: float


class _CacheState:
    """
    Plain class (not dataclass) so asyncio.Lock fields are created cleanly
    per-instance without dataclass field-type inspection touching sys.modules.
    """
    def __init__(self) -> None:
        self.entries:        dict[str, "CacheEntry"]   = {}
        self.per_file_locks: dict[str, asyncio.Lock]   = {}
        self.meta_lock:      asyncio.Lock               = asyncio.Lock()


_cache = _CacheState()


async def _get_or_create_file_lock(resolved_path: str) -> asyncio.Lock:
    """Return the per-file lock, creating it if it doesn't exist yet."""
    async with _cache.meta_lock:
        if resolved_path not in _cache.per_file_locks:
            _cache.per_file_locks[resolved_path] = asyncio.Lock()
        return _cache.per_file_locks[resolved_path]


async def _get_cached(resolved_path: str, mtime: float) -> Optional[CacheEntry]:
    """Return a cache hit only if the mtime still matches."""
    entry = _cache.entries.get(resolved_path)
    if entry is not None and entry.file_mtime == mtime:
        return entry
    return None


def _store_cache(resolved_path: str, entry: CacheEntry) -> None:
    _cache.entries[resolved_path] = entry


# ---------------------------------------------------------------------------
# Path security
# ---------------------------------------------------------------------------

def _resolve_and_validate(file_param: str) -> Path:
    """
    Resolve the user-supplied `file` parameter to a safe absolute path.

    Rules
    -----
    1. Path is joined to DATA_DIR and resolved (symlinks followed, .. collapsed).
    2. The resolved path must be inside DATA_DIR — rejects traversal attacks.
    3. Must exist and be a regular file.
    4. Must have a .csv extension.

    Raises HTTPException (403 / 404 / 400) on any violation.
    """
    # [SECURITY] Never trust user input directly as a filesystem path.
    # Joining with DATA_DIR and resolving collapses all traversal sequences.
    try:
        resolved = (DATA_DIR / file_param).resolve()
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path: {exc}") from exc

    # [SECURITY] is_relative_to ensures the resolved path is strictly inside
    # DATA_DIR — even after symlink resolution.
    if not resolved.is_relative_to(DATA_DIR):
        log.warning("Path traversal attempt: %r → %s", file_param, resolved)
        raise HTTPException(
            status_code=403,
            detail="Path traversal is not allowed.",
        )

    if not resolved.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {file_param}",
        )

    if not resolved.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a regular file: {file_param}",
        )

    if resolved.suffix.lower() != ".csv":
        raise HTTPException(
            status_code=400,
            detail="Only .csv files are accepted.",
        )

    return resolved


# ---------------------------------------------------------------------------
# Thread pool (for running build_report off the event loop)
# ---------------------------------------------------------------------------

_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    """Return the module-level ThreadPoolExecutor (created at startup)."""
    if _executor is None:
        raise RuntimeError("Executor not initialised — app not started?")
    return _executor


async def _build_in_executor(path: str, chunksize: Optional[int]) -> dict:
    """
    Run build_report in the thread pool.

    [DESIGN] build_report blocks for ~16 s on large files (pandas + numpy).
    Calling it directly in an async handler would freeze the entire event
    loop.  run_in_executor schedules it on a background thread while the loop
    remains free to serve other requests.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _get_executor(),
        lambda: build_report(path, chunksize=chunksize),
    )


# ---------------------------------------------------------------------------
# Lifespan  (replaces deprecated on_startup / on_shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Create the thread pool on startup; shut it down cleanly on exit."""
    global _executor
    _executor = ThreadPoolExecutor(
        max_workers=MAX_WORKERS,
        thread_name_prefix="build_report",
    )
    log.info(
        "API starting  |  DATA_DIR=%s  MAX_WORKERS=%d  LOG_LEVEL=%s",
        DATA_DIR, MAX_WORKERS, LOG_LEVEL,
    )
    yield
    _executor.shutdown(wait=True)
    log.info("API shut down cleanly.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Events Analytics API",
    description=(
        "Wraps pipeline.build_report to expose pre-computed analytics reports "
        "over HTTP with in-memory caching and automatic cache invalidation."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)

_START_TIME: float = time.perf_counter()

# ---------------------------------------------------------------------------
# Middleware – response-time logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _log_response_time(request, call_next):
    """
    Log every request with method, path, status code, and wall-clock duration.

    [DESIGN] Middleware is the right place for cross-cutting concerns.
    Handler code stays clean; the timing wraps the complete request lifecycle
    including response streaming.

    Example log line:
        GET /report?file=events.csv  →  200  in 0.003s  (cache hit)
    """
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0

    cache_note = ""
    # Propagate a custom header set by the route handler.
    if "X-Cache" in response.headers:
        cache_note = f"  ({response.headers['X-Cache']})"

    log.info(
        "%s %s  →  %d  in %.3fs%s",
        request.method,
        request.url.path
        + (f"?{request.url.query}" if request.url.query else ""),
        response.status_code,
        elapsed,
        cache_note,
    )
    # [DESIGN] Expose timing to the caller — useful for client-side debugging
    # and SLO monitoring without needing access to server logs.
    response.headers["X-Response-Time"] = f"{elapsed:.4f}s"
    return response


# ---------------------------------------------------------------------------
# Response schemas  (Pydantic → OpenAPI docs)
# ---------------------------------------------------------------------------

class _DateRange(BaseModel):
    start: str
    end: str

class _Counts(BaseModel):
    raw_rows:     int
    valid_rows:   int
    dropped_rows: int
    breakdown:    dict[str, int]

class _DAUEntry(BaseModel):
    date: str
    dau:  int

class _FunnelEntry(BaseModel):
    date:               str
    pv:                 int
    signup:             int
    purchase:           int
    pv_to_signup:       float
    signup_to_purchase: float

class _RevenueEntry(BaseModel):
    date:        str
    net_revenue: float

class _CountryRevenue(BaseModel):
    country:     str
    net_revenue: float

class _AnomalyEntry(BaseModel):
    date:        str
    net_revenue: float
    z:           float

class _RetentionEntry(BaseModel):
    cohort_date: str
    users:       int
    retained:    int
    rate:        float

class ReportResponse(BaseModel):
    """Full analytics report returned by GET /report."""
    range:          _DateRange
    counts:         _Counts
    dau:            list[_DAUEntry]
    funnel:         list[_FunnelEntry]
    revenue_daily:  list[_RevenueEntry]
    top_countries:  list[_CountryRevenue]
    anomalies:      list[_AnomalyEntry]
    retention_d1:   list[_RetentionEntry]
    # Cache metadata — injected by the route handler, not by build_report.
    _meta:          Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    status:     str
    version:    str
    uptime_s:   float
    python:     str
    data_dir:   str
    cache:      dict[str, Any]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service liveness check",
    tags=["ops"],
)
async def health() -> JSONResponse:
    """
    Returns 200 while the service is running.

    Includes uptime, Python version, DATA_DIR, and a summary of the
    in-memory cache (number of entries and their file paths).

    [DESIGN] /health intentionally never calls build_report and never touches
    the filesystem — it must be fast and always-available for load-balancer
    health checks.
    """
    cached_files = [
        {
            "path":             e.file_path,
            "built_at":         e.built_at.isoformat(),
            "build_duration_s": round(e.build_duration_s, 2),
        }
        for e in _cache.entries.values()
    ]
    body = {
        "status":   "ok",
        "version":  app.version,
        "uptime_s": round(time.perf_counter() - _START_TIME, 2),
        "python":   sys.version.split()[0],
        "data_dir": str(DATA_DIR),
        "cache":    {
            "entries": len(cached_files),
            "files":   cached_files,
        },
    }
    return JSONResponse(content=body)


@app.get(
    "/report",
    response_model=ReportResponse,
    summary="Analytics report for a CSV file",
    tags=["analytics"],
    responses={
        200: {"description": "Report built successfully (or served from cache)"},
        400: {"description": "Invalid file path or extension"},
        403: {"description": "Path traversal attempt"},
        404: {"description": "File not found"},
        422: {"description": "File parsed but produced no valid rows"},
        500: {"description": "Unexpected error during report generation"},
    },
)
async def report(
    file: str = Query(
        ...,
        description=(
            "Path to a CSV file relative to DATA_DIR.  "
            "Example: events.csv  or  subdir/mydata.csv"
        ),
        example="events.csv",
    ),
    chunksize: Optional[int] = Query(
        default=None,
        ge=10_000,
        description=(
            "Read CSV in chunks of this many rows.  "
            "Reduces peak RAM at the cost of slightly higher wall-clock time.  "
            "Recommended for files > 1 GB."
        ),
        example=500_000,
    ),
) -> JSONResponse:
    """
    Build (or return a cached) analytics report for the given CSV file.

    Cache behaviour
    ---------------
    • First call: builds the report and caches it.
    • Subsequent calls with the same file: returns the cached report instantly,
      unless the file has been modified since the last build (mtime changed).
    • The response header **X-Cache** indicates HIT or MISS.
    • The response header **X-Response-Time** shows total handler time in seconds.

    The `chunksize` parameter is forwarded to `build_report`.  A different
    `chunksize` value does NOT bust the cache — the cached report was built
    from the same underlying file regardless of how it was read.
    """
    # ── Validate path ─────────────────────────────────────────────────────
    resolved: Path = _resolve_and_validate(file)
    path_key: str  = str(resolved)

    try:
        mtime: float = resolved.stat().st_mtime
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Cannot stat file: {exc}") from exc

    # ── Check cache (fast path, no lock needed for a pure read) ───────────
    cached = await _get_cached(path_key, mtime)
    if cached is not None:
        log.debug("Cache HIT: %s", path_key)
        body = dict(cached.report)
        body["_meta"] = {
            "cache":            "HIT",
            "built_at":         cached.built_at.isoformat(),
            "build_duration_s": round(cached.build_duration_s, 2),
        }
        return JSONResponse(content=body, headers={"X-Cache": "HIT"})

    # ── Cache miss — acquire per-file lock to prevent stampede ───────────
    # [DESIGN] Two concurrent requests for the same uncached file:
    #   Request A acquires the lock, builds the report, stores it.
    #   Request B waits on the lock.  On wakeup it checks the cache again
    #   (double-checked locking) and finds the warm entry — no second build.
    file_lock = await _get_or_create_file_lock(path_key)

    async with file_lock:
        # Double-check: another coroutine may have built it while we waited.
        cached = await _get_cached(path_key, mtime)
        if cached is not None:
            log.debug("Cache HIT (post-lock): %s", path_key)
            body = dict(cached.report)
            body["_meta"] = {
                "cache":            "HIT",
                "built_at":         cached.built_at.isoformat(),
                "build_duration_s": round(cached.build_duration_s, 2),
            }
            return JSONResponse(content=body, headers={"X-Cache": "HIT"})

        # Genuine miss — build the report in the thread pool.
        log.info("Cache MISS: building report for %s", path_key)
        t0 = time.perf_counter()

        try:
            result: dict = await _build_in_executor(path_key, chunksize)
        except FileNotFoundError:
            # Unlikely (we validated above) but possible if file deleted mid-request.
            raise HTTPException(status_code=404, detail=f"File not found: {file}")
        except ValueError as exc:
            # build_report raises ValueError when no valid rows remain.
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            log.exception("Unexpected error building report for %s", path_key)
            raise HTTPException(
                status_code=500,
                detail=f"Report generation failed: {type(exc).__name__}: {exc}",
            ) from exc

        duration = time.perf_counter() - t0
        entry = CacheEntry(
            report           = result,
            file_path        = path_key,
            file_mtime       = mtime,
            built_at         = datetime.now(timezone.utc),
            build_duration_s = duration,
        )
        _store_cache(path_key, entry)
        log.info("Report built and cached in %.2fs: %s", duration, path_key)

    body = dict(result)
    body["_meta"] = {
        "cache":            "MISS",
        "built_at":         entry.built_at.isoformat(),
        "build_duration_s": round(duration, 2),
    }
    return JSONResponse(content=body, headers={"X-Cache": "MISS"})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Events Analytics API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Hot-reload on code changes")
    args = parser.parse_args()

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        # [DESIGN] workers=1 is intentional — the cache is process-local.
        # Scale horizontally with a shared cache (Redis) if multi-process is needed.
        workers=1,
        log_level=LOG_LEVEL.lower(),
    )