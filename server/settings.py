"""Runtime settings from environment variables (no secrets in the repo)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("STM_DATA", ROOT / "data" / "build"))
OCTREE = DATA / "octree"
PARQUET = DATA / "stars.parquet"
CACHE_DB = Path(os.environ.get("STM_CACHE_DB", DATA / "cache.sqlite"))
# sub-path under which the app is mounted behind the reverse proxy, e.g. "/stm"
ROOT_PATH = os.environ.get("STM_ROOT_PATH", "")
DUCKDB_MEMORY = os.environ.get("STM_DUCKDB_MEMORY", "1GB")
DUCKDB_THREADS = int(os.environ.get("STM_DUCKDB_THREADS", str(min(4, os.cpu_count() or 2))))
OFFLINE = os.environ.get("STM_OFFLINE", "0") == "1"  # disable external enrichment
WEB_DIST = Path(os.environ.get("STM_WEB_DIST", ROOT / "web" / "dist"))
ORBIT_WORKERS = int(os.environ.get("STM_ORBIT_WORKERS", "2"))
ORBIT_TIMEOUT = float(os.environ.get("STM_ORBIT_TIMEOUT", "20"))
RATE_PER_MIN = int(os.environ.get("STM_RATE_PER_MIN", "120"))  # expensive endpoints, per client IP
WARM_STATS = os.environ.get("STM_WARM_STATS", "1") == "1"
# in-process LRU caches (entries): filtered octree nodes, 2-D histograms
STATS_THREADS = int(os.environ.get("STM_STATS_THREADS", str(min(4, os.cpu_count() or 2))))
HEAVY_QUERIES = int(os.environ.get("STM_HEAVY_QUERIES", "2"))
NODE_CACHE = int(os.environ.get("STM_NODE_CACHE", "1024"))
STATS_CACHE = int(os.environ.get("STM_STATS_CACHE", "512"))
