"""DuckDB over the 6-D Parquet catalogue (data/build/stars.parquet).

The Parquet file is sorted by (node, rank) in 65,536-row groups, so the row-group
min/max statistics let `WHERE node = ?` read one or two groups. Filters are whitelisted
numeric ranges, never raw SQL from the client.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import duckdb
import numpy as np

from .. import settings

def _d(col: str) -> str:
    return f"CAST({col} AS DOUBLE)"


# client-visible filter fields -> SQL expressions. Everything is compared and
# binned in float64 of the stored value (FLOAT columns are widened first), so
# the semantics do not depend on how DuckDB types the bound parameters, and
# services/stats.py reproduces them exactly.
FIELDS = {
    "dist": "dist",
    "bp_rp": _d("bp_rp"),
    "absg": _d("absg"),
    "g": _d("g"),
    "teff": _d("teff"),
    "vtot": _d("v_tot"),
    "u": _d("u_lsr"),
    "v": _d("v_lsr"),
    "w": _d("w_lsr"),
    "uw": f"sqrt({_d('u_lsr')}*{_d('u_lsr')} + {_d('w_lsr')}*{_d('w_lsr')})",
    "z": _d("gal_z"),
    "ruwe": _d("ruwe"),
}


# the analysis panel's preset filters (web/src/ui/panels.ts PRESETS)
PRESET_FILTERS = ["dist:0:15.33", "bp_rp::0.82", "bp_rp:0.9:2.5,absg:-3:2", "vtot:180:"]


@dataclass(frozen=True)
class Range:
    field: str
    lo: float | None
    hi: float | None


def parse_filter(q: str | None) -> list[Range]:
    """'dist:0:50,bp_rp:1.2:' -> [Range(...)]; unknown fields are rejected."""
    if not q:
        return []
    out = []
    for part in q.split(","):
        bits = part.split(":")
        if len(bits) != 3 or bits[0] not in FIELDS:
            raise ValueError(f"bad filter term {part!r}")
        lo = float(bits[1]) if bits[1] else None
        hi = float(bits[2]) if bits[2] else None
        out.append(Range(bits[0], lo, hi))
    return out


def where(ranges: list[Range]) -> tuple[str, list[float]]:
    conds, args = [], []
    for r in ranges:
        e = FIELDS[r.field]
        if r.lo is not None:
            conds.append(f"{e} >= ?")
            args.append(r.lo)
        if r.hi is not None:
            conds.append(f"{e} <= ?")
            args.append(r.hi)
    return (" AND ".join(conds) if conds else "TRUE"), args


class Catalog:
    """One DuckDB database instance (one memory limit, one buffer cache);
    each worker thread gets its own cursor on it."""

    def __init__(self) -> None:
        self._local = threading.local()
        self.path = str(settings.PARQUET)
        self._db: duckdb.DuckDBPyConnection | None = None
        self._lock = threading.Lock()
        # whole-table scans that the column store cannot answer: at most this
        # many at once, so node and star requests stay responsive
        self._heavy = threading.BoundedSemaphore(settings.HEAVY_QUERIES)

    @property
    def db(self) -> duckdb.DuckDBPyConnection:
        with self._lock:
            if self._db is None:
                self._db = duckdb.connect(config={"memory_limit": settings.DUCKDB_MEMORY, "threads": settings.DUCKDB_THREADS})
                self._db.execute(f"CREATE VIEW stars AS SELECT * FROM read_parquet('{self.path}')")
        return self._db

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        c = getattr(self._local, "con", None)
        if c is None:
            c = self.db.cursor()
            self._local.con = c
        return c

    def node_rows(self, node: int, ranges: list[Range]) -> dict[str, np.ndarray]:
        w, args = where(ranges)
        cols = "x, y, z, vx, vy, vz, absg, bp_rp, flags, rank"
        res = self.con.execute(f"SELECT {cols} FROM stars WHERE node = ? AND {w} ORDER BY rank", [node, *args]).fetchnumpy()
        return res

    def _one(self, sql: str, args: list) -> dict | None:
        cur = self.con.execute(sql, args)
        row = cur.fetchone()
        if row is None:
            return None
        return {d[0]: v for d, v in zip(cur.description, row)}

    def star(self, node: int, rank: int) -> dict | None:
        return self._one("SELECT * FROM stars WHERE node = ? AND rank = ?", [node, rank])

    def star_by_sid(self, sid: int) -> dict | None:
        return self._one("SELECT * FROM stars WHERE sid = ?", [sid])

    def hist2d(self, xf: str, yf: str, xr: tuple[float, float], yr: tuple[float, float], bins: tuple[int, int], ranges: list[Range]) -> dict:
        from .stats import store

        if store.available and store.covers([xf, yf, *(r.field for r in ranges)]):
            return store.hist2d(xf, yf, xr, yr, bins, ranges)
        with self._heavy:
            return self.hist2d_sql(xf, yf, xr, yr, bins, ranges)

    def count(self, ranges: list[Range]) -> int:
        from .stats import store

        if store.available and store.covers([r.field for r in ranges]):
            return store.count(ranges)
        with self._heavy:
            return self.count_sql(ranges)

    def hist2d_sql(self, xf: str, yf: str, xr: tuple[float, float], yr: tuple[float, float], bins: tuple[int, int], ranges: list[Range]) -> dict:
        xe, ye = FIELDS[xf], FIELDS[yf]
        w, args = where(ranges)
        nx, ny = bins
        q = f"""
        SELECT CAST(floor(({xe} - ?) / ? ) AS INTEGER) AS ix,
               CAST(floor(({ye} - ?) / ? ) AS INTEGER) AS iy,
               COUNT(*) AS n
        FROM stars
        WHERE {xe} BETWEEN ? AND ? AND {ye} BETWEEN ? AND ? AND {w}
        GROUP BY ix, iy"""
        dx = (xr[1] - xr[0]) / nx
        dy = (yr[1] - yr[0]) / ny
        res = self.con.execute(q, [xr[0], dx, yr[0], dy, xr[0], xr[1], yr[0], yr[1], *args]).fetchnumpy()
        grid = np.zeros((ny, nx), dtype=np.int64)
        ix = np.clip(res["ix"], 0, nx - 1)
        iy = np.clip(res["iy"], 0, ny - 1)
        np.add.at(grid, (iy, ix), res["n"])
        return {"x": xf, "y": yf, "xr": xr, "yr": yr, "nx": nx, "ny": ny, "counts": grid.ravel().tolist(), "total": int(grid.sum())}

    def count_sql(self, ranges: list[Range]) -> int:
        w, args = where(ranges)
        return int(self.con.execute(f"SELECT COUNT(*) FROM stars WHERE {w}", args).fetchone()[0])


catalog = Catalog()
