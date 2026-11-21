"""Whole-catalogue histograms and counts for the analysis panel.

The column store (pipeline/build_stats.py, memory-mapped .npy files) is
scanned by one fused, parallel numba kernel: filter, bin, count. It gives
exactly the same numbers as the DuckDB path in services/db.py (both compare
and bin in float64 of the stored values; tests/server/test_stats.py checks
them against each other), in tens of milliseconds instead of a full Parquet
scan. Queries that touch a field outside the store fall back to DuckDB.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from .. import settings

os.environ.setdefault("NUMBA_CACHE_DIR", str(settings.DATA / "numba_cache"))
import numba as nb  # noqa: E402

# kernel slot order; "uw" (|(U, W)|, the Toomre ordinate) is computed in-kernel
FAST = ("dist", "bp_rp", "absg", "vtot", "u", "v", "w", "uw")
STORED = FAST[:-1]
SLOT = {f: i for i, f in enumerate(FAST)}


@nb.njit(inline="always")
def _get(k, i, dist, bprp, absg, vtot, u, v, w):
    if k == 0:
        return dist[i]
    if k == 1:
        return np.float64(bprp[i])
    if k == 2:
        return np.float64(absg[i])
    if k == 3:
        return np.float64(vtot[i])
    if k == 4:
        return np.float64(u[i])
    if k == 5:
        return np.float64(v[i])
    if k == 6:
        return np.float64(w[i])
    uu = np.float64(u[i])
    ww = np.float64(w[i])
    return np.sqrt(uu * uu + ww * ww)


@nb.njit(parallel=True, nogil=True, cache=True)
def _scan(dist, bprp, absg, vtot, u, v, w, act, lo, hi, xi, yi, x0, dx, y0, dy, xlo, xhi, ylo, yhi, nx, ny, nchunk):
    """act: slots with a filter (lo/hi aligned with it). xi < 0: count only.
    Only the columns a query uses are read. NaN (NULL) fails every comparison,
    as in SQL; bin = floor((x - x0) / dx) clipped to the grid, as in
    db.Catalog.hist2d_sql."""
    n = dist.shape[0]
    cells = nx * ny if xi >= 0 else 1
    out = np.zeros((nchunk, cells), np.int64)
    step = (n + nchunk - 1) // nchunk
    na = act.shape[0]
    for c in nb.prange(nchunk):
        for i in range(c * step, min(n, (c + 1) * step)):
            ok = True
            for j in range(na):
                x = _get(act[j], i, dist, bprp, absg, vtot, u, v, w)
                if not (x >= lo[j] and x <= hi[j]):
                    ok = False
                    break
            if not ok:
                continue
            if xi < 0:
                out[c, 0] += 1
                continue
            xv = _get(xi, i, dist, bprp, absg, vtot, u, v, w)
            if not (xv >= xlo and xv <= xhi):
                continue
            yv = _get(yi, i, dist, bprp, absg, vtot, u, v, w)
            if not (yv >= ylo and yv <= yhi):
                continue
            ix = min(max(int(np.floor((xv - x0) / dx)), 0), nx - 1)
            iy = min(max(int(np.floor((yv - y0) / dy)), 0), ny - 1)
            out[c, iy * nx + ix] += 1
    return out.sum(axis=0)


class Store:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or settings.DATA / "stats")
        self._cols: dict[str, np.ndarray] | None = None
        self._lock = threading.Lock()  # the parallel kernel is not re-entrant
        self._init = threading.Lock()

    @property
    def available(self) -> bool:
        return (self.path / "meta.json").exists()

    def cols(self) -> dict[str, np.ndarray]:
        with self._init:
            if self._cols is None:
                meta = json.loads((self.path / "meta.json").read_text())
                cols = {f: np.load(self.path / f"{f}.npy", mmap_mode="r") for f in STORED}
                if any(a.shape[0] != meta["rows"] for a in cols.values()):
                    raise RuntimeError("stats store is inconsistent; rerun pipeline/build_stats.py")
                nb.set_num_threads(max(1, min(settings.STATS_THREADS, nb.config.NUMBA_NUM_THREADS)))
                self._cols = cols
        return self._cols

    @staticmethod
    def covers(fields) -> bool:
        return all(f in SLOT for f in fields)

    def _run(self, ranges, xi=-1, yi=-1, grid=(0.0, 1.0, 0.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1, 1)) -> np.ndarray:
        c = self.cols()
        bounds: dict[int, list[float]] = {}
        for r in ranges:
            b = bounds.setdefault(SLOT[r.field], [-np.inf, np.inf])
            # several terms on one field intersect, as the SQL AND does
            if r.lo is not None:
                b[0] = max(b[0], float(r.lo))
            if r.hi is not None:
                b[1] = min(b[1], float(r.hi))
        act = np.array(list(bounds), np.int64)
        lo = np.array([b[0] for b in bounds.values()], np.float64)
        hi = np.array([b[1] for b in bounds.values()], np.float64)
        x0, dx, y0, dy, xlo, xhi, ylo, yhi, nx, ny = grid
        with self._lock:
            return _scan(c["dist"], c["bp_rp"], c["absg"], c["vtot"], c["u"], c["v"], c["w"], act, lo, hi,
                         xi, yi, x0, dx, y0, dy, xlo, xhi, ylo, yhi, nx, ny, 64)

    def count(self, ranges) -> int:
        return int(self._run(ranges)[0])

    def hist2d(self, xf: str, yf: str, xr, yr, bins, ranges) -> dict:
        nx, ny = bins
        dx = (xr[1] - xr[0]) / nx
        dy = (yr[1] - yr[0]) / ny
        counts = self._run(ranges, SLOT[xf], SLOT[yf], (float(xr[0]), dx, float(yr[0]), dy, float(xr[0]), float(xr[1]),
                                                        float(yr[0]), float(yr[1]), int(nx), int(ny)))
        return {"x": xf, "y": yf, "xr": xr, "yr": yr, "nx": nx, "ny": ny, "counts": counts.tolist(), "total": int(counts.sum())}


store = Store()
