"""Claims, orbits and stellar encounters.

Orbits are integrated with galpy (MWPotential2014, the same rescaling as
stm/galaxy.py) in a small process pool so a slow request never blocks the
event loop and memory stays bounded.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
import multiprocessing as mp
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from .. import settings
from ..services.db import catalog

router = APIRouter()
_pool: cf.ProcessPoolExecutor | None = None


def pool() -> cf.ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = cf.ProcessPoolExecutor(max_workers=settings.ORBIT_WORKERS, mp_context=mp.get_context("spawn"))
    return _pool


def _orbit_job(r0: list[float], v: list[float], span: float, n: int) -> dict:
    """Runs in a worker process: galpy orbit of a star and of the Sun."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from stm import galaxy as gx

    ts = np.linspace(-span, span, n)
    rel = gx.galpy_relative(np.array([r0]), np.array([v]), ts)[0]
    m, r_sun, _ = gx.galcen_affine()
    # absolute galactocentric track of the star (for the galaxy view)
    R, phi, z = gx.orbit_galpy(np.array([r0]), np.array([v]), ts)
    xyz = np.stack([R[0] * np.cos(phi[0]), -R[0] * np.sin(phi[0]), z[0]], axis=-1)
    i0 = int(np.argmin(np.abs(ts)))
    d = np.linalg.norm(rel, axis=-1)
    k = int(np.argmin(d))
    return {
        "t_myr": ts.round(5).tolist(),
        "rel_icrs_pc": rel.round(4).tolist(),
        "galcen_pc": xyz.round(2).tolist(),
        "now_index": i0,
        "closest": {"t_myr": float(ts[k]), "d_pc": float(d[k])},
        "R_range_pc": [float(R[0].min()), float(R[0].max())],
        "z_max_pc": float(np.abs(z[0]).max()),
        "model": "galpy MWPotential2014, R0 = 8.122 kpc, v_c(R0) = 233.1 km/s, dop853_c",
    }


@router.get("/api/claims")
def claims():
    p = settings.ROOT / "web" / "public" / "data" / "claims.json"
    if not p.exists():
        raise HTTPException(503, "claims not computed")
    return FileResponse(p, media_type="application/json", headers={"Cache-Control": "public, max-age=300"})


@router.get("/api/orbit")
async def orbit(node: int | None = None, rank: int | None = None, sid: int | None = None,
                span: float = Query(10.0, gt=0, le=50), n: int = Query(401, ge=11, le=2001)):
    row = None
    if sid is not None:
        row = await asyncio.to_thread(catalog.star_by_sid, sid)
    elif node is not None and rank is not None:
        row = await asyncio.to_thread(catalog.star, node, rank)
    if row is None:
        raise HTTPException(404, "star not found")
    r0 = [row["x"], row["y"], row["z"]]
    v = [row["vx"], row["vy"], row["vz"]]
    loop = asyncio.get_running_loop()
    try:
        res = await asyncio.wait_for(loop.run_in_executor(pool(), _orbit_job, r0, v, span, n), timeout=settings.ORBIT_TIMEOUT)
    except TimeoutError as e:
        raise HTTPException(504, "orbit integration timed out") from e
    res["sid"] = str(int(row["sid"]))  # 64-bit: a string for JavaScript
    return res


@lru_cache(maxsize=1)
def _encounters() -> dict:
    p = settings.DATA / "encounters.json"
    if not p.exists():
        return {"encounters": [], "note": "not computed yet"}
    # NaN (no RUWE / distance interval for Hipparcos-only stars) is not JSON;
    # Gaia source_ids are 64-bit and must reach JavaScript as strings
    d = json.loads(p.read_text(), parse_constant=lambda c: None)
    for key in ("encounters", "nearest_succession"):
        for x in d.get(key, []):
            x["sid"] = str(x["sid"])
    return d


@router.get("/api/encounters")
def encounters(max_pc: float = Query(1.0, gt=0, le=5)):
    e = _encounters()
    return {**e, "encounters": [x for x in e.get("encounters", []) if x["d_pc"] <= max_pc]}
