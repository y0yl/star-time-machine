"""Data stream, single-star lookup and analysis statistics."""

from __future__ import annotations

import asyncio
import functools

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response

from stm.nodeformat import encode

from .. import settings
from ..services.db import FIELDS, catalog, parse_filter
from ..services.enrich import enrich

router = APIRouter()
KMS = 1.022712165045695e-06


def _filters(f: str | None):
    try:
        return parse_filter(f)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/api/octree/index.json")
def octree_index():
    p = settings.OCTREE / "index.json"
    if not p.exists():
        raise HTTPException(503, "octree not built")
    return FileResponse(p, media_type="application/json", headers={"Cache-Control": "public, max-age=300"})


@functools.lru_cache(maxsize=settings.NODE_CACHE)
def _filtered_node(node: int, f: str) -> bytes:
    r = catalog.node_rows(node, parse_filter(f))
    return encode(node, r["x"], r["y"], r["z"], r["vx"] / KMS, r["vy"] / KMS, r["vz"] / KMS,
                  r["absg"], r["bp_rp"], r["flags"], ranks=r["rank"])


@router.get("/api/nodes/{node_id}")
async def node(node_id: int, request: Request, f: str | None = Query(None, description="filter, e.g. dist:0:50,bp_rp:1.2:")):
    if f:
        _filters(f)
        data = await asyncio.to_thread(_filtered_node, node_id, f)
        return Response(data, media_type="application/octet-stream", headers={"Cache-Control": "public, max-age=3600"})
    base = settings.OCTREE / "nodes" / f"{node_id}.bin"
    br = base.with_suffix(".bin.br")
    if "br" in request.headers.get("accept-encoding", "") and br.exists():
        return Response(br.read_bytes(), media_type="application/octet-stream",
                        headers={"Content-Encoding": "br", "Cache-Control": "public, max-age=86400", "Vary": "Accept-Encoding"})
    if not base.exists():
        raise HTTPException(404, f"node {node_id} not found")
    return Response(base.read_bytes(), media_type="application/octet-stream", headers={"Cache-Control": "public, max-age=86400"})


def _clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if k == "sid" and v is not None:
            # Gaia source_id is a 64-bit integer: above 2^53 a JSON number is
            # rounded by JavaScript, so it travels as a string
            out[k] = str(int(v))
        elif isinstance(v, (np.floating, float)):
            out[k] = None if not np.isfinite(v) else float(v)
        elif v is None:
            out[k] = None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


@router.get("/api/stars/at")
async def star_at(node: int, rank: int):
    r = await asyncio.to_thread(catalog.star, node, rank)
    if r is None:
        raise HTTPException(404, "no such star")
    return _clean(r)


@router.get("/api/stars/sid/{sid}")
async def star_by_sid(sid: int):
    """One star by Gaia source_id (negative: -HIP for Hipparcos-only stars),
    e.g. to follow an encounter star whatever its octree node."""
    r = await asyncio.to_thread(catalog.star_by_sid, sid)
    if r is None:
        raise HTTPException(404, "no such star")
    return _clean(r)


@functools.lru_cache(maxsize=settings.STATS_CACHE)
def _hist2d_cached(x: str, y: str, rx: tuple, ry: tuple, nx: int, ny: int, f: str) -> dict:
    return catalog.hist2d(x, y, rx, ry, (nx, ny), parse_filter(f))


def warm_stats() -> None:
    """Precompute the charts the interface asks for first (unfiltered and presets)."""
    from ..services.db import PRESET_FILTERS

    for f in ["", *PRESET_FILTERS]:
        for (x, y, xr, yr) in (("bp_rp", "absg", (-0.5, 4.0), (-6.0, 16.0)), ("v", "uw", (-400.0, 100.0), (0.0, 400.0))):
            _hist2d_cached(x, y, xr, yr, 160, 120, f)


@router.get("/api/stats/hist2d")
async def hist2d(x: str = "bp_rp", y: str = "absg", xr: str = "-0.5,4.0", yr: str = "-6,16",
                 nx: int = Query(180, le=512), ny: int = Query(180, le=512), f: str | None = None):
    if x not in FIELDS or y not in FIELDS:
        raise HTTPException(400, "unknown field")
    rx = tuple(float(v) for v in xr.split(","))
    ry = tuple(float(v) for v in yr.split(","))
    _filters(f)
    return await asyncio.to_thread(_hist2d_cached, x, y, rx, ry, nx, ny, f or "")


@router.get("/api/stats/count")
async def count(f: str | None = None):
    return {"count": await asyncio.to_thread(catalog.count, _filters(f))}


@router.get("/api/stars/{sid}/enrich")
async def star_enrich(sid: int, hip: int | None = None):
    """External data for one star (Gaia AP, SIMBAD, Wikidata); cached."""
    gaia_sid = sid if sid > 0 else None
    return await enrich(gaia_sid, hip if hip else (-sid if 0 > sid > -1_000_000_000 else None))
