"""External enrichment for a single star: Gaia DR3 astrophysical parameters
(FLAME mass/age/radius/luminosity), SIMBAD (identifiers, spectral type) and
Wikidata (Chinese label and description).

Every call is cached in SQLite, has a timeout, and degrades to "unavailable"
without affecting the core experience. With STM_OFFLINE=1 only the cache
is used.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time

import httpx

from .. import settings

GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
WIKIDATA = "https://query.wikidata.org/sparql"
UA = {"User-Agent": "star-time-machine/0.1 (student competition project; contact via project page)"}

_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    settings.CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(settings.CACHE_DB, check_same_thread=False)
    con.execute("CREATE TABLE IF NOT EXISTS enrich (key TEXT PRIMARY KEY, value TEXT, fetched REAL)")
    return con


_con = _db()


def cache_get(key: str) -> dict | None:
    with _lock:
        row = _con.execute("SELECT value FROM enrich WHERE key = ?", (key,)).fetchone()
    return None if row is None else json.loads(row[0])


def cache_put(key: str, value: dict) -> None:
    with _lock:
        _con.execute("INSERT OR REPLACE INTO enrich VALUES (?, ?, ?)", (key, json.dumps(value, ensure_ascii=False), time.time()))
        _con.commit()


async def _tap(client: httpx.AsyncClient, url: str, adql: str, timeout: float) -> list[dict]:
    r = await client.post(url, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql}, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    cols = [c["name"] for c in j["metadata"]]
    return [dict(zip(cols, row)) for row in j["data"]]


async def gaia_ap(client: httpx.AsyncClient, sid: int) -> dict:
    rows = await _tap(client, GAIA_TAP, f"""SELECT mass_flame, mass_flame_lower, mass_flame_upper, age_flame, age_flame_lower,
        age_flame_upper, radius_flame, lum_flame, teff_gspspec, spectraltype_esphs, evolstage_flame
        FROM gaiadr3.astrophysical_parameters WHERE source_id = {int(sid)}""", 60)
    return rows[0] if rows else {}


async def simbad(client: httpx.AsyncClient, sid: int | None, hip: int | None) -> dict:
    ident = f"Gaia DR3 {sid}" if sid and sid > 0 else f"HIP {hip}"
    rows = await _tap(client, SIMBAD_TAP, f"""SELECT b.main_id, b.sp_type, b.otype, b.ra, b.dec
        FROM basic AS b JOIN ident AS i ON i.oidref = b.oid WHERE i.id = '{ident}'""", 15)
    if not rows:
        return {}
    out = rows[0]
    ids = await _tap(client, SIMBAD_TAP, f"""SELECT i2.id FROM ident AS i JOIN ident AS i2 ON i.oidref = i2.oidref
        WHERE i.id = '{ident}'""", 15)
    out["ids"] = [r["id"] for r in ids][:30]
    return out


async def wikidata(client: httpx.AsyncClient, simbad_main_id: str | None) -> dict:
    """Chinese label/description via the SIMBAD ID property (P3083)."""
    if not simbad_main_id:
        return {}
    mid = simbad_main_id.replace('"', "")
    q = f"""SELECT ?item ?zh ?zhdesc ?en WHERE {{
      ?item wdt:P3083 "{mid}" .
      OPTIONAL {{ ?item rdfs:label ?zh FILTER(lang(?zh) = "zh-hans" || lang(?zh) = "zh") }}
      OPTIONAL {{ ?item schema:description ?zhdesc FILTER(lang(?zhdesc) = "zh-hans" || lang(?zhdesc) = "zh") }}
      OPTIONAL {{ ?item rdfs:label ?en FILTER(lang(?en) = "en") }}
    }} LIMIT 1"""
    r = await client.get(WIKIDATA, params={"query": q, "format": "json"}, headers={**UA, "Accept": "application/sparql-results+json"}, timeout=12)
    r.raise_for_status()
    b = r.json()["results"]["bindings"]
    if not b:
        return {}
    return {k: v["value"] for k, v in b[0].items()}


async def enrich(sid: int | None, hip: int | None) -> dict:
    key = f"v1:{sid}:{hip}"
    hit = cache_get(key)
    if hit is not None:
        return {**hit, "cached": True}
    if settings.OFFLINE:
        return {"offline": True}
    out: dict = {"sources": {}}
    complete = True

    def record(name, res):
        nonlocal complete
        if isinstance(res, Exception):
            out["sources"][name] = {"ok": False, "error": type(res).__name__}
            complete = False
        else:
            out[name] = res
            out["sources"][name] = {"ok": True}

    async with httpx.AsyncClient(headers=UA) as client:
        async def simbad_then_wikidata():
            sb = await simbad(client, sid, hip)
            record("simbad", sb)
            try:
                record("wikidata", await wikidata(client, sb.get("main_id")))
            except Exception as e:  # noqa: BLE001
                record("wikidata", e)

        async def ap():
            try:
                record("gaia_ap", await gaia_ap(client, sid))
            except Exception as e:  # noqa: BLE001
                record("gaia_ap", e)

        jobs = [simbad_then_wikidata()]
        if sid and sid > 0:
            jobs.append(ap())
        res = await asyncio.gather(*jobs, return_exceptions=True)
        for r in res:
            if isinstance(r, Exception):
                record("simbad", r)
    out["fetched_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if complete:
        cache_put(key, out)
    return out
