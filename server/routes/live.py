"""Share links, the presenter WebSocket, health and metrics."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import string
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse

from .. import settings

router = APIRouter()
_lock = threading.Lock()
_db = sqlite3.connect(settings.CACHE_DB, check_same_thread=False)
_db.execute("CREATE TABLE IF NOT EXISTS shares (id TEXT PRIMARY KEY, state TEXT, created REAL)")
_db.commit()
ALPHABET = string.ascii_letters + string.digits
STARTED = time.time()

# ---------------------------------------------------------------- metrics
LAT: dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))
COUNT: dict[str, int] = defaultdict(int)


def route_group(path: str) -> str:
    parts = path.strip("/").split("/")
    return "/".join(parts[:2]) if parts and parts[0] == "api" else (parts[0] if parts else "/")


def record(path: str, ms: float) -> None:
    g = route_group(path)
    LAT[g].append(ms)
    COUNT[g] += 1


def _rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def _children(pid: int) -> list[int]:
    out = []
    try:
        for p in Path("/proc").iterdir():
            if p.name.isdigit():
                try:
                    stat = (p / "stat").read_text()
                    ppid = int(stat.rsplit(")", 1)[1].split()[1])
                    if ppid == pid:
                        out.append(int(p.name))
                except OSError:
                    continue
    except OSError:
        pass
    return out


@router.get("/api/metrics")
def metrics():
    pid = os.getpid()
    kids = _children(pid)
    lat = {}
    for g, d in LAT.items():
        a = sorted(d)
        if a:
            lat[g] = {"n": COUNT[g], "p50_ms": round(a[len(a) // 2], 1), "p95_ms": round(a[int(len(a) * 0.95) - 1 if len(a) > 1 else 0], 1)}
    return {
        "uptime_s": round(time.time() - STARTED),
        "rss_mb": {"app": round(_rss_mb(pid), 1), "children": round(sum(_rss_mb(k) for k in kids), 1)},
        "latency": lat,
        "ws_rooms": {k: len(v) for k, v in ROOMS.items()},
    }


# ---------------------------------------------------------------- share
@router.post("/api/share")
async def share(request: Request):
    body = await request.body()
    if len(body) > 8192:
        raise HTTPException(413, "state too large")
    try:
        state = json.loads(body)
    except json.JSONDecodeError as e:
        raise HTTPException(400, "invalid JSON") from e
    sid = "".join(secrets.choice(ALPHABET) for _ in range(7))
    with _lock:
        _db.execute("INSERT INTO shares VALUES (?, ?, ?)", (sid, json.dumps(state, ensure_ascii=False), time.time()))
        _db.commit()
    return {"id": sid, "path": f"/s/{sid}"}


@router.get("/api/share/{sid}")
def share_get(sid: str):
    with _lock:
        row = _db.execute("SELECT state, created FROM shares WHERE id = ?", (sid,)).fetchone()
    if row is None:
        raise HTTPException(404, "no such share")
    return {"id": sid, "state": json.loads(row[0]), "created": row[1]}


@router.get("/s/{sid}")
def share_redirect(sid: str, request: Request):
    root = request.scope.get("root_path", "")
    return RedirectResponse(f"{root}/?v={sid}")


# ---------------------------------------------------------------- presenter
ROOMS: dict[str, set[WebSocket]] = defaultdict(set)


@router.websocket("/ws/session/{room}")
async def session(ws: WebSocket, room: str):
    if len(room) > 32:
        await ws.close(code=1008)
        return
    await ws.accept()
    ROOMS[room].add(ws)
    try:
        while True:
            msg = await ws.receive_text()
            if len(msg) > 16384:
                continue
            dead = []
            for peer in list(ROOMS[room]):
                if peer is ws:
                    continue
                try:
                    await peer.send_text(msg)
                except Exception:  # noqa: BLE001
                    dead.append(peer)
            for d in dead:
                ROOMS[room].discard(d)
    except WebSocketDisconnect:
        pass
    finally:
        ROOMS[room].discard(ws)
        if not ROOMS[room]:
            ROOMS.pop(room, None)


@router.get("/api/share/{sid}/qr.svg")
def share_qr(sid: str, request: Request):
    """QR code of the short link (segno, BSD-3)."""
    import io

    import segno
    from fastapi.responses import Response

    with _lock:
        row = _db.execute("SELECT 1 FROM shares WHERE id = ?", (sid,)).fetchone()
    if row is None:
        raise HTTPException(404, "no such share")
    base = str(request.base_url).rstrip("/")
    root = request.scope.get("root_path", "")
    url = f"{base}{root}/s/{sid}" if root and not base.endswith(root) else f"{base}/s/{sid}"
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="svg", scale=6, border=2, dark="#0b0d17", light="#eef3ff", xmldecl=False)
    return Response(buf.getvalue(), media_type="image/svg+xml")


@router.get("/api/share/{sid}/card.png")
async def share_card(sid: str, request: Request, style: str | None = None):
    """Server-rendered postcard of a shared view (今 or 古)."""
    import asyncio
    import io

    import segno
    from fastapi.responses import Response

    from ..services import postcard

    with _lock:
        row = _db.execute("SELECT state FROM shares WHERE id = ?", (sid,)).fetchone()
    if row is None:
        raise HTTPException(404, "no such share")
    state = json.loads(row[0])
    if style in ("obs", "atlas"):
        state["style"] = style
    base = str(request.base_url).rstrip("/")
    url = f"{base}/s/{sid}"
    q = io.BytesIO()
    dark = state.get("style") != "atlas"
    segno.make(url, error="m").save(q, kind="png", scale=8, border=2,
                                    dark="#0b0d17" if not dark else "#eef3ff", light="#ece2cc" if not dark else "#06070e")
    png = await asyncio.to_thread(postcard.render, state, url, q.getvalue())
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
