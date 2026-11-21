"""星空的时间机器 — application server (FastAPI).

Run (dev):  uvicorn server.app:app --port 8787 --reload
All routes live under /api; the built front end is served from / when
web/dist exists. Mounted under a sub-path via STM_ROOT_PATH.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import settings

app = FastAPI(title="星空的时间机器 API", root_path=settings.ROOT_PATH, docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(GZipMiddleware, minimum_size=2048)


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


@app.exception_handler(HTTPException)
async def http_exc(_: Request, exc: HTTPException):
    return error(exc.status_code, "http_error", str(exc.detail))


@app.on_event("startup")
async def _warm() -> None:
    """Fill the statistics cache in the background so the first visitor waits less."""
    import asyncio

    from .routes.data import warm_stats

    if settings.WARM_STATS:
        asyncio.get_running_loop().run_in_executor(None, warm_stats)


@app.get("/api/health")
def health():
    return {"ok": True}


from .routes import data as data_routes  # noqa: E402
from .routes import live as live_routes  # noqa: E402
from .routes import science as science_routes  # noqa: E402

app.include_router(data_routes.router)
app.include_router(science_routes.router)
app.include_router(live_routes.router)

# ---- latency metrics and a token bucket for the expensive endpoints --------
import time as _time  # noqa: E402
from collections import defaultdict as _dd  # noqa: E402

EXPENSIVE = ("/api/orbit", "/api/stats", "/api/stars/", "/api/share")
_buckets: dict[str, list[float]] = _dd(lambda: [float(settings.RATE_PER_MIN), _time.monotonic()])


@app.middleware("http")
async def metrics_and_limits(request: Request, call_next):
    path = request.url.path
    if any(path.startswith(p) for p in EXPENSIVE) and not path.endswith("/at"):
        ip = request.headers.get("x-real-ip") or (request.client.host if request.client else "?")
        b = _buckets[ip]
        now = _time.monotonic()
        b[0] = min(settings.RATE_PER_MIN, b[0] + (now - b[1]) * settings.RATE_PER_MIN / 60)
        b[1] = now
        if b[0] < 1:
            return error(429, "rate_limited", "请求太频繁，请稍后再试")
        b[0] -= 1
    t0 = _time.perf_counter()
    resp = await call_next(request)
    live_routes.record(path, (_time.perf_counter() - t0) * 1000)
    return resp


if settings.WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=settings.WEB_DIST, html=True), name="web")
