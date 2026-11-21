"""The space-time postcard: a poster-quality PNG of a saved view, rendered on
the server with the same formulas as everything else (stm/*).

今 (obs): the whole visible hemisphere above 登封 at that moment, zenith at
          the centre (equidistant fisheye), stars as glows coloured by
          temperature, the Milky Way omitted (stars only, honest at this size).
古 (atlas): the polar chart of that year — ink dots for 星官 stars,
          vermilion lines, 28-mansion radial lines and names, a vertical
          inscription and a seal.
"""

from __future__ import annotations

import functools
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stm import motion as mo  # noqa: E402
from stm import precession as pr  # noqa: E402
from stm import sky  # noqa: E402
from stm.calendar import to_display  # noqa: E402
from stm.constants import GAIA_EPOCH, KMS_TO_PCYR  # noqa: E402

FONT_DIR = ROOT / "data" / "frozen" / "fonts"
W, H = 1200, 1800
SS = 2  # supersampling
LAT = 34.4 * math.pi / 180
SEASONS = ["春分", "清明", "谷雨", "立夏", "小满", "芒种", "夏至", "小暑", "大暑", "立秋", "处暑", "白露",
           "秋分", "寒露", "霜降", "立冬", "小雪", "大雪", "冬至", "小寒", "大寒", "立春", "雨水", "惊蛰"]


@functools.lru_cache(maxsize=1)
def sky_data() -> dict:
    return json.loads((ROOT / "web" / "public" / "data" / "sky.json").read_text())


@functools.lru_cache(maxsize=1)
def bright_stars() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stars brighter than G = 7 from the catalogue: r0 (pc), v (pc/yr), M, bp_rp."""
    import duckdb

    from .. import settings

    con = duckdb.connect(config={"memory_limit": "200MB", "threads": 1})
    q = f"SELECT x, y, z, vx, vy, vz, absg, bp_rp FROM read_parquet('{settings.PARQUET}') WHERE g < 7.0"
    a = con.execute(q).fetchnumpy()
    r0 = np.stack([a["x"], a["y"], a["z"]], -1)
    v = np.stack([a["vx"], a["vy"], a["vz"]], -1)
    return r0, v, a["absg"].astype(float), np.nan_to_num(a["bp_rp"].astype(float), nan=0.8)


@functools.lru_cache(maxsize=32)
def font(kind: str, size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
    name = {"serif": "notoserifsc__NotoSerifSC[wght].ttf", "sans": "notosanssc__NotoSansSC[wght].ttf", "mono": "jetbrainsmono__JetBrainsMono[wght].ttf"}[kind]
    f = ImageFont.truetype(str(FONT_DIR / name), size * SS)
    try:
        f.set_variation_by_axes([weight])
    except Exception:  # noqa: BLE001
        pass
    return f


def star_rgb(bp_rp: float) -> tuple[int, int, int]:
    # same empirical BP-RP -> Teff table as the browser, coarse blackbody tint
    t = np.interp(bp_rp, [-0.3, 0.0, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5], [20000, 10200, 6800, 5800, 4800, 4100, 3550, 3000])
    if t > 9000:
        c = (198, 214, 255)
    elif t > 7000:
        c = (226, 232, 255)
    elif t > 5800:
        c = (255, 246, 232)
    elif t > 4800:
        c = (255, 226, 190)
    elif t > 3900:
        c = (255, 200, 150)
    else:
        c = (255, 176, 128)
    return c


def year_label(st: dict) -> str:
    if st.get("scale") == "deep":
        t = float(st.get("deep", 0))
        return "今天" if abs(t) < 1e-4 else f"{abs(t) * 100:.1f} 万年{'后' if t > 0 else '前'}"
    return to_display(math.floor(float(st.get("year", 2026)) + 1e-9))


def render(state: dict, url: str, qr_png: bytes | None) -> bytes:
    style = state.get("style", "obs")
    year = float(state.get("year", 2026.0)) if state.get("scale") != "deep" else 2026.0
    lam = float(state.get("lam", 203.5)) * math.pi / 180
    hsun = float(state.get("hSun", 131.0)) * math.pi / 180
    prec = state.get("precession", True)
    motion = state.get("motion", True)
    ep = year if prec else GAIA_EPOCH
    dt = (year - GAIA_EPOCH) if motion else 0.0
    P = pr.ltpb(ep)
    lst = sky.sidereal_angle(ep, lam, hsun)
    season = SEASONS[int(round(((lam * 180 / math.pi) % 360) / 15)) % 24]
    hrs = ((hsun * 180 / math.pi / 15 + 12) % 24)
    when = f"{season} · {int(hrs):02d}:{int((hrs % 1) * 60):02d} · 登封 北纬 34.4°"
    if style == "atlas":
        img = render_atlas(P, lst, dt, year)
    else:
        img = render_obs(P, lst, dt)
    d = ImageDraw.Draw(img)
    dark = style != "atlas"
    fg = (234, 240, 255) if dark else (29, 26, 22)
    fg2 = (150, 162, 196) if dark else (110, 95, 80)
    accent = (238, 244, 255) if dark else (178, 59, 41)
    # typography block
    d.text((90 * SS, 1330 * SS), "星空的时间机器", font=font("serif", 54, 600), fill=fg)
    d.text((92 * SS, 1420 * SS), year_label(state), font=font("serif", 40, 600), fill=accent)
    d.text((92 * SS, 1485 * SS), when, font=font("sans", 22, 400), fill=fg2)
    d.text((92 * SS, 1640 * SS), "数据：Gaia DR3（ESA/Gaia/DPAC）· Hipparcos/HYG · Stellarium 星官", font=font("sans", 16, 300), fill=fg2)
    d.text((92 * SS, 1668 * SS), url, font=font("mono", 16, 400), fill=fg2)
    if qr_png:
        q = Image.open(io.BytesIO(qr_png)).convert("RGB").resize((200 * SS, 200 * SS), Image.NEAREST)
        img.paste(q, ((W - 290) * SS, 1430 * SS))
    out = img.resize((W, H), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _project_fisheye(P: np.ndarray, lst: float, dirs: np.ndarray, R: float, cx: float, cy: float):
    Hm = sky.date_frame_to_horizon(lst, LAT) @ P
    enu = dirs @ Hm.T
    alt = np.arcsin(np.clip(enu[:, 2], -1, 1))
    az = np.arctan2(enu[:, 0], enu[:, 1])
    rr = (math.pi / 2 - alt) / (math.pi / 2) * R
    # north up, east left (as seen looking up)
    x = cx - rr * np.sin(az)
    y = cy - rr * np.cos(az)
    return x, y, alt


def render_obs(P: np.ndarray, lst: float, dt: float) -> Image.Image:
    img = Image.new("RGB", (W * SS, H * SS), (6, 7, 14))
    glow = Image.new("RGB", img.size, (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    dd = ImageDraw.Draw(img)
    r0, v, M, c = bright_stars()
    p = mo.propagate(r0, v, dt)
    d = np.linalg.norm(p, axis=1)
    dirs = p / d[:, None]
    m = M + 5 * np.log10(d / 10)
    cx, cy, R = W / 2 * SS, 640 * SS, 520 * SS
    x, y, alt = _project_fisheye(P, lst, dirs, R, cx, cy)
    order = np.argsort(-m)
    for i in order:
        if alt[i] < 0 or m[i] > 6.0:
            continue
        # naked-eye sky: size and brightness by magnitude, a soft halo only for the brightest
        rad = max(0.6, (6.0 - m[i]) * 0.62) * SS
        col = star_rgb(c[i])
        k = min(1.0, 0.3 + (6.0 - m[i]) / 5)
        cc = tuple(int(v_ * k) for v_ in col)
        dd.ellipse([x[i] - rad, y[i] - rad, x[i] + rad, y[i] + rad], fill=cc)
        if m[i] < 1.2:
            g = rad * 3
            gd.ellipse([x[i] - g, y[i] - g, x[i] + g, y[i] + g], fill=tuple(int(v_ * 0.22) for v_ in col))
    glow = glow.filter(ImageFilter.GaussianBlur(10 * SS))
    img = Image.fromarray(np.minimum(255, np.asarray(img, dtype=np.int16) + np.asarray(glow, dtype=np.int16)).astype(np.uint8))
    dd = ImageDraw.Draw(img)
    dd.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(70, 82, 120), width=SS)
    f = font("serif", 22, 500)
    for t, az in (("北", 0), ("东", 90), ("南", 180), ("西", 270)):
        a = math.radians(az)
        dd.text((cx - (R + 34 * SS) * math.sin(a) - 11 * SS, cy - (R + 34 * SS) * math.cos(a) - 14 * SS), t, font=f, fill=(150, 162, 196))
    return img


def render_atlas(P: np.ndarray, lst: float, dt: float, year: float) -> Image.Image:
    rng = np.random.default_rng(7)
    base = np.array([236, 226, 204], dtype=np.float32)
    noise = rng.normal(0, 1, (H * SS // 8, W * SS // 8)).astype(np.float32)
    noise = np.asarray(Image.fromarray(((noise + 3) * 40).clip(0, 255).astype(np.uint8)).resize((W * SS, H * SS), Image.BICUBIC), dtype=np.float32) / 255 - 0.47
    paper = (base[None, None, :] + noise[..., None] * 18).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(paper, "RGB")
    d = ImageDraw.Draw(img)
    sd = sky_data()
    cx, cy = W / 2 * SS, 640 * SS
    colat = math.pi / 2 - LAT
    max_polar = math.pi / 2 + colat
    R = 520 * SS
    k = R / max_polar
    c0, s0 = math.cos(-lst), math.sin(-lst)
    Rz = np.array([[c0, -s0, 0], [s0, c0, 0], [0, 0, 1]])
    A = Rz @ P

    def proj(dirs):
        e = dirs @ A.T
        polar = math.pi / 2 - np.arctan2(e[:, 2], np.hypot(e[:, 0], e[:, 1]))
        az = np.arctan2(e[:, 1], e[:, 0])
        return cx - np.sin(az) * polar * k, cy + np.cos(az) * polar * k, polar

    ochre, red, ink = (138, 106, 69), (178, 59, 41), (29, 26, 22)
    for dec, w in ((colat, 2), (0.0, 2), (-colat, 3), (-colat - 0.012, 2)):
        rr = (math.pi / 2 - dec) * k
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=ochre, width=w * SS // 2)
    stars = sd["stars"]

    def dir_of(h):
        s = stars.get(str(h))
        if not s:
            return None
        r0 = np.array([s["p"]])
        v = np.array([s["v"]]) * KMS_TO_PCYR
        p = mo.propagate(r0, v, dt)[0]
        return p / np.linalg.norm(p)

    # 28 mansions
    f = font("serif", 22, 600)
    ras = []
    for x in sd["xiu"]:
        u = dir_of(x["det"])
        if u is None:
            ras.append(None)
            continue
        e = P @ u
        ras.append(math.atan2(e[1], e[0]))
    for i, x in enumerate(sd["xiu"]):
        ra = ras[i]
        if ra is None:
            continue
        pts = []
        for dec in np.linspace(colat, -colat - 0.012, 40):
            pts.append([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
        X, Y, _ = proj(np.array(pts) @ P)
        d.line(list(zip(X, Y)), fill=ochre, width=SS)
        nxt = ras[(i + 1) % len(ras)] or ra
        mid = (ra + ((nxt - ra) % (2 * math.pi)) / 2)
        dec = -colat - 0.07
        X, Y, _ = proj(np.array([[math.cos(dec) * math.cos(mid), math.cos(dec) * math.sin(mid), math.sin(dec)]]) @ P)
        d.text((X[0] - 11 * SS, Y[0] - 14 * SS), x["short"], font=f, fill=red)
    # asterism lines and ink dots
    for a in sd["chinese"]:
        for line in a["lines"]:
            pts = [dir_of(h) for h in line]
            pts = [p_ for p_ in pts if p_ is not None]
            if len(pts) < 2:
                continue
            X, Y, pol = proj(np.array(pts))
            if np.any(pol > max_polar):
                continue
            d.line(list(zip(X, Y)), fill=red, width=2 * SS // 2 + 1)
    for h, s in stars.items():
        u = dir_of(int(h))
        if u is None or s["g"] > 6.5:
            continue
        X, Y, pol = proj(u[None])
        if pol[0] > max_polar:
            continue
        r = max(1.1, (6.5 - s["g"]) * 0.75 + 1.0) * SS
        d.ellipse([X[0] - r, Y[0] - r, X[0] + r, Y[0] + r], fill=ink)
    # seal
    sx, sy = (W - 150) * SS, 1340 * SS
    d.rounded_rectangle([sx, sy, sx + 70 * SS, sy + 70 * SS], radius=6 * SS, fill=red)
    d.text((sx + 13 * SS, sy + 6 * SS), "星", font=font("serif", 44, 700), fill=(244, 234, 214))
    return img
