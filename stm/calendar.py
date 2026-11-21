"""Year numbering.

Internally every time is a Julian epoch in *astronomical* year numbering:
year 0 = 1 BCE, year -1 = 2 BCE, ... The interface shows 公元前/公元.
"""

from __future__ import annotations


def to_display(year: float) -> str:
    """Astronomical year -> Chinese label, e.g. -2799 -> '公元前 2800 年'."""
    y = round(year)
    if abs(y) >= 100_000:
        k = y / 10_000
        return f"{'' if y > 0 else '前 '}{abs(k):g} 万年"
    if y <= 0:
        return f"公元前 {1 - y} 年"
    return f"公元 {y} 年"


def from_bce(bce: int) -> int:
    """'公元前 N 年' -> astronomical year (1 - N)."""
    if bce < 1:
        raise ValueError("BCE years start at 1")
    return 1 - bce


def relative_label(years_from_now: float) -> str:
    """Offsets for the deep-time scale, e.g. +1.29e6 -> '129 万年后'."""
    a = abs(years_from_now)
    if a < 1:
        return "今天"
    if a >= 1e8:
        s = f"{a / 1e8:.3g} 亿年"
    elif a >= 1e4:
        s = f"{a / 1e4:.4g} 万年"
    else:
        s = f"{a:.0f} 年"
    return s + ("后" if years_from_now > 0 else "前")
