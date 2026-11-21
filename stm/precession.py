"""Long-term precession, Vondrák, Capitaine & Wallace (2011, A&A 534, A22).

Valid for +/- 200 000 years around J2000. This is a line-by-line port of
the ERFA routines eraLtpecl, eraLtpequ, eraLtp and eraLtpb (BSD-3,
derived from IAU SOFA); ``tests/py/test_precession.py`` checks it
against pyerfa.

Matrices act on column vectors: ``v_date = P @ v_j2000``.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import DAS2R

_EPS0 = 84381.406 * DAS2R

_PQPOL = (
    (5851.607687, -0.1189000, -0.00028913, 0.000000101),
    (-1600.886300, 1.1689818, -0.00000020, -0.000000437),
)
_PQPER = (
    (708.15, -5486.751211, -684.661560, 667.666730, -5523.863691),
    (2309.00, -17.127623, 2446.283880, -2354.886252, -549.747450),
    (1620.00, -617.517403, 399.671049, -428.152441, -310.998056),
    (492.20, 413.442940, -356.652376, 376.202861, 421.535876),
    (1183.00, 78.614193, -186.387003, 184.778874, -36.776172),
    (622.00, -180.732815, -316.800070, 335.321713, -145.278396),
    (882.00, -87.676083, 198.296701, -185.138669, -34.744450),
    (547.00, 46.140315, 101.135679, -120.972830, 22.885731),
)
_XYPOL = (
    (5453.282155, 0.4252841, -0.00037173, -0.000000152),
    (-73750.930350, -0.7675452, -0.00018725, 0.000000231),
)
_XYPER = (
    (256.75, -819.940624, 75004.344875, 81491.287984, 1558.515853),
    (708.15, -8444.676815, 624.033993, 787.163481, 7774.939698),
    (274.20, 2600.009459, 1251.136893, 1251.296102, -2219.534038),
    (241.45, 2755.175630, -1102.212834, -1257.950837, -2523.969396),
    (2309.00, -167.659835, -2660.664980, -2966.799730, 247.850422),
    (492.20, 871.855056, 699.291817, 639.744522, -846.485643),
    (396.10, 44.769698, 153.167220, 131.600209, -1393.124055),
    (288.90, -512.313065, -950.865637, -445.040117, 368.526116),
    (231.10, -819.415595, 499.754645, 584.522874, 749.045012),
    (1610.00, -538.071099, -145.188210, -89.756563, 444.704518),
    (620.00, -189.793622, 558.116553, 524.429630, 235.934465),
    (157.87, -402.922932, -23.923029, -13.549067, 374.049623),
    (220.30, 179.516345, -165.405086, -210.157124, -171.330180),
    (1200.00, -9.814756, 9.344131, -44.919798, -22.899655),
)

# Frame bias, IERS Conventions 2010 eqs. 5.21 and 5.33
_DX = -0.016617 * DAS2R
_DE = -0.0068192 * DAS2R
_DR = -0.0146 * DAS2R


def _series(epj: float, pol, per) -> tuple[float, float]:
    t = (epj - 2000.0) / 100.0
    a_acc = 0.0
    b_acc = 0.0
    w = 2.0 * math.pi * t
    for period, ca, cb, sa, sb in per:
        ang = w / period
        s, c = math.sin(ang), math.cos(ang)
        a_acc += c * ca + s * sa
        b_acc += c * cb + s * sb
    w = 1.0
    for i in range(4):
        a_acc += pol[0][i] * w
        b_acc += pol[1][i] * w
        w *= t
    return a_acc * DAS2R, b_acc * DAS2R


def ecliptic_pole(epj: float) -> np.ndarray:
    """Unit vector of the ecliptic pole of date, in the J2000 mean frame."""
    p, q = _series(epj, _PQPOL, _PQPER)
    w = 1.0 - p * p - q * q
    w = math.sqrt(w) if w > 0 else 0.0
    s, c = math.sin(_EPS0), math.cos(_EPS0)
    return np.array([p, -q * c - w * s, -q * s + w * c])


def equator_pole(epj: float) -> np.ndarray:
    """Unit vector of the mean equator pole of date, in the J2000 mean frame."""
    x, y = _series(epj, _XYPOL, _XYPER)
    w = 1.0 - x * x - y * y
    return np.array([x, y, math.sqrt(w) if w > 0 else 0.0])


def ltp(epj: float) -> np.ndarray:
    """Precession matrix, J2000 mean equator/equinox -> mean of date."""
    peq = equator_pole(epj)
    pec = ecliptic_pole(epj)
    eqx = np.cross(peq, pec)
    eqx /= np.linalg.norm(eqx)
    mid = np.cross(peq, eqx)
    return np.array([eqx, mid, peq])


def ltpb(epj: float) -> np.ndarray:
    """Precession-bias matrix, ICRS (GCRS) -> mean equator/equinox of date."""
    rp = ltp(epj)
    out = np.empty((3, 3))
    for i in range(3):
        out[i, 0] = rp[i, 0] - rp[i, 1] * _DR + rp[i, 2] * _DX
        out[i, 1] = rp[i, 0] * _DR + rp[i, 1] + rp[i, 2] * _DE
        out[i, 2] = -rp[i, 0] * _DX - rp[i, 1] * _DE + rp[i, 2]
    return out


def obliquity(epj: float) -> float:
    """Mean obliquity of the ecliptic of date (rad), from the two poles."""
    return math.acos(float(np.clip(np.dot(equator_pole(epj), ecliptic_pole(epj)), -1, 1)))


def ecliptic_in_date_frame(epj: float) -> np.ndarray:
    """Ecliptic pole of date expressed in the mean-of-date equatorial frame.

    By construction of :func:`ltp` the equinox is the x axis, so this vector
    is (0, -sin eps, cos eps).
    """
    return ltp(epj) @ ecliptic_pole(epj)


def pole_icrs(epj: float) -> np.ndarray:
    """Mean celestial pole of date as an ICRS unit vector (row 3 of ltpb)."""
    return ltpb(epj)[2].copy()


def equinox_icrs(epj: float) -> np.ndarray:
    """Mean equinox of date as an ICRS unit vector (row 1 of ltpb)."""
    return ltpb(epj)[0].copy()
