"""Rigorous linear space motion of stars (historical scale, +/- 200 kyr).

This follows ERFA eraStarpv + eraStarpm (IAU SOFA):

1. the catalogue ("observed") place and motion are turned into an
   *inertial* space-motion vector, removing the light-time compression of
   the observed proper motion and radial velocity (the special-relativity
   correction terms ``d`` and ``del`` of eraStarpv);
2. the star moves in a straight line, and the light-time to the observer
   is accounted for at both epochs (eraStarpm).

Without step 1 and the light-time terms, the position of a nearby fast
star such as Barnard's star drifts by several arcminutes from the SOFA
result after 100 000 years.

Units: positions in pc, velocities in pc per Julian year, times in Julian
years. Everything is vectorised over stars with numpy.
"""

from __future__ import annotations

import numpy as np

from .constants import C_PCYR, DEG, KMS_TO_PCYR

MAS = DEG / 3.6e6


def unit_vectors(ra_deg, dec_deg):
    a = np.asarray(ra_deg, dtype=float) * DEG
    d = np.asarray(dec_deg, dtype=float) * DEG
    ca, sa, cd, sd = np.cos(a), np.sin(a), np.cos(d), np.sin(d)
    u = np.stack([cd * ca, cd * sa, sd], axis=-1)
    e_ra = np.stack([-sa, ca, np.zeros_like(a)], axis=-1)
    e_dec = np.stack([-sd * ca, -sd * sa, cd], axis=-1)
    return u, e_ra, e_dec


def observed_to_inertial(r0, v_obs):
    """eraStarpv's relativistic light-time correction, vectorised.

    ``r0`` (N,3) position, ``v_obs`` (N,3) velocity built naively from the
    observed proper motion and radial velocity. Returns the inertial
    velocity (N,3).
    """
    r0 = np.atleast_2d(r0)
    v_obs = np.atleast_2d(v_obs)
    pu = r0 / np.linalg.norm(r0, axis=-1, keepdims=True)
    vsr = np.sum(pu * v_obs, axis=-1)
    usr = vsr[:, None] * pu
    ust = v_obs - usr
    vst = np.linalg.norm(ust, axis=-1)
    betsr = vsr / C_PCYR
    betst = vst / C_PCYR
    betr, bett = betsr.copy(), betst.copy()
    d = np.ones_like(betr)
    dl = np.zeros_like(betr)
    for _ in range(8):  # converges to machine precision in < 5 iterations for v << c
        d = 1.0 + betr
        w = betr * betr + bett * bett
        dl = -w / (np.sqrt(1.0 - w) + 1.0)
        betr = d * betsr + dl
        bett = d * betst
    ut = d[:, None] * ust
    ur = (C_PCYR * (d * betsr + dl))[:, None] * pu
    return ur + ut


def catalogue_to_pv(ra_deg, dec_deg, dist_pc, pmra_masyr, pmdec_masyr, rv_kms):
    """Catalogue astrometry -> (r0, v_inertial) in pc and pc/yr (ICRS axes).

    ``pmra`` includes the cos(dec) factor, as in Gaia. Missing radial
    velocities must be passed as 0.
    """
    u, e_ra, e_dec = unit_vectors(ra_deg, dec_deg)
    r = np.asarray(dist_pc, dtype=float)
    r0 = u * r[..., None]
    v_t = r[..., None] * (
        np.asarray(pmra_masyr, dtype=float)[..., None] * MAS * e_ra
        + np.asarray(pmdec_masyr, dtype=float)[..., None] * MAS * e_dec
    )
    v_r = (np.asarray(rv_kms, dtype=float) * KMS_TO_PCYR)[..., None] * u
    return r0, observed_to_inertial(r0, v_t + v_r)


def light_time_tau(r0, v, dt):
    """Effective time of flight (yr) along the track, eraStarpm style.

    The observed place at epoch ``t0 + dt`` is ``r0 + v * tau``.
    ``dt`` is a scalar or an array broadcastable to the number of stars.
    """
    r0 = np.atleast_2d(r0)
    v = np.atleast_2d(v)
    tl1 = np.linalg.norm(r0, axis=-1) / C_PCYR
    p = r0 + v * (np.asarray(dt) + tl1)[..., None]
    r2 = np.sum(p * p, axis=-1)
    rdv = np.sum(p * v, axis=-1)
    v2 = np.sum(v * v, axis=-1)
    c2mv2 = C_PCYR * C_PCYR - v2
    tl2 = (-rdv + np.sqrt(rdv * rdv + c2mv2 * r2)) / c2mv2
    return dt + (tl1 - tl2)


def propagate(r0, v, dt):
    """Observed position (pc, ICRS axes) after ``dt`` Julian years."""
    tau = light_time_tau(r0, v, dt)
    return np.atleast_2d(r0) + np.atleast_2d(v) * np.asarray(tau)[..., None]


def to_radec(p):
    p = np.atleast_2d(p)
    r = np.linalg.norm(p, axis=-1)
    ra = np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(p[:, 2] / r, -1, 1)))
    return ra, dec, r


def apparent_mag(m0, r0, p):
    """Magnitude at the new distance (no change in extinction)."""
    return np.asarray(m0) + 5.0 * np.log10(
        np.linalg.norm(np.atleast_2d(p), axis=-1) / np.linalg.norm(np.atleast_2d(r0), axis=-1)
    )


def angular_sep_deg(a, b):
    """Angle between direction vectors (degrees), robust for tiny angles."""
    a = np.atleast_2d(a) / np.linalg.norm(np.atleast_2d(a), axis=-1, keepdims=True)
    b = np.atleast_2d(b) / np.linalg.norm(np.atleast_2d(b), axis=-1, keepdims=True)
    return np.degrees(
        np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))
    )
