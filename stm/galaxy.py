"""Deep time (+/- 10 Myr): the Galaxy, epicycles and reference orbits.

Frames
------
* ICRS heliocentric cartesian (pc, pc/yr) - what the catalogue gives.
* Galactocentric (astropy ``Galactocentric`` v4.0 defaults): the Sun at
  x ~ -8.12 kpc, rotation towards +y. We use the *rotation azimuth*
  ``phi = atan2(-y, x)``, which increases in the direction of Galactic
  rotation, and ``v_phi > 0`` for disc stars (the galpy convention).

Model
-----
Epicycle approximation (Binney & Tremaine 2008, sec. 3.2.3) in galpy's
MWPotential2014 (Bovy 2015), rescaled to R0 = 8.122 kpc and
v_c(R0) = 233.1 km/s:

    R(t)   = Rg + X cos(kappa t + alpha)
    phi(t) = phi0 + Omega t - (2 Omega / kappa) (X / Rg) [sin(kappa t + alpha) - sin alpha]
    z(t)   = Z cos(nu t + beta)

where Rg solves Rg * v_c(Rg) = L (the star's angular momentum) and
Omega, kappa, nu are the circular, epicyclic and vertical frequencies at
Rg, tabulated from galpy. The browser evaluates the same formulas on the
GPU; ``orbit_galpy`` integrates the full equations of motion as the
reference.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np

from .constants import (
    GALCEN_DEC_DEG,
    GALCEN_RA_DEG,
    KMS_TO_PCYR,
    R0_PC,
    V_SUN_KMS,
    VC0_KMS,
    Z_SUN_PC,
)

# km/s/kpc -> rad/Myr
KMSKPC_TO_RADMYR = KMS_TO_PCYR * 1e6 / 1000.0
MYR = 1e6


GALCEN_CACHE = Path(__file__).with_name("galcen.json")


@functools.lru_cache(maxsize=1)
def galcen_affine() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (M, r_sun, v_sun): x_gc = M @ x_icrs + r_sun, v_gc = M @ v_icrs + v_sun.

    Positions in pc, velocities in km/s. Derived from astropy so that our
    frame is exactly astropy's Galactocentric with the constants above; the
    result is cached in stm/galcen.json so servers need not import astropy.
    """
    if GALCEN_CACHE.exists():
        j = json.loads(GALCEN_CACHE.read_text())
        return np.array(j["M"]), np.array(j["r_sun"]), np.array(j["v_sun"])
    m, r, v = _galcen_affine_astropy()
    GALCEN_CACHE.write_text(json.dumps({"M": m.tolist(), "r_sun": r.tolist(), "v_sun": v.tolist(),
                                        "note": "astropy Galactocentric with stm.constants; regenerate by deleting this file"}))
    return m, r, v


def _galcen_affine_astropy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import astropy.units as u
    from astropy.coordinates import (
        ICRS,
        CartesianDifferential,
        CartesianRepresentation,
        Galactocentric,
    )

    frame = Galactocentric(
        galcen_coord=ICRS(ra=GALCEN_RA_DEG * u.deg, dec=GALCEN_DEC_DEG * u.deg),
        galcen_distance=R0_PC * u.pc,
        z_sun=Z_SUN_PC * u.pc,
        galcen_v_sun=CartesianDifferential(V_SUN_KMS * u.km / u.s),
    )

    def tf(xyz_pc, v_kms=(0.0, 0.0, 0.0)):
        rep = CartesianRepresentation(np.asarray(xyz_pc, float) * u.pc).with_differentials(
            CartesianDifferential(np.asarray(v_kms, float) * u.km / u.s)
        )
        g = ICRS(rep).transform_to(frame)
        return (
            g.cartesian.xyz.to_value(u.pc),
            g.cartesian.differentials["s"].d_xyz.to_value(u.km / u.s),
        )

    r_sun, v_sun = tf([0.0, 0.0, 0.0])
    m = np.empty((3, 3))
    for i, e in enumerate(np.eye(3)):
        p, _ = tf(e * 1000.0)
        m[:, i] = (p - r_sun) / 1000.0
    return m, r_sun, v_sun


def icrs_to_cyl(r_icrs_pc, v_icrs_pcyr):
    """Heliocentric ICRS (pc, pc/yr) -> galactocentric cylindrical.

    Returns R, phi, z (pc, rad) and vR, vphi, vz (km/s).
    """
    m, r_sun, v_sun = galcen_affine()
    p = np.atleast_2d(r_icrs_pc) @ m.T + r_sun
    v = np.atleast_2d(v_icrs_pcyr) / KMS_TO_PCYR @ m.T + v_sun
    x, y, z = p.T
    vx, vy, vz = v.T
    R = np.hypot(x, y)
    phi = np.arctan2(-y, x)
    vR = (x * vx + y * vy) / R
    vphi = (-x * vy + y * vx) / R
    return R, phi, z, vR, vphi, vz


def cyl_to_galcen(R, phi, z):
    return np.stack([R * np.cos(phi), -R * np.sin(phi), z], axis=-1)


def galcen_to_icrs_helio(p_gc):
    m, r_sun, _ = galcen_affine()
    return (np.atleast_2d(p_gc) - r_sun) @ m  # inverse of orthonormal M is M.T


# --------------------------------------------------------------------- tables
@functools.lru_cache(maxsize=1)
def frequency_table(n: int = 2048):
    """Tabulate L -> (Rg, Omega, kappa, nu) for the rescaled MWPotential2014.

    Units: L in kpc km/s, Rg in pc, frequencies in rad/Myr. The grid is
    uniform in log R from 0.2 to 60 kpc.
    """
    from galpy.potential import MWPotential2014, epifreq, omegac, vcirc, verticalfreq

    ro = R0_PC / 1000.0
    rr = np.geomspace(0.2, 60.0, n) / ro  # natural units
    vc = np.array([vcirc(MWPotential2014, r) for r in rr]) * VC0_KMS
    om = np.array([omegac(MWPotential2014, r) for r in rr]) * VC0_KMS / ro
    ka = np.array([epifreq(MWPotential2014, r) for r in rr]) * VC0_KMS / ro
    nu = np.array([verticalfreq(MWPotential2014, r) for r in rr]) * VC0_KMS / ro
    R_kpc = rr * ro
    L = R_kpc * vc
    assert np.all(np.diff(L) > 0), "L(R) must be monotonic"
    return {
        "L": L,
        "Rg": R_kpc * 1000.0,
        "Omega": om * KMSKPC_TO_RADMYR,
        "kappa": ka * KMSKPC_TO_RADMYR,
        "nu": nu * KMSKPC_TO_RADMYR,
        "vc": vc,
    }


def epicycle_params(R, phi, z, vR, vphi, vz):
    """Per-star epicycle constants. Returns a dict of arrays.

    ``ok`` is False where the approximation is not meaningful
    (counter-rotating or nearly radial orbits, large amplitudes).
    """
    tab = frequency_table()
    L = R / 1000.0 * vphi
    Lc = np.clip(L, tab["L"][0], tab["L"][-1])
    Rg = np.interp(Lc, tab["L"], tab["Rg"])
    Om = np.interp(Lc, tab["L"], tab["Omega"])
    ka = np.interp(Lc, tab["L"], tab["kappa"])
    nu = np.interp(Lc, tab["L"], tab["nu"])
    vR_pcmyr = vR * KMS_TO_PCYR * MYR
    vz_pcmyr = vz * KMS_TO_PCYR * MYR
    x0 = R - Rg
    X = np.hypot(x0, vR_pcmyr / ka)
    alpha = np.arctan2(-vR_pcmyr / ka, x0)
    Z = np.hypot(z, vz_pcmyr / nu)
    beta = np.arctan2(-vz_pcmyr / nu, z)
    ok = (L > tab["L"][0]) & (X < 0.2 * Rg) & (Z < 1500.0)
    return {
        "Rg": Rg, "Omega": Om, "kappa": ka, "nu": nu, "X": X, "alpha": alpha,
        "Z": Z, "beta": beta, "phi0": phi, "R0": R, "z0": z, "ok": ok,
    }


def epicycle_at(par, t_myr):
    """Galactocentric cylindrical position at time ``t_myr`` (Myr)."""
    t = np.asarray(t_myr, dtype=float)
    a = par["kappa"] * t + par["alpha"]
    R = par["Rg"] + par["X"] * np.cos(a)
    gamma = 2.0 * par["Omega"] / par["kappa"]
    phi = par["phi0"] + par["Omega"] * t - gamma * par["X"] / par["Rg"] * (np.sin(a) - np.sin(par["alpha"]))
    z = par["Z"] * np.cos(par["nu"] * t + par["beta"])
    return R, phi, z


def sun_cyl():
    """The Sun's galactocentric cylindrical state (pc, rad, km/s)."""
    R, phi, z, vR, vphi, vz = icrs_to_cyl(np.zeros((1, 3)), np.zeros((1, 3)))
    return R[0], phi[0], z[0], vR[0], vphi[0], vz[0]


def epicycle_helio_icrs(r_icrs_pc, v_icrs_pcyr, t_myr):
    """Heliocentric ICRS position (pc) of stars *relative to the Sun* at t.

    Both the stars and the Sun follow their own epicycles; the result is
    expressed in the inertial galactocentric axes rotated back to ICRS
    orientation (no co-rotation).
    """
    st = icrs_to_cyl(r_icrs_pc, v_icrs_pcyr)
    par = epicycle_params(*st)
    R, phi, z = epicycle_at(par, t_myr)
    sp = epicycle_params(*[np.atleast_1d(q) for q in sun_cyl()])
    Rs, phis, zs = epicycle_at(sp, t_myr)
    m, _, _ = galcen_affine()
    d = cyl_to_galcen(R, phi, z) - cyl_to_galcen(Rs, phis, zs)
    return np.atleast_2d(d) @ m


def orbit_galpy(r_icrs_pc, v_icrs_pcyr, t_myr):
    """Reference: integrate orbits in MWPotential2014 with galpy.

    Returns galactocentric cylindrical R (pc), phi (rad), z (pc) with shape
    (n_stars, n_times).
    """
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    ro = R0_PC / 1000.0
    R, phi, z, vR, vphi, vz = icrs_to_cyl(r_icrs_pc, v_icrs_pcyr)
    vxvv = np.stack([R / 1000.0 / ro, vR / VC0_KMS, vphi / VC0_KMS, z / 1000.0 / ro, vz / VC0_KMS, phi], axis=-1)
    t = np.atleast_1d(np.asarray(t_myr, dtype=float))
    # galpy's natural time unit is ro/vo
    tunit_myr = ro * 1000.0 / (VC0_KMS * KMS_TO_PCYR * MYR)
    out = np.empty((3, len(R), len(t)))
    for mask in (t >= 0, t < 0):
        if not mask.any():
            continue
        tt = t[mask]
        order = np.argsort(np.abs(tt))
        grid = np.concatenate([[0.0], tt[order]]) / tunit_myr
        o = Orbit(vxvv)
        o.integrate(grid, MWPotential2014, method="dop853_c")
        idx = np.where(mask)[0][order]
        out[0][:, idx] = np.atleast_2d(o.R(grid, use_physical=False))[:, 1:] * ro * 1000.0
        out[1][:, idx] = np.atleast_2d(o.phi(grid, use_physical=False))[:, 1:]
        out[2][:, idx] = np.atleast_2d(o.z(grid, use_physical=False))[:, 1:] * ro * 1000.0
    return out[0], out[1], out[2]


# ------------------------------------------------- relative motion (browser model)
def _galpy_orbits_xyz(vxvv_list, ts_myr):
    """Integrate several galactocentric states; return inertial xyz (pc)."""
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    ro = R0_PC / 1000.0
    tunit_myr = ro * 1000.0 / (VC0_KMS * KMS_TO_PCYR * MYR)
    out = np.empty((len(vxvv_list), len(ts_myr), 3))
    for mask in (ts_myr >= 0, ts_myr < 0):
        if not mask.any():
            continue
        tt = ts_myr[mask]
        order = np.argsort(np.abs(tt))
        grid = np.concatenate([[0.0], tt[order]]) / tunit_myr
        o = Orbit(np.asarray(vxvv_list))
        o.integrate(grid, MWPotential2014, method="dop853_c")
        R = np.atleast_2d(o.R(grid, use_physical=False))[:, 1:] * ro * 1000.0
        ph = np.atleast_2d(o.phi(grid, use_physical=False))[:, 1:]
        z = np.atleast_2d(o.z(grid, use_physical=False))[:, 1:] * ro * 1000.0
        idx = np.where(mask)[0][order]
        out[:, idx, :] = np.stack([R * np.cos(ph), -R * np.sin(ph), z], axis=-1)
    return out


def _gc_xyz_to_vxvv(p_pc, v_kms):
    ro = R0_PC / 1000.0
    p = np.atleast_2d(p_pc)
    v = np.atleast_2d(v_kms)
    x, y, z = p.T
    vx, vy, vz = v.T
    R = np.hypot(x, y)
    phi = np.arctan2(-y, x)
    vR = (x * vx + y * vy) / R
    vphi = (-x * vy + y * vx) / R
    return np.stack([R / 1000.0 / ro, vR / VC0_KMS, vphi / VC0_KMS, z / 1000.0 / ro, vz / VC0_KMS, phi], axis=-1)


@functools.lru_cache(maxsize=4)
def relative_motion_table(t_max_myr: float = 12.0, step_myr: float = 0.05, eps_pc: float = 5.0, eps_kms: float = 0.5):
    """State-transition matrices of motion relative to the Sun.

    For a star at heliocentric ICRS position r0 (pc) with velocity v0
    (pc/Myr) relative to the Sun, its position relative to the Sun at time
    t is, to first order in (r0, v0),

        d(t) = A(t) @ r0 + B(t) @ v0          (ICRS axes, pc)

    A and B are obtained by central finite differences of galpy orbits in
    MWPotential2014 around the Sun's own orbit. This is the linearised
    (Hill / epicycle) relative motion but about the Sun's *actual*
    non-circular orbit, so its error grows with the star's distance from
    the Sun rather than with the Sun's epicycle amplitude.

    Returns dict with t (Myr), A, B (n,3,3), sun_gc (n,3) galactocentric
    position of the Sun (pc, inertial axes).
    """
    m, r_sun, v_sun = galcen_affine()
    ts = np.round(np.arange(-t_max_myr, t_max_myr + step_myr / 2, step_myr), 10)
    states = [(r_sun, v_sun)]
    for k in range(3):
        for sgn in (1, -1):
            e = np.zeros(3)
            e[k] = sgn * eps_pc
            states.append((r_sun + e, v_sun))
    for k in range(3):
        for sgn in (1, -1):
            e = np.zeros(3)
            e[k] = sgn * eps_kms
            states.append((r_sun, v_sun + e))
    vxvv = np.vstack([_gc_xyz_to_vxvv(p, v) for p, v in states])
    xyz = _galpy_orbits_xyz(vxvv, ts)  # (13, nt, 3)
    sun = xyz[0]
    A_gc = np.empty((len(ts), 3, 3))
    B_gc = np.empty((len(ts), 3, 3))
    kms_to_pcmyr = KMS_TO_PCYR * MYR
    for k in range(3):
        A_gc[:, :, k] = (xyz[1 + 2 * k] - xyz[2 + 2 * k]) / (2 * eps_pc)
        B_gc[:, :, k] = (xyz[7 + 2 * k] - xyz[8 + 2 * k]) / (2 * eps_kms * kms_to_pcmyr)
    # ICRS axes: d_icrs = M^T d_gc, r0_gc = M r0_icrs
    A = np.einsum("ji,tjk,kl->til", m, A_gc, m)
    B = np.einsum("ji,tjk,kl->til", m, B_gc, m)
    i0 = int(np.argmin(np.abs(ts)))
    A[i0] = np.eye(3)
    B[i0] = 0.0
    return {"t": ts, "A": A, "B": B, "sun_gc": sun}


def relative_position(r0_icrs_pc, v0_icrs_pcyr, t_myr, table=None):
    """Browser model in float64: position relative to the Sun at ``t_myr``."""
    tab = table or relative_motion_table()
    t = float(t_myr)
    ts = tab["t"]
    i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
    f = (t - ts[i]) / (ts[i + 1] - ts[i])
    A = _interp_mat(tab["A"], i, f, ts)
    B = _interp_mat(tab["B"], i, f, ts)
    v_pcmyr = np.atleast_2d(v0_icrs_pcyr) * MYR
    return np.atleast_2d(r0_icrs_pc) @ A.T + v_pcmyr @ B.T


def _interp_mat(arr, i, f, ts):
    """Cubic Hermite (Catmull-Rom) interpolation of tabulated matrices."""
    n = len(ts)
    p0 = arr[max(i - 1, 0)]
    p1 = arr[i]
    p2 = arr[i + 1]
    p3 = arr[min(i + 2, n - 1)]
    f2, f3 = f * f, f * f * f
    return (
        p1 * (1 - 3 * f2 + 2 * f3)
        + p2 * (3 * f2 - 2 * f3)
        + 0.5 * (p2 - p0) * (f - 2 * f2 + f3)
        + 0.5 * (p3 - p1) * (f3 - f2)
    )


def galpy_relative(r0_icrs_pc, v0_icrs_pcyr, ts_myr):
    """Reference for :func:`relative_position`: full galpy orbits of the
    stars and the Sun, differenced. Returns (n_stars, n_t, 3) ICRS axes."""
    m, r_sun, v_sun = galcen_affine()
    p = np.atleast_2d(r0_icrs_pc) @ m.T + r_sun
    v = np.atleast_2d(v0_icrs_pcyr) / KMS_TO_PCYR @ m.T + v_sun
    vxvv = np.vstack([_gc_xyz_to_vxvv(r_sun, v_sun), _gc_xyz_to_vxvv(p, v)])
    xyz = _galpy_orbits_xyz(vxvv, np.atleast_1d(np.asarray(ts_myr, float)))
    rel = xyz[1:] - xyz[0][None]
    return rel @ m
