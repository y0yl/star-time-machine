"""Clear-sky atmosphere for the Earth-bound view.

Model: the spherical, horizontally uniform atmosphere of Bruneton & Neyret
(2008) with the multiple-scattering approximation of Hillaire (2020):
Rayleigh + Mie scattering, an ozone absorption layer (which keeps the
twilight zenith blue), the Earth's shadow, and ground albedo. Light is
traced in 20 spectral bands (380-780 nm) with the solar spectrum and the
ozone Chappuis band (stm/atmosphere_spectra.py), then converted through the
CIE 1931 colour-matching functions to linear sRGB, as in Bruneton's
reference implementation; three RGB "wavelengths" would turn the twilight
zenith lavender.

Units are the renderer's: radiance in "G = 0 stars per steradian", i.e. the
Sun is a source of flux 10**(0.4 * 26.90) in the same units as a star of
apparent magnitude G. The sky and the stars are therefore drawn on one scale,
and stars fade into a bright sky because they are fainter than it, not by a
rule.

Aerosol optical depth 0.10 at 550 nm (scale height 1.2 km, Angstrom exponent
1.3) gives a zenith extinction of about 0.22 mag at the observer, close to
the 0.25 mag/airmass previously used for a dark site.

Not modelled: the Moon, light pollution, clouds, refraction, the zodiacal
light and airglow (the renderer adds a separate night-sky glow).
"""

from __future__ import annotations

import numpy as np

from .atmosphere_spectra import CIE_XYZ, OZONE_CROSS_SECTION, SOLAR_IRRADIANCE, WAVELENGTH_NM

RG = 6360.0  # ground radius, km
RT = 6460.0  # top of the atmosphere, km
H_OBS = 0.35  # observer height above the ground, km (登封 ~ 350 m)
LAMBDA_NM = WAVELENGTH_NM

# Rayleigh: 1.24062e-6 m^-1 um^4 (Bruneton 2017; 13.56e-6 /m at 550 nm)
RAY_SCAT = 1.24062e-3 * (LAMBDA_NM / 1000.0) ** -4  # 1/km at sea level
RAY_H = 8.0
MIE_TAU_550 = 0.10
MIE_H = 1.2
MIE_ANGSTROM = 1.3
MIE_EXT = MIE_TAU_550 / MIE_H * (LAMBDA_NM / 550.0) ** -MIE_ANGSTROM
MIE_SCAT = 0.9 * MIE_EXT
MIE_G = 0.8
# 300 Dobson units spread over a tent profile whose integral is 15 km
OZONE_ABS = 300 * 2.687e20 / 15000.0 * OZONE_CROSS_SECTION * 1e3  # 1/km at the layer peak
OZONE_CENTER = 25.0
OZONE_WIDTH = 15.0
GROUND_ALBEDO = 0.3

SUN_MAG_G = -26.90  # apparent G magnitude of the Sun (M_G = 4.67)
SUN_FLUX = 10 ** (-0.4 * SUN_MAG_G)
SUN_RADIUS = np.radians(0.2666)
LUM = np.array([0.2126, 0.7152, 0.0722])  # linear sRGB -> luminance
XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660, 1.8760108, 0.0415560],
                        [0.0556434, -0.2040259, 1.0572252]])
# solar spectrum scaled so that the Sun's luminance is SUN_FLUX (G = 0 stars)
SUN_SPECTRUM = SOLAR_IRRADIANCE * SUN_FLUX / float(SOLAR_IRRADIANCE @ CIE_XYZ[:, 1])


def spectrum_to_rgb(spec):
    """Spectral radiance (..., 20) -> linear sRGB (..., 3) with Y preserved.
    Colours outside the sRGB gamut (the deepest twilight blue) are
    desaturated towards grey at constant luminance instead of clipped."""
    xyz = np.asarray(spec) @ CIE_XYZ
    rgb = xyz @ XYZ_TO_SRGB.T
    Y = xyz[..., 1:2]
    lo = rgb.min(axis=-1, keepdims=True)
    t = np.where(lo < 0, Y / np.maximum(Y - lo, 1e-300), 1.0)
    return Y + (rgb - Y) * np.clip(t, 0, 1)


def densities(h):
    h = np.asarray(h, dtype=float)
    return np.exp(-h / RAY_H), np.exp(-h / MIE_H), np.clip(1 - np.abs(h - OZONE_CENTER) / OZONE_WIDTH, 0, None)


def extinction(h):
    r, m, o = densities(h)
    return r[..., None] * RAY_SCAT + m[..., None] * MIE_EXT + o[..., None] * OZONE_ABS


def _disc(r, mu, R):
    return r * r * (mu * mu - 1) + R * R


def dist_to_top(r, mu):
    return -r * mu + np.sqrt(np.maximum(_disc(r, mu, RT), 0))


def hits_ground(r, mu):
    return (mu < 0) & (_disc(r, mu, RG) >= 0)


def dist_to_ground(r, mu):
    return -r * mu - np.sqrt(np.maximum(_disc(r, mu, RG), 0))


def ray_length(r, mu):
    g = hits_ground(r, mu)
    return np.where(g, dist_to_ground(r, mu), dist_to_top(r, mu)), g


def phase_rayleigh(c):
    return 3 / (16 * np.pi) * (1 + c * c)


def phase_mie(c, g=MIE_G):
    k = 3 / (8 * np.pi) * (1 - g * g) / (2 + g * g)
    return k * (1 + c * c) / (1 + g * g - 2 * g * c) ** 1.5


def optical_depth_to_top(r, mu, n=96):
    """Optical depth (3 channels) from radius r along mu to the top (ray assumed not to hit the ground)."""
    r = np.asarray(r, float)
    mu = np.asarray(mu, float)
    L = dist_to_top(r, mu)
    t = (np.arange(n) + 0.5) / n
    s = L[..., None] * t
    ri = np.sqrt(r[..., None] ** 2 + s * s + 2 * r[..., None] * mu[..., None] * s)
    return extinction(ri - RG).sum(axis=-2) * (L / n)[..., None]


# ---- transmittance LUT (Bruneton & Neyret 2008 parameterisation) -----------
_HTOP = np.sqrt(RT * RT - RG * RG)


def _tlut_uv_to_r_mu(xm, xr):
    rho = _HTOP * xr
    r = np.sqrt(rho * rho + RG * RG)
    dmin = RT - r
    dmax = rho + _HTOP
    d = dmin + xm * (dmax - dmin)
    mu = np.where(d == 0, 1.0, (_HTOP**2 - rho * rho - d * d) / np.maximum(2 * r * d, 1e-12))
    return r, np.clip(mu, -1, 1)


def _tlut_r_mu_to_uv(r, mu):
    rho = np.sqrt(np.maximum(r * r - RG * RG, 0))
    d = dist_to_top(r, mu)
    dmin = RT - r
    dmax = rho + _HTOP
    return (d - dmin) / np.maximum(dmax - dmin, 1e-12), rho / _HTOP


class Transmittance:
    def __init__(self, nm: int = 512, nr: int = 128) -> None:
        xm = np.arange(nm) / (nm - 1)
        xr = np.arange(nr) / (nr - 1)
        XM, XR = np.meshgrid(xm, xr)
        r, mu = _tlut_uv_to_r_mu(XM, XR)
        # optical depth, not transmittance: grazing sunlight changes T by
        # orders of magnitude between columns, tau varies smoothly
        self.tau = optical_depth_to_top(r, mu)  # (nr, nm, 3)
        self.nm, self.nr = nm, nr

    def lookup(self, r, mu):
        """Bilinear lookup; valid for rays that do not hit the ground."""
        u, v = _tlut_r_mu_to_uv(np.asarray(r, float), np.asarray(mu, float))
        x = np.clip(u, 0, 1) * (self.nm - 1)
        y = np.clip(v, 0, 1) * (self.nr - 1)
        x0 = np.clip(np.floor(x).astype(int), 0, self.nm - 2)
        y0 = np.clip(np.floor(y).astype(int), 0, self.nr - 2)
        fx = (x - x0)[..., None]
        fy = (y - y0)[..., None]
        t = self.tau
        a = t[y0, x0] * (1 - fx) + t[y0, x0 + 1] * fx
        b = t[y0 + 1, x0] * (1 - fx) + t[y0 + 1, x0 + 1] * fx
        return np.exp(-(a * (1 - fy) + b * fy))

    def sun(self, r, mu_s):
        """Sunlight reaching radius r with the Sun at cos-zenith mu_s, with the
        Earth's shadow softened over the solar disc."""
        r = np.asarray(r, float)
        mu_s = np.asarray(mu_s, float)
        mu_h = -np.sqrt(np.maximum(0, 1 - (RG / r) ** 2))
        e = SUN_RADIUS * np.sqrt(np.maximum(1 - mu_h * mu_h, 1e-6))
        vis = np.clip((mu_s - (mu_h - e)) / (2 * e), 0, 1)
        vis = vis * vis * (3 - 2 * vis)
        return self.lookup(r, np.maximum(mu_s, mu_h + 1e-4)) * vis[..., None]


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5**0.5) * i
    s = np.sqrt(1 - z * z)
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], -1)


class MultipleScattering:
    """Psi_ms(r, mu_s) of Hillaire (2020), section 5.5: second-order
    isotropic scattering summed as a geometric series 1/(1 - f_ms)."""

    def __init__(self, T: Transmittance, n: int = 32, nmu: int = 128, ndir: int = 256, steps: int = 32) -> None:
        self.n, self.nmu = n, nmu
        h = (np.arange(n) + 0.5) / n * (RT - RG)
        mus = -1 + 2 * (np.arange(nmu) + 0.5) / nmu
        H, MU = np.meshgrid(h, mus, indexing="ij")
        r = RG + H.ravel()
        mu_s = MU.ravel()
        dirs = _fibonacci_sphere(ndir)
        sun = np.stack([np.sqrt(1 - mu_s**2), np.zeros_like(mu_s), mu_s], -1)  # (C, 3)
        C = r.size
        nl = LAMBDA_NM.size
        L2 = np.zeros((C, nl))
        F = np.zeros((C, nl))
        for w in dirs:
            mu = np.full(C, w[2])
            length, ground = ray_length(r, mu)
            dt = length / steps
            Tacc = np.ones((C, nl))
            for k in range(steps):
                s = (k + 0.5) * dt
                p = np.stack([w[0] * s, w[1] * s, r + w[2] * s], -1)
                ri = np.linalg.norm(p, axis=-1)
                mus_i = (p * sun).sum(-1) / ri
                rr, mm, _ = densities(ri - RG)
                sig_s = rr[:, None] * RAY_SCAT + mm[:, None] * MIE_SCAT
                sig_t = extinction(ri - RG)
                Tstep = np.exp(-sig_t * dt[:, None])
                S = sig_s * T.sun(ri, mus_i) / (4 * np.pi)
                L2 += Tacc * (S - S * Tstep) / sig_t
                F += Tacc * (sig_s - sig_s * Tstep) / sig_t
                Tacc *= Tstep
            # light reflected by the ground at the end of the ray
            pg = np.stack([w[0] * length, w[1] * length, r + w[2] * length], -1)
            ng = pg / np.linalg.norm(pg, axis=-1, keepdims=True)
            mug = (ng * sun).sum(-1)
            gl = T.sun(np.full(C, RG + 1e-3), mug) * (GROUND_ALBEDO / np.pi * np.maximum(mug, 0))[:, None]
            L2 += np.where(ground[:, None], Tacc * gl, 0)
        L2 /= ndir
        F /= ndir
        self.table = (L2 / (1 - F)).reshape(n, nmu, nl)
        # deep in the Earth's shadow Psi falls off exponentially with the
        # solar depression: interpolate its logarithm
        self.log = np.log(self.table + 1e-30)

    def lookup(self, r, mu_s):
        x = np.clip((np.asarray(r, float) - RG) / (RT - RG) * self.n - 0.5, 0, self.n - 1)
        y = np.clip((np.asarray(mu_s, float) + 1) / 2 * self.nmu - 0.5, 0, self.nmu - 1)
        x0 = np.clip(np.floor(x).astype(int), 0, self.n - 2)
        y0 = np.clip(np.floor(y).astype(int), 0, self.nmu - 2)
        fx = (x - x0)[..., None]
        fy = (y - y0)[..., None]
        t = self.log
        a = t[x0, y0] * (1 - fy) + t[x0, y0 + 1] * fy
        b = t[x0 + 1, y0] * (1 - fy) + t[x0 + 1, y0 + 1] * fy
        return np.exp(a * (1 - fx) + b * fx)


class Atmosphere:
    def __init__(self) -> None:
        self.T = Transmittance()
        self.M = MultipleScattering(self.T)
        self.r_obs = RG + H_OBS

    def sky(self, alt, daz, sun_alt, steps: int = 48):
        """Sky radiance [G=0 stars / sr], 3 channels, seen by the observer
        looking at altitude `alt`, azimuth `daz` from the Sun's azimuth, with
        the Sun at altitude `sun_alt` (radians; arrays broadcast)."""
        alt, daz, sun_alt = np.broadcast_arrays(np.asarray(alt, float), np.asarray(daz, float), np.asarray(sun_alt, float))
        shape = alt.shape
        alt, daz, sun_alt = alt.ravel(), daz.ravel(), sun_alt.ravel()
        d = np.stack([np.sin(daz) * np.cos(alt), np.cos(daz) * np.cos(alt), np.sin(alt)], -1)
        s = np.stack([np.zeros_like(sun_alt), np.cos(sun_alt), np.sin(sun_alt)], -1)
        cos_t = (d * s).sum(-1)
        pr = phase_rayleigh(cos_t)[:, None]
        pm = phase_mie(cos_t)[:, None]
        r0 = np.full(alt.size, self.r_obs)
        length, ground = ray_length(r0, d[:, 2])
        # march non-uniformly: short steps near the observer where the air is dense
        u = (np.arange(steps + 1) / steps) ** 2
        C = LAMBDA_NM.size
        L = np.zeros((alt.size, C))
        Tacc = np.ones((alt.size, C))
        for k in range(steps):
            s0, s1 = length * u[k], length * u[k + 1]
            dt = (s1 - s0)[:, None]
            sm = 0.5 * (s0 + s1)
            p = d * sm[:, None]
            p[:, 2] += self.r_obs
            ri = np.linalg.norm(p, axis=-1)
            mus_i = (p * s).sum(-1) / ri
            rr, mm, _ = densities(ri - RG)
            sr = rr[:, None] * RAY_SCAT
            sm_ = mm[:, None] * MIE_SCAT
            sig_t = extinction(ri - RG)
            Tstep = np.exp(-sig_t * dt)
            S = self.T.sun(ri, mus_i) * (sr * pr + sm_ * pm) + (sr + sm_) * self.M.lookup(ri, mus_i)
            L += Tacc * (S - S * Tstep) / sig_t
            Tacc *= Tstep
        return spectrum_to_rgb(L * SUN_SPECTRUM).reshape(*shape, 3)

    def transmittance_spectral(self, alt):
        alt = np.asarray(alt, float)
        mu = np.sin(alt)
        t = self.T.lookup(np.full(alt.shape, self.r_obs), mu)
        return np.where(hits_ground(self.r_obs, mu)[..., None], 0.0, t)

    def transmittance_view(self, alt):
        """RGB transmittance from the observer to space at altitude `alt`
        (rad), for sunlight-coloured light: rgb(S T) / rgb(S) per channel."""
        t = self.transmittance_spectral(alt)
        ref = spectrum_to_rgb(SUN_SPECTRUM)
        return spectrum_to_rgb(t * SUN_SPECTRUM) / ref


def surface_brightness_mag(radiance_lum):
    """G=0-stars/sr -> mag/arcsec^2."""
    return -2.5 * np.log10(np.asarray(radiance_lum) / (180 * 3600 / np.pi) ** 2)
