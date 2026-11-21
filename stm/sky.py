"""Observer's sky: Sun by solar longitude, horizon coordinates, meridian.

The historical sky avoids civil calendars and Delta T on purpose. A moment
is given as
  * ``epj``   - Julian epoch (astronomical year numbering, see calendar.py),
  * ``lam``   - the Sun's mean-ecliptic longitude *of date* (0 = March
                equinox, 90 deg = June solstice, ...), i.e. the 节气,
  * ``h_sun`` - the Sun's hour angle (0 = local apparent noon), i.e. the
                local apparent solar time,
  * ``lat``   - the observer's geographic latitude.
That is exactly the information an ancient observation records ("at dusk
on the day of the summer solstice, Huo culminates"), so the reconstruction
does not depend on how that year maps to a modern date.

Approximations (documented in docs/method.md): mean equator/ecliptic of
date (nutation <= 17" ignored), no annual aberration (<= 20.5") or
refraction, Sun on the ecliptic.
"""

from __future__ import annotations

import math

import numpy as np

from . import precession as pr


def sun_dir_of_date(epj: float, lam_rad: float) -> np.ndarray:
    """Sun's unit vector in the mean equatorial frame of date."""
    eps = pr.obliquity(epj)
    return np.array([math.cos(lam_rad), math.sin(lam_rad) * math.cos(eps), math.sin(lam_rad) * math.sin(eps)])


def radec_of(v) -> tuple[float, float]:
    v = np.asarray(v, dtype=float)
    return math.atan2(v[1], v[0]) % (2 * math.pi), math.asin(v[2] / np.linalg.norm(v))


def sun_radec_of_date(epj: float, lam_rad: float) -> tuple[float, float]:
    return radec_of(sun_dir_of_date(epj, lam_rad))


def sidereal_angle(epj: float, lam_rad: float, h_sun_rad: float) -> float:
    """Local sidereal angle (hour angle of the equinox of date)."""
    a_sun, _ = sun_radec_of_date(epj, lam_rad)
    return (a_sun + h_sun_rad) % (2 * math.pi)


def hour_angle_at_altitude(dec_rad: float, lat_rad: float, alt_rad: float) -> float | None:
    """Positive hour angle (rad) at which a body of declination ``dec``
    reaches altitude ``alt`` (afternoon side). None if it never does."""
    c = (math.sin(alt_rad) - math.sin(lat_rad) * math.sin(dec_rad)) / (
        math.cos(lat_rad) * math.cos(dec_rad)
    )
    if c < -1 or c > 1:
        return None
    return math.acos(c)


def dusk_hour_angle(epj: float, lam_rad: float, lat_rad: float, sun_alt_rad: float) -> float:
    """Sun's hour angle at dusk, defined by the Sun's altitude below horizon."""
    _, d = sun_radec_of_date(epj, lam_rad)
    h = hour_angle_at_altitude(d, lat_rad, sun_alt_rad)
    if h is None:
        raise ValueError("Sun never reaches that altitude on this day")
    return h


def date_frame_to_horizon(lst: float, lat_rad: float) -> np.ndarray:
    """Rotation: mean equatorial of date -> horizon (x=N, y=E... see below).

    Horizon frame axes: x = south meridian point... We use
    (east, north, up) so that azimuth = atan2(east, north).
    """
    # hour-angle frame: rotate about z by local sidereal angle
    c, s = math.cos(lst), math.sin(lst)
    to_ha = np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, 1.0]])  # (x toward meridian, y toward west)
    sl, cl = math.sin(lat_rad), math.cos(lat_rad)
    # hour-angle (x: H=0 on equator, y: H=-6h? -> west positive) to (east, north, up)
    ha_to_enu = np.array([
        [0.0, -1.0, 0.0],
        [-sl, 0.0, cl],
        [cl, 0.0, sl],
    ])
    return ha_to_enu @ to_ha


def alt_az(v_date: np.ndarray, lst: float, lat_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Altitude and azimuth (rad, azimuth from north through east)."""
    enu = np.atleast_2d(v_date) @ date_frame_to_horizon(lst, lat_rad).T
    enu = enu / np.linalg.norm(enu, axis=-1, keepdims=True)
    return np.arcsin(np.clip(enu[:, 2], -1, 1)), np.arctan2(enu[:, 0], enu[:, 1]) % (2 * math.pi)


def culminating_ra(epj: float, lam_rad: float, h_sun_rad: float) -> float:
    """Right ascension of date on the meridian (rad) = local sidereal angle."""
    return sidereal_angle(epj, lam_rad, h_sun_rad)
