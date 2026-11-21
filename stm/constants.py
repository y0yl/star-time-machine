"""Physical constants and adopted model parameters (single source of truth).

The same numbers are exported to ``web/src/astro/constants.gen.ts`` by
``pipeline/export_constants.py`` so the browser and server never drift.
"""

import math

# --- units -------------------------------------------------------------------
DAS2R = math.pi / (180.0 * 3600.0)  # arcsec -> rad
DEG = math.pi / 180.0
AU_M = 149_597_870_700.0  # IAU 2012
PC_AU = 648_000.0 / math.pi  # 1 pc in au
PC_KM = PC_AU * AU_M / 1000.0
JULIAN_YEAR_S = 365.25 * 86400.0
C_KMS = 299_792.458
# km/s -> pc per Julian year
KMS_TO_PCYR = JULIAN_YEAR_S / PC_KM
# speed of light in pc per Julian year
C_PCYR = C_KMS * KMS_TO_PCYR
# mas/yr at 1 pc -> km/s  (4.74047...)
K_PM = AU_M / 1000.0 / JULIAN_YEAR_S

# --- epochs ------------------------------------------------------------------
GAIA_EPOCH = 2016.0  # Gaia DR3 reference epoch (Julian year, TCB)
HIP_EPOCH = 1991.25

# --- Galaxy model (astropy Galactocentric v4.0 defaults + galpy MWPotential2014)
R0_PC = 8122.0  # GRAVITY Collaboration 2018
Z_SUN_PC = 20.8  # Bennett & Bovy 2019
V_SUN_KMS = (12.9, 245.6, 7.78)  # Drimmel & Poggio 2018 / Reid & Brunthaler 2004
# MWPotential2014 (Bovy 2015) rescaled so that v_c(R0) = VC0
VC0_KMS = 233.1  # McMillan 2017; leaves V_sun - v_c = 12.5 km/s (Schonrich+ 2010: 12.24)
GALCEN_RA_DEG = 266.4051
GALCEN_DEC_DEG = -28.936175
