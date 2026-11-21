"""Binary node payload shared by the pipeline (static files) and the server
(filtered streams). See web/src/net/nodeFormat.ts for the decoder.

STM1: header "STM1" | u32 node | u32 count | count x 22-byte records
STM2: same records followed by count x u16 rank (index of each star in the
      unfiltered node), used when the server filters a node.
record: f32 x,y,z (pc, ICRS J2016) | f16 vx,vy,vz (km/s) | f16 absG | u8 colour | u8 flags
"""

from __future__ import annotations

import struct

import numpy as np

REC = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("vx", "<f2"), ("vy", "<f2"), ("vz", "<f2"),
                ("absg", "<f2"), ("col", "u1"), ("flags", "u1")])
COLOR_MIN, COLOR_STEP = -0.6, 0.02
KMS_TO_PCYR = 1.022712165045695e-06


def color_byte(bp_rp: np.ndarray) -> np.ndarray:
    c = np.where(np.isfinite(bp_rp), bp_rp, 0.8)
    return np.clip(np.round((c - COLOR_MIN) / COLOR_STEP), 0, 255).astype(np.uint8)


def encode(node: int, x, y, z, vx_kms, vy_kms, vz_kms, absg, bp_rp, flags, ranks=None) -> bytes:
    n = len(x)
    rec = np.empty(n, dtype=REC)
    rec["x"], rec["y"], rec["z"] = x, y, z
    rec["vx"], rec["vy"], rec["vz"] = vx_kms, vy_kms, vz_kms
    rec["absg"] = absg
    rec["col"] = color_byte(np.asarray(bp_rp, dtype=float))
    rec["flags"] = np.asarray(flags).astype(np.uint8)
    if ranks is None:
        return b"STM1" + struct.pack("<II", node, n) + rec.tobytes()
    return b"STM2" + struct.pack("<II", node, n) + rec.tobytes() + np.asarray(ranks, dtype="<u2").tobytes()
