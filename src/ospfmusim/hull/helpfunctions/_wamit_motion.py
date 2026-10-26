"""Frequency/heading interpolation of already processed MSS vessel RAOs.

This module DOES NOT read raw WAMIT files, transform axes, or create JSON.
All input arrays are already in the simulator's body frame, in SI units,
using a complex phase-lag convention. Interpolate complex Cartesian values
instead of wrapped phase angles.

License: GPL-3.0-or-later (see LICENSE and NOTICE.md in this folder)
Authors: Araz, Olve Grønås Birkeland (2026)
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionRAOs:
    omega: np.ndarray         # (nf,), rad/s, increasing
    heading_deg: np.ndarray   # (nh,), deg, increasing in [0, 360)
    values: np.ndarray        # (ndof, nf, nh), complex or real


def make_frequency_grid(table: MotionRAOs, count: int = 50,
                        spacing: str = "log") -> np.ndarray:
    """Select points WITHIN the original calculated frequency range."""
    if count < 2:
        raise ValueError("At least two frequency points are required")
    lo, hi = float(table.omega[0]), float(table.omega[-1])
    if lo <= 0 or hi <= lo:
        raise ValueError("Input frequencies must be positive and increasing")
    if spacing == "log":
        return np.geomspace(lo, hi, count)
    if spacing == "linear":
        return np.linspace(lo, hi, count)
    raise ValueError("spacing must be log or linear")


def interpolate(table: MotionRAOs, target_omega: np.ndarray,
                target_heading: np.ndarray) -> np.ndarray:
    """Interpolate existing table to (DOF, nfreq, nheading), periodic in angle.

    Missing directional coverage is an error, not a license to invent RAOs.
    """
    f = np.asarray(table.omega, dtype=float)
    h = np.asarray(table.heading_deg, dtype=float)
    v = np.asarray(table.values)
    w = np.asarray(target_omega, dtype=float)
    a = np.asarray(target_heading, dtype=float)
    if (f.ndim != 1 or len(f) < 2 or not np.isfinite(f).all()
            or f[0] <= 0 or np.any(np.diff(f) <= 0)):
        raise ValueError("Source frequencies must be positive and increasing")
    if (h.ndim != 1 or len(h) < 3 or not np.isfinite(h).all()
            or h[0] < 0 or h[-1] >= 360 or np.any(np.diff(h) <= 0)
            or max(np.diff(np.r_[h, h[0] + 360])) > 90):
        raise ValueError("Full-circle source headings are required")
    if (v.ndim != 3 or v.shape[1:] != (len(f), len(h))
            or not np.isfinite(v).all()):
        raise ValueError("RAO table must be (DOF, frequency, heading) and finite")
    if (w.ndim != 1 or len(w) < 2 or not np.isfinite(w).all()
            or w[0] < f[0] - 1e-12 or w[-1] > f[-1] + 1e-12
            or np.any(np.diff(w) <= 0)):
        raise ValueError("Target frequencies must be increasing and within source data")
    if a.ndim != 1 or len(a) < 1 or not np.isfinite(a).all():
        raise ValueError("Target headings must be finite")

    lo = np.clip(np.searchsorted(f, w, side="right") - 1, 0, len(f) - 2)
    t = (w - f[lo]) / (f[lo + 1] - f[lo])
    vf = v[:, lo, :] * (1 - t)[None, :, None] + v[:, lo + 1, :] * t[None, :, None]

    angles = (a - h[0]) % 360 + h[0]
    lower_i = np.clip(np.searchsorted(h, angles, side="right") - 1, 0, len(h) - 1)
    upper_i = (lower_i + 1) % len(h)
    lower_h = h[lower_i]
    upper_h = np.where(upper_i == 0, h[0] + 360, h[upper_i])
    t_h = (angles - lower_h) / (upper_h - lower_h)
    return (vf[:, :, lower_i] * (1 - t_h)[None, None, :]
            + vf[:, :, upper_i] * t_h[None, None, :])


def checked_matrix(values, shape, name):
    """Validate an MSS matrix; the optional final axis holds vessel speeds."""
    a = np.asarray(values, dtype=float)
    if a.ndim == 4:
        a = a[:, :, :, 0]  # zero forward speed
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected {shape}, received {a.shape}")
    return a


def read_response(vessel, name, signs, headings):
    """Read Gunnerus/tanker MSS response cells at zero forward speed.

    Gunnerus stores 19 heading columns (0..180) but lists 36 in metadata.
    Fill 190..350 by explicitly assuming port/starboard hull symmetry.
    """
    raw = vessel[name]
    omega = np.asarray(raw["w"], dtype=float).ravel()
    n = len(signs)

    def first_speed(cell):
        a = np.asarray(cell, dtype=float)
        return a[:, :, 0] if a.ndim == 3 else a

    if len(raw["amp"]) < n:
        raise ValueError(f"{name}: missing DOFs")
    amplitude = np.stack([first_speed(c) for c in raw["amp"][:n]])
    values = amplitude
    if "phase" in raw:
        phase = np.stack([first_speed(c) for c in raw["phase"][:n]])
        values = amplitude * np.exp(-1j * phase)

    if values.shape[-1] == 19:
        if not np.allclose(headings, np.arange(0, 360, 10)):
            raise ValueError(f"{name}: unexpected 19-column heading convention")
        mirror = values[:, :, 17:0:-1] * np.asarray(signs)[:, None, None]
        values = np.concatenate((values, mirror), axis=2)

    if values.shape != (n, len(omega), len(headings)):
        raise ValueError(f"{name}: unexpected response shape {values.shape}")
    if name == "driftfrc" and any(np.any(np.asarray(c)) for c in raw["amp"][3:]):
        raise ValueError("driftfrc: nonzero extra DOFs")
    return MotionRAOs(omega, headings, values)


def interpolate_matrices(frequencies, matrix, target):
    """Linearly interpolate the 36 coefficients of a 6x6 hydrodynamic table."""
    return np.array([
        [np.interp(target, frequencies, matrix[i, j]) for j in range(6)]
        for i in range(6)
    ])
