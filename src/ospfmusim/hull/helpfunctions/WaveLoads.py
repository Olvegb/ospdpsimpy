"""Wave load calculations adapted from Viljar Simonnes Tveit.
Source: https://github.com/viljarst/DP_cosim/tree/wave-loads/wave_forces
Revision: 668b1c4a419ff9cbdc6ca28d348c5c62301d3bab

Upstream license:
MIT License

Copyright (c) 2026 Viljar Simonnes Tveit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import json
from dataclasses import dataclass

import numpy as np


# ---------------- WAVE DRIFT CONTROL: edit here ----------------
DRIFT_METHOD = "newman"  # "newman" or "sorensen"
# Newman (6.97) handles signed Gunnerus drift coefficients.
# Sorensen (6.99) will reject negative active drift coefficients.
# --------------------------------------------------------------


def selected_drift_method(method=None):
    # Convert the simple choice into the equation identifier used by the model.
    options = {
        "newman": "newman_6_97",
        "sorensen": "sorensen_6_99",
        "newman_6_97": "newman_6_97",
        "sorensen_6_99": "sorensen_6_99",
    }
    method = DRIFT_METHOD if method is None else method
    if method not in options:
        raise ValueError("DRIFT_METHOD must be 'newman' or 'sorensen'")
    return options[method]


def finite_vector(value, name):
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must be a finite one-dimensional array')
    return a.copy()


@dataclass(frozen=True)
class WaveComponents:
    # INPUT: amplitude [m], omega [rad/s], phase and travel direction [rad].
    amplitude: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    direction: np.ndarray

    def __post_init__(self):
        for name in ('amplitude', 'omega', 'phase', 'direction'):
            a = finite_vector(getattr(self, name), name)
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        if len({len(self.amplitude), len(self.omega), len(self.phase), len(self.direction)}) != 1:
            raise ValueError('The four wave arrays must have equal lengths')
        if np.any(self.amplitude < 0) or np.any(self.omega < 0):
            raise ValueError('Wave amplitude and frequency must be nonnegative')
        if np.any((self.amplitude > 0) & (self.omega == 0)):
            raise ValueError('Nonzero wave amplitudes require positive frequency')


def amplitude_phase(amplitude, phase, *, phase_unit, amplitude_mode):
    a, p = np.asarray(amplitude, dtype=float), np.asarray(phase, dtype=float)
    if a.shape != p.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(p)):
        raise ValueError('RAO amplitudes and phases must have equal, finite shapes')
    if phase_unit not in ('rad', 'deg') or amplitude_mode not in ('absolute', 'signed'):
        raise ValueError('Specify phase_unit rad/deg and amplitude_mode absolute/signed')
    if phase_unit == 'deg':
        p = np.deg2rad(p)
    return (np.abs(a) if amplitude_mode == 'absolute' else a) * np.exp(1j*p)


class ResponseTable:
    # Data values: [DOF, wave frequency, wave heading], full 360 degrees.

    def __init__(self, frequencies, headings, values, *, kind):
        self.frequencies = finite_vector(frequencies, 'frequencies')
        self.headings = finite_vector(headings, 'headings')
        self.values = np.asarray(values).copy()
        if kind not in ('rao', 'drift'):
            raise ValueError('Table kind must be rao or drift')
        self.kind = kind
        if (len(self.frequencies) < 2 or self.frequencies[0] <= 0
                or np.any(np.diff(self.frequencies) <= 0)):
            raise ValueError('Need at least two increasing positive table frequencies')
        if (len(self.headings) < 3 or self.headings[0] != 0
                or np.any(np.diff(self.headings) <= 0)
                or not np.pi < self.headings[-1] < 2*np.pi):
            raise ValueError('Need full-circle headings starting at 0, without duplicate 2*pi')
        if (self.values.ndim != 3 or self.values.shape[1:] !=
                (len(self.frequencies), len(self.headings))
                or not np.all(np.isfinite(self.values))):
            raise ValueError('Table must have finite shape (DOF, frequency, heading)')
        if kind == 'drift' and np.iscomplexobj(self.values):
            raise ValueError('Drift coefficients must be real')

    def at_frequencies(self, omega):
        # RAOs hold below table, zero above. Drift tends to zero at omega=0, holds above.
        w = finite_vector(omega, 'omega')
        if np.any(w < 0):
            raise ValueError('Frequency must be nonnegative')
        f, v = self.frequencies, self.values
        hi = np.clip(np.searchsorted(f, w, side='right'), 1, len(f)-1)
        lo = hi - 1
        weight = np.clip((w-f[lo])/(f[hi]-f[lo]), 0, 1)
        out = v[:, lo, :]*(1-weight)[None, :, None] + v[:, hi, :]*weight[None, :, None]
        if self.kind == 'rao':
            out[:, w > f[-1], :] = 0
        else:
            below = w < f[0]
            out[:, below, :] *= (w[below]/f[0])[None, :, None]
        return out

    def at_headings(self, frequency_values, beta):
        b = finite_vector(beta, 'beta') % (2*np.pi)
        v = np.asarray(frequency_values)
        if v.shape != (self.values.shape[0], len(b), len(self.headings)):
            raise ValueError('Frequency cache does not match component headings')
        h = self.headings
        lo = np.searchsorted(h, b, side='right')-1
        hi = (lo+1) % len(h)
        upper = np.where(hi == 0, 2*np.pi, h[hi])
        weight = (b-h[lo])/(upper-h[lo])
        j = np.arange(len(b))
        return v[:, j, lo]*(1-weight) + v[:, j, hi]*weight


def spatial_phase(components, time, eta, gravity):
    distance = eta[0]*np.cos(components.direction) + eta[1]*np.sin(components.direction)
    return components.omega*time - components.omega**2/gravity*distance - components.phase


def check_pose_time(time, eta):
    pose = finite_vector(eta, 'eta_lf')
    if pose.shape != (6,) or not np.isfinite(time):
        raise ValueError('Need finite time and six LF pose values')
    return pose


@dataclass(frozen=True)
class WaveLoadResult:
    # OUTPUT: six body loads [X,Y,Z,K,M,N]; only X, Y, N are modelled.
    tau_first_order: np.ndarray   # Diagnostic first-order excitation (Force RAOs).
    tau_second_order: np.ndarray  # Mean + slowly varying drift -> LF.py (once).
    tau_mean: np.ndarray          # Diagnostic, already included in tau_second_order.


# --- Viljar: wave_forces/vessel_data.py ---
DOFS = ("surge", "sway", "yaw")


def interpolate_frequency(x_table, table, x_new, left, right):
    x_new = np.asarray(x_new, dtype=float)
    nf = len(x_table)
    i = np.clip(np.searchsorted(x_table, x_new) - 1, 0, nf - 2)
    w = (x_new - x_table[i]) / (x_table[i + 1] - x_table[i])
    out = table[:, i, :] * (1.0 - w)[None, :, None] + table[:, i + 1, :] * w[None, :, None]

    below = x_new < x_table[0]
    above = x_new > x_table[-1]
    for mask, rule, edge in ((below, left, 0), (above, right, -1)):
        if mask.any():
            out[:, mask, :] = 0.0 if rule == "zero" else table[:, [edge], :]
    return out


class VesselData:

    def __init__(self, data_file, speed_index, force_rao_rows, drift_rows, rao_amplitude):
        if rao_amplitude not in ("absolute", "signed"):
            raise ValueError("rao_amplitude must be 'absolute' or 'signed', "
                             f"got '{rao_amplitude}'")
        with open(data_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        freqs = np.asarray(raw["freqs"], dtype=float)
        headings = np.asarray(raw["headings"], dtype=float) % (2.0 * np.pi)
        order = np.argsort(headings)
        if np.any(np.diff(freqs) <= 0):
            raise ValueError(f"{data_file}: the frequencies must be increasing")

        # Older JSON/ShipX loading path; Gunnerus uses in-memory ResponseTables.
        rao_rows = [force_rao_rows[dof] for dof in DOFS]
        drift_row_list = [drift_rows[dof] for dof in DOFS]
        amp = np.asarray(raw["forceRAO"]["amp"], dtype=float)[rao_rows, :, :, speed_index]
        phase_deg = np.asarray(raw["forceRAO"]["phase"], dtype=float)
        phase = np.deg2rad(phase_deg)[rao_rows, :, :, speed_index]
        drift = np.asarray(raw["driftfrc"]["amp"], dtype=float)[drift_row_list, :, :, speed_index]
        if rao_amplitude == "absolute":
            amp = np.abs(amp)

        self.freqs = freqs                                     # (nf,)  [rad/s]
        self.headings = headings[order]                        # (nh,)  [rad]
        self.rao = (amp * np.exp(1j * phase))[:, :, order]     # (3, nf, nh) [N/m, Nm/m]
        self.drift = drift[:, :, order]                        # (3, nf, nh) [N/m^2, Nm/m^2]

    def tables_at(self, frequencies):
        rao = interpolate_frequency(self.freqs, self.rao, frequencies, left="hold", right="zero")
        # Drift should tend to zero for very long waves (omega -> 0).
        f0 = np.concatenate(([0.0], self.freqs))
        d0 = np.concatenate((np.zeros_like(self.drift[:, :1, :]), self.drift), axis=1)
        drift = interpolate_frequency(f0, d0, frequencies, left="hold", right="hold")
        return rao, drift

    def heading_weights(self, beta):
        h = self.headings
        nh = len(h)
        beta = (np.asarray(beta, dtype=float) - h[0]) % (2.0 * np.pi) + h[0]
        h_ext = np.append(h, h[0] + 2.0 * np.pi)
        lo = np.clip(np.searchsorted(h_ext, beta, side="right") - 1, 0, nh - 1)
        w = (beta - h_ext[lo]) / (h_ext[lo + 1] - h_ext[lo])
        hi = (lo + 1) % nh
        return lo, hi, w


# --- Viljar: wave_forces/model.py ---
class WaveLoadModel:
    # Calculate first-order wave excitation and second-order drift in surge, sway, yaw.

    def __init__(self, vessel, gravity, *, drift_method=None,
                 remove_sum_frequency=True):
        self.drift_method = selected_drift_method(drift_method)
        self.remove_sum_frequency = bool(remove_sum_frequency)
        self.vessel = vessel
        self.gravity = gravity
        self.frequencies = None
        self.set_waves([], [], [], [])  # No waves until set_waves() is called.

    def set_waves(self, amplitudes, frequencies, phases, directions):
        # INPUT: wave amplitude [m], frequency [rad/s], phase [rad], direction TO [rad].
        frequencies = np.asarray(frequencies, dtype=float)
        if self.frequencies is None or not np.array_equal(frequencies, self.frequencies):
            self.frequencies = frequencies
            self._rao, self._drift = self.vessel.tables_at(frequencies)
            self._k = frequencies**2 / self.gravity
            self._idx = np.arange(len(frequencies))
        self.amplitudes = np.asarray(amplitudes, dtype=float)
        self.phases = np.asarray(phases, dtype=float)
        self.directions = np.asarray(directions, dtype=float)

    def _interpolate_heading(self, table, beta):
        lo, hi, w = self.vessel.heading_weights(beta)
        j = self._idx
        return table[:, j, lo] * (1.0 - w) + table[:, j, hi] * w

    def first_order(self, t, eta):
        # FORMULA: tau_1 = Re{conj(H_force) * exp(i*phase)} @ amplitudes.
        if len(self.amplitudes) == 0:
            return np.zeros(3)
        north, east, psi = eta
        H = self._interpolate_heading(self._rao, self.directions - psi)
        arg = (self.frequencies * t
               - self._k * (north * np.cos(self.directions) + east * np.sin(self.directions))
               - self.phases)
        return (np.conj(H) * np.exp(1j * arg)).real @ self.amplitudes

    def drift(self, t, eta):
        # INPUT: Gunnerus drift coefficients T and the current wave components.
        if len(self.amplitudes) == 0:
            return np.zeros(3)
        psi = eta[2]
        T = self._interpolate_heading(self._drift, self.directions - psi)
        e = self.amplitudes * np.exp(1j * (self.frequencies * t - self.phases))

        # FORMULA: Newman (6.97), accepts signed Gunnerus drift coefficients.
        if self.drift_method == "newman_6_97":
            return ((T * e).sum(axis=1) * np.conj(e.sum())).real

        # FORMULA: Sorensen (6.99), requires nonnegative active drift coefficients.
        active = self.amplitudes > 0
        T_active = T[:, active]
        tolerance = 1e-12 * max(1.0, float(np.max(np.abs(T_active), initial=0)))
        if np.any(T_active < -tolerance):
            raise ValueError(
                "Sørensen Eq. 6.99 requires nonnegative wave-drift coefficients "
                "for every active component. This vessel has negative coefficients. "
                "Use drift_method='newman_6_97' for signed Tjj or provide "
                "a physically validated signed-QTF approximation."
            )

        # Clip inactive-wave / roundoff negatives only; never take abs(T) for active waves.
        nonnegative_T = np.maximum(np.where(active[None, :], T, 0.0), 0.0)
        z = np.sum(np.sqrt(nonnegative_T) * e[None, :], axis=1)
        if self.remove_sum_frequency:
            # FORMULA: |sum(A_j sqrt(T_j) exp(i*phase_j))|^2; mean + difference terms.
            return np.abs(z)**2
        # FORMULA: literal 6.99 = 2*Re(sum(A_j sqrt(T_j) exp(i*phase_j)))^2.
        return 2.0 * np.real(z)**2

    def mean_drift(self, psi):
        # FORMULA: mean = T @ amplitude^2 (diagnostic, do not add to LF twice).
        if len(self.amplitudes) == 0:
            return np.zeros(3)
        T = self._interpolate_heading(self._drift, self.directions - psi)
        return T @ self.amplitudes**2

    def loads(self, t, eta):
        return self.first_order(t, eta), self.drift(t, eta)


def build_model(p):
    vessel = VesselData(
        p["vessel_data_file"], int(p["speed_index"]),
        force_rao_rows={"surge": int(p["force_rao_row_surge"]),
                        "sway": int(p["force_rao_row_sway"]),
                        "yaw": int(p["force_rao_row_yaw"])},
        drift_rows={"surge": int(p["drift_row_surge"]),
                    "sway": int(p["drift_row_sway"]),
                    "yaw": int(p["drift_row_yaw"])},
        rao_amplitude=p["rao_amplitude"],
    )
    return WaveLoadModel(
        vessel, float(p["gravity"]),
        drift_method=p.get("drift_method"),
        remove_sum_frequency=p.get("remove_sum_frequency", True),
    )


class _ResponseTableVessel:

    def __init__(self, force_table, drift_table):
        self.force_table = force_table
        self.drift_table = drift_table
        self.headings = force_table.headings
        if not np.array_equal(self.headings, drift_table.headings):
            raise ValueError('Force and drift tables must share a heading grid')

    def tables_at(self, frequencies):
        return (self.force_table.at_frequencies(frequencies),
                self.drift_table.at_frequencies(frequencies))

    heading_weights = VesselData.heading_weights


class WaveLoads:
    # Adapter between original Viljar wave model and hull/vessel.py.

    def __init__(self, force_table, drift_table, *, gravity=9.81,
                 drift_method=None, remove_sum_frequency=True):
        # INPUT from Wamit.py: result["force_table"] (X,Y,N Force RAOs).
        # INPUT from Wamit.py: result["drift_table"] (X,Y,N drift coefficients).
        # Also gravity from Wamit.py; drift method comes from the control panel above.
        if (force_table.kind != 'rao' or drift_table.kind != 'drift'
                or force_table.values.shape[0] != 3 or drift_table.values.shape[0] != 3):
            raise ValueError('Supply X,Y,N force-RAO and real drift tables')
        self.force_table, self.drift_table = force_table, drift_table
        self._initialize(_ResponseTableVessel(force_table, drift_table), gravity,
                         drift_method, remove_sum_frequency)

    def _initialize(self, vessel, gravity, drift_method, remove_sum_frequency):
        if not np.isfinite(gravity) or gravity <= 0:
            raise ValueError('gravity must be positive')
        self.gravity = gravity
        self.model = WaveLoadModel(
            vessel, gravity, drift_method=drift_method,
            remove_sum_frequency=remove_sum_frequency,
        )
        self.set_components(WaveComponents([], [], [], []))

    # @classmethod
    # def from_shipx(cls, path, *, phase_unit, amplitude_mode, gravity=9.81,
    #                drift_method=None, remove_sum_frequency=True):
    #     if phase_unit != 'deg':
    #         raise ValueError("Viljar's vessel loader requires phase_unit='deg'")
    #     raw = json.loads(Path(path).read_text())
    #     if not np.isclose(np.asarray(raw['velocities'])[0], 0):
    #         raise ValueError('This implementation requires zero-speed data in entry 0')
    #     vessel = VesselData(str(path), 0,
    #                         {'surge': 0, 'sway': 1, 'yaw': 5},
    #                         {'surge': 0, 'sway': 1, 'yaw': 2}, amplitude_mode)
    #     obj = cls.__new__(cls)
    #     obj._initialize(vessel, gravity, drift_method, remove_sum_frequency)
    #     return obj

    def set_components(self, components):
        # INPUT via hull/vessel.py: WaveComponents(amplitude, omega, phase, direction).
        if not isinstance(components, WaveComponents):
            raise TypeError('Expected WaveComponents')
        self.components = components
        self.model.set_waves(components.amplitude, components.omega,
                             components.phase, components.direction)

    def calculate(self, time, eta_lf):
        # INPUT: simulation time [s], LF pose (north/east/yaw used for wave loading).
        eta = check_pose_time(time, eta_lf)
        first, drift = self.model.loads(time, (eta[0], eta[1], eta[5]))
        mean = self.model.mean_drift(eta[5])
        outputs = []
        for value in (first, drift, mean):
            load = np.zeros(6)
            load[[0, 1, 5]] = value
            if not np.all(np.isfinite(load)):
                raise ValueError('Wave calculation produced nonfinite loads')
            outputs.append(load)
        # OUTPUT: tau_first_order (diagnostic), tau_second_order (sent to LF),
        # tau_mean (diagnostic).
        return WaveLoadResult(*outputs)
