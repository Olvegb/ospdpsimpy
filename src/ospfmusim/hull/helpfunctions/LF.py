"""Low-frequency (LF) 6-DOF vessel motion.

License: GPL-3.0-or-later (see LICENSE and NOTICE.md in this folder)
Authors: Araz, Olve Grønås Birkeland (2026)
"""
import numpy as np


def checked(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return result.copy()


def rotation(angles):
    phi, theta, psi = angles
    cp, ct, cs = np.cos([phi, theta, psi])
    sp, st, ss = np.sin([phi, theta, psi])
    return np.array([
        [cs*ct, cs*st*sp-ss*cp, cs*st*cp+ss*sp],
        [ss*ct, ss*st*sp+cs*cp, ss*st*cp-cs*sp],
        [-st, ct*sp, ct*cp],
    ])


def kinematics(eta):
    phi, theta, _ = eta[3:]
    if abs(np.cos(theta)) < 1e-6:
        raise ValueError("Euler angles are singular near pitch +/-90 degrees")
    s, c, t = np.sin(phi), np.cos(phi), np.tan(theta)
    J = np.zeros((6, 6))
    J[:3, :3] = rotation(eta[3:])
    J[3:, 3:] = [[1., s*t, c*t], [0., c, -s],
                 [0., s/np.cos(theta), c/np.cos(theta)]]
    return J


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def coriolis(mass, velocity):
    momentum = mass @ velocity
    p, h = skew(momentum[:3]), skew(momentum[3:])
    return np.block([[np.zeros((3, 3)), -p], [-p, -h]])


class LF:
    # State: [north, east, down, roll, pitch, yaw, u, v, w, p, q, r]
    # Body axes: forward, starboard, down. Positions: NED, angles: radians.

    def __init__(self, rigid_mass, added_mass, damping, restoring):
        # Inputs from Wamit.py result, passed by hull/vessel.py:
        self.M_RB = checked(rigid_mass, (6, 6), "rigid_mass")  # "rigid_mass": MRB
        self.A = checked(added_mass, (6, 6), "added_mass")    # "added_mass": A(omega_LF)
        self.D = checked(damping, (6, 6), "damping")          # "damping": B(omega_LF) + Bv
        self.G = checked(restoring, (6, 6), "restoring")      # "restoring": C(omega_LF)
        self.M = self.M_RB + self.A
        if np.linalg.matrix_rank(self.M) != 6:
            raise ValueError("Total mass matrix must be invertible")

    # @classmethod
    # def from_shipx(cls, path, *, frequency):
    #     # Legacy JSON loader; Gunnerus .mat is supplied through hull/vessel.py.
    #     import json
    #     from pathlib import Path
    #     data = json.loads(Path(path).read_text())
    #     freqs = np.asarray(data['freqs'], dtype=float)
    #     if (freqs.ndim != 1 or len(freqs) == 0 or not np.isfinite(freqs).all()
    #             or np.any(np.diff(freqs) <= 0)):
    #         raise ValueError('ShipX frequencies must be finite and increasing')
    #     if not np.isfinite(frequency) or not freqs[0] <= frequency <= freqs[-1]:
    #         raise ValueError('Choose a frequency inside the table; '
    #                          'zero-frequency LF coefficients cannot be inferred')
    #     if not np.isclose(np.asarray(data['velocities'])[0], 0):
    #         raise ValueError('Expected zero speed at table index 0')
    #     index = int(np.argmin(abs(freqs-frequency)))
    #     obj = cls(np.asarray(data['MRB']), np.asarray(data['A'])[:, :, index, 0],
    #               np.asarray(data['B'])[:, :, index, 0]+np.asarray(data['Bv']),
    #               np.asarray(data['C'])[:, :, index, 0])
    #     obj.selected_frequency = float(freqs[index])
    #     obj.data_source = str(Path(path).resolve())
    #     return obj

    def derivative(self, state, tau, current_ned=(0., 0., 0.)):
        state = checked(state, (12,), "state")
        tau = checked(tau, (6,), "tau")
        current_ned = checked(current_ned, (3,), "current_ned")
        eta, nu = state[:6], state[6:]
        current_body = rotation(eta[3:]).T @ current_ned
        nu_c = np.r_[current_body, np.zeros(3)]
        nu_r = nu - nu_c
        nu_c_dot = np.r_[-np.cross(nu[3:], current_body), np.zeros(3)]

        # (MRB + A) nu_dot = tau - CRB*nu - CA*nu_r - D*nu_r - G*eta + A*nu_c_dot
        net_load = (tau - coriolis(self.M_RB, nu) @ nu
                    - coriolis(self.A, nu_r) @ nu_r
                    - self.D @ nu_r - self.G @ eta + self.A @ nu_c_dot)
        eta_dot = kinematics(eta) @ nu
        nu_dot = np.linalg.solve(self.M, net_load)
        return np.r_[eta_dot, nu_dot]

    def step(self, state, dt, *, tau_wind=(0.,)*6, tau_thruster=(0.,)*6,
             tau_wave_drift=(0.,)*6, current_ned=(0., 0., 0.),
             tau_external=(0.,)*6):
        # Runtime inputs: 6-DOF body forces/moments [X,Y,Z,K,M,N] and NED current.
        # tau_wave_drift = WaveLoads.py "tau_second_order" (mean already included).
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Finite time and positive step size required")
        state = checked(state, (12,), "state")
        tau = (checked(tau_wind, (6,), "tau_wind")
               + checked(tau_thruster, (6,), "tau_thruster")
               + checked(tau_wave_drift, (6,), "tau_wave_drift")
               + checked(tau_external, (6,), "tau_external"))
        current = checked(current_ned, (3,), "current_ned")

        # RK4: integrate low-frequency motion; first-order wave motion is handled by WF.py.
        k1 = self.derivative(state, tau, current)
        k2 = self.derivative(state + dt*k1/2, tau, current)
        k3 = self.derivative(state + dt*k2/2, tau, current)
        k4 = self.derivative(state + dt*k3, tau, current)
        return checked(state + dt*(k1 + 2*k2 + 2*k3 + k4)/6,
                       (12,), "updated LF state")

    def step_with_waves(self, state, dt, *, time, wave_loads,
                        tau_wind=(0.,)*6, tau_thruster=(0.,)*6,
                        current_ned=(0., 0., 0.), tau_external=(0.,)*6):
        # Alternate caller: calculate second-order drift before the LF step.
        state = checked(state, (12,), "state")
        drift = wave_loads.calculate(time, state[:6]).tau_second_order
        return self.step(state, dt, tau_wind=tau_wind,
                         tau_thruster=tau_thruster, tau_wave_drift=drift,
                         current_ned=current_ned, tau_external=tau_external)
