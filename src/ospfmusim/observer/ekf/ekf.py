"""Extended Kalman filter (EKF) FMU.

Estimates the vessel's LF and WF motion and slowly varying disturbances.

A 3-DOF DP observer (surge, sway, yaw) after Fossen, with 15 states:

    x = [eta, nu, b, xi_1, xi_2]
    eta  = [north, east, psi]   low-frequency (LF) position and heading, NED [m, m, rad]
    nu   = [u, v, r]            LF velocity in the body frame [m/s, m/s, rad/s]
    b    = [b_x, b_y, b_n]      bias in NED [N, N, Nm]: every force the model
                                does not know about (current, wave drift, model errors)
    xi_1, xi_2                  wave-frequency (WF) states, 2 per DOF;
                                the WF motion is eta_w = xi_2

Process model (Fossen, DP observer with a wave model):

    eta_dot  = R(psi) nu
    nu_dot   = M^-1 (tau_thr + tau_wind + R(psi)^T b - D nu)
    b_dot    = -T_b^-1 b                                      (+ noise)
    xi_1_dot = xi_2
    xi_2_dot = -Omega^2 xi_1 - 2 Lambda Omega xi_2            (+ noise)

The WF model is a linear second-order system per DOF, tuned to the wave peak
frequency omega_0 = 2 pi / T_p and a relative damping lambda. The noise on
xi_2 is chosen so the stationary std of eta_w is wave_motion_std.

Measurement model: GNSS position and gyro heading measure the TOTAL motion.

    y = eta + eta_w = H x,   H = [I 0 0 0 I]

The controller gets only the LF estimate, so it does not try to counteract
the oscillating wave motion.

Every time step the filter
  1. predicts the state and covariance one step ahead (continuous-discrete
     EKF): x_dot = f(x), P_dot = F P + P F^T + Q, integrated with RK4, and
  2. corrects the prediction with the new measurement.

Units: SI, angles in radians. The bias is in NED, the forces tau in the body frame.
"""
import tomllib
from math import pi
from pathlib import Path

import numpy as np
from pythonfmu import Fmi2Causality, Fmi2Slave, Fmi2Variability, Real, String

N_STATES = 15
ETA, NU, BIAS, XI1, XI2 = (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15))


def rotation(psi):
    """R(psi): rotates a body-frame vector [x, y, n] into NED."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_derivative(psi):
    """dR/dpsi, used in the Jacobian of the process model."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]])


def wrap_angle(angle):
    """Wraps an angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


INPUTS = (
    # Measurements of the total motion: GNSS [m] and gyro [rad]
    "north_meas", "east_meas", "psi_meas",
    "tau_thr_x", "tau_thr_y", "tau_thr_n",   # thruster forces, body frame [N, N, Nm]
    "tau_wind_x", "tau_wind_y", "tau_wind_n",  # wind forces (feed-forward), body frame [N, N, Nm]
)
OUTPUTS = (
    "north_hat", "east_hat", "psi_hat",      # estimated LF position and heading [m, m, rad]
    "u_hat", "v_hat", "r_hat",               # estimated LF velocity [m/s, m/s, rad/s]
    "b_x_hat", "b_y_hat", "b_n_hat",         # estimated bias in NED [N, N, Nm]
    "wf_north_hat", "wf_east_hat", "wf_psi_hat",  # estimated WF motion eta_w [m, m, rad]
)


class ExtendedKalmanFilter(Fmi2Slave):

    author = "Olve Grønås Birkeland"
    description = ("Extended Kalman filter: 3-DOF DP observer with LF, WF and bias estimates "
                   "(RK4 prediction)")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # PARAMETER: path to the config file.
        self.config_path = ""
        self.register_variable(String("config_path", causality=Fmi2Causality.parameter,
                                      variability=Fmi2Variability.fixed))

        # INPUTS: measurements and known forces.
        for name in INPUTS:
            setattr(self, name, 0.0)
            self.register_variable(Real(name, causality=Fmi2Causality.input))

        # OUTPUTS: the estimates.
        for name in OUTPUTS:
            setattr(self, name, 0.0)
            self.register_variable(Real(name, causality=Fmi2Causality.output))

        # CONSTANTS and filter state: set in exit_initialization_mode.
        self.M_inv = self.D = self.T_b = None   # LF vessel model
        self.omega_0 = self.wave_damping = None  # WF model, per DOF
        self.Q = self.R = None                  # noise covariances
        self.x = self.P = None                  # state estimate and its covariance
        self.H = np.hstack([np.eye(3), np.zeros((3, 9)), np.eye(3)])  # y = eta + xi_2

    def exit_initialization_mode(self):
        if not self.config_path:
            raise ValueError("config_path is not set. Set it in OspSystemStructure.xml.")
        config = tomllib.loads(Path(self.config_path).read_text(encoding="utf-8-sig"))

        # LF vessel model, from [vessel]. Shared with other FMUs that need the same matrices.
        vessel = config["vessel"]
        self.M_inv = np.linalg.inv(np.array(vessel["mass_matrix"], dtype=float))
        self.D = np.array(vessel["damping_matrix"], dtype=float)

        # Filter settings, from [kalman_filter].
        kf = config["kalman_filter"]
        self.T_b = np.array(kf["bias_time_constants"], dtype=float)

        # WF model: peak frequency from the sea state in [waves] (zero speed, so
        # the encounter frequency equals the wave frequency).
        self.omega_0 = np.full(3, 2.0 * pi / float(config["waves"]["tp"]))
        self.wave_damping = np.array(kf["wave_damping_ratio"], dtype=float)
        wave_std = np.array(kf["wave_motion_std"], dtype=float)
        # For xi_2_dot = ... + w with noise intensity q, the stationary variances are
        # var(xi_2) = q / (4 lambda omega_0) and var(xi_1) = var(xi_2) / omega_0^2.
        q_wave = 4.0 * self.wave_damping * self.omega_0 * wave_std ** 2

        self.Q = np.diag(np.concatenate([np.array(kf["process_noise"], dtype=float),
                                         np.zeros(3), q_wave]))
        self.R = np.diag(np.array(kf["measurement_noise"], dtype=float))
        # The WF states start in their stationary distribution.
        self.P = np.diag(np.concatenate([np.array(kf["initial_covariance"], dtype=float),
                                         (wave_std / self.omega_0) ** 2, wave_std ** 2]))
        self.x = np.zeros(N_STATES)

        self.write_outputs()

    def f(self, x, tau):
        """The process model x_dot = f(x, tau)."""
        R = rotation(x[2])
        nu, b, xi1, xi2 = x[NU], x[BIAS], x[XI1], x[XI2]
        return np.concatenate([
            R @ nu,
            self.M_inv @ (tau + R.T @ b - self.D @ nu),
            -b / self.T_b,
            xi2,
            -self.omega_0 ** 2 * xi1 - 2.0 * self.wave_damping * self.omega_0 * xi2,
        ])

    def jacobian(self, x):
        """F = df/dx of the process model."""
        R, dR = rotation(x[2]), rotation_derivative(x[2])
        F = np.zeros((N_STATES, N_STATES))
        F[ETA, 2] = dR @ x[NU]
        F[ETA, NU] = R
        F[NU, 2] = self.M_inv @ (dR.T @ x[BIAS])
        F[NU, NU] = -self.M_inv @ self.D
        F[NU, BIAS] = self.M_inv @ R.T
        F[BIAS, BIAS] = -np.diag(1.0 / self.T_b)
        F[XI1, XI2] = np.eye(3)
        F[XI2, XI1] = -np.diag(self.omega_0 ** 2)
        F[XI2, XI2] = -np.diag(2.0 * self.wave_damping * self.omega_0)
        return F

    def derivatives(self, x, P, tau):
        """Time derivatives of the state and the covariance (continuous-discrete EKF)."""
        F = self.jacobian(x)
        return self.f(x, tau), F @ P + P @ F.T + self.Q

    def predict(self, tau, dt):
        """Moves the estimate and covariance one time step ahead with RK4."""
        x, P = self.x, self.P
        k1x, k1P = self.derivatives(x, P, tau)
        k2x, k2P = self.derivatives(x + 0.5 * dt * k1x, P + 0.5 * dt * k1P, tau)
        k3x, k3P = self.derivatives(x + 0.5 * dt * k2x, P + 0.5 * dt * k2P, tau)
        k4x, k4P = self.derivatives(x + dt * k3x, P + dt * k3P, tau)
        self.x = x + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        self.P = P + dt / 6.0 * (k1P + 2.0 * k2P + 2.0 * k3P + k4P)
        self.x[2] = wrap_angle(self.x[2])
        self.P = 0.5 * (self.P + self.P.T)             # remove round-off asymmetry

    def correct(self, y):
        """Corrects the estimate with a measurement y = [north, east, psi] of the total motion."""
        H = self.H
        innovation = y - H @ self.x
        innovation[2] = wrap_angle(innovation[2])      # 359 deg vs 1 deg is 2 deg, not 358

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)            # Kalman gain
        self.x = self.x + K @ innovation
        self.x[2] = wrap_angle(self.x[2])
        I_KH = np.eye(N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form, keeps P symmetric

    def write_outputs(self):
        values = np.concatenate([self.x[ETA], self.x[NU], self.x[BIAS], self.x[XI2]])
        for name, value in zip(OUTPUTS, values):
            setattr(self, name, float(value))

    def do_step(self, current_time, step_size):
        tau = np.array([self.tau_thr_x + self.tau_wind_x,
                        self.tau_thr_y + self.tau_wind_y,
                        self.tau_thr_n + self.tau_wind_n])
        y = np.array([self.north_meas, self.east_meas, self.psi_meas])

        self.predict(tau, step_size)
        self.correct(y)
        self.write_outputs()
        return True
