"""Tests of the ExtendedKalmanFilter FMU code (observer/ekf/ekf.py), without OSP.

A simple "true" 3-DOF vessel, M nu_dot + D nu = tau + R^T b, is simulated in
plain Python with the matrices in the config the simulation uses, a constant
disturbance b in NED and a made-up wave-frequency motion added on top. The
filter gets the total motion as measurement and must split it into LF and WF.

  1. The LF estimate follows the true LF motion, and is much smoother than
     the measurement (the WF motion is filtered out).
  2. The WF estimate follows the true WF motion.
  3. The bias estimate converges to the true disturbance.
  4. Heading wrap: a heading near +-pi gives no jump in the estimate.

License: MIT (see LICENSE in the repository root)
Authors: Olve Grønås Birkeland (2026)
"""
import tomllib

import numpy as np
import pytest

from app.build.data import default_config
from ospfmusim.observer.ekf.ekf import ExtendedKalmanFilter, rotation, wrap_angle

CONFIG = default_config()   # the config the simulation uses (OspSystemStructure.xml)
DT = 0.1


def make_filter():
    kf = ExtendedKalmanFilter(instance_name="check", resources=".")
    kf.config_path = str(CONFIG)
    kf.exit_initialization_mode()
    return kf


def wave_motion(t, omega_0):
    """A made-up WF motion: two harmonics near the peak frequency [m, m, rad]."""
    s = np.sin(omega_0 * t) + 0.5 * np.sin(1.3 * omega_0 * t + 1.0)
    return np.array([0.4 * s, 0.6 * s, 0.008 * s])


def simulate(duration, psi_0=0.0, bias_ned=(2.0e4, -3.0e4, 1.0e5)):
    """Runs truth and filter side by side. Returns time, truth and estimates."""
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    M = np.array(config["vessel"]["mass_matrix"], dtype=float)
    D = np.array(config["vessel"]["damping_matrix"], dtype=float)
    omega_0 = 2.0 * np.pi / config["waves"]["tp"]
    b = np.array(bias_ned)

    kf = make_filter()
    kf.x[2] = psi_0                     # start the estimate at the true heading
    eta, nu = np.array([0.0, 0.0, psi_0]), np.zeros(3)

    # A simple P controller on the true LF position keeps the vessel near the origin.
    Kp, Kd = np.diag([2.0e4, 2.0e4, 1.0e7]), np.diag([5.0e5, 6.0e5, 3.0e8])

    log = {k: [] for k in ("t", "eta", "eta_w", "eta_hat", "eta_w_hat", "b_hat", "y")}
    for k in range(int(duration / DT)):
        t = k * DT
        err = np.r_[eta[:2], wrap_angle(eta[2] - psi_0)]
        tau = -rotation(eta[2]).T @ (Kp @ err) - Kd @ nu

        # Truth: integrate the LF vessel with small RK4 steps.
        def f(state):
            e, n = state[:3], state[3:]
            R = rotation(e[2])
            return np.r_[R @ n, np.linalg.solve(M, tau + R.T @ b - D @ n)]

        state = np.r_[eta, nu]
        for _ in range(10):
            h = DT / 10
            k1 = f(state)
            k2 = f(state + h / 2 * k1)
            k3 = f(state + h / 2 * k2)
            k4 = f(state + h * k3)
            state = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        eta, nu = state[:3], state[3:]
        eta[2] = wrap_angle(eta[2])

        eta_w = wave_motion(t + DT, omega_0)
        y = eta + eta_w
        y[2] = wrap_angle(y[2])

        kf.north_meas, kf.east_meas, kf.psi_meas = y
        kf.tau_thr_x, kf.tau_thr_y, kf.tau_thr_n = tau
        kf.do_step(t, DT)

        log["t"].append(t + DT)
        log["eta"].append(eta.copy())
        log["eta_w"].append(eta_w)
        log["y"].append(y)
        log["eta_hat"].append([kf.north_hat, kf.east_hat, kf.psi_hat])
        log["eta_w_hat"].append([kf.wf_north_hat, kf.wf_east_hat, kf.wf_psi_hat])
        log["b_hat"].append([kf.b_x_hat, kf.b_y_hat, kf.b_n_hat])
    return {k: np.array(v) for k, v in log.items()}, b


def rms(x):
    return np.sqrt(np.mean(x ** 2, axis=0))


@pytest.fixture(scope="module")
def ten_minutes():
    """Ten minutes of truth and filter. Returns the last five minutes, and the true bias."""
    log, b = simulate(600.0)
    late = log["t"] > 300.0
    return {k: v[late] for k, v in log.items()}, b


def test_lf_estimate_follows_the_lf_motion(ten_minutes):
    log, _ = ten_minutes
    lf_error = log["eta_hat"] - log["eta"]
    lf_error[:, 2] = wrap_angle(lf_error[:, 2])
    assert np.all(rms(lf_error) < 0.3 * rms(log["eta_w"])), \
        "the LF estimate still contains much of the WF motion"


def test_wf_estimate_follows_the_wf_motion(ten_minutes):
    log, _ = ten_minutes
    wf_error = log["eta_w_hat"] - log["eta_w"]
    assert np.all(rms(wf_error) < 0.3 * rms(log["eta_w"])), \
        "the WF estimate does not follow the WF motion"


def test_bias_estimate_converges(ten_minutes):
    log, b = ten_minutes
    b_hat = log["b_hat"].mean(axis=0)
    assert np.all(np.abs(b_hat - b) < 0.15 * np.abs(b)), "the bias estimate is off by more than 15 %"


def test_heading_near_pi_gives_no_jump():
    log, _ = simulate(120.0, psi_0=np.pi - 0.005)
    jumps = np.abs(np.diff(np.unwrap(log["eta_hat"][:, 2])))
    assert jumps.max() < 0.05, "psi_hat jumps when the heading crosses +-pi"
