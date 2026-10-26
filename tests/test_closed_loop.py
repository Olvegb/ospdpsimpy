"""Tests that the DP controller can hold a vessel, WITHOUT OSP and without a hull FMU.

A very simple 3-DOF vessel, only for this test (not the hull model):

    eta_dot = R(psi) nu
    M nu_dot + D nu = tau + R(psi)^T b

M and D from [vessel] in the config the simulation uses, b a constant force in
NED (current, wave drift, wind). The controller gets the true position and
velocity (no Kalman filter) and the thrusters give exactly the commanded force.

The vessel starts away from the set-point and must come back and stay there,
also with the constant force pushing it away. That needs the I term.

License: MIT (see LICENSE in the repository root)
Authors: Olve Grønås Birkeland, viljartveit (2026)
"""
import tomllib

import numpy as np
import pytest

from app.build.data import default_config
from ospfmusim.controller.dp_controller import DPController, rotation, total_tau

CONFIG = default_config()   # the config the simulation uses (OspSystemStructure.xml)
DT = 0.1                    # time step [s], as in OspSystemStructure.xml
DURATION = 30 * 60.0        # 30 minutes

config = tomllib.loads(CONFIG.read_text(encoding="utf-8-sig"))
M = np.asarray(config["vessel"]["mass_matrix"], dtype=float)
D = np.asarray(config["vessel"]["damping_matrix"], dtype=float)
M_inv = np.linalg.inv(M)


def simulate(eta0, b_ned):
    """Run the closed loop. Returns times, positions (eta) and commanded forces (tau)."""
    fmu = DPController(instance_name="dp_controller")
    fmu.config_path = str(CONFIG)
    eta, nu = np.array(eta0, dtype=float), np.zeros(3)
    fmu.north_hat, fmu.east_hat, fmu.psi_hat = eta
    fmu.exit_initialization_mode()
    times, etas, taus = [], [], []
    t = 0.0
    while t < DURATION:
        # The controller sees the true state.
        fmu.north_hat, fmu.east_hat, fmu.psi_hat = eta
        fmu.u_hat, fmu.v_hat, fmu.r_hat = nu
        fmu.do_step(t, DT)
        tau = total_tau(fmu)
        # The vessel: one step (semi-implicit Euler: new velocity first, then position).
        nu = nu + DT * M_inv @ (tau + rotation(eta[2]).T @ b_ned - D @ nu)
        eta = eta + DT * rotation(eta[2]) @ nu
        t += DT
        times.append(t), etas.append(eta.copy()), taus.append(tau)
    return np.array(times), np.array(etas), np.array(taus)


@pytest.mark.parametrize("eta0, b_ned", [
    pytest.param([10.0, -5.0, np.radians(20.0)], np.zeros(3), id="start_away_no_force"),
    pytest.param([0.0, 0.0, 0.0], np.array([-35355.0, -35355.0, 0.0]),
                 id="start_at_set_point_50kN_from_north_east"),
])
def test_vessel_returns_to_and_holds_the_set_point(eta0, b_ned):
    times, etas, taus = simulate(eta0, b_ned)
    error = etas - np.array([config["controller"]["north_ref"], config["controller"]["east_ref"],
                             np.radians(config["controller"]["heading_ref_deg"])])
    error[:, 2] = np.degrees((error[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    distance = np.hypot(error[:, 0], error[:, 1])
    after_10_min = int(10 * 60 / DT) - 1
    last = times > DURATION - 5 * 60.0                    # the last 5 minutes
    assert distance[after_10_min] < 0.5
    assert distance[last].max() < 0.01
    assert np.abs(error[last, 2]).max() < 0.01
    # tau at the end cancels the constant force.
    force_body = -(rotation(etas[-1, 2]).T @ b_ned)
    np.testing.assert_allclose(taus[-1], force_body, rtol=1e-3, atol=10.0)
