"""Check that the DP controller can hold a vessel, WITHOUT OSP and without a hull FMU.
Run from the repo folder:  python src/ospfmusim/controller/tests/check_closed_loop.py

A very simple 3-DOF vessel, only for this check (not the hull model):

    eta_dot = R(psi) nu
    M nu_dot + D nu = tau + R(psi)^T b

M and D from [vessel] in the config the simulation uses, b a constant force in NED (current, wave
drift, wind). The controller gets the true position and velocity (no Kalman
filter) and the thrusters give exactly the commanded force.

The vessel starts away from the set-point and must come back and stay there,
also with the constant force pushing it away. That needs the I term.
"""
import tomllib

import numpy as np

from app.build.data import default_config
from ospfmusim.controller.dp_controller import DPController, rotation, total_tau

CONFIG = default_config()   # the config the simulation uses (OspSystemStructure.xml)
DT = 0.1                    # time step [s], as in OspSystemStructure.xml
DURATION = 30 * 60.0                             # 30 minutes

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


def report(title, eta0, b_ned):
    times, etas, taus = simulate(eta0, b_ned)
    error = etas - np.array([config["controller"]["north_ref"], config["controller"]["east_ref"],
                             np.radians(config["controller"]["heading_ref_deg"])])
    error[:, 2] = np.degrees((error[:, 2] + np.pi) % (2 * np.pi) - np.pi)
    distance = np.hypot(error[:, 0], error[:, 1])
    last = times > DURATION - 5 * 60.0                    # the last 5 minutes
    print(title)
    print(f"  start: {distance[0]:.1f} m from the set-point, heading error {error[0, 2]:+.1f} deg")
    for minutes in (2, 5, 10):
        k = int(minutes * 60 / DT) - 1
        print(f"  after {minutes:2d} min: {distance[k]:5.2f} m, "
              f"heading error {error[k, 2]:+6.2f} deg")
    print(f"  last 5 min: at most {distance[last].max():.3f} m and "
          f"{np.abs(error[last, 2]).max():.3f} deg from the set-point")
    print(f"  largest distance from the set-point: {distance.max():.2f} m")
    tau_end = taus[-1] + 0.0                              # + 0.0 turns -0 into 0
    force_body = -(rotation(etas[-1, 2]).T @ b_ned) + 0.0
    print(f"  tau at the end: X = {tau_end[0]:.0f} N, Y = {tau_end[1]:.0f} N, "
          f"N = {tau_end[2]:.0f} Nm"
          f"  (should cancel the force: X = {force_body[0]:.0f} N, Y = {force_body[1]:.0f} N)")


report("CHECK: START AWAY FROM THE SET-POINT, NO FORCE:",
       eta0=[10.0, -5.0, np.radians(20.0)], b_ned=np.zeros(3))
report("CHECK: START AT THE SET-POINT, CONSTANT FORCE 50 kN FROM NORTH-EAST:",
       eta0=[0.0, 0.0, 0.0], b_ned=np.array([-35355.0, -35355.0, 0.0]))
