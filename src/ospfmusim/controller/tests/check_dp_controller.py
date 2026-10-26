"""Check the DPController FMU code WITHOUT building an FMU and without OSP.
Run from the repo folder:  python src/ospfmusim/controller/tests/check_dp_controller.py

Creates the DPController class from controller/dp_controller.py and checks the
signs: the commanded force must push the vessel back towards the set-point,
also when the vessel has another heading.

Every case is placed relative to the set-point in the config, and the force is
checked in NED, so the checks hold for any set-point and heading. Exits with
code 1 if a check fails.
"""
import sys
import tomllib

import numpy as np

from app.build.data import default_config
from ospfmusim.controller.dp_controller import (OUTPUTS, DPController, rotation, total_tau,
                                                wrap_angle)

CONFIG = default_config()   # the config the simulation uses (OspSystemStructure.xml)
DT = 0.1                    # time step [s], as in OspSystemStructure.xml
# Vessel heading relative to the set-point heading [deg]
HEADINGS = (0.0, 45.0, 90.0, 180.0, 270.0)

failures = []


def controller():
    """A started controller with the config the simulation uses."""
    fmu = DPController(instance_name="dp_controller")
    fmu.config_path = str(CONFIG)
    fmu.exit_initialization_mode()
    return fmu


def tau(north=0.0, east=0.0, heading=0.0, **velocities):
    """tau [X, Y, N] (body) after one step, with the vessel at the set-point.

    The vessel is at the set-point + (north [m], east [m], heading [deg]) and has
    the given body velocities (u_hat, v_hat, r_hat). Also returns psi_hat [rad].
    """
    fmu = controller()
    north_ref, east_ref, psi_ref = fmu.reference
    fmu.north_hat, fmu.east_hat = north_ref + north, east_ref + east
    fmu.psi_hat = wrap_angle(psi_ref + np.radians(heading))
    for name, value in velocities.items():
        setattr(fmu, name, value)
    fmu.do_step(0.0, DT)
    return total_tau(fmu) + 0.0, fmu.psi_hat     # + 0.0 turns -0 into 0


def check(text, ok, t):
    print(f"  [{'OK' if ok else 'FAIL'}] {text:52s} tau_X = {t[0]:10.0f} N,  "
          f"tau_Y = {t[1]:10.0f} N,  tau_N = {t[2]:12.0f} Nm")
    if not ok:
        failures.append(text)


config = tomllib.loads(CONFIG.read_text(encoding="utf-8-sig"))
fmu = controller()
print(f"CONFIG: {CONFIG}")
print(f"  set-point: north {fmu.reference[0]} m, east {fmu.reference[1]} m, "
      f"heading {np.degrees(fmu.reference[2]):.1f} deg")
print("GAINS FROM [vessel] AND [controller]:")
for name, gain in (("Kp", fmu.kp), ("Ki", fmu.ki), ("Kd", fmu.kd)):
    print(f"  {name} = {gain}  [surge, sway, yaw]")
print(f"  {len(OUTPUTS)} outputs: {', '.join(OUTPUTS)}")

print("CHECK AT THE SET-POINT:")
t, _ = tau()
check("At the set-point, not moving: tau = 0", np.allclose(t, 0.0), t)

print("CHECK THE P TERM ON POSITION: the force in NED must point back towards the set-point")
for heading in HEADINGS:
    for north, east, where in ((10.0, 0.0, "10 m north"), (0.0, 10.0, "10 m east"),
                               (-7.0, 7.0, "7 m south, 7 m east")):
        t, psi = tau(north, east, heading)
        force_ned = (rotation(psi) @ t)[:2]
        check(f"{where}, heading set-point {heading:+.0f} deg", force_ned @ [north, east] < 0.0, t)

print("CHECK THE P TERM ON HEADING: turn back the shortest way")
for heading, expected in ((10.0, -1), (-10.0, +1), (179.0, -1), (181.0, +1), (359.0, +1)):
    t, _ = tau(heading=heading)
    check(f"Heading set-point {heading:+.0f} deg: tau_N {'<' if expected < 0 else '>'} 0",
          np.sign(t[2]) == expected, t)
t1, _ = tau(heading=1.0)
t359, _ = tau(heading=359.0)
check("Heading set-point +359 deg gives the same |tau_N| as -1 deg",
      np.isclose(t359[2], -t1[2]), t359)

print("CHECK THE D TERM (at the set-point, moving): brake")
for name, value, dof, unit in (("u_hat", 1.0, 0, "m/s"), ("v_hat", 1.0, 1, "m/s"),
                               ("r_hat", 0.01, 2, "rad/s")):
    t, _ = tau(**{name: value})
    check(f"{name} = {value} {unit}: {('tau_X', 'tau_Y', 'tau_N')[dof]} < 0", t[dof] < 0.0, t)

print("CHECK THE I TERM AND ANTI-WINDUP (10 m north of the set-point, held there):")
limit = np.asarray(config["controller"]["integral_force_limit"], dtype=float)
fmu = controller()
fmu.north_hat, fmu.east_hat, fmu.psi_hat = fmu.reference + [10.0, 0.0, 0.0]
time = 0.0
for minutes in (1, 10, 100):
    while time < minutes * 60.0 - 1e-9:
        fmu.do_step(time, DT)
        time += DT
    integral = -fmu.ki * fmu.z                   # body frame [N, N, Nm]
    integral_ned = (rotation(fmu.psi_hat) @ integral)[:2]
    ok = np.all(np.abs(integral) <= limit * (1 + 1e-9)) and integral_ned @ [10.0, 0.0] <= 0.0
    check(f"after {minutes:3d} min: within the limit {limit}, pointing back", ok, integral)
held = np.isclose(np.abs(integral), limit) | (fmu.ki == 0.0)
check("after 100 min: the integral term is at its limit (anti-windup holds it there)",
      bool(held[np.abs(rotation(fmu.psi_hat).T @ [10.0, 0.0, 0.0]) > 1e-9].all()), integral)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for text in failures:
        print(f"  - {text}")
    sys.exit(1)
print("All DPController checks passed.")
