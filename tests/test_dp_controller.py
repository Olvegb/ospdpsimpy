"""Tests of the DPController FMU code (controller/dp_controller.py) WITHOUT building an FMU
and without OSP.

Checks the signs: the commanded force must push the vessel back towards the
set-point, also when the vessel has another heading. Every case is placed
relative to the set-point in the config, and the force is checked in NED, so
the tests hold for any set-point and heading.

License: MIT (see LICENSE in the repository root)
Authors: Olve Grønås Birkeland, viljartveit (2026)
"""
import tomllib

import numpy as np
import pytest

from app.build.data import default_config
from ospfmusim.controller.dp_controller import DPController, rotation, total_tau, wrap_angle

CONFIG = default_config()   # the config the simulation uses (OspSystemStructure.xml)
DT = 0.1                    # time step [s], as in OspSystemStructure.xml
# Vessel heading relative to the set-point heading [deg]
HEADINGS = (0.0, 45.0, 90.0, 180.0, 270.0)


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


def test_no_force_at_the_set_point():
    t, _ = tau()
    np.testing.assert_allclose(t, 0.0)


@pytest.mark.parametrize("heading", HEADINGS)
@pytest.mark.parametrize("north, east", [(10.0, 0.0), (0.0, 10.0), (-7.0, 7.0)])
def test_p_term_pushes_back_towards_the_set_point(north, east, heading):
    t, psi = tau(north, east, heading)
    force_ned = (rotation(psi) @ t)[:2]
    assert force_ned @ [north, east] < 0.0


@pytest.mark.parametrize("heading, sign", [(10.0, -1), (-10.0, +1), (179.0, -1), (181.0, +1),
                                           (359.0, +1)])
def test_p_term_turns_back_the_shortest_way(heading, sign):
    t, _ = tau(heading=heading)
    assert np.sign(t[2]) == sign


def test_heading_error_is_wrapped():
    t1, _ = tau(heading=1.0)
    t359, _ = tau(heading=359.0)
    assert t359[2] == pytest.approx(-t1[2])


@pytest.mark.parametrize("name, value, dof", [("u_hat", 1.0, 0), ("v_hat", 1.0, 1),
                                              ("r_hat", 0.01, 2)])
def test_d_term_brakes(name, value, dof):
    t, _ = tau(**{name: value})
    assert t[dof] < 0.0


def test_i_term_and_anti_windup():
    """10 m north of the set-point, held there for 100 minutes."""
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8-sig"))
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
        assert np.all(np.abs(integral) <= limit * (1 + 1e-9)), f"after {minutes} min"
        assert integral_ned @ [10.0, 0.0] <= 0.0, f"after {minutes} min"
    # After 100 min the integral term is held at its limit, in every direction with an error.
    held = np.isclose(np.abs(integral), limit) | (fmu.ki == 0.0)
    assert held[np.abs(rotation(fmu.psi_hat).T @ [10.0, 0.0, 0.0]) > 1e-9].all()
