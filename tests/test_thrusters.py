"""Tests that turning azimuths get commands they can follow, and don't push the wrong way.

1. ThrusterDynamics alone: one azimuth at rest pointing ahead (0 deg) is told
   to push 100 kN astern (180 deg). At 10 deg/s it needs 18 s to turn.
2. ThrustAllocator + ThrusterDynamics, a Gunnerus-like layout without thrust
   lag, starting at rest with an astern command: the force the dynamics give
   must be the one the allocator says it commands, every step.

License: MIT (see LICENSE in the repository root)
Authors: William Boland (2026)
"""
import numpy as np
import pytest

from ospfmusim.thruster.helpfunctions.allocation import ThrustAllocator
from ospfmusim.thruster.helpfunctions.dynamics import ThrusterDynamics
from ospfmusim.thruster.helpfunctions.layout import Layout, Thruster

RATE = 10.0      # [deg/s]
THRUST = 100e3   # [N]
DT = 0.1         # [s]


def azimuth(**kwargs):
    return ThrusterDynamics(Layout((Thruster("AZ", "azimuth", 0.0, 0.0, 200e3, **kwargs),)))


@pytest.fixture(scope="module")
def turning():
    """The states of one azimuth turning from ahead to astern, one per second."""
    model = azimuth(max_azimuth_rate=RATE)
    return [model.step([THRUST], [180.0], dt=1.0) for _ in range(18)]


def test_turning_azimuth_never_pushes_forward(turning):
    assert all(s.fx[0] <= 1e-6 for s in turning)


def test_no_thrust_while_more_than_90_deg_off(turning):
    assert all(s.thrust[0] < 1e-6 for s in turning[:9])


def test_full_thrust_astern_once_turned(turning):
    assert turning[-1].azimuth_deg[0] == pytest.approx(180.0)
    assert turning[-1].thrust[0] == pytest.approx(THRUST)


def test_without_rate_limit_full_thrust_astern_at_once():
    state = azimuth().step([THRUST], [180.0], dt=1.0)
    assert state.fx[0] == pytest.approx(-THRUST)


@pytest.fixture(scope="module")
def allocated():
    """Allocator and dynamics together for 25 s. Returns tau, commands, states and turns."""
    layout = Layout((
        Thruster("AZ1", "azimuth", -16.95, 2.7, 50e3, max_azimuth_rate=RATE),
        Thruster("AZ2", "azimuth", -16.95, -2.7, 50e3, max_azimuth_rate=RATE),
        Thruster("BT1", "tunnel", 14.5, 0.0, 20e3),
    ))
    tau = np.array([-30e3, 5e3, 20e3])
    allocator, model = ThrustAllocator(layout), ThrusterDynamics(layout)
    angle = np.array([0.0, 0.0, 90.0])
    commands, states, turns = [], [], []
    for _ in range(250):
        command = allocator.step(tau, dt=DT)
        turns.append(np.abs((command.azimuth_deg - angle + 180.0) % 360.0 - 180.0).max())
        angle = command.azimuth_deg
        commands.append(command)
        states.append(model.step(command.thrust, command.azimuth_deg, dt=DT))
    return tau, commands, states, turns


def test_angle_commands_respect_the_rate_limit(allocated):
    _, _, _, turns = allocated
    assert max(turns) <= RATE * DT + 1e-9


def test_saturated_at_the_start(allocated):
    _, commands, _, _ = allocated
    assert commands[0].saturated         # the azimuths can't push astern yet


def test_dynamics_give_the_commanded_force_every_step(allocated):
    _, commands, states, _ = allocated
    for state, command in zip(states, commands):
        np.testing.assert_allclose(state.tau, command.tau, atol=1.0)


def test_tau_reached_and_delivered_after_25_s(allocated):
    tau, commands, states, _ = allocated
    assert not commands[-1].saturated
    np.testing.assert_allclose(states[-1].tau, tau, atol=1.0)
