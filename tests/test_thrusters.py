"""Tests that turning azimuths get commands they can follow, and don't push the wrong way.

1. ThrusterDynamics alone: one azimuth at rest pointing ahead (0 deg) is told
   to push 100 kN astern (180 deg). At 10 deg/s it needs 18 s to turn.
2. ThrustAllocator + ThrusterDynamics, a Gunnerus-like layout without thrust
   lag, starting at rest with an astern command: the force the dynamics give
   must be the one the allocator says it commands, every step.
3. Open-water curves (ThrusterDynamicsKtKq): an azimuth at bollard pull
   gives the commanded thrust, and loses thrust when it moves through the
   water. The two FMUs take the same inputs; only the KT/KQ one loses thrust.

License: MIT (see LICENSE in the repository root)
Authors: William Boland (2026)
"""
import numpy as np
import pytest

from ospfmusim.thruster import thruster_dynamics
from ospfmusim.thruster.helpfunctions.allocation import ThrustAllocator
from ospfmusim.thruster.helpfunctions.dynamics import ThrusterDynamics
from ospfmusim.thruster.helpfunctions.layout import Layout, Thruster, load_layout
from ospfmusim.thruster_ktkq import thruster_dynamics_ktkq
from ospfmusim.thruster_ktkq.helpfunctions.dynamics_ktkq import RHO, ThrusterDynamicsKtKq

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


def test_turn_limited_not_saturated_at_the_start(allocated):
    _, commands, _, _ = allocated
    assert commands[0].turn_limited      # the azimuths can't push astern yet
    assert not commands[0].saturated     # but they are strong enough
    assert commands[0].allocation.utilisation < 1


def test_dynamics_give_the_commanded_force_every_step(allocated):
    _, commands, states, _ = allocated
    for state, command in zip(states, commands):
        np.testing.assert_allclose(state.tau, command.tau, atol=1.0)


def test_tau_reached_and_delivered_after_25_s(allocated):
    tau, commands, states, _ = allocated
    assert not commands[-1].saturated and not commands[-1].turn_limited
    np.testing.assert_allclose(states[-1].tau, tau, atol=1.0)


# Linear open-water curves KT = 0.36 - 0.32 J, KQ = 0.046 - 0.028 J (Sørensen 2018, (8.10)).
OPEN_WATER = ((-0.5, 0.52, 0.060), (1.0, 0.04, 0.018))
D = 2.0


def propeller_azimuth():
    return ThrusterDynamicsKtKq(Layout((Thruster("AZ", "azimuth", 0.0, 0.0, 200e3, diameter=D,
                                                 open_water=OPEN_WATER),)))


def test_bollard_pull_gives_the_commanded_thrust():
    state = propeller_azimuth().step([THRUST], [0.0])
    n = np.sqrt(THRUST / (RHO * D**4 * 0.36))                     # (8.38)
    assert state.thrust[0] == pytest.approx(THRUST)
    assert state.shaft_speed[0] == pytest.approx(n)
    assert state.torque[0] == pytest.approx(RHO * D**5 * 0.046 * n**2)
    assert state.power_kw[0] == pytest.approx(2 * np.pi * n * state.torque[0] / 1000)


def test_speed_through_the_water_loses_thrust():
    model = propeller_azimuth()
    model.velocity = (2.0, 0.0, 0.0)
    state = model.step([THRUST], [0.0])
    n = state.shaft_speed[0]
    ja = 2.0 / (n * D)                                             # (8.6)
    assert state.thrust[0] < THRUST
    assert state.thrust[0] == pytest.approx(RHO * D**4 * (0.36 - 0.32 * ja) * n**2)


def test_only_the_ktkq_fmu_loses_thrust_with_speed(tmp_path):
    """The same config and inputs in both FMUs, as when one replaces the other."""
    config = tmp_path / "config.toml"
    config.write_text(f"""
[[thrusters]]
name = "AZ"
kind = "azimuth"
x = 0.0
y = 0.0
max_thrust = 200e3
diameter = {D}
open_water = {[list(row) for row in OPEN_WATER]}
""")
    fmus = {}
    for cls in (thruster_dynamics.ThrusterDynamics, thruster_dynamics_ktkq.ThrusterDynamicsKtKq):
        fmu = fmus[cls] = cls(instance_name="dynamics")
        fmu.config_path = str(config)
        fmu.thruster1_thrust_command = THRUST
        fmu.u = 2.0
        fmu.exit_initialization_mode()
    model = propeller_azimuth()
    model.velocity = (2.0, 0.0, 0.0)
    expected = model.step([THRUST], [0.0])
    old, ktkq = fmus.values()
    assert old.thruster1_thrust == pytest.approx(THRUST)
    assert ktkq.thruster1_thrust == pytest.approx(expected.thrust[0])
    assert ktkq.thruster1_shaft_speed == pytest.approx(expected.shaft_speed[0])
    assert ktkq.X == pytest.approx(expected.thrust[0])


def test_max_azimuth_rpm_is_read_as_deg_per_s(tmp_path):
    config = tmp_path / "config.toml"
    entry = '[[thrusters]]\nname = "AZ"\nkind = "azimuth"\nx = 0.0\ny = 0.0\nmax_thrust = 1e3\n'
    config.write_text(entry + "max_azimuth_rpm = 2.0\n")
    assert load_layout(config).thrusters[0].max_azimuth_rate == pytest.approx(12.0)
    config.write_text(entry + "max_azimuth_rpm = 2.0\nmax_azimuth_rate = 12.0\n")
    with pytest.raises(ValueError, match="not both"):
        load_layout(config)
