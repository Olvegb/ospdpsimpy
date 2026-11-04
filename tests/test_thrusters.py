"""Tests that turning azimuths get commands they can follow, and don't push the wrong way.

1. ThrusterDynamics alone: one azimuth at rest pointing ahead (0 deg) is told
   to push 100 kN astern (180 deg). At 10 deg/s it needs 18 s to turn.
2. ThrustAllocator + ThrusterDynamics, a Gunnerus-like layout without thrust
   lag, starting at rest with an astern command: the force the dynamics give
   must be the one the allocator says it commands, every step.
3. Open-water curves (propeller_curves = "open_water"): an azimuth at bollard
   pull gives the commanded thrust, and loses thrust when it moves through the
   water. With the same inputs and "none" in the config, it doesn't.
4. ThrustAllocation with enabled = false in [allocation]: every thrust command is 0.
5. Ducted series (propeller_curves = "ducted_series"): the Ka 4-70/19A polynomials
   match the series, and a unit at its rated rpm in bollard pull draws its rated power.
6. propeller_curves chooses the curves, a thruster without the data falls back,
   and an unknown value is an error.
7. The allocation calls HiGHS directly: every LP gets the same answer as with linprog.
8. method = "pseudo_inverse": within reach it gives tau exactly, out of reach it keeps
   the direction of tau, and it stays out of the zones and within the power limits.

License: MIT (see LICENSE in the repository root)
Authors: William Boland (2026)
"""
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import linprog

from app.build.data import default_config
from ospfmusim.thruster import thrust_allocation, thruster_dynamics
from ospfmusim.thruster.helpfunctions import allocation
from ospfmusim.thruster.helpfunctions.allocation import ThrustAllocator
from ospfmusim.thruster.helpfunctions.dynamics import (KQ_19A, KT_19A, RHO, ThrusterDynamics,
                                                       _poly)
from ospfmusim.thruster.helpfunctions.layout import Layout, PowerSource, Thruster, load_layout
from ospfmusim.thruster.helpfunctions.zones import forbidden_zones

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


@pytest.fixture(scope="module", params=("linear_program", "pseudo_inverse"))
def allocated(request):
    """Allocator and dynamics together for 25 s. Returns tau, commands, states and turns."""
    layout = Layout((
        Thruster("AZ1", "azimuth", -16.95, 2.7, 50e3, max_azimuth_rate=RATE),
        Thruster("AZ2", "azimuth", -16.95, -2.7, 50e3, max_azimuth_rate=RATE),
        Thruster("BT1", "tunnel", 14.5, 0.0, 20e3),
    ), method=request.param)
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
    return ThrusterDynamics(Layout((Thruster("AZ", "azimuth", 0.0, 0.0, 200e3, diameter=D,
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


def test_only_propeller_curves_lose_thrust_with_speed(tmp_path):
    """The same inputs in the FMU, with only propeller_curves in the config changed."""
    fmus = {}
    for source in ("none", "open_water"):
        config = tmp_path / f"{source}.toml"
        config.write_text(f"""
[allocation]
propeller_curves = "{source}"

[[thrusters]]
name = "AZ"
kind = "azimuth"
x = 0.0
y = 0.0
max_thrust = 200e3
diameter = {D}
open_water = {[list(row) for row in OPEN_WATER]}
""")
        fmu = fmus[source] = thruster_dynamics.ThrusterDynamics(instance_name="dynamics")
        fmu.config_path = str(config)
        fmu.thruster1_thrust_command = THRUST
        fmu.u = 2.0
        fmu.exit_initialization_mode()
    model = propeller_azimuth()
    model.velocity = (2.0, 0.0, 0.0)
    expected = model.step([THRUST], [0.0])
    assert fmus["none"].thruster1_thrust == pytest.approx(THRUST)
    assert fmus["none"].thruster1_shaft_speed == 0.0
    assert fmus["open_water"].thruster1_thrust == pytest.approx(expected.thrust[0])
    assert fmus["open_water"].thruster1_shaft_speed == pytest.approx(expected.shaft_speed[0])
    assert fmus["open_water"].X == pytest.approx(expected.thrust[0])


def test_switched_off_allocation_gives_no_thrust(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("""
[allocation]
enabled = false

[[thrusters]]
name = "AZ"
kind = "azimuth"
x = 0.0
y = 0.0
max_thrust = 200e3
""")
    fmu = thrust_allocation.ThrustAllocation(instance_name="allocation")
    fmu.config_path = str(config)
    fmu.tau_X_P = 50e3
    fmu.exit_initialization_mode()
    fmu.do_step(0.0, DT)
    assert fmu.tau_X == 50e3
    assert fmu.thruster1_thrust_command == 0.0


def test_series_matches_ka_4_70_in_nozzle_19a():
    assert _poly(KT_19A, 0.0, 1.0) == pytest.approx(0.526, abs=5e-4)
    assert _poly(KQ_19A, 0.0, 1.0) == pytest.approx(0.044, abs=5e-4)


def ducted_azimuth():
    """A Gunnerus-like ducted azimuth: 1.9 m, 500 kW at 198 rpm."""
    return ThrusterDynamics(Layout((Thruster("AZ", "azimuth", 0.0, 0.0, 200e3, diameter=1.9,
                                             power_kw=500.0, rated_rpm=198.0),),
                                   propeller_curves="ducted_series"))


def test_ducted_draws_rated_power_at_rated_rpm():
    model = ducted_azimuth()
    n = 198.0 / 60
    thrust = RHO * 1.9**4 * model._curves[0][0](0.0) * n**2      # the Tn that gives rated n
    state = model.step([thrust], [0.0])
    assert state.thrust[0] == pytest.approx(thrust)
    assert state.shaft_speed[0] == pytest.approx(n)
    assert state.power_kw[0] == pytest.approx(500.0)


def test_ducted_loses_thrust_with_speed():
    model = ducted_azimuth()
    model.velocity = (2.0, 0.0, 0.0)
    assert model.step([THRUST], [0.0]).thrust[0] < THRUST


@pytest.mark.parametrize("source, data, kt0", [
    ("none", {"open_water": OPEN_WATER, "rated_rpm": 198.0}, None),
    ("open_water", {"open_water": OPEN_WATER, "rated_rpm": 198.0}, 0.36),
    ("ducted_series", {"open_water": OPEN_WATER, "rated_rpm": 198.0}, 0.83),
    ("ducted_series", {"open_water": OPEN_WATER}, 0.36),     # no rated_rpm: the table
    ("ducted_series", {}, None),                             # no data: no curves
    ("open_water", {"rated_rpm": 198.0}, None),              # the series only when chosen
])
def test_propeller_curves_chooses_the_curves(source, data, kt0):
    thruster = Thruster("AZ", "azimuth", 0.0, 0.0, 200e3, diameter=1.9, power_kw=500.0, **data)
    [curves] = ThrusterDynamics(Layout((thruster,), propeller_curves=source))._curves
    if kt0 is None:
        assert curves is None
    else:
        assert curves[0](0.0) == pytest.approx(kt0, abs=5e-3)


def test_unknown_propeller_curves_stops_the_run(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("""
[allocation]
propeller_curves = "open-water"

[[thrusters]]
name = "AZ"
kind = "azimuth"
x = 0.0
y = 0.0
max_thrust = 200e3
""")
    with pytest.raises(ValueError, match="propeller_curves"):
        load_layout(config)


def test_every_lp_gets_the_same_answer_as_with_linprog(monkeypatch):
    # Fails first if a scipy upgrade changes the private binding _solve() uses.
    solve = allocation._solve
    solved, infeasible = [], []

    def both(c, a_ub, b_ub, a_eq, b_eq, bounds):
        x = solve(c, a_ub, b_ub, a_eq, b_eq, bounds)
        expected = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds,
                           method="highs")
        if expected.status == 2:
            assert x is None
            infeasible.append(c)
        else:
            assert np.array_equal(x, expected.x)
            solved.append(c)
        return x

    monkeypatch.setattr(allocation, "_solve", both)
    # The config's layout, with its zones and power limits: within reach, then saturated.
    allocator = ThrustAllocator(replace(load_layout(default_config()), method="linear_program"))
    for tau in ([2e4, 1e4, 1e5], [2.5e4, 0.5e4, -1e5], [5e6, 5e6, 5e7]):
        allocator.step(tau, dt=DT)
    assert solved
    # A tunnel thruster cannot give a surge force at all.
    tunnel = ThrustAllocator(Layout((Thruster("T", "tunnel", 10.0, 0.0, 1e5),)))
    assert tunnel.step([1e3, 0.0, 0.0]).saturated
    assert infeasible


def test_unknown_method_stops_the_run(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("""
[allocation]
method = "pinv"

[[thrusters]]
name = "AZ"
kind = "azimuth"
x = 0.0
y = 0.0
max_thrust = 200e3
""")
    with pytest.raises(ValueError, match="method"):
        load_layout(config)


def gunnerus():
    """The Gunnerus layout, its zones and power limits, with the pseudo-inverse."""
    return replace(load_layout(default_config().parent / "gunnerus.toml"), method="pseudo_inverse")


def test_pinv_gives_tau_within_reach():
    command = ThrustAllocator(gunnerus()).step([20e3, 10e3, 100e3])
    np.testing.assert_allclose(command.tau, [20e3, 10e3, 100e3], atol=1e-6)
    assert not command.saturated
    assert command.allocation.utilisation < 1


def test_pinv_uses_the_thrusters_beyond_where_the_plain_inverse_stops():
    """The plain pseudo-inverse with all forces scaled reaches 40.3 kN of pure sway here."""
    command = ThrustAllocator(gunnerus()).step([0.0, 50e3, 0.0])
    np.testing.assert_allclose(command.tau, [0.0, 50e3, 0.0], atol=1e-6)
    assert not command.saturated


def test_pinv_keeps_the_direction_of_a_tau_out_of_reach():
    """Redistributing alone would give 42 kN of sway here, and 185 kNm of yaw for nothing."""
    command = ThrustAllocator(gunnerus()).step([0.0, 80e3, 0.0])
    assert command.saturated
    np.testing.assert_allclose(command.tau, [0.0, 52.2e3, 0.0], atol=100.0)
    assert command.allocation.utilisation == pytest.approx(1.53, abs=0.01)
    assert command.allocation.thrust_fraction.max() <= 1 + 1e-9


def test_pinv_with_a_lost_thruster_keeps_the_direction():
    """AZ2 lost: the tunnel and one azimuth can still give all three, with less room."""
    command = ThrustAllocator(gunnerus()).step([60e3, 10e3, 100e3], [True, False, True])
    assert command.saturated
    assert command.allocation.thrust_fraction[1] == 0.0
    assert command.allocation.thrust_fraction.max() <= 1 + 1e-9
    np.testing.assert_allclose(command.tau / np.array([60e3, 10e3, 100e3]),
                               command.tau[0] / 60e3, atol=1e-3)


def test_pinv_stays_out_of_the_flushing_sectors():
    layout = gunnerus()
    zones = forbidden_zones(layout.thrusters, flushing=True)
    assert zones[0] and zones[1]
    for yaw in (0.0, 1e5, -1e5):
        for angle in np.deg2rad(np.arange(0.0, 360.0, 15.0)):
            tau = [10e3 * np.cos(angle), 10e3 * np.sin(angle), yaw]
            result = ThrustAllocator(layout).step(tau).allocation
            for i in (0, 1):
                for start, end in zones[i]:
                    # Up to an edge, not past it. The margin is only for rounding.
                    assert not 1e-6 < (result.angle_deg[i] - start) % 360.0 < end - start - 1e-6


def test_pinv_is_not_thrown_by_noise_around_zero():
    """Nearly parallel azimuths must not push against each other during a turn."""
    rng = np.random.default_rng(1)
    allocator = ThrustAllocator(gunnerus())
    for _ in range(500):
        tau = np.array([300.0, 300.0, 1000.0]) * rng.standard_normal(3)
        error = np.abs(allocator.step(tau, dt=DT).tau - tau)
        assert error[:2].max() < 2e3 and error[2] < 60e3


def test_pinv_stays_within_the_power_limits():
    """Thrust alone would allow this: only the switchboard stops it."""
    layout = Layout((
        Thruster("AZ1", "azimuth", -16.95, 2.7, 50e3, power_kw=500.0,
                 power_supply=(("SWBD", 1.0),)),
        Thruster("AZ2", "azimuth", -16.95, -2.7, 50e3, power_kw=500.0,
                 power_supply=(("SWBD", 1.0),)),
    ), power_sources=(PowerSource("SWBD", 100.0),), method="pseudo_inverse")
    command = ThrustAllocator(layout).step([60e3, 0.0, 0.0])
    assert command.saturated
    assert command.allocation.source_power_kw[0] == pytest.approx(100.0)
    assert command.allocation.utilisation > 1
    np.testing.assert_allclose(command.tau[1:], 0.0, atol=1e-6)


def test_pinv_tunnel_alone_against_surge():
    layout = Layout((Thruster("T", "tunnel", 10.0, 0.0, 1e5),), method="pseudo_inverse")
    command = ThrustAllocator(layout).step([1e3, 0.0, 0.0])
    assert command.saturated
    assert command.allocation.utilisation == np.inf
    np.testing.assert_allclose(command.thrust, 0.0, atol=1e-6)
    # It still gives the part of tau it can push along.
    command = ThrustAllocator(layout).step([1e3, 5e3, 50e3])
    assert command.saturated
    np.testing.assert_allclose(command.tau, [0.0, 5e3, 50e3], atol=1e-6)


def test_pinv_keeps_within_a_smaller_reverse_thrust():
    layout = Layout((Thruster("T", "tunnel", 0.0, 0.0, 1e5, max_reverse_thrust=4e4),),
                    method="pseudo_inverse")
    command = ThrustAllocator(layout).step([0.0, -8e4, 0.0])
    assert command.saturated
    assert command.thrust[0] == pytest.approx(-4e4, rel=1e-3)
    assert command.allocation.thrust_fraction[0] == pytest.approx(1.0, rel=1e-3)
