"""Tests of the Environment FMU code (enviroment/environment.py) WITHOUT building an FMU
and without OSP.

Creates the Environment class with the config the simulation uses, or with a
copy of it where some values are changed, and checks:

  1. Everything switched off: the outputs are constant and equal to the config.
  2. Slowly varying wind (Gauss-Markov): zero mean, the right std and time
     constant, and the same statistics for another step size.
  3. Gusts: the std of the gust signal matches the NORSOK spectrum
     integrated over [f_min, f_max]. Zero mean.
  4. Wind direction: a constant turning rate gives mean + rate * t, and the
     output always stays in [0, 360).
  5. Current: north and east components match speed and direction.
     enabled = false in [wind], [waves] or [current] gives 0 for that part.
  6. Seed: the same seed gives the same wind, another seed another wind.
  7. Mean wind height profile (Sorensen 2018, eq. 6.25): U10 at 10 m, less
     below and more above, and the log law between two heights.
  8. Saturation of the slowly varying wind (eq. 6.27): mean + slowly varying
     stays inside [min_speed, max_speed].

License: MIT (see LICENSE in the repository root)
Authors: Olve Grønås Birkeland (2026)
"""
import random
from math import cos, exp, log, radians, sin

import numpy as np
import pytest

from app.build.data import default_config
from ospfmusim.enviroment.environment import (Environment, GaussMarkov, mean_wind_profile,
                                              norsok_wind_spectrum)

CONFIG = default_config()   # the config the simulation uses (OspSystemStructure.xml)
DT = 0.1                    # time step [s], as in OspSystemStructure.xml
ALL_OFF = {
    ("wind.slowly_varying", "std"): "0.0",
    ("wind.gust", "n_components"): "0",
    ("wind.direction_variation", "rate_deg_per_s"): "0.0",
    ("wind.direction_variation", "std_deg"): "0.0",
}
KAPPA = 0.0026              # surface drag coefficient in the height profile


def config_with(changes, folder):
    """A copy of the config the simulation uses with some values changed, written to folder.

    changes: {("section", "key"): "new value"}.
    """
    lines, section, done = [], None, set()
    for line in CONFIG.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and not stripped.startswith("[["):
            section = stripped.strip("[]")
        is_key_line = "=" in stripped and not stripped.startswith("#")
        key = stripped.split("=")[0].strip() if is_key_line else None
        if (section, key) in changes:
            line = f"{key} = {changes[(section, key)]}"
            done.add((section, key))
        lines.append(line)
    missing = set(changes) - done
    assert not missing, f"not found in {CONFIG}: {missing}"
    path = folder / f"config_{len(list(folder.iterdir()))}.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def environment(tmp_path):
    """Starts an Environment FMU, with the config the simulation uses or a changed copy of it."""
    def start(changes=None):
        fmu = Environment(instance_name="environment")
        fmu.config_path = str(config_with(changes, tmp_path) if changes else CONFIG)
        fmu.exit_initialization_mode()
        return fmu
    return start


def run(fmu, duration, names):
    """Steps the FMU and returns the outputs in names as arrays, one row per step."""
    rows = []
    for k in range(int(round(duration / DT))):
        fmu.do_step(k * DT, DT)
        rows.append([getattr(fmu, name) for name in names])
    return np.array(rows)


def test_all_off_gives_constant_outputs(environment):
    fmu = environment(ALL_OFF)
    names = ("mean_wind_speed", "slowly_varying_wind_speed", "gust_wind_speed",
             "wind_direction_deg")
    out = run(fmu, 60.0, names)
    expected = [fmu.mean_wind_speed, 0.0, 0.0, fmu.mean_wind_direction_deg % 360.0]
    np.testing.assert_allclose(out, np.broadcast_to(expected, out.shape))


@pytest.mark.parametrize("dt", [0.1, 1.0])
def test_slowly_varying_wind_statistics(dt):
    T, std = 300.0, 1.0
    duration = 1000 * T                          # many time constants, so the statistics settle
    process = GaussMarkov(T, std, random.Random(7))
    x = np.array([process.step(dt) for _ in range(int(duration / dt))])
    lag = int(T / dt)
    correlation = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    assert abs(x.mean()) < 0.2 * std, "the slowly varying wind does not have zero mean"
    assert abs(x.std() - std) < 0.1 * std, "the std is off by more than 10 %"
    assert abs(correlation - exp(-1)) < 0.1, "the time constant is wrong"


def test_gusts_match_the_spectrum(environment):
    fmu = environment({("wind.slowly_varying", "std"): "0.0"})
    gust = run(fmu, 3600.0, ("gust_wind_speed",))[:, 0]
    # The variance of the spectrum, integrated over [f_min, f_max] (log-spaced trapezoid).
    f = np.geomspace(0.005, 0.5, 5000)
    S = np.array([norsok_wind_spectrum(fi, fmu.mean_speed_10m, 10.0) for fi in f])
    std_spectrum = np.sqrt(np.trapezoid(S, f))
    std_components = np.sqrt(sum(a ** 2 / 2.0 for _, a, _ in fmu.gust_components))
    assert abs(std_components - std_spectrum) < 0.1 * std_spectrum, \
        "the gust amplitudes do not match the spectrum"
    assert abs(gust.std() - std_spectrum) < 0.15 * std_spectrum, \
        "the gust signal does not match the spectrum"
    assert abs(gust.mean()) < 0.1 * std_spectrum, "the gusts do not have zero mean"


def test_direction_turns_at_a_constant_rate(environment):
    rate = 0.5                                   # deg/s, so the direction passes 360 several times
    fmu = environment({**ALL_OFF, ("wind.direction_variation", "rate_deg_per_s"): str(rate)})
    direction = run(fmu, 2000.0, ("wind_direction_deg",))[:, 0]
    t = DT * np.arange(1, len(direction) + 1)
    expected = (fmu.mean_wind_direction_deg + rate * t) % 360.0
    error = (direction - expected + 180.0) % 360.0 - 180.0
    assert np.abs(error).max() < 1e-9, "the direction does not follow mean + rate * t"
    assert direction.min() >= 0.0 and direction.max() < 360.0


def test_random_direction_stays_in_range(environment):
    direction = run(environment(), 3600.0, ("wind_direction_deg",))[:, 0]
    assert direction.min() >= 0.0 and direction.max() < 360.0


@pytest.mark.parametrize("speed, from_deg", [(0.5, 0.0), (0.5, 90.0), (1.0, 225.0),
                                             (0.8, 370.0)])
def test_current_components(environment, speed, from_deg):
    fmu = environment({("current", "speed"): str(speed),
                       ("current", "direction_deg"): str(from_deg)})
    # The current flows the opposite way of where it comes from.
    expected = (-speed * cos(radians(from_deg)), -speed * sin(radians(from_deg)))
    np.testing.assert_allclose((fmu.current_north, fmu.current_east), expected, atol=1e-12)
    assert 0.0 <= fmu.current_direction_deg < 360.0


def test_switched_off_parts_are_zero(environment):
    fmu = environment({("wind", "enabled"): "false", ("waves", "enabled"): "false",
                       ("current", "enabled"): "false"})
    names = ("mean_wind_speed", "slowly_varying_wind_speed", "gust_wind_speed", "H_s",
             "current_speed", "current_north", "current_east")
    np.testing.assert_array_equal(run(fmu, 100.0, names), 0.0)


def test_seed(environment):
    names = ("slowly_varying_wind_speed", "gust_wind_speed", "wind_direction_deg")
    a = run(environment(), 60.0, names)
    b = run(environment(), 60.0, names)
    c = run(environment({("wind", "seed"): "2"}), 60.0, names)
    assert np.array_equal(a, b), "the same seed gives different wind"
    assert not np.array_equal(a, c), "another seed gives the same wind"


@pytest.mark.parametrize("z", [3.0, 10.0, 30.0])
def test_mean_wind_is_u10_times_the_profile(environment, z):
    fmu = environment({("wind", "reference_height"): str(z)})
    assert fmu.mean_wind_speed == pytest.approx(fmu.mean_speed_10m * mean_wind_profile(z, KAPPA),
                                                abs=1e-12)


def test_height_profile():
    assert mean_wind_profile(10.0, KAPPA) == pytest.approx(1.0, abs=1e-12)
    assert mean_wind_profile(3.0, KAPPA) < 1.0 < mean_wind_profile(30.0, KAPPA)
    # Log law: U(z2) - U(z1) = U10 * 5/2 sqrt(kappa) ln(z2 / z1).
    difference = mean_wind_profile(30.0, KAPPA) - mean_wind_profile(3.0, KAPPA)
    assert difference == pytest.approx(2.5 * KAPPA ** 0.5 * log(10.0), abs=1e-12)


def test_saturation_of_the_slowly_varying_wind(environment):
    mean = environment().mean_wind_speed
    low, high = mean - 1.0, mean + 1.0
    fmu = environment({("wind.slowly_varying", "std"): "3.0",
                       ("wind.slowly_varying", "time_constant"): "10.0",
                       ("wind.slowly_varying", "min_speed"): str(low),
                       ("wind.slowly_varying", "max_speed"): str(high)})
    speed = fmu.mean_wind_speed + run(fmu, 600.0, ("slowly_varying_wind_speed",))[:, 0]
    assert speed.min() >= low - 1e-9 and speed.max() <= high + 1e-9
    # Both limits are reached, so the test really tests the saturation.
    assert speed.min() == pytest.approx(low) and speed.max() == pytest.approx(high)
