"""Check the Environment FMU code WITHOUT building an FMU and without OSP.
Run from the repo folder:  python src/ospfmusim/enviroment/tests/check_environment.py

Creates the Environment class from enviroment/environment.py with the config
the simulation uses, or with a copy of it where some values are changed, and
checks:

  1. Everything switched off: the outputs are constant and equal to the config.
  2. Slowly varying wind (Gauss-Markov): zero mean, the right std and time
     constant, and the same statistics for another step size.
  3. Gusts: the std of the gust signal matches the NORSOK spectrum
     integrated over [f_min, f_max]. Zero mean.
  4. Wind direction: a constant turning rate gives mean + rate * t, and the
     output always stays in [0, 360).
  5. Current: north and east components match speed and direction.
  6. Seed: the same seed gives the same wind, another seed another wind.
  7. Mean wind height profile (Sorensen 2018, eq. 6.25): U10 at 10 m, less
     below and more above, and the log law between two heights.
  8. Saturation of the slowly varying wind (eq. 6.27): mean + slowly varying
     stays inside [min_speed, max_speed].
"""
import random
import tempfile
from math import cos, exp, log, radians, sin
from pathlib import Path

import numpy as np

from app.build.data import default_config
from ospfmusim.enviroment.environment import (Environment, GaussMarkov, mean_wind_profile,
                                              norsok_wind_spectrum)

CONFIG = default_config()   # the config the simulation uses (OspSystemStructure.xml)
DT = 0.1                    # time step [s], as in OspSystemStructure.xml
TMP = Path(tempfile.mkdtemp())


def config_with(changes):
    """A copy of the config the simulation uses with some values changed.

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
    path = TMP / f"config_{len(list(TMP.iterdir()))}.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def environment(changes=None):
    """A started Environment FMU, with the config the simulation uses or a changed copy of it."""
    fmu = Environment(instance_name="environment")
    fmu.config_path = str(config_with(changes) if changes else CONFIG)
    fmu.exit_initialization_mode()
    return fmu


def run(fmu, duration, names):
    """Steps the FMU and returns the outputs in names as arrays, one row per step."""
    rows = []
    for k in range(int(round(duration / DT))):
        fmu.do_step(k * DT, DT)
        rows.append([getattr(fmu, name) for name in names])
    return np.array(rows)


ALL_OFF = {
    ("wind.slowly_varying", "std"): "0.0",
    ("wind.gust", "n_components"): "0",
    ("wind.direction_variation", "rate_deg_per_s"): "0.0",
    ("wind.direction_variation", "std_deg"): "0.0",
}


def check_all_off():
    fmu = environment(ALL_OFF)
    names = ("mean_wind_speed", "slowly_varying_wind_speed", "gust_wind_speed",
             "wind_direction_deg")
    out = run(fmu, 60.0, names)
    expected = [fmu.mean_wind_speed, 0.0, 0.0, fmu.mean_wind_direction_deg % 360.0]
    print("1. Everything switched off (60 s)")
    print(f"   outputs: {dict(zip(names, [round(float(v), 6) for v in out[-1]]))}")
    assert np.allclose(out, expected), \
        "the outputs are not constant when every part is switched off"


def gauss_markov_series(time_constant, std, dt, duration, seed):
    process = GaussMarkov(time_constant, std, random.Random(seed))
    return np.array([process.step(dt) for _ in range(int(duration / dt))])


def check_slowly_varying():
    T, std = 300.0, 1.0
    duration = 1000 * T                          # many time constants, so the statistics settle
    print("2. Slowly varying wind (Gauss-Markov), T = 300 s, std = 1.0 m/s")
    for dt in (0.1, 1.0):
        x = gauss_markov_series(T, std, dt, duration, seed=7)
        lag = int(T / dt)
        correlation = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        print(f"   dt = {dt:3}: mean = {x.mean():6.3f}, std = {x.std():5.3f}, "
              f"correlation after T = {correlation:5.3f} (expected {exp(-1):.3f})")
        assert abs(x.mean()) < 0.2 * std, "the slowly varying wind does not have zero mean"
        assert abs(x.std() - std) < 0.1 * std, \
            "the std of the slowly varying wind is off by more than 10 %"
        assert abs(correlation - exp(-1)) < 0.1, \
            "the time constant of the slowly varying wind is wrong"


def check_gusts():
    fmu = environment({("wind.slowly_varying", "std"): "0.0"})
    gust = run(fmu, 3600.0, ("gust_wind_speed",))[:, 0]

    # The variance of the spectrum, integrated over [f_min, f_max] (log-spaced trapezoid).
    f = np.geomspace(0.005, 0.5, 5000)
    S = np.array([norsok_wind_spectrum(fi, fmu.mean_speed_10m, 10.0) for fi in f])
    std_spectrum = np.sqrt(np.trapezoid(S, f))
    std_components = np.sqrt(sum(a ** 2 / 2.0 for _, a, _ in fmu.gust_components))

    print("3. Gusts (one hour)")
    print(f"   std from the spectrum = {std_spectrum:.3f} m/s, "
          f"from the components = {std_components:.3f} m/s, "
          f"of the signal = {gust.std():.3f} m/s, mean = {gust.mean():.3f} m/s")
    assert abs(std_components - std_spectrum) < 0.1 * std_spectrum, \
        "the gust amplitudes do not match the spectrum"
    assert abs(gust.std() - std_spectrum) < 0.15 * std_spectrum, \
        "the gust signal does not match the spectrum"
    assert abs(gust.mean()) < 0.1 * std_spectrum, "the gusts do not have zero mean"


def check_direction():
    rate = 0.5                                   # deg/s, so the direction passes 360 several times
    fmu = environment({**ALL_OFF, ("wind.direction_variation", "rate_deg_per_s"): str(rate)})
    duration = 2000.0
    direction = run(fmu, duration, ("wind_direction_deg",))[:, 0]
    t = DT * np.arange(1, len(direction) + 1)
    expected = (fmu.mean_wind_direction_deg + rate * t) % 360.0
    error = (direction - expected + 180.0) % 360.0 - 180.0

    print(f"4. Wind direction, turning rate {rate} deg/s for {duration:.0f} s")
    print(f"   largest error = {np.abs(error).max():.2e} deg, "
          f"range = [{direction.min():.2f}, {direction.max():.2f}] deg")
    assert np.abs(error).max() < 1e-9, "the direction does not follow mean + rate * t"
    assert direction.min() >= 0.0 and direction.max() < 360.0, "the direction is outside [0, 360)"

    fmu = environment()                          # default config: slow random variation
    direction = run(fmu, 3600.0, ("wind_direction_deg",))[:, 0]
    assert direction.min() >= 0.0 and direction.max() < 360.0, "the direction is outside [0, 360)"


def check_current():
    print("5. Current components")
    for speed, towards in ((0.5, 0.0), (0.5, 90.0), (1.0, 225.0), (0.8, 370.0)):
        fmu = environment({("current", "speed"): str(speed),
                           ("current", "direction_deg"): str(towards)})
        expected = (speed * cos(radians(towards)), speed * sin(radians(towards)))
        print(f"   {speed} m/s towards {towards:5.1f} deg -> north = {fmu.current_north:6.3f}, "
              f"east = {fmu.current_east:6.3f}, "
              f"direction out = {fmu.current_direction_deg:5.1f} deg")
        assert np.allclose((fmu.current_north, fmu.current_east), expected), \
            "wrong current components"
        assert 0.0 <= fmu.current_direction_deg < 360.0, \
            "the current direction is outside [0, 360)"


def check_seed():
    names = ("slowly_varying_wind_speed", "gust_wind_speed", "wind_direction_deg")
    a = run(environment(), 60.0, names)
    b = run(environment(), 60.0, names)
    c = run(environment({("wind", "seed"): "2"}), 60.0, names)
    print("6. Seed")
    print(f"   same seed: equal = {np.array_equal(a, b)};  "
          f"other seed: equal = {np.array_equal(a, c)}")
    assert np.array_equal(a, b), "the same seed gives different wind"
    assert not np.array_equal(a, c), "another seed gives the same wind"


def check_height_profile():
    kappa = 0.0026
    print("7. Mean wind height profile, kappa = 0.0026")
    for z in (3.0, 10.0, 30.0):
        fmu = environment({("wind", "reference_height"): str(z)})
        print(f"   z = {z:4.1f} m: U10 = {fmu.mean_speed_10m:.2f} m/s -> "
              f"mean = {fmu.mean_wind_speed:.3f} m/s")
        expected = fmu.mean_speed_10m * mean_wind_profile(z, kappa)
        assert abs(fmu.mean_wind_speed - expected) < 1e-12, \
            "mean_wind_speed is not U10 times the profile"
    assert abs(mean_wind_profile(10.0, kappa) - 1.0) < 1e-12, "the profile is not 1 at 10 m"
    assert mean_wind_profile(3.0, kappa) < 1.0 < mean_wind_profile(30.0, kappa), \
        "the mean wind does not increase with the height"
    # Log law: U(z2) - U(z1) = U10 * 5/2 sqrt(kappa) ln(z2 / z1).
    difference = mean_wind_profile(30.0, kappa) - mean_wind_profile(3.0, kappa)
    assert abs(difference - 2.5 * kappa ** 0.5 * log(10.0)) < 1e-12, \
        "the profile does not follow the log law"


def check_saturation():
    mean = environment().mean_wind_speed
    low, high = mean - 1.0, mean + 1.0
    fmu = environment({("wind.slowly_varying", "std"): "3.0",
                       ("wind.slowly_varying", "time_constant"): "10.0",
                       ("wind.slowly_varying", "min_speed"): str(low),
                       ("wind.slowly_varying", "max_speed"): str(high)})
    speed = fmu.mean_wind_speed + run(fmu, 600.0, ("slowly_varying_wind_speed",))[:, 0]
    print("8. Saturation, std = 3 m/s, limits mean -/+ 1 m/s")
    print(f"   mean + slowly varying in [{speed.min():.3f}, {speed.max():.3f}] m/s, "
          f"limits [{low:.3f}, {high:.3f}] m/s")
    assert speed.min() >= low - 1e-9 and speed.max() <= high + 1e-9, \
        "mean + slowly varying is outside the limits"
    assert np.isclose(speed.min(), low) and np.isclose(speed.max(), high), \
        "the limits are never reached, so the check does not test the saturation"


if __name__ == "__main__":
    check_all_off()
    check_slowly_varying()
    check_gusts()
    check_direction()
    check_current()
    check_seed()
    check_height_profile()
    check_saturation()
    print("All Environment checks passed.")
