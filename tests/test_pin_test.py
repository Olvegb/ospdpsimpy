"""The pin test: with pin_test = true in [simulation] the vessel stays at its initial pose and
still reports the loads on it. Runs the VesselBlock class with the .mat file the config reads,
no FMU build.

The wave drift is checked against physics, not against numbers from the same file, so these
tests fail if the file is wrong. They check symmetry and direction, not the size of the drift.

License: MIT (see LICENSE in the repository root)
Authors: William Boland (2026)
"""
import numpy as np
import pytest

from app.build.config_tools import make_paths_absolute, set_values
from app.build.data import default_config, load_config
from ospfmusim.hull.vessel import POSE, VELOCITY, VesselBlock

CONFIG = default_config()   # the config the simulation uses (OspSystemStructure.xml)
MAT = CONFIG.parent / load_config(CONFIG)["vessel_fmu"]["mat_file"]
PERIOD = 5.0                # wave length about the hull length: large drift, clear sign [s]

pytestmark = pytest.mark.skipif(not MAT.is_file(),
                                reason="Requires external MATLAB datasets, "
                                       "which are not committed to Git")


def run_vessel(tmp_path, pin_test, wave_from_deg=10.0):
    """2 s with a regular wave with H = 5 m from wave_from_deg, wind and thrust."""
    text = make_paths_absolute(CONFIG.read_text(encoding="utf-8-sig"), CONFIG.parent)
    path = tmp_path / "config.toml"
    path.write_text(set_values(text, {("simulation", "pin_test"): pin_test}), encoding="utf-8")
    vessel = VesselBlock(instance_name="pin_test")
    vessel.config_path = str(path)
    vessel.wave1_amplitude, vessel.wave1_omega = 2.5, 2*np.pi/PERIOD
    vessel.wave1_direction_deg = (wave_from_deg + 180.0) % 360  # the input is where it travels to
    vessel.mean_wind_speed, vessel.X_thrusters = 15.0, 1e5
    vessel.setup_experiment(0.0)
    vessel.exit_initialization_mode()
    for k in range(20):
        vessel.do_step(k*0.1, 0.1)
    return vessel


def drift(vessel, prefix="wave_mean_"):
    return np.array([getattr(vessel, prefix + dof) for dof in "XYN"])


def test_pinned_vessel_stays_still_and_reports_the_loads(tmp_path):
    vessel = run_vessel(tmp_path, True)
    names = [p + s for p in ("", "lf_", "wf_") for s in POSE] + ["lf_" + s for s in VELOCITY]
    assert {name: getattr(vessel, name) for name in names} == dict.fromkeys(names, 0.0)
    # In a regular wave the drift is constant and equal to its mean.
    np.testing.assert_allclose(drift(vessel, "wave_drift_"), drift(vessel), rtol=1e-3)
    assert np.all(drift(vessel) != 0)
    assert vessel.wind_X != 0 and vessel.wave_first_X != 0


def test_without_the_pin_the_same_inputs_move_the_vessel(tmp_path):
    vessel = run_vessel(tmp_path, False)
    assert vessel.lf_north != 0 and vessel.wf_north != 0


@pytest.mark.parametrize("wave_from_deg", [10.0, 45.0, 135.0, 170.0])
def test_mirrored_waves_give_mirrored_drift(tmp_path, wave_from_deg):
    """The hull is symmetric: from the other side, X stays and Y and N change sign."""
    starboard = drift(run_vessel(tmp_path, True, wave_from_deg))
    port = drift(run_vessel(tmp_path, True, 360.0 - wave_from_deg))
    np.testing.assert_allclose(starboard, port*[1, -1, -1], rtol=1e-3)


# Body frame: X forward, Y starboard. 0 = not checked, the drift is about zero there.
@pytest.mark.parametrize("wave_from_deg, x_sign, y_sign", [
    (0.0, -1, 0), (45.0, -1, -1), (90.0, 0, -1), (135.0, 1, -1),
    (180.0, 1, 0), (225.0, 1, 1), (270.0, 0, 1), (315.0, -1, 1)])
def test_drift_pushes_the_vessel_the_way_the_waves_travel(tmp_path, wave_from_deg, x_sign, y_sign):
    x, y, _ = drift(run_vessel(tmp_path, True, wave_from_deg))
    assert x*x_sign >= 0 and y*y_sign >= 0
