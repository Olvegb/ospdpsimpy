"""Tests of the Waves FMU code (waves/waves.py) WITHOUT building an FMU and without OSP.

Creates the Waves class directly in Python and calls it the same way OSP would:
set config_path and the inputs -> exit_initialization_mode() -> do_step() again
and again. This works on any computer, also on Mac.

License: MIT (see LICENSE in the repository root)
Authors: Olve Grønås Birkeland, viljartveit (2026)
"""
import re
import tomllib

import numpy as np
import pytest

from app.build.data import default_config
from ospfmusim.waves.helpfunctions.sea_state import directions_to, sea_from_settings
from ospfmusim.waves.helpfunctions.settings import load_settings
from ospfmusim.waves.waves import COMPONENT_OUTPUTS, N_WAVES, SLOTS, Waves

CONFIG = default_config()   # the config the simulation uses (OspSystemStructure.xml)
# H_s, T_p and direction, as Environment sends them
WAVES = tomllib.loads(CONFIG.read_text(encoding="utf-8-sig"))["waves"]
DT = 0.1                    # time step [s], as in OspSystemStructure.xml


def make_fmu(config_path):
    """Create the FMU object and set what OSP sets: config_path and the inputs from Environment."""
    fmu = Waves(instance_name="waves")
    fmu.config_path = str(config_path)
    fmu.H_s, fmu.T_p, fmu.wave_direction_deg = WAVES["hs"], WAVES["tp"], WAVES["direction_deg"]
    return fmu


def read_outputs(fmu):
    """Read the wave component outputs back into four arrays.

    The arrays are amplitude, omega, phase and direction_deg.
    """
    return [np.array([getattr(fmu, f"wave{k}_{quantity}") for k in SLOTS])
            for quantity in COMPONENT_OUTPUTS]


def config_with(folder, **changes):
    """A copy of the config the simulation uses with some [waves] values changed.

    The copy is written to folder.
    """
    text = CONFIG.read_text(encoding="utf-8-sig")
    for key, value in changes.items():
        text, n = re.subn(rf'^{key} = "[^"]*"', f'{key} = "{value}"', text, count=1,
                          flags=re.MULTILINE)
        assert n == 1, f"{key} not found in {CONFIG}"
    file = folder / "config.toml"
    file.write_text(text, encoding="utf-8")
    return file


@pytest.fixture
def fmu():
    """A started Waves FMU, stepped for one minute."""
    fmu = make_fmu(CONFIG)
    fmu.exit_initialization_mode()
    t = 0.0
    while t < 60.0:
        fmu.do_step(t, DT)
        t += DT
    return fmu


def test_variables():
    fmu = make_fmu(CONFIG)
    n_parameters = sum(1 for v in fmu.vars.values() if v.causality.name == "parameter")
    n_outputs = sum(1 for v in fmu.vars.values() if v.causality.name == "output")
    assert n_parameters == 1                     # config_path
    assert n_outputs == 4 * N_WAVES + 1          # wave components x 4 + kept_energy_fraction


def test_outputs_are_set_before_the_first_step():
    fmu = make_fmu(CONFIG)
    fmu.exit_initialization_mode()
    amplitude, *_ = read_outputs(fmu)
    assert amplitude[0] > 0.0


def test_fmu_gives_the_same_components_as_the_helpfunctions(fmu):
    amplitude, omega, phase, direction = read_outputs(fmu)
    sea, kept = sea_from_settings(WAVES["hs"], WAVES["tp"], load_settings(CONFIG), N_WAVES)
    np.testing.assert_allclose(amplitude, sea.amplitudes)
    np.testing.assert_allclose(omega, sea.frequencies)
    np.testing.assert_allclose(phase, sea.phases)
    np.testing.assert_allclose(direction,
                               np.degrees(directions_to(sea, WAVES["direction_deg"], True)))
    assert fmu.kept_energy_fraction == pytest.approx(kept)


def test_new_direction_only_turns_the_waves(fmu):
    amplitude, omega, phase, direction = read_outputs(fmu)
    fmu.wave_direction_deg = WAVES["direction_deg"] + 90.0
    fmu.do_step(60.0, DT)
    amplitude2, omega2, phase2, direction2 = read_outputs(fmu)
    np.testing.assert_array_equal(amplitude2, amplitude)
    np.testing.assert_array_equal(omega2, omega)
    np.testing.assert_array_equal(phase2, phase)
    np.testing.assert_allclose((direction2 - direction) % 360.0, 90.0)


def test_calm_sea(fmu):
    fmu.H_s = 0.0
    fmu.do_step(60.0, DT)
    amplitude, *_ = read_outputs(fmu)
    assert np.all(amplitude == 0.0)


def test_regular_sea(tmp_path):
    fmu = make_fmu(config_with(tmp_path, sea_type="regular"))
    fmu.exit_initialization_mode()
    fmu.do_step(0.0, DT)
    amplitude, omega, phase, direction = read_outputs(fmu)
    assert amplitude[0] == pytest.approx(WAVES["hs"] / 2)
    assert omega[0] == pytest.approx(2 * np.pi / WAVES["tp"])
    assert phase[0] == 0.0
    assert direction[0] == pytest.approx((WAVES["direction_deg"] + 180) % 360)
    assert np.all(amplitude[1:] == 0.0)
    assert fmu.kept_energy_fraction == 1.0


def test_missing_config_path_is_caught():
    with pytest.raises(ValueError, match="config_path"):
        Waves(instance_name="no_config").exit_initialization_mode()


def test_wrong_name_in_config_is_caught_at_the_start(tmp_path):
    with pytest.raises(ValueError, match="long_crested"):
        make_fmu(config_with(tmp_path, sea_type="long_crested")).exit_initialization_mode()
