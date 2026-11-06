"""Regression tests for the central hull control panel.

Default behavior must preserve original Gunnerus hydrodynamic matrices and
all existing LF dynamics. Explicit choices must reach the FMU.
"""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from ospfmusim.hull.Control_Panel_Hull import (
    apply_hull_damping, resolve_hull_controls,
)
from ospfmusim.hull.helpfunctions.LF import LF
from ospfmusim.hull.helpfunctions.Wamit import load_vessel
from ospfmusim.hull.vessel import VesselBlock


REPO = Path(__file__).resolve().parents[1]
GUNNERUS = REPO / "data_examples" / "gunnerus2.mat"
CONFIG = REPO / "configs" / "gunnerus.toml"


def test_panel_baseline_preserves_matlab_damping():
    c = resolve_hull_controls({})
    data = load_vessel()
    expected = data["damping"]
    np.testing.assert_array_equal(apply_hull_damping(expected, c), expected)
    assert c["drift_method"] == "newman_absolute"
    assert c["frequency_count"] == 100


def test_add_and_replace_damping_and_overrides():
    base = np.diag([2., 3., 4., 5., 6., 7.])
    override = {"hull_control": {
        "damping_mode": "add",
        "damping_diagonal_si": [10., 0., 0., 0., 0., 0.],
        "frequency_count": 50,
        "heading_step_deg": 20,
        "watch_roll_deg": 25,
    }}
    c = resolve_hull_controls(override)
    np.testing.assert_array_equal(
        np.diag(apply_hull_damping(base, c)),
        [12., 3., 4., 5., 6., 7.],
    )
    assert c["frequency_count"] == 50
    assert c["heading_step_deg"] == 20
    assert c["watch"]["roll_deg"] == 25

    c = resolve_hull_controls({"hull_control": {
        "damping_mode": "replace",
        "damping_diagonal_si": [1., 2., 3., 4., 5., 6.],
    }})
    np.testing.assert_array_equal(
        np.diag(apply_hull_damping(base, c)),
        [1., 2., 3., 4., 5., 6.],
    )


def test_rejects_invalid_configuration():
    for config in (
        {"heading_step_deg": 11},
        {"frequency_count": 1},
        {"damping_mode": "unknown"},
        {"log_interval_s": 0},
        {"damping_matrix_si": [[1, 2]]},
    ):
        with pytest.raises(ValueError):
            resolve_hull_controls({"hull_control": config})


def test_default_lf_dynamics_matches_pre_refactor_golden_state():
    # Reference generated from the user's LF.py BEFORE control-panel edits.
    # Persisted here so the regression survives removal of /tmp backups.
    mass = np.diag([5e5, 8e5, 1e6, 4e6, 1e7, 5e7])
    added = np.diag([1e5, 2e5, 2e5, 5e5, 1e6, 2e6])
    damping = np.diag([1000., 1000., 2000., 3000., 4000., 5000.])
    restoring = np.diag([0., 0., 2e5, 2e6, 3e6, 0.])
    x = np.array([2., -1., .1, .01, -.02, .1, .2, -.1, .01, .005, -.004, .006])
    force = np.array([1000., -700., 0., 100., 0., 400.])

    model = LF(mass, added, damping, restoring)
    for _ in range(100):
        x = model.step(x, .02, tau_wave_drift=force)

    golden = np.array([
        2.4201167989430545, -1.1588443146155882, 0.09169022007207135,
        0.009468683479899452, -0.016752885050503125, 0.11236676456543986,
        0.20075673214080614, -0.10260362995120496, -0.023530654698746167,
        -0.0054651795726743885, 0.006864237530191147, 0.006329020402465457,
    ])
    np.testing.assert_allclose(x, golden, rtol=1e-13, atol=1e-13)


def test_fmu_obeys_panel_config(tmp_path):
    text = CONFIG.read_text()
    text = text.replace(
        'mat_file = "../data_examples/gunnerus2.mat"',
        f'mat_file = "{GUNNERUS}"',
    )
    assert f'mat_file = "{GUNNERUS}"' in text
    text += ("\n[hull_control]\n"
             "damping_mode = 'add'\n"
             "damping_diagonal_si = [900, 0, 0, 0, 0, 0]\n"
             "drift_method = 'newman'\n"
             "max_internal_step_s = 0.0125\n"
             "watch_roll_deg = 30\n")
    config = tmp_path / "config.toml"
    config.write_text(text)
    fm = VesselBlock(instance_name="hull_control_test")
    fm.config_path = str(config)
    fm.setup_experiment(0.)
    fm.exit_initialization_mode()
    assert fm.ready
    assert fm.max_internal_step_s == 0.0125
    assert fm.hull_controls["watch"]["roll_deg"] == 30
    assert fm.wave_model.model.drift_method == "newman_6_97"
    assert fm.lf_model.D[0, 0] == pytest.approx(900.)
    assert fm.lf_model.D[3, 3] == pytest.approx(1e6)
    assert fm.do_step(0., 0.025)
