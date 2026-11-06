"""Regression checks for the independent MSS WAMIT tanker validation case.

The tanker is deliberately *not* a calibrated Gunnerus-style DP model.
Its additional viscous damping is an explicit exploratory assumption.
"""
from pathlib import Path
import tomllib

import numpy as np
import pytest

from ospfmusim.hull.helpfunctions.Wamit import load_mat_vessel
from ospfmusim.hull.helpfunctions.WaveLoads import WaveComponents, WaveLoads

ROOT = Path(__file__).resolve().parents[1]
MAT = ROOT / "data_examples" / "tanker.mat"
CONFIG = ROOT / "configs" / "tanker_wave_free.toml"


@pytest.fixture(scope="module")
def hydro():
    with CONFIG.open("rb") as stream:
        config = tomllib.load(stream)["vessel_fmu"]
    return load_mat_vessel(
        MAT,
        frequency_count=config["frequency_count"],
        heading_step_deg=config["heading_step_deg"],
        frequency_spacing=config["frequency_spacing"],
        lf_reference_omega_rad_s=config["lf_reference_omega_rad_s"],
        additional_linear_damping_matrix_si=np.diag(
            config["lf_viscous_damping_diagonal_si"]
        ),
        use_mat_viscous_damping=False,
        drift_method="newman_absolute",
        require_simulation_ready=True,
    )


def test_tanker_loaded_and_finite(hydro):
    info = hydro["frequency_info"]
    assert hydro["vessel_name"].lower() == "tanker"
    assert info["source_calculated_count"] == 58
    assert info["output_count"] == 100
    assert info["source_heading_count"] == 36
    assert hydro["added_mass_by_frequency"].shape == (6, 6, 100)
    assert np.all(np.isfinite(hydro["added_mass_by_frequency"]))
    assert np.all(np.isfinite(hydro["radiation_damping_by_frequency"]))
    M = hydro["rigid_mass"] + hydro["added_mass"]
    assert np.linalg.eigvalsh(0.5 * (M + M.T)).min() > 0
    assert hydro["drift_table"].values.min() < 0
    assert hydro["drift_table"].values.max() > 0


@pytest.mark.parametrize("method", ["newman_6_97", "newman_absolute"])
def test_regular_wave_drift_matches_diagonal_mean(hydro, method):
    wave = WaveLoads(hydro["force_table"], hydro["drift_table"],
                     gravity=hydro["gravity"], drift_method=method)
    wave.set_components(WaveComponents(
        [1.0], [2 * np.pi / 9], [0.3], [np.deg2rad(225)]
    ))
    for t in (1.0, 4.5, 10.0):
        forces = wave.calculate(t, np.zeros(6))
        np.testing.assert_allclose(forces.tau_second_order,
                                   forces.tau_mean, rtol=1e-12, atol=1e-6)
        assert np.all(np.isfinite(forces.tau_first_order))


def test_two_newman_methods_are_not_same_formula(hydro):
    wave_loads = []
    for method in ("newman_6_97", "newman_absolute"):
        wave = WaveLoads(hydro["force_table"], hydro["drift_table"],
                         gravity=hydro["gravity"], drift_method=method)
        wave.set_components(WaveComponents(
            [0.7, 0.6], [0.62, 0.78], [0.1, 1.1],
            np.deg2rad([225.0, 235.0])
        ))
        wave_loads.append(wave.calculate(5.0, np.zeros(6)).tau_second_order)
    assert np.linalg.norm(wave_loads[0] - wave_loads[1]) > 1e-6
