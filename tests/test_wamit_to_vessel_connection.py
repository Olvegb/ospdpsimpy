"""One targeted test: Gunnerus .mat -> Wamit.py -> vessel.py -> LF and WF.

This initializes only the VesselBlock module, NOT the full seven-FMU
co-simulation. WindModel and WaveLoads are left unchanged. It checks data
delivery rather than claiming full hydrodynamic validation.

License: MIT (see repository LICENSE).
"""
from pathlib import Path
import tomllib

import numpy as np
from scipy.io import loadmat

from ospfmusim.hull.vessel import VesselBlock
from ospfmusim.hull.helpfunctions.Wamit import (
    FREQUENCY_COUNT, load_mat_vessel
)
from ospfmusim.hull.helpfunctions.WaveLoads import selected_drift_method


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "gunnerus.toml"


def test_wamit_to_vessel_lf_wf_connection():
    with CONFIG.open("rb") as f:
        settings = tomllib.load(f)["vessel_fmu"]

    assert "frequency_count" not in settings, (
        "The only frequency-count user setting should be FREQUENCY_COUNT in Wamit.py"
    )
    source = (CONFIG.parent / settings["mat_file"]).resolve()
    assert source.is_file(), f"MATLAB file missing: {source}"
    assert source.name == "GunnerusL35_20201112.mat"

    raw = loadmat(source, simplify_cells=True)["vessel"]
    source_omega = np.asarray(raw["motionRAO"]["w"], dtype=float).ravel()
    print(f"\nMAT file: {source}")
    print(f"WAMIT source: {len(source_omega)} calculated frequencies, "
          f"{source_omega[0]:.9f} to {source_omega[-1]:.9f} rad/s")
    print(f"Selected FREQUENCY_COUNT: {FREQUENCY_COUNT}")

    expected = load_mat_vessel(
        source,
        frequency_count=FREQUENCY_COUNT,
        use_mat_viscous_damping=True,
        drift_method=selected_drift_method(),
        require_simulation_ready=True,
    )
    info = expected["frequency_info"]
    print("Wamit.py decision:", info["grid_mode"])
    print("Input Bv matrix shape:", np.shape(raw["Bv"]))
    print("LF damping source:", info["viscous_damping_source"])

    # Exercise Olve's real vessel.py initialization path; no OSP orchestration.
    vessel = VesselBlock(instance_name="wamit_lf_wf_connection")
    vessel.config_path = str(CONFIG)
    vessel.setup_experiment(0.0)
    vessel.exit_initialization_mode()
    assert vessel.ready

    # LF.py received exactly the source matrices chosen by Wamit.py.
    lf = vessel.lf_model
    for model_attr, loader_field in (
        ("M_RB", "rigid_mass"),
        ("A", "added_mass"),
        ("D", "damping"),
        ("G", "restoring"),
    ):
        np.testing.assert_array_equal(
            getattr(lf, model_attr), expected[loader_field],
            err_msg=f"LF.{model_attr} does not match Wamit.py {loader_field}"
        )
        print(f"LF.{model_attr} OK: {getattr(lf, model_attr).shape}")

    # WF.py received the complete six-DOF complex motion RAOs and grids.
    wf = vessel.wf_model
    np.testing.assert_array_equal(
        wf.table.frequencies, expected["motion_table"].frequencies
    )
    np.testing.assert_array_equal(
        wf.table.headings, expected["motion_table"].headings
    )
    np.testing.assert_array_equal(
        wf.table.values, expected["motion_table"].values
    )
    assert wf.table.values.shape == (6, len(expected["frequency_grid"]), 36)
    print(f"WF motion RAOs OK: {wf.table.values.shape}")

    if len(source_omega) == FREQUENCY_COUNT:
        # Extra assurance: no accidental interpolation when counts match.
        np.testing.assert_array_equal(expected["frequency_grid"], source_omega)
        np.testing.assert_allclose(lf.M_RB, raw["MRB"], rtol=0, atol=0)
        np.testing.assert_allclose(lf.A, raw["A"][:, :, 1], rtol=0, atol=0)
        np.testing.assert_allclose(
            lf.D, raw["B"][:, :, 1] + raw["Bv"][:, :, 1], rtol=0, atol=0
        )
        np.testing.assert_allclose(lf.G, raw["C"][:, :, 1], rtol=0, atol=0)
        # Test the stored motion amplitude and phase, not just table size.
        amplitude = np.asarray(raw["motionRAO"]["amp"][0])[:, :, 0]
        phase = np.asarray(raw["motionRAO"]["phase"][0])[:, :, 0]
        expected_surge = amplitude * np.exp(-1j * phase)
        np.testing.assert_array_equal(wf.table.values[0], expected_surge)
        print("Original WAMIT omega and source surge amplitudes/phases: exact match")

    assert np.isfinite(lf.M).all()
    assert np.isfinite(wf.table.values).all()
    print("PASS: Wamit.py -> vessel.py -> LF.py and WF.py data connection")
    print("No full simulation run; wind and wave-load algorithms unchanged.")
