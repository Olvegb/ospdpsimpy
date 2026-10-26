"""Regression checks for direct MSS tanker.mat -> LF/WF/WaveLoads pipeline.

The explicit zero-viscous-damping choice below is ONLY a code-path smoke test,
not a physically calibrated tanker model.
"""
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from app.build.data import default_config
from ospfmusim.hull.helpfunctions.Wamit import (load_mat_vessel, FREQUENCY_COUNT, HEADING_STEP_DEG,
                                                FREQUENCY_SPACING)
from ospfmusim.hull.helpfunctions.WaveLoads import ResponseTable, selected_drift_method

MAT = Path(__file__).resolve().parents[1] / "helpfunctions" / "data" / "tanker.mat"


@unittest.skipUnless(MAT.exists() and (MAT.parent / "GunnerusL35_20201112.mat").exists(),
                     "Requires external MATLAB datasets, which are not committed to Git")
class TestMatLoader(unittest.TestCase):
    def test_direct_mat_has_expected_shapes(self):
        data = load_mat_vessel(MAT)
        self.assertEqual(data["vessel_name"], "tanker")
        self.assertEqual(len(data["frequency_grid"]), FREQUENCY_COUNT)
        np.testing.assert_allclose(data["headings_deg"], np.arange(0, 360, 10))
        self.assertEqual(data["rigid_mass"].shape, (6, 6))
        self.assertEqual(data["added_mass_by_frequency"].shape, (6, 6, FREQUENCY_COUNT))
        self.assertEqual(data["radiation_damping_by_frequency"].shape, (6, 6, FREQUENCY_COUNT))
        self.assertIsInstance(data["motion_table"], ResponseTable)
        self.assertEqual(data["motion_table"].values.shape, (6, FREQUENCY_COUNT, 36))
        self.assertEqual(data["force_table"].values.shape, (3, FREQUENCY_COUNT, 36))
        self.assertEqual(data["drift_table"].values.shape, (3, FREQUENCY_COUNT, 36))
        for kind in ("motion_table", "force_table", "drift_table"):
            self.assertTrue(np.isfinite(data[kind].values).all())

    def test_mat_rao_matches_original_complex_source_endpoint(self):
        source = loadmat(MAT, simplify_cells=True)["vessel"]
        generated = load_mat_vessel(MAT)
        for src, output in (("motionRAO", "motion_table"),
                            ("forceRAO", "force_table")):
            original = source[src]
            expected = np.asarray(original["amp"][0])[:, :, 0][0, 0] * np.exp(
                -1j * np.asarray(original["phase"][0])[:, :, 0][0, 0])
            self.assertAlmostEqual(abs(generated[output].values[0, 0, 0] - expected), 0, places=8)
        self.assertAlmostEqual(
            generated["drift_table"].values[0, 0, 0],
            float(np.asarray(source["driftfrc"]["amp"][0])[0, 0, 0]), places=8)

    def test_lf_selection_is_not_invented(self):
        with self.assertRaisesRegex(ValueError, "LF frequency and viscous damping"):
            load_mat_vessel(MAT, drift_method="newman_6_97",
                            require_simulation_ready=True)
        with self.assertRaisesRegex(ValueError, "wave-drift approximation"):
            load_mat_vessel(
                MAT, lf_reference_omega_rad_s=0.8,
                additional_linear_damping_matrix_si=np.zeros((6, 6)),
                require_simulation_ready=True)

    def test_explicit_lf_values_are_finite(self):
        data = load_mat_vessel(MAT, lf_reference_omega_rad_s=0.8,
                               additional_linear_damping_matrix_si=np.zeros((6, 6)),
                               drift_method="newman_6_97",
                               require_simulation_ready=True)
        self.assertEqual(data["added_mass"].shape, (6, 6))
        self.assertEqual(data["damping"].shape, (6, 6))
        self.assertTrue(np.isfinite(data["damping"]).all())

    def test_fmu_can_initialize_from_mat_with_explicit_test_settings(self):
        from ospfmusim.hull.vessel import VesselBlock
        shared = default_config().read_text(encoding="utf-8-sig")
        self.assertIn("[vessel_fmu]", shared)
        test_block = (
            f'mat_file = "{MAT}"\n'
            'frequency_count = 50\nheading_step_deg = 10\n'
            'lf_reference_omega_rad_s = 0.8\n'
            'lf_viscous_damping_diagonal_si = [0, 0, 0, 0, 0, 0]\n'
            '# Drift method is selected in WaveLoads.py'
        )
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "config.toml"
            # Override only the active vessel_fmu block for this tanker smoke test.
            p.write_text(re.sub(r"^\[vessel_fmu\]\n.*?(?=^\[)",
                                lambda _: "[vessel_fmu]\n" + test_block + "\n\n",
                                shared, count=1, flags=re.MULTILINE | re.DOTALL))
            fm = VesselBlock(instance_name="mat_tanker_smoke")
            fm.config_path = str(p)
            fm.setup_experiment(0)
            fm.exit_initialization_mode()
            self.assertTrue(fm.ready)
            self.assertTrue(fm.do_step(0, 0.1))
            self.assertTrue(np.isfinite([fm.north, fm.east, fm.yaw]).all())

    def test_data_directory_contains_both_mat_files(self):
        names = {f.name for f in MAT.parent.iterdir()}
        self.assertEqual(names, {"tanker.mat", "GunnerusL35_20201112.mat"})

    def test_active_gunnerus_lf_receives_exact_loader_matrices(self):
        import tomllib
        from ospfmusim.hull.vessel import VesselBlock
        config = default_config()                # the config the simulation uses
        with config.open("rb") as stream:
            s = tomllib.load(stream)["vessel_fmu"]
        if "mat_file" not in s:
            self.skipTest(f"{config.name} uses no .mat file")
        mat_file = (config.parent / s["mat_file"]).resolve()     # relative to the config file
        self.assertTrue(mat_file.is_file(), mat_file)
        self.assertTrue(s["use_mat_viscous_damping"])
        expected = load_mat_vessel(
            mat_file,
            frequency_count=s.get("frequency_count", FREQUENCY_COUNT),
            heading_step_deg=s.get("heading_step_deg", HEADING_STEP_DEG),
            frequency_spacing=s.get("frequency_spacing", FREQUENCY_SPACING),
            lf_reference_omega_rad_s=s["lf_reference_omega_rad_s"],
            use_mat_viscous_damping=s["use_mat_viscous_damping"],
            drift_method=selected_drift_method(),
            require_simulation_ready=True,
        )
        fm = VesselBlock(instance_name="gunnerus_lf_regression")
        fm.config_path = str(config)
        fm.setup_experiment(0.0)
        fm.exit_initialization_mode()
        self.assertTrue(fm.ready)
        for key, attr in (
            ("rigid_mass", "M_RB"), ("added_mass", "A"),
            ("damping", "D"), ("restoring", "G"),
        ):
            np.testing.assert_array_equal(getattr(fm.lf_model, attr), expected[key])
        self.assertTrue(fm.do_step(0.0, 0.05))
        self.assertEqual(fm.wave_model.model.drift_method, selected_drift_method())


if __name__ == "__main__":
    unittest.main()
