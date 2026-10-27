"""Regression tests for automatic MSS frequency and Bv dimension detection.

Tests are local if the untracked data_examples/gunnerus.mat is available.
No external WAMIT or MATLAB process is needed to run them.

License: MIT (see LICENSE in the repository root)
"""
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ospfmusim.hull.helpfunctions.Wamit import (
    _frequency_axis,
    _mat_viscous_at,
    load_mat_vessel,
)

MAT = Path(__file__).resolve().parents[1] / "data_examples" / "gunnerus.mat"


@unittest.skipUnless(MAT.is_file(), "Optional Gunnerus .mat is not included in Git")
class TestWamitFrequencyAutodetect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = loadmat(MAT, simplify_cells=True)["vessel"]
        cls.actual = np.asarray(cls.v["motionRAO"]["w"], dtype=float).ravel()

    def load(self, count):
        return load_mat_vessel(
            MAT, frequency_count=count, use_mat_viscous_damping=True,
            drift_method="newman_6_97", require_simulation_ready=True,
        )

    def test_100_uses_original_nonuniform_grid_without_resampling(self):
        data = self.load(100)
        np.testing.assert_array_equal(data["frequency_grid"], self.actual)
        np.testing.assert_array_equal(data["added_mass_by_frequency"],
                                      self.v["A"][:, :, 1:-1])
        np.testing.assert_array_equal(data["radiation_damping_by_frequency"],
                                      self.v["B"][:, :, 1:-1])
        np.testing.assert_array_equal(data["restoring_by_frequency"],
                                      self.v["C"][:, :, 1:-1])
        self.assertIn("no frequency resampling", data["frequency_info"]["grid_mode"])
        self.assertEqual(data["frequency_info"]["source_calculated_count"], 100)
        self.assertEqual(data["frequency_info"]["matrix_frequency_count"], 102)
        np.testing.assert_allclose(data["frequency_info"]["matrix_only_frequency_values_rad_s"],
                                   [0, 10])

    def test_100_does_not_change_original_motion_amplitude_or_phase(self):
        data = self.load(100)
        amplitude = np.asarray(self.v["motionRAO"]["amp"][0])[:, :, 0]
        phase = np.asarray(self.v["motionRAO"]["phase"][0])[:, :, 0]
        expected = amplitude * np.exp(-1j * phase)
        np.testing.assert_array_equal(data["motion_table"].values[0], expected)
        self.assertEqual(data["force_table"].values.shape, (3, 100, 36))
        self.assertEqual(data["drift_table"].values.shape, (3, 100, 36))

    def test_auto_100_count_using_none(self):
        data = self.load(None)
        np.testing.assert_array_equal(data["frequency_grid"], self.actual)

    def test_50_and_150_interpolation_has_correct_range_and_dimensions(self):
        for count in (50, 150):
            with self.subTest(requested_frequency_count=count):
                data = self.load(count)
                w = data["frequency_grid"]
                self.assertEqual(len(w), count)
                self.assertGreater(np.min(np.diff(w)), 0)
                self.assertEqual(w[0], self.actual[0])
                self.assertAlmostEqual(w[-1], self.actual[-1], places=12)
                self.assertIn("resampling", data["frequency_info"]["grid_mode"])
                self.assertEqual(data["added_mass_by_frequency"].shape, (6, 6, count))
                self.assertEqual(data["radiation_damping_by_frequency"].shape, (6, 6, count))
                self.assertEqual(data["motion_table"].values.shape, (6, count, 36))
                self.assertTrue(np.isfinite(data["motion_table"].values).all())

    def test_linear_resampling_is_supported(self):
        data = load_mat_vessel(MAT, frequency_count=50, frequency_spacing="linear")
        np.testing.assert_allclose(np.diff(data["frequency_grid"]),
                                   (self.actual[-1]-self.actual[0])/49)

    def test_bv_at_auto_lf_matches_actual_first_frequency_slice(self):
        data = self.load(100)
        np.testing.assert_allclose(data["added_mass"], self.v["A"][:, :, 1])
        np.testing.assert_allclose(data["damping"],
                                   self.v["B"][:, :, 1] + self.v["Bv"][:, :, 1])
        np.testing.assert_allclose(data["restoring"], self.v["C"][:, :, 1])
        self.assertEqual(data["frequency_info"]["lf_reference_rad_s"], self.actual[0])

    def test_custom_lf_reference_samples_bv_at_that_frequency(self):
        idx = 25
        w = float(self.actual[idx])
        data = load_mat_vessel(
            MAT, frequency_count=100, lf_reference_omega_rad_s=w,
            use_mat_viscous_damping=True, drift_method="newman_6_97",
            require_simulation_ready=True,
        )
        np.testing.assert_allclose(data["damping"],
                                   self.v["B"][:, :, idx+1] + self.v["Bv"][:, :, idx+1])

    def test_constant_bv_and_frequency_dependent_bv_are_supported(self):
        freqs = np.asarray(self.v["freqs"], dtype=float)
        test = np.eye(6)
        arr, origin = _mat_viscous_at(test, freqs, self.actual[0])
        np.testing.assert_array_equal(arr, test)
        self.assertIn("constant", origin)
        arr, origin = _mat_viscous_at(self.v["Bv"], freqs, self.actual[0])
        np.testing.assert_allclose(arr, self.v["Bv"][:, :, 1])
        self.assertIn("frequency-dependent", origin)

    def test_bad_frequency_and_bad_lf_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "frequency_count"):
            self.load(1)
        with self.assertRaisesRegex(ValueError, "LF reference"):
            load_mat_vessel(MAT, lf_reference_omega_rad_s=10,
                            use_mat_viscous_damping=True)
        with self.assertRaisesRegex(ValueError, "increasing"):
            _frequency_axis("bad", np.array([0.3, 0.2]))
        with self.assertRaisesRegex(ValueError, "increasing"):
            _frequency_axis("bad", np.array([-1, 0, 1]), allow_zero=True)


if __name__ == "__main__":
    unittest.main()
