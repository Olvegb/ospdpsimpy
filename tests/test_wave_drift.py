"""Analytical tests of the wave drift, independent of the fast single-sum implementation.

License: MIT (see LICENSE in the repository root)
Authors: Araz, Olve Grønås Birkeland (2026)
"""
import unittest
from unittest.mock import patch

import numpy as np

from ospfmusim.hull.helpfunctions.WaveLoads import WaveLoadModel, selected_drift_method


class Tables:
    def __init__(self, coefficients):
        self.coefficients = np.asarray(coefficients, float)

    def tables_at(self, frequencies):
        table = np.repeat(self.coefficients[:, :, None], 3, axis=2)
        return np.zeros_like(table, dtype=complex), table

    def heading_weights(self, beta):
        return np.zeros(len(beta), int), np.ones(len(beta), int), np.zeros(len(beta))


def model(coefficients, method='sorensen_6_99', amplitudes=None, filtered=True):
    result = WaveLoadModel(Tables(coefficients), 9.81, drift_method=method,
                           remove_sum_frequency=filtered)
    n = len(coefficients[0])
    result.set_waves(np.linspace(.3, .9, n) if amplitudes is None else amplitudes,
                     np.linspace(.7, 1.3, n), np.linspace(.2, .8, n), np.zeros(n))
    return result


class DriftChecks(unittest.TestCase):
    def test_top_level_control_selects_both_methods(self):
        with patch("ospfmusim.hull.helpfunctions.WaveLoads.DRIFT_METHOD", "newman"):
            self.assertEqual(selected_drift_method(), "newman_6_97")
            self.assertEqual(WaveLoadModel(Tables([[2.], [3.], [-4.]]), 9.81).drift_method,
                             "newman_6_97")
        with patch("ospfmusim.hull.helpfunctions.WaveLoads.DRIFT_METHOD", "sorensen"):
            self.assertEqual(selected_drift_method(), "sorensen_6_99")
            self.assertEqual(WaveLoadModel(Tables([[2.], [3.], [4.]]), 9.81).drift_method,
                             "sorensen_6_99")
        with patch("ospfmusim.hull.helpfunctions.WaveLoads.DRIFT_METHOD", "invalid"):
            with self.assertRaisesRegex(ValueError, "DRIFT_METHOD"):
                selected_drift_method()

    def test_sorensen_control_refuses_signed_gunnerus(self):
        from ospfmusim.hull.helpfunctions.Wamit import load_gunnerus
        with patch("ospfmusim.hull.helpfunctions.WaveLoads.DRIFT_METHOD", "sorensen"):
            with self.assertRaisesRegex(ValueError, "Signed drift"):
                load_gunnerus()

    def test_filtered_double_sum(self):
        diagonal = np.array([[4, 9, 16], [1, 9, 25], [0, 9, 16.]])
        m = model(diagonal)
        self.assertEqual(m.drift_method, 'sorensen_6_99')
        for t in (0., 1.3, 5.7, 30.):
            phase = m.frequencies*t-m.phases
            expected = np.zeros(3)
            for dof in range(3):
                for i in range(3):
                    for j in range(3):
                        q = np.sqrt(diagonal[dof, i]*diagonal[dof, j])
                        expected[dof] += (q*m.amplitudes[i]*m.amplitudes[j]
                                          * np.cos(phase[i] - phase[j]))
            np.testing.assert_allclose(m.drift(t, (0, 0, 0)), expected, atol=1e-13)
        np.testing.assert_allclose(m.mean_drift(0), diagonal @ m.amplitudes**2)

    def test_literal_and_filtered_single_wave(self):
        single = model([[4], [9], [0]], amplitudes=[.5])
        raw = model([[4], [9], [0]], amplitudes=[.5], filtered=False)
        for t in (0., 1., 5.):
            np.testing.assert_allclose(single.drift(t, (0, 0, 0)), [1, 2.25, 0])
            np.testing.assert_allclose(raw.drift(t, (0, 0, 0)),
                                       2*np.array([1, 2.25, 0])*np.cos(.7*t-.2)**2)

    def test_strict_negative_rejection_and_inactive_slots(self):
        coefficients = [[4, -9], [1, -16], [0, -25]]
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            model(coefficients, 'sorensen_6_99').drift(0, (0, 0, 0))
        inactive = model(coefficients, 'sorensen_6_99', amplitudes=[.5, 0])
        np.testing.assert_allclose(inactive.drift(0, (0, 0, 0)), [1, .25, 0])
        calm = model(coefficients, amplitudes=[0, 0])
        np.testing.assert_array_equal(calm.drift(0, (0, 0, 0)), 0)


if __name__ == '__main__':
    unittest.main()
