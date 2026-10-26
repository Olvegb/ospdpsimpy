"""Tests of the wave spectra (waves/helpfunctions/spectra.py).

License: MIT (see LICENSE in the repository root)
Authors: Viljar Simonnes Tveit, Olve Grønås Birkeland (2026)
"""
import numpy as np
import pytest

from ospfmusim.waves.helpfunctions.spectra import jonswap, pierson_moskowitz

HS = 3.0                                      # significant wave height [m]
TP = 9.0                                      # peak period [s]
OMEGA = np.linspace(0.01, 5.0, 5000)          # frequencies [rad/s]


def test_pierson_moskowitz_gives_hs_and_peak():
    S = pierson_moskowitz(OMEGA, HS, TP)
    m0 = np.trapezoid(S, OMEGA)               # area under the spectrum
    assert 4 * np.sqrt(m0) == pytest.approx(HS, rel=2e-3)
    assert OMEGA[np.argmax(S)] == pytest.approx(2 * np.pi / TP, abs=2e-3)


@pytest.mark.parametrize("gamma", [1.0, 3.3, 5.0])
def test_jonswap_gives_hs(gamma):
    m0 = np.trapezoid(jonswap(OMEGA, HS, TP, gamma), OMEGA)
    assert 4 * np.sqrt(m0) == pytest.approx(HS, rel=2e-3)


def test_jonswap_peak_grows_with_gamma():
    peaks = [jonswap(OMEGA, HS, TP, gamma).max() for gamma in (1.0, 3.3, 5.0)]
    assert peaks == sorted(peaks)


def test_jonswap_with_gamma_1_is_pierson_moskowitz():
    np.testing.assert_allclose(jonswap(OMEGA, HS, TP, 1.0), pierson_moskowitz(OMEGA, HS, TP))
