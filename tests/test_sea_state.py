"""Tests of the sea state (waves/helpfunctions/sea_state.py).

License: MIT (see LICENSE in the repository root)
Authors: Viljar Simonnes Tveit, Olve Grønås Birkeland (2026)
"""
import numpy as np
import pytest

from ospfmusim.waves.helpfunctions.sea_state import (directions_to, elevation, keep_strongest,
                                                     realize, sea_from_settings)
from ospfmusim.waves.helpfunctions.spectra import cos2s

HS = 3.0     # significant wave height [m]
TP = 9.0     # peak period [s]
SETTINGS = {"sea_type": "irregular", "spectrum_type": "jonswap", "gamma": 3.3,
            "spreading_type": "cos2s", "spreading_s": 2, "n_directions": 11,
            "direction_limit_deg": 90, "n_frequencies": 30, "omega_min_factor": 0.25,
            "omega_max_factor": 3.0, "random_frequencies": True, "random_directions": True,
            "seed": 123}


def make_sea(spreading_type, random_frequencies=True, seed=123):
    """Short helper so each test only shows what it changes."""
    return realize(HS, TP, "jonswap", 3.3,
                   spreading_type=spreading_type, s=2, n_directions=11, direction_limit_deg=90,
                   n_frequencies=30, omega_min_factor=0.25, omega_max_factor=3.0,
                   random_frequencies=random_frequencies, random_directions=True, seed=seed)


def envelope(t, sea):
    """Height of the wave groups (the slowly varying envelope of the waves)."""
    t = t[:, None]
    return np.abs(np.sum(sea.amplitudes * np.exp(1j * (sea.frequencies * t - sea.phases)), axis=1))


@pytest.mark.parametrize("s", [1, 2, 10])
def test_spreading_function_has_area_1(s):
    theta = np.linspace(-np.pi / 2, np.pi / 2, 2001)
    assert np.trapezoid(cos2s(theta, s), theta) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("spreading_type", ["none", "cos2s"])
def test_components_give_hs(spreading_type):
    sea = make_sea(spreading_type)
    m0 = np.sum(sea.amplitudes**2 / 2)                     # energy in the components
    zeta = elevation(np.arange(0.0, 3 * 3600.0, 0.5), sea)  # three hours of wave elevation
    assert 4 * np.sqrt(m0) == pytest.approx(HS, rel=0.02)
    assert 4 * np.std(zeta) == pytest.approx(4 * np.sqrt(m0), rel=0.01)


def test_cos2s_directions_spread_around_the_mean():
    sea = make_sea("cos2s")
    energy = sea.amplitudes**2 / 2
    mean_offset = np.degrees(np.sum(energy * sea.direction_offsets) / np.sum(energy))
    within_30 = np.sum(energy[np.abs(sea.direction_offsets) <= np.radians(30)]) / np.sum(energy)
    assert abs(mean_offset) < 2.0
    assert within_30 > 0.5
    assert np.abs(sea.direction_offsets).max() <= np.radians(90)


@pytest.mark.parametrize("random_frequencies, repeats", [(False, True), (True, False)])
def test_wave_groups_repeat_only_on_an_even_grid(random_frequencies, repeats):
    d_omega = (3.0 - 0.25) * (2 * np.pi / TP) / (30 - 1)
    t_repeat = 2 * np.pi / d_omega               # repeat period of an evenly spaced grid
    t = np.arange(0.0, 200.0, 0.5)
    sea = make_sea("none", random_frequencies)
    assert np.allclose(envelope(t, sea), envelope(t + t_repeat, sea)) == repeats


def test_wrong_spreading_name_is_caught():
    with pytest.raises(ValueError, match="cos-2s"):
        realize(HS, TP, "jonswap", 3.3, "cos-2s", 2, 11, 90, 30, 0.25, 3.0, True, True, seed=123)


@pytest.mark.parametrize("n_waves, n_with_waves", [(100, 100), (330, 330), (400, 330)])
def test_keep_strongest_sends_out_n_waves(n_waves, n_with_waves):
    sea = make_sea("cos2s")                                 # 30 x 11 = 330 components
    strongest, kept = keep_strongest(sea, n_waves)
    hs_kept = 4 * np.sqrt(np.sum(0.5 * strongest.amplitudes**2))
    assert len(strongest.amplitudes) == n_waves
    assert np.count_nonzero(strongest.amplitudes) == n_with_waves
    assert hs_kept == pytest.approx(4 * np.sqrt(np.sum(0.5 * sea.amplitudes**2) * kept))
    if n_with_waves == 330:
        assert kept == pytest.approx(1.0)
    else:
        assert kept < 1.0


def test_keep_strongest_keeps_the_strongest_first():
    sea = make_sea("cos2s")
    strongest, _ = keep_strongest(sea, 100)
    energy = 0.5 * strongest.amplitudes**2
    assert np.all(np.diff(energy) <= 0)
    assert energy[-1] >= np.sort(0.5 * sea.amplitudes**2)[::-1][100]


@pytest.mark.parametrize("from_deg", [0.0, 45.0, 270.0])
def test_waves_travel_away_from_where_they_come_from(from_deg):
    long_crested, _ = keep_strongest(make_sea("none"), 5)
    to_deg = np.degrees(directions_to(long_crested, from_deg, direction_is_from=True)[0])
    assert to_deg == pytest.approx((from_deg + 180) % 360)


def test_regular_sea_is_one_wave():
    regular, kept = sea_from_settings(HS, TP, dict(SETTINGS, sea_type="regular"), 100)
    assert np.count_nonzero(regular.amplitudes) == 1
    assert regular.amplitudes[0] == pytest.approx(HS / 2)
    assert regular.frequencies[0] == pytest.approx(2 * np.pi / TP)
    assert regular.direction_offsets[0] == 0.0
    assert kept == pytest.approx(1.0)
    zeta = elevation(np.arange(0.0, TP, 0.01), regular)
    assert zeta.max() == pytest.approx(HS / 2, abs=1e-3)
    assert zeta.min() == pytest.approx(-HS / 2, abs=1e-3)


def test_long_crested_sea_has_one_direction():
    long_crested, _ = sea_from_settings(HS, TP, dict(SETTINGS, spreading_type="none"), 100)
    assert np.count_nonzero(long_crested.amplitudes) == 30
    assert np.all(long_crested.direction_offsets == 0.0)


def test_short_crested_sea_has_many_directions():
    short_crested, _ = sea_from_settings(HS, TP, SETTINGS, 100)
    assert np.count_nonzero(short_crested.amplitudes) == 100
    assert len(np.unique(np.round(short_crested.direction_offsets, 6))) > 1


def test_wrong_sea_type_is_caught():
    with pytest.raises(ValueError, match="long_crested"):
        sea_from_settings(HS, TP, dict(SETTINGS, sea_type="long_crested"), 100)
