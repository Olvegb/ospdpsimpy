"""Tests of the wave settings (waves/helpfunctions/settings.py).

Reads [waves] from the config the simulation uses and builds the sea ONLY from
that, the same way the Waves FMU does it.

License: MIT (see LICENSE in the repository root)
Authors: Viljar Simonnes Tveit, Olve Grønås Birkeland (2026)
"""
import tomllib

import numpy as np
import pytest

from app.build.data import default_config
from ospfmusim.waves.helpfunctions.sea_state import sea_from_settings
from ospfmusim.waves.helpfunctions.settings import load_settings, settings_from_config
from ospfmusim.waves.waves import N_WAVES

CONFIG = default_config()   # the config the simulation uses (OspSystemStructure.xml)


def read_config():
    return tomllib.loads(CONFIG.read_text(encoding="utf-8-sig"))


def test_config_builds_the_sea():
    waves = read_config()["waves"]
    settings = load_settings(CONFIG)
    sea, kept = sea_from_settings(waves["hs"], waves["tp"], settings, N_WAVES)
    hs_kept = 4 * np.sqrt(np.sum(0.5 * sea.amplitudes**2))
    assert len(sea.amplitudes) == N_WAVES
    assert 0.0 < kept <= 1.0
    assert hs_kept == pytest.approx(waves["hs"] * np.sqrt(kept), rel=0.02)
    omega_peak = 2 * np.pi / waves["tp"]
    sent_out = sea.frequencies[sea.amplitudes > 0]
    if settings["sea_type"] == "irregular":
        assert sent_out.min() >= settings["omega_min_factor"] * omega_peak - 1e-9
        assert sent_out.max() <= settings["omega_max_factor"] * omega_peak + 1e-9


def test_wrong_name_in_config_is_caught():
    wrong = read_config()
    wrong["waves"]["spreading"]["type"] = "cos-2s"
    with pytest.raises(ValueError, match="cos-2s"):
        settings_from_config(wrong)
