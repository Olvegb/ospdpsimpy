"""Reads the wave settings from config.toml.

The Waves FMU calls load_settings() in exit_initialization_mode(), the same
way the thruster FMUs call helpfunctions.layout.load_layout().

Hs, Tp and the wave direction are NOT read here: they come from the
Environment FMU as inputs. Environment reads them from the same [waves] section.
"""
import tomllib
from pathlib import Path

SEA_TYPES = ("regular", "irregular")
SPECTRUM_TYPES = ("jonswap", "pierson_moskowitz")
SPREADING_TYPES = ("none", "cos2s")


def settings_from_config(config):
    """Turn the [waves] section of the config into a flat dict for sea_from_settings().

    config: the whole config file, as returned by tomllib.
    """
    waves = config["waves"]
    spectrum = waves["spectrum"]
    spreading = waves["spreading"]
    discretization = waves["discretization"]
    settings = {
        "sea_type": waves["sea_type"],
        "seed": int(waves["seed"]),
        "spectrum_type": spectrum["type"],
        "gamma": float(spectrum["gamma"]),
        "spreading_type": spreading["type"],
        "spreading_s": float(spreading["s"]),
        "n_directions": int(spreading["n_directions"]),
        "direction_limit_deg": float(spreading["direction_limit_deg"]),
        "n_frequencies": int(discretization["n_frequencies"]),
        "omega_min_factor": float(discretization["omega_min_factor"]),
        "omega_max_factor": float(discretization["omega_max_factor"]),
        "random_frequencies": bool(discretization["random_frequencies"]),
        "random_directions": bool(discretization["random_directions"]),
    }
    # Check the names now, so a typo stops the simulation at the start
    # and not only when the first waves are made.
    for key, where, valid in (("sea_type", "[waves] sea_type", SEA_TYPES),
                              ("spectrum_type", "[waves.spectrum] type", SPECTRUM_TYPES),
                              ("spreading_type", "[waves.spreading] type", SPREADING_TYPES)):
        if settings[key] not in valid:
            raise ValueError(f"{where} = '{settings[key]}' in the config is not valid. "
                             f"Valid: {', '.join(valid)}")
    return settings


def load_settings(config_path):
    """Read config.toml and return the wave settings (see settings_from_config())."""
    config = tomllib.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    return settings_from_config(config)
