"""Quick check of waves/helpfunctions/settings.py.
Run from the repo folder:  python src/ospfmusim/waves/tests/check_settings.py

Reads [waves] from the config the simulation uses and builds the sea ONLY from
that, the same way the Waves FMU does it.
"""
import tomllib

import numpy as np

from app.build.data import default_config
from ospfmusim.waves.helpfunctions.sea_state import sea_from_settings
from ospfmusim.waves.helpfunctions.settings import load_settings, settings_from_config
from ospfmusim.waves.waves import N_WAVES

CONFIG = default_config()   # the config the simulation uses (OspSystemStructure.xml)
waves = tomllib.loads(CONFIG.read_text(encoding="utf-8-sig"))["waves"]
settings = load_settings(CONFIG)

print(f"SETTINGS FROM configs/{CONFIG.name}:")
for name, value in settings.items():
    print(f"  {name:20s} = {value}")

print(f"SEA STATE FROM [waves]: H_s = {waves['hs']} m, T_p = {waves['tp']} s, "
      f"from {waves['direction_deg']} deg")
sea, kept = sea_from_settings(waves["hs"], waves["tp"], settings, N_WAVES)
n_directions = settings["n_directions"] if settings["spreading_type"] != "none" else 1
n_grid = settings["n_frequencies"] * n_directions
hs_kept = 4 * np.sqrt(np.sum(0.5 * sea.amplitudes**2))
print(f"  Sea type: {settings['sea_type']}")
if settings["sea_type"] == "irregular":
    print(f"  Grid: {n_grid} components. Sent out: the {N_WAVES} strongest")
print(f"  Kept energy: {kept:.1%}  ->  H_s of the components sent out = {hs_kept:.2f} m "
      f"(full sea: {waves['hs']} m)")
nonzero = sea.frequencies[sea.amplitudes > 0]
print(f"  Frequencies sent out: {nonzero.min():.2f} to {nonzero.max():.2f} rad/s "
      f"(peak frequency {2 * np.pi / waves['tp']:.2f} rad/s)")

print("CHECK THAT A WRONG NAME IN THE CONFIG IS CAUGHT:")
wrong = tomllib.loads(CONFIG.read_text(encoding="utf-8-sig"))
wrong["waves"]["spreading"]["type"] = "cos-2s"
try:
    settings_from_config(wrong)
except ValueError as error:
    print(f"  {error}")
