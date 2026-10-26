"""Check the Waves FMU code WITHOUT building an FMU and without OSP.
Run from the repo folder:  python src/ospfmusim/waves/tests/check_waves_fmu.py

It creates the Waves class from waves/waves.py directly in Python and calls it
the same way OSP would: set config_path and the inputs -> exit_initialization_mode()
-> do_step() again and again. This works on any computer, also on Mac.
"""
import re
import tempfile
import tomllib
from pathlib import Path

import numpy as np

from app.build.data import default_config
from ospfmusim.waves.helpfunctions.sea_state import directions_to, sea_from_settings
from ospfmusim.waves.helpfunctions.settings import load_settings
from ospfmusim.waves.waves import COMPONENT_OUTPUTS, N_WAVES, SLOTS, Waves

CONFIG = default_config()   # the config the simulation uses (OspSystemStructure.xml)
# H_s, T_p and direction, as Environment sends them
WAVES = tomllib.loads(CONFIG.read_text(encoding="utf-8-sig"))["waves"]
DT = 0.1                    # time step [s], as in OspSystemStructure.xml


def make_fmu(config_path):
    """Create the FMU object and set what OSP sets: config_path and the inputs from Environment."""
    fmu = Waves(instance_name="waves")
    fmu.config_path = str(config_path)
    fmu.H_s, fmu.T_p, fmu.wave_direction_deg = WAVES["hs"], WAVES["tp"], WAVES["direction_deg"]
    return fmu


def read_outputs(fmu):
    """Read the wave component outputs back into four arrays.

    The arrays are amplitude, omega, phase and direction_deg.
    """
    return [np.array([getattr(fmu, f"wave{k}_{quantity}") for k in SLOTS])
            for quantity in COMPONENT_OUTPUTS]


def config_with(**changes):
    """A copy of the config the simulation uses with some [waves] values changed.

    The copy is written to a temporary file.
    """
    text = CONFIG.read_text(encoding="utf-8-sig")
    for key, value in changes.items():
        text, n = re.subn(rf'^{key} = "[^"]*"', f'{key} = "{value}"', text, count=1,
                          flags=re.MULTILINE)
        assert n == 1, f"{key} not found in {CONFIG}"
    file = Path(tempfile.mkdtemp()) / "config.toml"
    file.write_text(text, encoding="utf-8")
    return file


print("CHECK THE OUTPUTS:")
fmu = make_fmu(CONFIG)
n_parameters = sum(1 for v in fmu.vars.values() if v.causality.name == "parameter")
n_outputs = sum(1 for v in fmu.vars.values() if v.causality.name == "output")
print(f"  {n_parameters} parameter (config_path), {n_outputs} outputs "
      f"({N_WAVES} wave components x 4 + kept_energy_fraction)")

fmu.exit_initialization_mode()
amplitude, omega, phase, direction = read_outputs(fmu)
print(f"  Right from time 0 (before the first step): wave1_amplitude = {amplitude[0]:.3f} m")
t = 0.0
while t < 60.0:                                  # 1 minute of time steps
    fmu.do_step(t, DT)
    t += DT
amplitude, omega, phase, direction = read_outputs(fmu)
print(f"  wave1 (strongest): amplitude = {amplitude[0]:.3f} m,  omega = {omega[0]:.3f} rad/s,  "
      f"phase = {phase[0]:.3f} rad,  direction = {direction[0]:.1f} deg")
print(f"  kept_energy_fraction = {fmu.kept_energy_fraction:.3f}")

print("COMPARE WITH helpfunctions CALLED DIRECTLY:")
sea, kept = sea_from_settings(WAVES["hs"], WAVES["tp"], load_settings(CONFIG), N_WAVES)
same = (np.allclose(amplitude, sea.amplitudes) and np.allclose(omega, sea.frequencies)
        and np.allclose(phase, sea.phases)
        and np.allclose(direction, np.degrees(directions_to(sea, WAVES["direction_deg"], True))))
print(f"  FMU gives the same wave components: {same}")

print("CHECK THAT A NEW DIRECTION ONLY TURNS THE WAVES:")
fmu.wave_direction_deg = WAVES["direction_deg"] + 90.0
fmu.do_step(t, DT)
amplitude2, omega2, phase2, direction2 = read_outputs(fmu)
unchanged = (np.array_equal(amplitude, amplitude2) and np.array_equal(omega, omega2)
             and np.array_equal(phase, phase2))
print(f"  Amplitudes, frequencies and phases unchanged: {unchanged}")
print(f"  All directions turned by 90 deg: {np.allclose((direction2 - direction) % 360.0, 90.0)}")

print("CHECK CALM SEA (H_s = 0):")
fmu.H_s = 0.0
fmu.do_step(t, DT)
amplitude, *_ = read_outputs(fmu)
print(f"  All amplitudes zero: {np.all(amplitude == 0.0)}")

print('CHECK REGULAR SEA (sea_type = "regular" in the config):')
fmu = make_fmu(config_with(sea_type="regular"))
fmu.exit_initialization_mode()
fmu.do_step(0.0, DT)
amplitude, omega, phase, direction = read_outputs(fmu)
print(f"  wave1: amplitude = {amplitude[0]:.3f} m (H/2 = {WAVES['hs'] / 2}),  "
      f"omega = {omega[0]:.4f} rad/s (2 pi/T = {2 * np.pi / WAVES['tp']:.4f}),  "
      f"phase = {phase[0]:.1f},  "
      f"direction = {direction[0]:.1f} deg (to {(WAVES['direction_deg'] + 180) % 360:.1f})")
print(f"  All other amplitudes zero: {np.all(amplitude[1:] == 0.0)},  "
      f"kept_energy_fraction = {fmu.kept_energy_fraction}")

print("CHECK THAT A MISSING config_path IS CAUGHT:")
try:
    Waves(instance_name="no_config").exit_initialization_mode()
except ValueError as error:
    print(f"  {error}")

print("CHECK THAT A WRONG NAME IN THE CONFIG IS CAUGHT AT THE START:")
try:
    make_fmu(config_with(sea_type="long_crested")).exit_initialization_mode()
except ValueError as error:
    print(f"  {error}")
