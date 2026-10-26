"""Quick check of sea_state.py.

Run from the repo folder:  python src/ospfmusim/waves/tests/check_sea_state.py
"""
import numpy as np

from ospfmusim.waves.helpfunctions.sea_state import (directions_to, elevation, keep_strongest,
                                                     realize, sea_from_settings)
from ospfmusim.waves.helpfunctions.spectra import cos2s

hs = 3.0     # significant wave height [m]
tp = 9.0     # peak period [s]
t = np.arange(0.0, 3 * 3600.0, 0.5)          # three hours, 0.5 s steps


def make_sea(spreading_type, random_frequencies=True, seed=123):
    """Short helper so each check only shows what it changes."""
    return realize(hs, tp, "jonswap", 3.3,
                   spreading_type=spreading_type, s=2, n_directions=11, direction_limit_deg=90,
                   n_frequencies=30, omega_min_factor=0.25, omega_max_factor=3.0,
                   random_frequencies=random_frequencies, random_directions=True, seed=seed)


print("CHECK SPREADING FUNCTION:")
theta = np.linspace(-np.pi / 2, np.pi / 2, 2001)
for s in (1, 2, 10):
    area = np.trapezoid(cos2s(theta, s), theta)
    print(f"  s = {s:2d}: area under D = {area:.4f} (should be 1),  "
          f"D at mean direction = {cos2s(0.0, s):.3f}")

print("CHECK HS:")
for spreading_type in ("none", "cos2s"):
    sea = make_sea(spreading_type)
    m0 = np.sum(sea.amplitudes**2 / 2)       # energy in the components
    zeta = elevation(t, sea)                 # wave elevation time series
    print(f"  {spreading_type:5s}: {len(sea.amplitudes):3d} components,  "
          f"Hs from components = {4 * np.sqrt(m0):.3f} m,  "
          f"from time series = {4 * np.std(zeta):.3f} m")

print("CHECK DIRECTIONS (cos2s):")
sea = make_sea("cos2s")
energy = sea.amplitudes**2 / 2
mean_offset = np.degrees(np.sum(energy * sea.direction_offsets) / np.sum(energy))
within_30 = np.sum(energy[np.abs(sea.direction_offsets) <= np.radians(30)]) / np.sum(energy)
print(f"  Energy-weighted mean direction offset: {mean_offset:+.1f} deg  (should be close to 0)")
print(f"  Share of energy within +-30 deg:        {within_30:.0%}")
widest_deg = np.degrees(np.abs(sea.direction_offsets).max())
print(f"  Widest direction used:                  {widest_deg:.1f} deg")


def envelope(t, sea):
    """Height of the wave groups (the slowly varying envelope of the waves)."""
    t = t[:, None]
    return np.abs(np.sum(sea.amplitudes * np.exp(1j * (sea.frequencies * t - sea.phases)), axis=1))


print("CHECK REPETITION OF WAVE GROUPS:")
d_omega = (3.0 - 0.25) * (2 * np.pi / tp) / (30 - 1)
t_repeat = 2 * np.pi / d_omega               # repeat period of an evenly spaced grid
t_short = np.arange(0.0, 200.0, 0.5)
for random_frequencies in (False, True):
    sea = make_sea("none", random_frequencies)
    same = np.allclose(envelope(t_short, sea), envelope(t_short + t_repeat, sea))
    print(f"  random_frequencies = {random_frequencies}:  "
          f"groups repeat after {t_repeat:.1f} s? {same}")

print("CHECK WRONG NAME:")
try:
    realize(hs, tp, "jonswap", 3.3, "cos-2s", 2, 11, 90, 30, 0.25, 3.0, True, True, seed=123)
except ValueError as error:
    print(f"  {error}")

print("CHECK KEEP THE STRONGEST COMPONENTS:")
sea = make_sea("cos2s")                                   # 30 x 11 = 330 components
for n_waves in (100, 330, 400):
    strongest, kept = keep_strongest(sea, n_waves)
    energy = 0.5 * strongest.amplitudes**2
    print(f"  n_waves = {n_waves}:  {len(strongest.amplitudes)} components sent out, "
          f"{np.count_nonzero(strongest.amplitudes)} with waves,  kept energy = {kept:.1%},  "
          f"Hs = {4 * np.sqrt(energy.sum()):.2f} m (full sea: {hs} m)")
strongest, _ = keep_strongest(sea, 100)
energy = 0.5 * strongest.amplitudes**2
print(f"  Sorted, strongest first: {bool(np.all(np.diff(energy) <= 0))}")
print(f"  Weakest kept is stronger than strongest removed: "
      f"{energy[-1] >= np.sort(0.5 * sea.amplitudes**2)[::-1][100]}")

print("CHECK DIRECTIONS SENT OUT:")
long_crested, _ = keep_strongest(make_sea("none"), 5)
for from_deg in (0.0, 45.0, 270.0):
    to_deg = np.degrees(directions_to(long_crested, from_deg, direction_is_from=True)[0])
    print(f"  Waves from {from_deg:5.1f} deg travel to {to_deg:5.1f} deg  "
          f"(should be {(from_deg + 180) % 360:5.1f})")

print("CHECK SEA TYPES (sea_from_settings):")
p = {"sea_type": "irregular", "spectrum_type": "jonswap", "gamma": 3.3, "spreading_type": "cos2s",
     "spreading_s": 2, "n_directions": 11, "direction_limit_deg": 90, "n_frequencies": 30,
     "omega_min_factor": 0.25, "omega_max_factor": 3.0, "random_frequencies": True,
     "random_directions": True, "seed": 123}
regular, kept = sea_from_settings(hs, tp, dict(p, sea_type="regular"), 100)
print(f"  regular:               {np.count_nonzero(regular.amplitudes)} wave sent out,  "
      f"amplitude = {regular.amplitudes[0]:.3f} m (should be H/2 = {hs / 2}),  "
      f"omega = {regular.frequencies[0]:.4f} rad/s (should be 2 pi/T = {2 * np.pi / tp:.4f}),  "
      f"direction offset = {regular.direction_offsets[0]:.1f},  kept energy = {kept:.0%}")
zeta = elevation(np.arange(0.0, tp, 0.01), regular)
print(f"                         highest crest = {zeta.max():.3f} m, "
      f"deepest trough = {zeta.min():.3f} m (should be +-{hs / 2})")
long_crested, _ = sea_from_settings(hs, tp, dict(p, spreading_type="none"), 100)
print(f"  irregular, long-crested:  {np.count_nonzero(long_crested.amplitudes)} waves,  "
      f"all in the mean direction: {np.all(long_crested.direction_offsets == 0.0)}")
short_crested, _ = sea_from_settings(hs, tp, p, 100)
print(f"  irregular, short-crested: {np.count_nonzero(short_crested.amplitudes)} waves,  "
      f"{len(np.unique(np.round(short_crested.direction_offsets, 6)))} different directions")
try:
    sea_from_settings(hs, tp, dict(p, sea_type="long_crested"), 100)
except ValueError as error:
    print(f"  Wrong name: {error}")
