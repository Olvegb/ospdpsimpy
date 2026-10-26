"""Sea state: turns a wave spectrum into a set of regular wave components.

Each component is a cosine wave with its own amplitude, frequency, phase and
direction. Adding all the components together gives an irregular sea.

Regular sea:       one regular wave (sea_type "regular"), see regular_wave().
Irregular sea (sea_type "irregular"), from a wave spectrum:
  Long-crested sea:  all waves travel in the mean direction (spreading "none").
  Short-crested sea: the energy is spread over several directions around the
                     mean direction (spreading "cos2s").

The directions are stored RELATIVE to the mean direction (direction_offsets).
The mean direction itself is added last, in directions_to(), so the
environment block can change it without building a new sea.

The waves block sends out a fixed number of components (n_waves): the ones
with the most energy (keep_strongest). This is what sea_from_settings()
does, and both the FMU (waves/waves.py) and the tests use that function.

References:
- Sørensen, Marine Control Systems, section 6.2.1 (eq. 6.1, 6.6, 6.22)
"""
from dataclasses import dataclass

import numpy as np

from .spectra import cos2s, jonswap, pierson_moskowitz


@dataclass
class SeaState:
    """One realisation of the sea. All arrays have length n_components."""
    amplitudes: np.ndarray          # [m]
    frequencies: np.ndarray         # [rad/s]
    phases: np.ndarray              # [rad]
    direction_offsets: np.ndarray   # [rad], relative to the mean wave direction


def frequency_spectrum(omega, hs, tp, spectrum_type, gamma):
    """Pick the wave spectrum by name.

    spectrum_type: "pierson_moskowitz" or "jonswap".
    gamma is only used for "jonswap".
    """
    if spectrum_type == "pierson_moskowitz":
        return pierson_moskowitz(omega, hs, tp)
    if spectrum_type == "jonswap":
        return jonswap(omega, hs, tp, gamma)
    raise ValueError(
        f"Unknown spectrum type '{spectrum_type}'. Valid: pierson_moskowitz, jonswap"
    )


def direction_bins(spreading_type, s, n_directions, direction_limit_deg):
    """Split the directions around the mean direction into bins.

    Returns (offsets, weights, d_theta):
      offsets: centre of each direction bin, relative to the mean direction [rad]
      weights: share of the energy in each bin (they add up to 1)
      d_theta: width of each bin [rad]

    spreading_type "none" gives one bin straight along the mean direction.
    """
    if spreading_type == "none":
        return np.zeros(1), np.ones(1), 0.0
    if spreading_type != "cos2s":
        raise ValueError(f"Unknown spreading type '{spreading_type}'. Valid: none, cos2s")

    limit = np.deg2rad(direction_limit_deg)                  # +- limit around the mean
    d_theta = 2.0 * limit / n_directions
    offsets = -limit + (np.arange(n_directions) + 0.5) * d_theta
    weights = cos2s(offsets, s) * d_theta                    # energy share = D * d_theta
    return offsets, weights / weights.sum(), d_theta         # normalise so the sum is exactly 1


def realize(hs, tp, spectrum_type, gamma,
            spreading_type, s, n_directions, direction_limit_deg,
            n_frequencies, omega_min_factor, omega_max_factor,
            random_frequencies, random_directions, seed):
    """Split a wave spectrum into n_frequencies x n_directions wave components.

    random_frequencies / random_directions = True moves each component to a
    random point inside its bin, so the sea does not repeat itself.
    The same seed always gives the same sea.
    """
    # 1. Frequency bins
    wp = 2.0 * np.pi / tp                                   # peak frequency [rad/s]
    omega = np.linspace(omega_min_factor * wp, omega_max_factor * wp, n_frequencies)
    d_omega = omega[1] - omega[0]                           # width of each frequency bin
    S = frequency_spectrum(omega, hs, tp, spectrum_type, gamma)   # energy density per bin

    # 2. Direction bins
    offsets, dir_weights, d_theta = direction_bins(
        spreading_type, s, n_directions, direction_limit_deg)
    m = len(offsets)                                        # number of directions actually used

    # 3. Every combination of frequency and direction becomes one component
    #    (frequency in the outer loop, direction in the inner loop)
    freqs = np.repeat(omega, m)                             # f1 f1 f1 f2 f2 f2 ...
    dirs = np.tile(offsets, n_frequencies)                  # d1 d2 d3 d1 d2 d3 ...
    energy_share = np.repeat(S * d_omega, m) * np.tile(dir_weights, n_frequencies)
    amplitudes = np.sqrt(2.0 * energy_share)                # Sørensen eq. 6.6

    # 4. Random numbers: phases first, then frequencies, then directions
    n_total = n_frequencies * m
    rng = np.random.default_rng(seed)                       # random number generator
    phases = rng.uniform(0.0, 2.0 * np.pi, n_total)         # random phase per component
    if random_frequencies:
        # move each frequency randomly within its own bin: +- half a bin width
        freqs = freqs + rng.uniform(-0.5 * d_omega, 0.5 * d_omega, n_total)
    if random_directions and m > 1:
        # move each direction randomly within its own bin: +- half a bin width
        dirs = dirs + rng.uniform(-0.5 * d_theta, 0.5 * d_theta, n_total)

    return SeaState(amplitudes, freqs, phases, dirs)


def elevation(t, sea):
    """Wave elevation zeta [m] at the origin at time(s) t.

    zeta(t) = sum_j a_j * cos(omega_j * t - eps_j)
    (At the origin the direction of each wave does not matter.)
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))[:, None]  # column, so each row is one time
    return np.sum(sea.amplitudes * np.cos(sea.frequencies * t - sea.phases), axis=1)


# -- What the waves block sends out ----------------------------------------------------

def calm_sea(n_waves):
    """A sea with n_waves components that all have amplitude 0 (no waves)."""
    zeros = np.zeros(n_waves)
    return SeaState(zeros, zeros.copy(), zeros.copy(), zeros.copy())


def regular_wave(height, period, n_waves):
    """One regular wave as the first component (wave1_ in the FMU).

    The other n_waves - 1 get amplitude 0.

    height: wave height H [m] (crest to trough), so the amplitude is H / 2
    period: wave period T [s], so the frequency is 2 * pi / T
    The wave travels in the mean wave direction (offset 0) and has phase 0.
    """
    sea = calm_sea(n_waves)
    sea.amplitudes[0] = height / 2.0
    sea.frequencies[0] = 2.0 * np.pi / period
    return sea


def keep_strongest(sea, n_waves):
    """Keep the n_waves components with the most energy (Sørensen section 6.2.1, method 1).

    Returns (sea, kept_energy_fraction):
      sea:                  exactly n_waves components, strongest first. If the
                            sea has fewer components, the rest get amplitude 0.
      kept_energy_fraction: share of the total energy that was kept (1.0 = all).
                            The amplitudes are NOT scaled up, so Hs of the kept
                            components is sqrt(kept_energy_fraction) * Hs.
    """
    energy = 0.5 * sea.amplitudes**2                        # energy of each component
    order = np.argsort(-energy, kind="stable")[:n_waves]    # index of the strongest, largest first
    kept_energy_fraction = float(energy[order].sum() / energy.sum())

    strongest = calm_sea(n_waves)                           # start with n_waves empty slots ...
    n = len(order)                                          # ... and fill the first n of them
    strongest.amplitudes[:n] = sea.amplitudes[order]
    strongest.frequencies[:n] = sea.frequencies[order]
    strongest.phases[:n] = sea.phases[order]
    strongest.direction_offsets[:n] = sea.direction_offsets[order]
    return strongest, kept_energy_fraction


def sea_from_settings(hs, tp, p, n_waves):
    """Build the sea exactly as the Waves FMU does.

    p is the settings from settings.load_settings() ([waves] in config.toml).
    n_waves is the number of components sent out (N_WAVES in waves/waves.py).
    p["sea_type"] decides what kind of sea:
      "regular":   one regular wave with height H = hs and period T = tp
                   (regular_wave()). kept_energy_fraction is 1.
      "irregular": splits the spectrum into n_frequencies x n_directions
                   components and keeps the n_waves strongest.
    Hs <= 0 or Tp <= 0 gives a calm sea.

    Returns (sea, kept_energy_fraction), see keep_strongest().
    """
    sea_type = p["sea_type"]
    if sea_type not in ("regular", "irregular"):
        raise ValueError(f"Unknown sea type '{sea_type}'. Valid: regular, irregular")
    if hs <= 0.0 or tp <= 0.0:
        return calm_sea(n_waves), 1.0
    if sea_type == "regular":
        return regular_wave(hs, tp, n_waves), 1.0
    sea = realize(hs, tp,
                  spectrum_type=p["spectrum_type"], gamma=float(p["gamma"]),
                  spreading_type=p["spreading_type"], s=float(p["spreading_s"]),
                  n_directions=int(p["n_directions"]),
                  direction_limit_deg=float(p["direction_limit_deg"]),
                  n_frequencies=int(p["n_frequencies"]),
                  omega_min_factor=float(p["omega_min_factor"]),
                  omega_max_factor=float(p["omega_max_factor"]),
                  random_frequencies=bool(p["random_frequencies"]),
                  random_directions=bool(p["random_directions"]),
                  seed=int(p["seed"]))
    return keep_strongest(sea, n_waves)


def directions_to(sea, wave_direction_deg, direction_is_from):
    """The direction each component travels TO [rad, NED, 0 = north, between 0 and 2*pi].

    wave_direction_deg: mean wave direction from the environment block [deg]
    direction_is_from:  True if wave_direction_deg is where the waves come FROM
    """
    mean = np.radians(wave_direction_deg)
    if direction_is_from:
        mean = mean + np.pi                                 # coming from north = going to south
    return (mean + sea.direction_offsets) % (2.0 * np.pi)
