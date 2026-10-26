# Waves FMU

One FMU (FMI 2.0 co-simulation, built with pythonfmu 0.7.0, run in OSP) that turns the sea state
from the Environment FMU into a set of regular wave components for the hull FMU. Owner: Viljar.

```mermaid
flowchart LR
    config["[waves] in configs/*.toml"] --> Waves
    Environment -->|"H_s, T_p, wave_direction_deg"| Waves
    Waves -->|"wave#lt;k#gt;_amplitude, _omega, _phase, _direction_deg<br>k = 1 ... N_WAVES (= 100)"| hull["hull FMU"]
```

The FMU only describes the **sea**. It does **not** calculate any forces: the wave loads depend on
the vessel, so the hull FMU calculates them from the wave components.

Three kinds of sea, chosen in `configs/*.toml` without rebuilding the FMU:

| Sea | `[waves] sea_type` | `[waves.spreading] type` | What is sent out |
|---|---|---|---|
| Regular | `"regular"` | (not used) | One wave (`wave1_`): amplitude H/2, frequency 2π/T, phase 0, mean direction. All other amplitudes 0 |
| Irregular, long-crested | `"irregular"` | `"none"` | The strongest components of the spectrum, all in the mean direction |
| Irregular, short-crested | `"irregular"` | `"cos2s"` | The strongest components of the spectrum, spread over directions |

For a regular sea, `H_s` is used as the **wave height H** and `T_p` as the **wave period T**.

Based on the TMR4240 / `mcsimpy` wave model, extended with directional spreading
(Sørensen, *Marine Control Systems*, section 6.2.1).

## 1. Inputs, outputs, parameter

**Parameter**

| Name | Description |
|---|---|
| `config_path` | Path to `configs/*.toml`, set in `OspSystemStructure.xml`. The FMU reads `[waves]` itself in `exit_initialization_mode()` |

**Inputs** (connected from the Environment FMU)

| Name | Unit | Description |
|---|---|---|
| `H_s` | m | Significant wave height (regular sea: wave height H) |
| `T_p` | s | Peak period (regular sea: wave period T) |
| `wave_direction_deg` | deg | Mean wave direction, where the waves come **from**, clockwise from north |

**Outputs**: four per wave component, `k = 1 ... N_WAVES`. `wave1` has the most energy.

| Name | Unit | Description |
|---|---|---|
| `wave<k>_amplitude` | m | Wave amplitude (half the wave height) |
| `wave<k>_omega` | rad/s | Wave frequency |
| `wave<k>_phase` | rad | Phase |
| `wave<k>_direction_deg` | deg | Direction the wave travels **to**, NED, clockwise from north, 0–360 |
| `kept_energy_fraction` | – | Share of the total wave energy in the components sent out (for logging) |

With `N_WAVES = 100` that is 401 outputs.

## 2. What the hull FMU must do with the outputs

**Use exactly this wave elevation** (same sign, cos, and `t` = the simulation time):

```
zeta(x, y, t) = sum_k  amplitude_k * cos(omega_k * t - kw_k * (x cos(theta_k) + y sin(theta_k)) - phase_k)

theta_k = radians(direction_deg_k)      the direction in RADIANS
kw_k    = omega_k^2 / g                 wave number, deep water
x = north, y = east                     the vessel position [m]
```

The wave components only change when `H_s` or `T_p` changes (then the sea is built again, with the
same seed) or when `wave_direction_deg` changes (then all directions turn). They do not change with
time; the time dependence is in `omega_k * t` above.

The direction relative to the vessel (for the RAO and drift tables) is `theta_k - psi`, with
`psi` the heading.

## 3. `configs/*.toml`

Everything is in `[waves]` and its sub-sections:

| Key | Read by | Description |
|---|---|---|
| `hs`, `tp`, `direction_deg` | Environment | The sea state, sent on to Waves as `H_s`, `T_p`, `wave_direction_deg` |
| `sea_type` | Waves | `"regular"` or `"irregular"` |
| `seed` | Waves | Random seed. Same seed = same sea |
| `[waves.spectrum] type`, `gamma` | Waves | `"jonswap"` or `"pierson_moskowitz"`, and the JONSWAP peak factor |
| `[waves.spreading] type`, `s`, `n_directions`, `direction_limit_deg` | Waves | `"none"` (long-crested) or `"cos2s"` (short-crested), the spreading exponent, and the direction grid |
| `[waves.discretization] n_frequencies`, `omega_min_factor`, `omega_max_factor`, `random_frequencies`, `random_directions` | Waves | The frequency grid, and whether frequencies and directions are drawn at random inside each bin |

Only `[waves.spectrum]`, `[waves.spreading]` and `[waves.discretization]` are used for an irregular
sea. A wrong name (for example `type = "cos-2s"`) stops the simulation at the start with a message.

**Changing `configs/*.toml` needs no rebuild.**

## 4. `N_WAVES` is special

An FMU has a fixed number of outputs, decided when it is **built**. So `N_WAVES` is a constant at
the top of `src/ospfmusim/waves/waves.py` (the same idea as `MAX_THRUSTERS` in `thrust_allocation.py`), not in
`configs/*.toml`. After changing it:

1. rebuild: `python -m ospfmusim.build` (or run the simulation, which rebuilds it),
2. redo the connections to the hull FMU in `OspSystemStructure.xml` (4 per wave component).

## 5. How it works

```mermaid
flowchart TD
    sea["H_s, T_p"] -->|"helpfunctions/spectra.py + sea_state.py: realize()"| grid
    grid["n_frequencies x n_directions wave components<br>(regular sea: regular_wave(), one wave)"]
    grid -->|"sea_state.py: keep_strongest()"| strongest
    strongest["the N_WAVES components with the most energy<br>(Sørensen 6.2.1, method 1)"]
    strongest -->|"sea_state.py: directions_to() (+ mean wave direction)"| outputs
    outputs["outputs: wave#lt;k#gt;_amplitude, _omega, _phase, _direction_deg"]
```

The choice between regular and irregular sea is made in `sea_state.py: sea_from_settings()`.

The amplitudes are **not** scaled up after the weakest components are removed, so H_s of the
components sent out is a little lower than the input H_s. `kept_energy_fraction` says how much of
the energy is kept (0.93 with the current config).

## 6. Files

| File | What it does |
|---|---|
| `src/ospfmusim/waves/waves.py` | The FMU: inputs, outputs, `config_path`, `do_step`. `N_WAVES` is set here |
| `src/ospfmusim/waves/helpfunctions/spectra.py` | Wave spectra (Pierson–Moskowitz, JONSWAP) and the cos-2s spreading function |
| `src/ospfmusim/waves/helpfunctions/sea_state.py` | Regular wave, or splits the spectrum into wave components and keeps the strongest. Adds the mean direction |
| `src/ospfmusim/waves/helpfunctions/settings.py` | Reads `[waves]` from `configs/*.toml` and checks the names |
| `tests/test_spectra.py`, `tests/test_sea_state.py`, `tests/test_wave_settings.py`, `tests/test_waves_fmu.py` | Tests, see below |

`src/ospfmusim/build.py` builds the model folders of `src/ospfmusim` (with `waves/helpfunctions/`) into every FMU's `resources/`.

## 7. Check

Run from the repo folder. All of these work on Mac (they do not need OSP).

```
python -m pytest tests/test_spectra.py        # H_s from the spectrum area, peak position
python -m pytest tests/test_sea_state.py      # H_s, spreading, repetition, strongest components, directions, sea types
python -m pytest tests/test_wave_settings.py  # [waves] in configs/*.toml builds the sea, wrong names are caught
python -m pytest tests/test_waves_fmu.py      # the FMU class: outputs, new direction, calm sea, regular sea, errors
```

Running the whole system (`python -m ospfmusim.run`) needs Windows or Linux: pythonfmu only
includes `win64` and `linux64` binaries, and libcosimpy is not available for Mac.

## 8. Known limitations

- Deep water (k = ω²/g).
- The spreading function depends only on direction, not on frequency.
- A change in T_p during the simulation moves all frequencies, which gives a jump in the wave elevation.
- At time 0 the inputs from Environment have not arrived yet, so the outputs are a calm sea for the
  first step; from the second step on they are right.
- Keeping only the strongest components removes the short waves first. For a small vessel these give
  much of the drift, so the drift loads are underestimated.
