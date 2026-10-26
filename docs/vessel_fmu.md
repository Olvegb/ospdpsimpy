# VesselBlock: 6-DOF hull FMU

**Owner: Araz. Status: software integration verified; NOT physically calibrated to Gunnerus.**

The hull is one FMI 2.0 co-simulation unit built with pythonfmu. It combines four editable calculation modules:

| Path | Responsibility |
| --- | --- |
| `src/ospfmusim/hull/vessel.py` | FMI input/output ports, shared configuration, initialization and stepping |
| `src/ospfmusim/hull/helpfunctions/LF.py` | 6-DOF low-frequency vessel dynamics and numerical integration |
| `src/ospfmusim/hull/helpfunctions/WF.py` | Wave-frequency motion from six-DOF motion response amplitude operators (RAOs) |
| `src/ospfmusim/hull/helpfunctions/WaveLoads.py` | First-order wave loads (diagnostic) and second-order wave drift |
| `src/ospfmusim/hull/helpfunctions/WindModel.py` | Apparent-wind force and moment, using Olve's Blendermann coefficients |
| `src/ospfmusim/hull/helpfunctions/example_test_config.json` | **SYNTHETIC TEST DATA ONLY**: placeholder six-DOF matrices, artificial RAO tables |
| `src/ospfmusim/fmus/VesselBlock.fmu` | Built FMI 2.0 co-simulation binary with bundled Python modules |

## Shared configuration

Like the other FMUs, VesselBlock exposes a single `config_path` string parameter. Point it to a config file in **configs/**. At initialization the wrapper reads `[vessel_fmu].mat_file` (or `data_file`) relative to that TOML file; the referenced JSON stores large 6-by-6 arrays and hydrodynamic tables that are impractical to inline in TOML.

The hull uses `[vessel]` (geometry and `type`), `[environment].air_density`, and `[blendermann][vessel.type]` for wind. It **does not use** the wind settings inside the synthetic JSON. It reads LF matrices, gravity, wave force/drift and motion tables from the specified hydrodynamic data file. Changes to TOML or referenced data are applied on next FMU initialization, without rebuilding. Changing Python requires a rebuild.

**The current `data_file` references an artificial OSV fixture**. Do not treat any trajectory or result as a prediction for RV Gunnerus. Existing `[vessel].mass_matrix` and `damping_matrix` are only 3-DOF placeholders used by the controller/filter; they are not silently expanded to the 6-DOF hull.

## Ports and conventions

| Port | Type | Unit / meaning |
| --- | --- | --- |
| `config_path` | fixed string parameter | Shared TOML file |
| `max_internal_step_s` | fixed real parameter | Maximum LF substep, default 0.05 s |
| `mean_wind_speed`, `slowly_varying_wind_speed`, `gust_wind_speed` | inputs | Wind components, m/s, from Environment |
| `wind_direction_deg` | input | Wind **FROM**, clockwise from north, degrees |
| `wave1_...` through `wave100_...` | inputs | amplitude [m], omega [rad/s], phase [rad], direction_deg [deg **TO**] |
| `X_thrusters`, `Y_thrusters`, `N_thrusters` | inputs | Delivered surge/sway force [N] and yaw moment [Nm] |
| `current_north`, `current_east`, `current_down` | inputs | Water velocity in NED [m/s] |
| `initial_north/east/down/roll/pitch/yaw/u/v/w/p/q/r` | fixed real parameters | Initial LF state; rotations in rad |
| `north/east/down/roll/pitch/yaw` | outputs | **Total** LF + WF pose, metres/radians |
| `lf_north...`, `wf_north...`, `lf_u/v/w/p/q/r` | outputs | LF pose, WF increments, LF-only body velocities |
| `wind_X/Y/Z/K/M/N`, `wave_first_X/Y/Z/K/M/N`, `wave_drift_X/Y/Z/K/M/N` | outputs | Diagnostic body loads [N] / moments [Nm] |

Body frame is forward/starboard/down. NED is north/east/down. Yaw is clockwise positive from north. The model uses the **same 100 wave component port names** as `src/ospfmusim/waves/waves.py`. Propagation directions are converted to radians inside the hull.

## How the loads are applied

LF integrates delivered thruster forces + wind + **second-order drift**. WF evaluates harmonic first-order displacement and is added only when publishing total pose. First-order wave forces are diagnostics only: integrating them again in LF would double-count WF motion. `wave_drift_*` already contains the mean drift, so do not add mean drift separately.

Wind is **already applied inside** VesselBlock. Do not separately connect `WindLoads.X/Y/N` to the vessel input. The observer may use `vessel.wind_X/Y/N` for estimated disturbance feedforward. `WindModel.py` intentionally reverses the supplied wind-angle convention to map wind FROM starboard to negative body Y. Confirm Olve's frame and moment sign convention before physical validation.

## Build and checks

From the repository root in a Python environment with `pythonfmu==0.7.0` and NumPy:

```sh
python -m ospfmusim.build                  # builds the FMUs that have changed (--force: all)
python -m pytest tests/test_vessel_fmu.py  # requires fmpy (pip install -e ".[test]")
```

The native FMI test checks the FMU schema, verifies the packaged module bytes against editable sources, and compares **60 varying-duration time steps** with direct module integration for each of calm water, thrusters, wind, waves, and combined input cases; FMI reset is exercised after each. Additional separately run analytical tests were performed during development; they are not replacements for calibrated measurements.

## Work still needed before a physical digital twin

1. Replace the synthetic fixture with identified **Gunnerus** matrices (including consistent body origin), verified force and drift coefficients, and verified six-axis **motion** RAOs.
2. The raw Gunnerus motion data has 19 heading columns while its heading metadata contains 36. Resolve this and verify rotational units and phase signs, **without guessing**.
3. Confirm with Olve the wind lateral/moment signs and thruster moment reference origin.
4. Test the connected OSP model with all group FMUs available, compare trajectories against data, and tune only after interfaces are verified.

**Provenance and license:** LF/WF source follows the NTNU mcsimpy GPL-3.0-or-later lineage (`src/ospfmusim/hull/helpfunctions/LICENSE` and `NOTICE.md`), WaveLoads contains a separately attributed MIT-licensed implementation from Viljar, and the WindModel adapter is based on Olve's supplied coefficients. Confirm redistribution expectations with the repository owner before merging.
