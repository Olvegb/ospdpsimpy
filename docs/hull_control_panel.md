# Gunnerus hull control panel

## Single place to edit model settings

Open `src/ospfmusim/hull/Control_Panel_Hull.py` in VS Code.

The first section, **USER CONTROL PANEL**, contains the editable defaults:

| Setting | Meaning | Effect |
| --- | --- | --- |
| `MAT_FILE_OVERRIDE` | Set to an absolute or config-relative .mat path to override each run's [vessel_fmu].mat_file | Changes input hydrodynamic file |
| `FREQUENCY_COUNT` | Number of WAMIT frequency points, or None for all calculated nodes | Frequency grid/interpolation |
| `HEADING_STEP_DEG` | Angular lookup grid spacing, must divide 360 | Heading interpolation |
| `FREQUENCY_SPACING` | log or linear (only when resampling) | Location of new frequency nodes |
| `DRIFT_METHOD` | newman, newman_absolute, sorensen | Second-order drift approximation |
| `REMOVE_SUM_FREQUENCY` | True / False | Include or remove sum-frequency components |
| `DAMPING_MODE` | matlab, add, replace | Selection of LF damping matrix |
| `DAMPING_DIAGONAL_SI` | Six SI damping entries | Used in add/replace mode |
| `DAMPING_MATRIX_SI` | Optional complete 6x6 SI matrix | Overrides the damping diagonal |
| `MAX_INTERNAL_STEP_S` | Maximum LF RK4 step size in seconds | Integrator time resolution |
| `WATCH_* / ENABLE_WATCH` | Arbitrary, non-stopping investigation thresholds | First warning on crash report |
| `LOG_DYNAMICS_TERMS` | Print each 6DOF term at interval | Diagnostics only |
| `LOG_INTERVAL_S` | Seconds between diagnostic samples | Diagnostics only |
| `REPORT_TERMS_ON_FAILURE` | Print 6DOF force contributions on numerical failure | Diagnostics only |
| `PRINT_WAMIT_MATRICES` | Print the full 6x6 matrices in Wamit CLI | Inspection only |
| `INSPECT_*` | Regular-wave period, height and direction used by Wamit CLI | Inspection only |

Order: surge, sway, heave, roll, pitch, yaw.

The explicit baseline is `DAMPING_MODE="matlab"`. It copies the existing MATLAB
D matrix unchanged (including any zero elements). `add` and `replace`
are experiments; coefficients must be physically justified and SI-consistent.

## Runtime configuration precedence

The vessel FMU reads an optional **[hull_control]** section in its active
run `config.toml`, which takes priority over the control-panel defaults.
Older `[vessel_fmu]` file/grid options continue to work as intermediate
overrides for backward compatibility.

To change one run without editing Python, append for example:

```toml
[hull_control]
frequency_count = 100
heading_step_deg = 10
drift_method = "newman_absolute"
damping_mode = "add"
damping_diagonal_si = [900.0, 0.0, 0.0, 0.0, 0.0, 0.0]
max_internal_step_s = 0.025
watch_roll_deg = 25.0
log_dynamics_terms = true
log_interval_s = 5.0
```

If you edit **the Python control panel**, run:

```bash
.venv/bin/python -m ospfmusim.build
```

Changing **only config.toml** requires starting a new simulation, not an FMU rebuild.

`Wamit.py` direct inspection:

```bash
.venv/bin/python src/ospfmusim/hull/helpfunctions/Wamit.py data_examples/gunnerus2.mat
```

The standalone Wamit.py inspection does **not** necessarily use the same
.mat file that an FMU run has selected through its active config.toml.

## Current numerical-drift investigation

The first warning at 15 degrees roll is **not** a stopping limit. LF stops
only when input/kinematic/numerical validation fails (e.g., Euler singularity,
NaN, infinite state, invalid time step, non-invertible mass matrix).

To investigate divergence, enable `LOG_DYNAMICS_TERMS` and compare the
separate external, Coriolis, damping, restoring and current terms before
the failed RK4 stage. The diagnostic terms are recomputed for observation
but do not modify dynamics.

The panel cannot validate the physical quality of `gunnerus2.mat`, and it
cannot fix unstable physics by itself. Keep baseline and experimental runs
with their exact configuration and results.
