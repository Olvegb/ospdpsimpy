"""Central, editable settings for the Gunnerus hull FMU.

Edit the constants in section 1, rebuild FMUs, then run the simulator.

The active config.toml may override settings in [hull_control] or existing
[vessel_fmu] keys (see resolve_hull_controls). Values changed in config.toml
are loaded on the next run without rebuilding.

This module configures models; it does not alter the vessel equations.
All damping entries are SI units in [surge, sway, heave, roll, pitch, yaw] order.

License: GPL-3.0-or-later
"""

from pathlib import Path

import numpy as np


# ======================================================================
# 1. USER CONTROL PANEL - EDIT THESE VALUES
# ======================================================================

# MATLAB input, WAMIT frequency/heading preparation.
MAT_FILE = Path(__file__).resolve().parents[3] / "data_examples" / "gunnerus2.mat"
# Set this to a .mat path to OVERRIDE the run config.toml [vessel_fmu].mat_file.
# Leave None to keep each run's selected hydrodynamic data file.
MAT_FILE_OVERRIDE = None
FREQUENCY_COUNT = 100            # None: original frequency nodes; integer >= 2
HEADING_STEP_DEG = 10            # Must divide 360 (deg)
FREQUENCY_SPACING = "log"        # "log" or "linear"; only used if resampling
LF_REFERENCE_OMEGA = None       # rad/s; only for non-M/D/G fallback path

# Printing/inspection when running Wamit.py directly (NOT during FMU runs).
PRINT_WAMIT_MATRICES = True
INSPECT_WAVE_FROM_DEG = 10.0
INSPECT_INITIAL_YAW_DEG = 0.0
INSPECT_WAVE_HEIGHT_M = 2.0
INSPECT_PERIODS_S = (2., 4., 5., 6., 7., 8., 9., 10.)

# Wave loads. "newman_absolute" is the currently used signed method.
DRIFT_METHOD = "newman_absolute" # "newman", "sorensen", "newman_absolute"
REMOVE_SUM_FREQUENCY = True     # second-order difference-frequency + mean only

# LF damping. Do not substitute arbitrary values for calibrated coefficients.
# "matlab": D_used = D_MAT (original baseline)
# "add":    D_used = D_MAT + D_custom
# "replace":D_used = D_custom
DAMPING_MODE = "add"  # "matlab", "add", or "replace"

DAMPING_DIAGONAL_SI = (
    50000.0,    # Surge
    100000,     # Sway
    800000.0,        # Heave
    1500000.0,        # Roll
    50000000.0,        # Pitch
    5000000.0, # Yaw
)

DAMPING_MATRIX_SI = None

# Maximum size of a single internal fixed-step RK4 LF step, seconds.
MAX_INTERNAL_STEP_S = 0.05

# Monitoring thresholds are NOT physical stopping conditions.
ENABLE_WATCH = True
WATCH_ROLL_DEG = 15.0
WATCH_PITCH_DEG = 15.0
WATCH_SPEED = 5.0               # body-frame linear DOFs, m/s
WATCH_RATE_DEG = 10.0           # body angular rates, deg/s

# Diagnostic-only calculation of each term of the 6DOF equation.
# When enabled, report a short term table to stderr at the selected interval.
# It never changes the forces used by the integrator.
LOG_DYNAMICS_TERMS = True
LOG_INTERVAL_S = 5.0
REPORT_TERMS_ON_FAILURE = True


# ======================================================================
# 2. INPUT VALIDATION AND PER-RUN OVERRIDES
# ======================================================================

def _matrix(data, name):
    array = np.asarray(data, dtype=float)
    if array.shape != (6, 6) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 6x6 SI matrix")
    return array.copy()


def _damping_entry(matrix, diagonal):
    if matrix is not None:
        return _matrix(matrix, "damping_matrix_si")
    values = np.asarray(diagonal, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("damping_diagonal_si must contain six finite SI coefficients")
    return np.diag(values)


def resolve_hull_controls(shared):
    """Resolve values for a single FMU run.

    Priority: [hull_control] > central MAT_FILE_OVERRIDE (when set) >
    legacy [vessel_fmu] > central defaults.
    The FMU's already-supplied max_internal_step_s parameter is preserved
    unless [hull_control] explicitly sets max_internal_step_s.

    This function does NOT read or modify hydrodynamic matrices.
    """
    main = dict(shared.get("vessel_fmu", {}))
    section = dict(shared.get("hull_control", {}))

    def pick(key, default, *, legacy=None):
        if key in section:
            return section[key]
        if legacy and legacy in main:
            return main[legacy]
        return default

    count = pick("frequency_count", FREQUENCY_COUNT, legacy="frequency_count")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 2):
        raise ValueError("frequency_count must be None or an integer >= 2")
    step = pick("heading_step_deg", HEADING_STEP_DEG, legacy="heading_step_deg")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0 or 360 % step:
        raise ValueError("heading_step_deg must be a positive divisor of 360")
    spacing = pick("frequency_spacing", FREQUENCY_SPACING, legacy="frequency_spacing")
    if spacing not in ("log", "linear"):
        raise ValueError("frequency_spacing must be 'log' or 'linear'")

    mode = pick("damping_mode", DAMPING_MODE)
    if mode not in ("matlab", "add", "replace"):
        raise ValueError("damping_mode must be 'matlab', 'add', or 'replace'")

    selected_method = pick("drift_method", DRIFT_METHOD)
    if selected_method not in (
        "newman", "sorensen", "newman_absolute",
        "newman_6_97", "sorensen_6_99",
    ):
        raise ValueError("Unknown drift_method")

    watch = {
        "roll_deg": float(pick("watch_roll_deg", WATCH_ROLL_DEG)),
        "pitch_deg": float(pick("watch_pitch_deg", WATCH_PITCH_DEG)),
        "speed_m_s": float(pick("watch_speed_m_s", WATCH_SPEED)),
        "rate_deg_s": float(pick("watch_rate_deg_s", WATCH_RATE_DEG)),
    }
    if any(not np.isfinite(x) or x < 0 for x in watch.values()):
        raise ValueError("Hull warning thresholds must be finite and nonnegative")

    interval = float(pick("log_interval_s", LOG_INTERVAL_S))
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("log_interval_s must be positive and finite")

    # Damping is applied only when the mode explicitly requests a change.
    custom = _damping_entry(
        pick("damping_matrix_si", DAMPING_MATRIX_SI),
        pick("damping_diagonal_si", DAMPING_DIAGONAL_SI),
    )

    config_max_step = section.get("max_internal_step_s")
    if config_max_step is not None:
        config_max_step = float(config_max_step)
        if not np.isfinite(config_max_step) or config_max_step <= 0:
            raise ValueError("max_internal_step_s must be positive and finite")

    return {
        "mat_file": section.get(
            "mat_file",
            MAT_FILE_OVERRIDE if MAT_FILE_OVERRIDE is not None
            else main.get("mat_file", str(MAT_FILE)),
        ),
        "frequency_count": count,
        "heading_step_deg": step,
        "frequency_spacing": spacing,
        "lf_reference_omega_rad_s": pick(
            "lf_reference_omega_rad_s", LF_REFERENCE_OMEGA,
            legacy="lf_reference_omega_rad_s",
        ),
        "drift_method": selected_method,
        "remove_sum_frequency": bool(pick("remove_sum_frequency", REMOVE_SUM_FREQUENCY)),
        "damping_mode": mode,
        "damping_custom": custom,
        "max_internal_step_s": config_max_step,
        "enable_watch": bool(pick("enable_watch", ENABLE_WATCH)),
        "watch": watch,
        "log_dynamics_terms": bool(pick("log_dynamics_terms", LOG_DYNAMICS_TERMS)),
        "log_interval_s": interval,
        "report_terms_on_failure": bool(pick("report_terms_on_failure", REPORT_TERMS_ON_FAILURE)),
    }


def apply_hull_damping(matlab_d, controls):
    """Return the exact LF damping matrix requested by the control panel.

    No effect at all in 'matlab' mode. The source .mat is never modified.
    """
    baseline = _matrix(matlab_d, "MATLAB damping")
    mode = controls["damping_mode"]
    if mode == "matlab":
        return baseline
    if mode == "add":
        return baseline + controls["damping_custom"]
    if mode == "replace":
        return controls["damping_custom"].copy()
    raise ValueError(f"Unknown damping mode: {mode}")


def show_hull_controls(controls, effective_damping=None):
    """Readable startup audit: confirms which controls actually reached FMU."""
    print("[HULL CONTROL] WAMIT file:", controls["mat_file"])
    print("[HULL CONTROL] Frequency count:", controls["frequency_count"])
    print("[HULL CONTROL] Heading step:", controls["heading_step_deg"], "deg")
    print("[HULL CONTROL] Frequency spacing:", controls["frequency_spacing"])
    print("[HULL CONTROL] Drift method:", controls["drift_method"])
    print("[HULL CONTROL] Damping mode:", controls["damping_mode"])
    print("[HULL CONTROL] Numerical monitoring:", controls["enable_watch"])
    print("[HULL CONTROL] Dynamics logging:", controls["log_dynamics_terms"])
    if effective_damping is not None:
        print("[HULL CONTROL] Effective D diagonal (SI):",
              np.diag(effective_damping).tolist())


if __name__ == "__main__":
    defaults = resolve_hull_controls({})
    show_hull_controls(defaults)
    print("Defaults only. The actual FMU also reads run config.toml overrides.")
