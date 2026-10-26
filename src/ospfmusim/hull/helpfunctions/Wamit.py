"""Load Gunnerus hydrodynamic data from MATLAB into the vessel simulator.

Click Run in VS Code to check every matrix. No FMU or JSON is created.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

try:
    from ._wamit_motion import checked_matrix, read_response, interpolate_matrices, interpolate
    from .WaveLoads import ResponseTable, selected_drift_method
except ImportError:  # VS Code: Run Python File
    from _wamit_motion import checked_matrix, read_response, interpolate_matrices, interpolate
    from WaveLoads import ResponseTable, selected_drift_method


# ---------------- SETTINGS: edit these ----------------
GUNNERUS_FILE = Path(__file__).resolve().parents[4] / "data_examples" / "GunnerusL35_20201112.mat"
FREQUENCY_COUNT = 100
HEADING_STEP_DEG = 10
FREQUENCY_SPACING = "log"       # "log" or "linear"
LF_REFERENCE_OMEGA = 0.2094395  # rad/s: first Gunnerus source frequency
# Wave drift method is selected at the top of WaveLoads.py.
# ------------------------------------------------------


def load_mat_vessel(
    filename=GUNNERUS_FILE, *, frequency_count=FREQUENCY_COUNT,
    heading_step_deg=HEADING_STEP_DEG, frequency_spacing=FREQUENCY_SPACING,
    lf_reference_omega_rad_s=None, additional_linear_damping_matrix_si=None,
    drift_method=None, require_simulation_ready=False,
    use_mat_viscous_damping=False,
):
    """Read, check and interpolate MSS hydrodynamic data; return Python tables."""
    filename = Path(filename).expanduser().resolve()
    v = loadmat(filename, simplify_cells=True)["vessel"]  # READ MATLAB HERE
    if use_mat_viscous_damping:
        if additional_linear_damping_matrix_si is not None:
            raise ValueError("Choose one viscous damping source")
        additional_linear_damping_matrix_si = checked_matrix(v["Bv"], (6, 6), "Bv")

    frequencies = np.asarray(v["freqs"], dtype=float).ravel()
    headings = np.rad2deg(np.asarray(v["headings"], dtype=float).ravel())
    if (not np.any(frequencies > 0) or np.any(np.diff(frequencies) <= 0)
            or heading_step_deg <= 0 or 360 % heading_step_deg):
        raise ValueError("Invalid source frequencies or output heading step")

    rigid = checked_matrix(v["MRB"], (6, 6), "MRB")
    A, B, C = (checked_matrix(v[name], (6, 6, len(frequencies)), name)
               for name in ("A", "B", "C"))
    motion = read_response(v, "motionRAO", [1, -1, 1, -1, 1, -1], headings)
    force = read_response(v, "forceRAO", [1, -1, 1, -1, 1, -1], headings)
    drift = read_response(v, "driftfrc", [1, -1, -1], headings)

    low = max(frequencies[frequencies > 0][0], motion.omega[0], force.omega[0], drift.omega[0])
    high = min(frequencies[-1], motion.omega[-1], force.omega[-1], drift.omega[-1])
    if high <= low or frequency_count < 2:
        raise ValueError("No common frequency range")
    if frequency_spacing not in ("log", "linear"):
        raise ValueError("frequency_spacing must be 'log' or 'linear'")
    make_grid = np.geomspace if frequency_spacing == "log" else np.linspace
    omega = make_grid(low, high, frequency_count)
    angles = np.arange(0, 360, heading_step_deg)

    result = {
        "schema_version": 1,
        "source": str(filename),
        "vessel_name": str(v["main"]["name"]),
        "frequency_grid": omega,
        "headings_deg": angles,
        "gravity": float(v["main"]["g"]),
        "rigid_mass": rigid,
        "added_mass_by_frequency": interpolate_matrices(frequencies, A, omega),
        "radiation_damping_by_frequency": interpolate_matrices(frequencies, B, omega),
        "restoring_by_frequency": interpolate_matrices(frequencies, C, omega),
        "restoring": C[:, :, 0],
        "motion_table": ResponseTable(omega, np.deg2rad(angles),
                                      interpolate(motion, omega, angles), kind="rao"),
        "force_table": ResponseTable(omega, np.deg2rad(angles),
                                     interpolate(force, omega, angles)[[0, 1, 5]], kind="rao"),
        "drift_table": ResponseTable(omega, np.deg2rad(angles),
                                     interpolate(drift, omega, angles), kind="drift"),
        "drift_method": drift_method,
    }

    # Low-frequency matrices: use stored M/D/G, or an explicitly selected omega.
    if all(key in v for key in ("M", "D", "G")):
        result["added_mass"] = checked_matrix(v["M"], (6, 6), "M") - rigid
        result["damping"] = checked_matrix(v["D"], (6, 6), "D")
        result["restoring"] = checked_matrix(v["G"], (6, 6), "G")
        result["lf_approximation"] = "MSS M/D/G"
    elif lf_reference_omega_rad_s is not None and additional_linear_damping_matrix_si is not None:
        w = float(lf_reference_omega_rad_s)
        if not low <= w <= high:
            raise ValueError("LF reference omega is outside the source range")
        viscous = checked_matrix(additional_linear_damping_matrix_si, (6, 6), "viscous damping")
        result["added_mass"] = interpolate_matrices(frequencies, A, [w])[:, :, 0]
        result["damping"] = interpolate_matrices(frequencies, B, [w])[:, :, 0] + viscous
        result["restoring"] = interpolate_matrices(frequencies, C, [w])[:, :, 0]
        result["lf_approximation"] = f"omega={w} rad/s + specified viscous damping"

    if drift_method not in (None, "newman_6_97", "sorensen_6_99"):
        raise ValueError("Unknown wave drift method")
    if drift_method == "sorensen_6_99" and np.any(result["drift_table"].values < -1e-12):
        raise ValueError("Signed drift requires newman_6_97")
    if require_simulation_ready:
        if "added_mass" not in result:
            raise ValueError("Missing LF frequency and viscous damping")
        if drift_method is None:
            raise ValueError("Missing wave-drift approximation")
    return result


def load_gunnerus():
    """Load the Gunnerus .mat file with its stored Bv damping matrix."""
    return load_mat_vessel(
        GUNNERUS_FILE, lf_reference_omega_rad_s=LF_REFERENCE_OMEGA,
        use_mat_viscous_damping=True,
        drift_method=selected_drift_method(), require_simulation_ready=True,
    )


def main():
    """Run this file in VS Code and verify actual matrix values in the terminal."""
    data = load_gunnerus()
    print("\n========== GUNNERUS MATLAB DATA TEST ==========")
    print("Source:", data["source"])
    print("Vessel:", data["vessel_name"])
    print(f"Output: {len(data['frequency_grid'])} frequencies, "
          f"{len(data['headings_deg'])} headings, {HEADING_STEP_DEG} degree step\n")

    matrices = {
        "MRB (rigid mass)": data["rigid_mass"],
        "A (added mass)": data["added_mass_by_frequency"],
        "B (radiation damping)": data["radiation_damping_by_frequency"],
        "C (restoring)": data["restoring_by_frequency"],
        "LF added mass": data["added_mass"],
        "LF damping": data["damping"],
        "LF restoring": data["restoring"],
        "Motion RAOs": data["motion_table"].values,
        "Force RAOs": data["force_table"].values,
        "Wave drift": data["drift_table"].values,
    }
    for name, values in matrices.items():
        numbers = np.abs(values) if np.iscomplexobj(values) else values
        print(f"[OK] {name:23} shape={str(values.shape):16} "
              f"min={numbers.min():.4g} max={numbers.max():.4g}")
    print("\nAll matrices loaded and checked. No simulation or JSON created.")


if __name__ == "__main__":
    main()
