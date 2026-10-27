"""Load processed MSS/WAMIT hydrodynamic data from a MATLAB vessel structure.

Run this file in VS Code to inspect the original frequencies and table dimensions.
Use the source frequency nodes when the user-selected count matches; otherwise
resample within the overlapping *calculated* frequency range. No FMU or JSON
files are produced by this script.

License: GPL-3.0-or-later (see LICENSE and NOTICE.md in this folder)
Authors: Araz, Olve Grønås Birkeland (2026)
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


# --------------- ONLY USER SETTING -------------------
FREQUENCY_COUNT = 100  # None = use all original calculated source frequencies
# -----------------------------------------------------

# Internal MSS loader defaults; normally no manual input is needed.
# The simulation's active MATLAB file is selected by configs/gunnerus.toml.
GUNNERUS_FILE = Path(__file__).resolve().parents[4] / "data_examples" / "gunnerus.mat"
HEADING_STEP_DEG = 10
FREQUENCY_SPACING = "log"  # Used only when the chosen count differs from source
LF_REFERENCE_OMEGA = None  # Select lowest calculated frequency automatically
# Drift formulation is selected separately in WaveLoads.py.


def _frequency_axis(name, values, *, allow_zero=False):
    """Check that an MSS frequency vector is finite, ordered, and nonempty."""
    omega = np.asarray(values, dtype=float).ravel()
    if omega.size < 2:
        raise ValueError(f"{name}: at least two frequency points are required")
    minimum_invalid = omega[0] < 0 if allow_zero else omega[0] <= 0
    if (not np.isfinite(omega).all()
            or np.any(np.diff(omega) <= 0) or minimum_invalid):
        expected = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name}: expected increasing {expected} frequencies")
    return omega


def _same_grid(source, target):
    """Match actual numerical frequency nodes, not just the number of entries."""
    return (len(source) == len(target)
            and np.allclose(source, target, rtol=1e-12, atol=1e-12))


def _matrix_at(frequencies, matrix, target):
    """Use stored matrix slices when exact nodes exist; interpolate otherwise."""
    positions = np.searchsorted(frequencies, target)
    if (np.all(positions < len(frequencies))
            and np.allclose(frequencies[positions], target, rtol=1e-12, atol=1e-12)):
        return matrix[:, :, positions].copy()
    return interpolate_matrices(frequencies, matrix, target)


def _response_at(table, target_omega, target_headings):
    """Return original complex RAOs unchanged if both axes already match."""
    if (_same_grid(table.omega, target_omega)
            and _same_grid(table.heading_deg, target_headings)):
        return table.values.copy()
    return interpolate(table, target_omega, target_headings)


def _mat_viscous_at(raw_bv, frequencies, omega):
    """Accept 6x6 constant Bv, or 6x6xN frequency-dependent MSS Bv.

    A four-dimensional MSS array may contain a final vessel-speed axis;
    checked_matrix selects its first entry (zero-speed convention).
    """
    array = np.asarray(raw_bv, dtype=float)
    if array.shape == (6, 6):
        return checked_matrix(array, (6, 6), "Bv"), "constant 6x6"
    if array.ndim == 4:
        array = array[:, :, :, 0]
    array = checked_matrix(array, (6, 6, len(frequencies)), "Bv")
    return _matrix_at(frequencies, array, np.array([omega]))[:, :, 0], (
        f"frequency-dependent at omega={omega:.9g} rad/s"
    )


def load_mat_vessel(
    filename=GUNNERUS_FILE, *, frequency_count=FREQUENCY_COUNT,
    heading_step_deg=HEADING_STEP_DEG, frequency_spacing=FREQUENCY_SPACING,
    lf_reference_omega_rad_s=None, additional_linear_damping_matrix_si=None,
    drift_method=None, require_simulation_ready=False,
    use_mat_viscous_damping=False,
):
    """Load a supported MSS vessel .mat file and select a validated output grid.

    The *calculated* frequencies come from motionRAO.w, intersected with the
    force/drift and matrix grids. The MSS matrix-only zero and infinity stand-ins
    are not counted as calculated wave frequencies.
    """
    filename = Path(filename).expanduser().resolve()
    if not filename.is_file():
        raise FileNotFoundError(f"MATLAB hydrodynamic file not found: {filename}")
    raw = loadmat(filename, simplify_cells=True)
    if "vessel" not in raw:
        raise ValueError(f"{filename.name} does not contain an MSS 'vessel' structure")
    v = raw["vessel"]

    frequencies = _frequency_axis("vessel.freqs", v["freqs"], allow_zero=True)
    headings = np.rad2deg(np.asarray(v["headings"], dtype=float).ravel())
    if (headings.size < 3 or not np.isfinite(headings).all()
            or np.any(np.diff(headings) <= 0)
            or headings[0] < -1e-8 or headings[-1] >= 360 + 1e-8):
        raise ValueError("Invalid vessel headings: expected increasing radians in [0,2*pi)")
    if (not isinstance(heading_step_deg, (int, np.integer))
            or heading_step_deg <= 0 or 360 % heading_step_deg):
        raise ValueError("heading_step_deg must be a positive integer dividing 360")
    if frequency_spacing not in ("log", "linear"):
        raise ValueError("frequency_spacing must be 'log' or 'linear'")

    rigid = checked_matrix(v["MRB"], (6, 6), "MRB")
    tables = {
        name: checked_matrix(v[name], (6, 6, len(frequencies)), name)
        for name in ("A", "B", "C")
    }
    motion = read_response(v, "motionRAO", [1, -1, 1, -1, 1, -1], headings)
    force = read_response(v, "forceRAO", [1, -1, 1, -1, 1, -1], headings)
    drift = read_response(v, "driftfrc", [1, -1, -1], headings)
    for name, table in (("motionRAO", motion), ("forceRAO", force), ("driftfrc", drift)):
        _frequency_axis(f"{name}.w", table.omega)

    positive_frequencies = frequencies[frequencies > 0]
    low = max(positive_frequencies[0], motion.omega[0], force.omega[0], drift.omega[0])
    high = min(frequencies[-1], motion.omega[-1], force.omega[-1], drift.omega[-1])
    if high <= low:
        raise ValueError(f"No overlapping calculated frequency range (low={low}, high={high})")

    # The motion RAO table defines available calculated nodes. Its frequencies
    # are never inferred from matrix-only limiting entries or from user count.
    calculated = motion.omega[(motion.omega >= low) & (motion.omega <= high)]
    if len(calculated) < 2:
        raise ValueError("Fewer than two motion RAO points in the common frequency range")
    if frequency_count is None:
        count = len(calculated)
    elif isinstance(frequency_count, (int, np.integer)) and frequency_count >= 2:
        count = int(frequency_count)
    else:
        raise ValueError("frequency_count must be an integer >= 2 or None")

    if count == len(calculated):
        omega = calculated.copy()
        grid_mode = "original calculated frequency nodes (no frequency resampling)"
    else:
        grid = np.geomspace if frequency_spacing == "log" else np.linspace
        omega = grid(calculated[0], calculated[-1], count)
        grid_mode = f"{frequency_spacing} resampling ({len(calculated)} source -> {count} output)"
    angles = np.arange(0, 360, heading_step_deg, dtype=float)
    matrix_shapes = {name: tuple(np.shape(v[name])) for name in
                     ("MRB", "A", "B", "C", "Bv") if name in v}
    for name in ("motionRAO", "forceRAO", "driftfrc"):
        response = v[name]
        matrix_shapes[f"{name}.amp[0]"] = tuple(np.shape(response["amp"][0]))
        if "phase" in response:
            matrix_shapes[f"{name}.phase[0]"] = tuple(np.shape(response["phase"][0]))

    # Original MSS hydro arrays may include (omega=0) and a synthetic
    # high-frequency stand-in. Report them, but do not count them as real RAOs.
    matrix_only = frequencies[(frequencies < calculated[0] - 1e-12)
                              | (frequencies > calculated[-1] + 1e-12)]
    info = {
        "matrix_frequency_count": len(frequencies),
        "matrix_frequency_min_rad_s": float(frequencies[0]),
        "matrix_frequency_max_rad_s": float(frequencies[-1]),
        "matrix_only_frequency_values_rad_s": matrix_only.copy(),
        "motion_frequency_count": len(motion.omega),
        "force_frequency_count": len(force.omega),
        "drift_frequency_count": len(drift.omega),
        "source_calculated_count": len(calculated),
        "source_calculated_min_rad_s": float(calculated[0]),
        "source_calculated_max_rad_s": float(calculated[-1]),
        "common_frequency_min_rad_s": float(low),
        "common_frequency_max_rad_s": float(high),
        "output_count": len(omega),
        "output_min_rad_s": float(omega[0]),
        "output_max_rad_s": float(omega[-1]),
        "grid_mode": grid_mode,
        "input_frequency_ranges_rad_s": {
            "motionRAO": (float(motion.omega[0]), float(motion.omega[-1])),
            "forceRAO": (float(force.omega[0]), float(force.omega[-1])),
            "driftfrc": (float(drift.omega[0]), float(drift.omega[-1])),
        },
        "source_heading_count": len(headings),
        "output_heading_count": len(angles),
        "matrix_shapes": matrix_shapes,
        "lf_reference_rad_s": None,
        "viscous_damping_source": "not selected",
    }

    result = {
        "schema_version": 1,
        "source": str(filename),
        "vessel_name": str(v["main"]["name"]),
        "frequency_grid": omega,
        "headings_deg": angles,
        "gravity": float(v["main"]["g"]),
        "rigid_mass": rigid,
        "added_mass_by_frequency": _matrix_at(frequencies, tables["A"], omega),
        "radiation_damping_by_frequency": _matrix_at(frequencies, tables["B"], omega),
        "restoring_by_frequency": _matrix_at(frequencies, tables["C"], omega),
        "restoring": tables["C"][:, :, 0],
        "motion_table": ResponseTable(omega, np.deg2rad(angles),
                                      _response_at(motion, omega, angles), kind="rao"),
        "force_table": ResponseTable(omega, np.deg2rad(angles),
                                     _response_at(force, omega, angles)[[0, 1, 5]], kind="rao"),
        "drift_table": ResponseTable(omega, np.deg2rad(angles),
                                     _response_at(drift, omega, angles), kind="drift"),
        "drift_method": drift_method,
        "frequency_info": info,
    }

    if all(key in v for key in ("M", "D", "G")):
        result["added_mass"] = checked_matrix(v["M"], (6, 6), "M") - rigid
        result["damping"] = checked_matrix(v["D"], (6, 6), "D")
        result["restoring"] = checked_matrix(v["G"], (6, 6), "G")
        result["lf_approximation"] = "MSS M/D/G"
        info["viscous_damping_source"] = "stored MSS M/D/G"
    else:
        if use_mat_viscous_damping and additional_linear_damping_matrix_si is not None:
            raise ValueError("Choose either stored MATLAB Bv or an explicit damping matrix")
        lf = (float(lf_reference_omega_rad_s)
              if lf_reference_omega_rad_s is not None else
              (float(calculated[0]) if use_mat_viscous_damping else None))
        if lf is not None:
            if not np.isfinite(lf) or lf < low or lf > high:
                raise ValueError(f"LF reference {lf:.9g} rad/s outside the calculated "
                                 f"frequency range [{low:.9g}, {high:.9g}] rad/s")
            info["lf_reference_rad_s"] = lf
        if use_mat_viscous_damping:
            if "Bv" not in v:
                raise ValueError("MATLAB vessel missing requested Bv viscous damping")
            viscous, info["viscous_damping_source"] = _mat_viscous_at(
                v["Bv"], frequencies, lf
            )
        elif additional_linear_damping_matrix_si is not None:
            viscous = checked_matrix(
                additional_linear_damping_matrix_si, (6, 6), "viscous damping"
            )
            info["viscous_damping_source"] = "explicit user-supplied matrix"
        else:
            viscous = None
        if lf is not None and viscous is not None:
            result["added_mass"] = _matrix_at(frequencies, tables["A"], np.array([lf]))[:, :, 0]
            result["damping"] = _matrix_at(frequencies, tables["B"], np.array([lf]))[:, :, 0] + viscous
            result["restoring"] = _matrix_at(frequencies, tables["C"], np.array([lf]))[:, :, 0]
            result["lf_approximation"] = (
                f"omega={lf:.9g} rad/s + {info['viscous_damping_source']}"
            )

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
    """Load the configured MATLAB vessel and auto-select its lowest REAL LF omega."""
    return load_mat_vessel(
        GUNNERUS_FILE, lf_reference_omega_rad_s=LF_REFERENCE_OMEGA,
        use_mat_viscous_damping=True,
        drift_method=selected_drift_method(), require_simulation_ready=True,
    )


def main():
    """Run this file in VS Code: show exactly what was read and what is used."""
    data = load_gunnerus()
    i = data["frequency_info"]
    fmt = lambda w: f"{w:.9f} rad/s (T={2*np.pi/w:.5f} s)" if w > 0 else "0 rad/s (T=infinity)"

    print("\n========== MSS / WAMIT MATLAB DATA INSPECTION ==========")
    print("MATLAB file:", data["source"])
    print("Vessel:", data["vessel_name"])
    print("Matrix table frequency axis:")
    print(f"  N = {i['matrix_frequency_count']}, from "
          f"{fmt(i['matrix_frequency_min_rad_s'])} to "
          f"{fmt(i['matrix_frequency_max_rad_s'])}")
    print("Motion / force / drift RAO frequency points:",
          i["motion_frequency_count"], "/", i["force_frequency_count"],
          "/", i["drift_frequency_count"])
    for kind, (w_min, w_max) in i["input_frequency_ranges_rad_s"].items():
        print(f"  {kind:10s} original range: {fmt(w_min)} to {fmt(w_max)}")
    print(f"Common usable source range: {fmt(i['common_frequency_min_rad_s'])}"
          f" to {fmt(i['common_frequency_max_rad_s'])}")
    print(f"CALCULATED wave range: {fmt(i['source_calculated_min_rad_s'])}"
          f" to {fmt(i['source_calculated_max_rad_s'])}")
    if len(i["matrix_only_frequency_values_rad_s"]):
        print("Extra matrix-only frequency entries (not counted as wave RAOs):",
              ", ".join(fmt(w) for w in i["matrix_only_frequency_values_rad_s"]))
        print("  NOTE: zero/infinity limiting values may be approximations; "
              "check WAMIT provenance.")
    print(f"Chosen output: {i['output_count']} frequencies, "
          f"{fmt(i['output_min_rad_s'])} to {fmt(i['output_max_rad_s'])}")
    print("Grid strategy:", i["grid_mode"])
    print(f"Headings: source {i['source_heading_count']}, "
          f"output {i['output_heading_count']} at {HEADING_STEP_DEG} degree steps")
    print("Original MATLAB field dimensions:")
    for name, shape in i["matrix_shapes"].items():
        print(f"  {name:10s}: {shape}")
    print("LF reference:", fmt(i["lf_reference_rad_s"]) if i["lf_reference_rad_s"] is not None else "not selected")
    print("LF viscous damping:", i["viscous_damping_source"])

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
