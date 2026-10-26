"""Runs the co-simulation and saves the results in a run folder in results/.

From Python:

    from ospfmusim.run import run_simulation

    run_dir = run_simulation("data_examples/GunnerusL35_20201112.mat", "configs/gunnerus.toml",
                             "src/ospfmusim/OspSystemStructure.xml", duration=60.0)

From the repo folder, with the conda environment master active:

    python -m ospfmusim.run                                 # the defaults, results/<time stamp>/
    python -m ospfmusim.run --name high_wind                # results/<time stamp>_high_wind/
    python -m ospfmusim.run --config configs/gunnerus.toml --duration 60
    python -m ospfmusim.run --mat other.mat --system other.xml --output results/my_run

The defaults are the OspSystemStructure.xml in src/ospfmusim, the config it
points config_path to, the .mat file in that config ([vessel_fmu] mat_file) and
the duration in that config ([simulation] duration).

run_simulation() first builds the FMUs the system uses, but only those whose
model code has changed (see build.py). Every run gets its own folder,
results/<YYYYMMDD_HHMMSS>[_name]/ in the repo folder, with one CSV per FMU and
a copy of the config (config.toml) and the system structure it was run with.
Nothing from earlier runs is overwritten.

License: MIT (see LICENSE in the repository root)
Authors: William Boland, Olve Grønås Birkeland (2026)
"""
import argparse
import re
import shutil
import tomllib
from datetime import datetime
from pathlib import Path

import pandas as pd
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimLogging import CosimLogLevel, log_output_level
from libcosimpy.CosimObserver import CosimObserver

from app.build.config_tools import make_paths_absolute, set_values
from app.build.data import default_config
from ospfmusim.build import SYSTEM_STRUCTURE, build_fmus

ROOT = Path(__file__).resolve().parents[2]       # the repo folder
RESULTS = ROOT / "results"


def run_folder(name):
    """results/<YYYYMMDD_HHMMSS>[_name], with the name made safe for a folder."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return RESULTS / (f"{stamp}_{name}" if name else stamp)


def set_config_path(execution, config_path):
    """Points config_path on every FMU to the run's own copy of the config."""
    for slave in execution.slave_infos():
        for variable in execution.slave_variables(slave.index):
            if variable.name == b"config_path":
                execution.string_initial_value(slave.index, variable.reference, str(config_path))


def run_simulation(mat_file, config_file, system_structure=SYSTEM_STRUCTURE, duration=None,
                   name="", run_dir=None):
    """Builds the FMUs, runs the simulation and returns the run folder with the results.

    mat_file          the vessel's hydrodynamic data (.mat), read by the vessel FMU
    config_file       the config file (.toml) the FMUs read their constants from
    system_structure  OspSystemStructure.xml: the FMUs, their connections and the step size
    duration          simulated time [s]; None uses [simulation] duration in the config
    name              short name added to the run folder: results/<time stamp>_<name>
    run_dir           a chosen run folder instead of results/<time stamp>[_name]

    Relative paths are read from the folder Python is started from.
    """
    mat_file, config_file, system_structure = (
        Path(path).resolve() for path in (mat_file, config_file, system_structure))
    for path in (mat_file, config_file, system_structure):
        if not path.is_file():
            raise FileNotFoundError(f"Not found: {path}")

    # 1. Build the FMUs the system uses, where it looks for them. Only those
    #    whose model code has changed are built again.
    build_fmus(system_structure)

    # 2. Make the run folder and save the config there. Relative file paths in
    #    the config are made absolute, because the FMUs read them relative to
    #    the config file, which now lies in the run folder. The .mat file and
    #    the duration are written into the copy, so it shows what was run.
    run_dir = Path(run_dir).resolve() if run_dir else run_folder(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    for old in [*run_dir.glob("*.csv"), *run_dir.glob("*.yaml")]:
        old.unlink()                             # only if an existing folder is reused
    text = make_paths_absolute(config_file.read_text(encoding="utf-8-sig"), config_file.parent)
    changes = {("vessel_fmu", "mat_file"): mat_file.as_posix()}
    if duration is not None:
        changes[("simulation", "duration")] = float(duration)
    text = set_values(text, changes)
    run_config = run_dir / "config.toml"
    run_config.write_text(text, encoding="utf-8")
    shutil.copy2(system_structure, run_dir / system_structure.name)
    duration = tomllib.loads(text)["simulation"]["duration"]

    # 3. Load the system, and point every FMU to the run's config. libcosim
    #    reads the FMU paths (source="fmus/...") relative to the system structure.
    log_output_level(CosimLogLevel.WARNING)  # hide routine messages, show only warnings and errors
    execution = CosimExecution.from_osp_config_file(osp_path=str(system_structure))
    set_config_path(execution, run_config)

    # 4. Save every variable to one CSV file per FMU in the run folder.
    observer = CosimObserver.create_to_dir(log_dir=str(run_dir))
    execution.add_observer(observer=observer)

    # 5. Run. The FMUs read the config here (in exit_initialization_mode).
    #    libcosim measures time in nanoseconds.
    try:
        finished = execution.simulate_until(target_time=int(duration * 1e9))
    finally:
        # Close the simulation, also when it failed, so libcosim finishes
        # writing the CSV files. A simulation that is still open when Python
        # exits makes Python hang.
        del execution, observer
    if not finished:
        raise RuntimeError("The simulation failed, see the error message above.")
    return run_dir


def configured_mat_file(config_file):
    """[vessel_fmu] mat_file in the config. A relative path is read from the config's folder."""
    config = tomllib.loads(Path(config_file).read_text(encoding="utf-8-sig"))
    return (Path(config_file).parent / config["vessel_fmu"]["mat_file"]).resolve()


def print_summary(run_dir):
    def last(simulator):
        return pd.read_csv(next(run_dir.glob(f"{simulator}_*.csv")), index_col="Time").iloc[-1]

    config = tomllib.loads((run_dir / "config.toml").read_text(encoding="utf-8"))
    environment, vessel = last("environment"), last("vessel")
    wind_speed = (environment["mean_wind_speed"] + environment["slowly_varying_wind_speed"]
                  + environment["gust_wind_speed"])
    print(f"Vessel:  {config['vessel']['type']}, "
          f"simulated for {config['simulation']['duration']} s")
    print(f"Wind:    {wind_speed:.1f} m/s from {environment['wind_direction_deg']:.1f} deg "
          "(NED, at the end)")
    print(f"Current: {environment['current_speed']} m/s towards "
          f"{environment['current_direction_deg']} deg (NED)")

    print("\nVessel at the end of the simulation:")
    for name, unit in [("north", "m"), ("east", "m"), ("yaw", "rad")]:
        print(f"  {name:14} = {vessel[name]:12.3f} {unit}")
    for name, unit in [("wind_X", "N"), ("wind_Y", "N"), ("wind_N", "Nm")]:
        print(f"  {name:14} = {vessel[name]:12.1f} {unit}")

    kalman = last("kalman_filter")
    print("\nKalman filter estimates at the end of the simulation:")
    for name, unit in [("north_hat", "m"), ("east_hat", "m"), ("psi_hat", "rad"),
                       ("u_hat", "m/s"), ("v_hat", "m/s"), ("r_hat", "rad/s"),
                       ("b_x_hat", "N"), ("b_y_hat", "N"), ("b_n_hat", "Nm")]:
        print(f"  {name:9} = {kalman[name]:12.4g} {unit}")

    allocation, dynamics = last("allocation"), last("dynamics")
    print("\nThrusters at the end of the simulation "
          f"(utilisation {allocation['utilisation']:.3f}, "
          f"saturated {bool(allocation['saturated'])}):")
    for name, unit in [("X", "N"), ("Y", "N"), ("N", "Nm"), ("total_power_kw", "kW")]:
        print(f"  {name:14} = {dynamics[name]:12.1f} {unit}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mat", default=None,
                        help="the .mat file (default: [vessel_fmu] mat_file in the config)")
    parser.add_argument("--config", default=None,
                        help="config file (default: the one the system structure points to)")
    parser.add_argument("--system", default=SYSTEM_STRUCTURE,
                        help="OspSystemStructure.xml (default: the one in src/ospfmusim)")
    parser.add_argument("--duration", type=float, default=None,
                        help="simulated time [s] (default: [simulation] duration in the config)")
    parser.add_argument("--name", default="", help="short name added to the run folder")
    parser.add_argument("--output", default=None,
                        help="run folder (default: results/<time stamp>[_name])")
    args = parser.parse_args(argv)

    config_file = Path(args.config).resolve() if args.config else default_config(args.system)
    mat_file = args.mat or configured_mat_file(config_file)
    run_dir = run_simulation(mat_file, config_file, args.system, args.duration, args.name,
                             args.output)

    print_summary(run_dir)
    print(f"\nResults: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
