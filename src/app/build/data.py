"""Reads the simulation results and the config files for the results app.

Plain Python and pandas, no Streamlit, so it can be used and checked on its own.

The config files are in configs/, one .toml file per set-up. The default one is
the file src/ospfmusim/OspSystemStructure.xml points config_path to.

ospfmusim/run.py puts every run in its own folder, results/<YYYYMMDD_HHMMSS>[_name]/,
with a copy of the config it was run with (config.toml). libcosim writes one
CSV per FMU there, named <simulator>_<YYYYMMDD>_<HHMMSS>_<microseconds>.csv,
where <simulator> is the name in OspSystemStructure.xml (environment, vessel, ...).
"""
import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]       # the repo folder
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
SYSTEM = ROOT / "src" / "ospfmusim" / "OspSystemStructure.xml"

CSV_NAME = re.compile(r"^(?P<simulator>.+)_\d{8}_\d{6}_\d+\.csv$")
WAVE_COLUMN = re.compile(r"^wave\d+_")


def result_files(folder):
    """{simulator: newest CSV file} in a results folder."""
    files = {}
    for path in sorted(Path(folder).glob("*.csv")):  # sorted by name = by time stamp
        match = CSV_NAME.match(path.name)
        if match:
            files[match["simulator"]] = path
    return files


def load_results(folder):
    """{simulator: DataFrame indexed by time [s]}. The config_path column is dropped."""
    results = {}
    for simulator, path in result_files(folder).items():
        df = pd.read_csv(path, index_col="Time")
        results[simulator] = df.drop(columns=["StepCount", "config_path"], errors="ignore")
    return results


def folder_stamp(folder):
    """Changes whenever a CSV file in the folder changes. Used as a cache key."""
    return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(Path(folder).glob("*.csv")))


def list_runs(results=RESULTS):
    """[(label, folder)] of every run folder with CSV files, newest first.

    Older flat results directly in results/ (from before run folders, or from
    cosim CLI) are listed last.
    """
    results = Path(results)
    if not results.is_dir():
        return []
    runs = [(d.name, d) for d in sorted(results.iterdir(), reverse=True)
            if d.is_dir() and any(d.glob("*.csv"))]
    if any(results.glob("*.csv")):
        runs.append(("results/ (loose files)", results))
    return runs


def run_config_path(folder):
    """The config a run was made with, or None for runs without a copy."""
    path = Path(folder) / "config.toml"
    return path if path.is_file() else None


def list_configs(configs=CONFIGS):
    """[config files] in configs/, sorted by name."""
    return sorted(Path(configs).glob("*.toml"))


def default_config(system=SYSTEM):
    """The config file OspSystemStructure.xml points config_path to.

    The path in the XML is relative to the repo folder, where the simulation is
    started. Falls back to the first file in configs/ if the XML has none.
    """
    for element in ET.parse(system).getroot().iter():
        if element.tag.endswith("InitialValue") and element.get("variable") == "config_path":
            for value in element:
                if value.tag.endswith("String") and value.get("value"):
                    return (ROOT / value.get("value")).resolve()
    configs = list_configs()
    if not configs:
        raise FileNotFoundError(f"No config file (*.toml) in {CONFIGS}")
    return configs[0]


def load_config(path):
    return tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))


def thruster_slots(config):
    """[(slot number, name)] for the thrusters in config.toml, in slot order."""
    thrusters = config.get("thrusters", [])
    return [(k, thruster["name"]) for k, thruster in enumerate(thrusters, start=1)]


def total_wind_speed(environment):
    return (environment["mean_wind_speed"] + environment["slowly_varying_wind_speed"]
            + environment["gust_wind_speed"])


# Arrow length per quantity in the footprint plot: an arrow is full length at this value.
ENVIRONMENT_REFERENCE = {"Wind": 20.0, "Waves": 5.0, "Current": 1.0}


def position_std(vessel):
    """Standard deviation of the distance from the mean position [m].

    sqrt(var north + var east).
    """
    return (vessel["north"].var() + vessel["east"].var()) ** 0.5


def summary(results, config):
    """Key numbers of a run, as {label: (value, unit)}. Missing data is skipped."""
    out = {}
    first = next(iter(results.values()), None)
    if first is not None and len(first):
        out["Simulated time"] = (first.index[-1], "s")
    dp = results.get("dp_controller")
    has_dp = dp is not None and len(dp)
    vessel = results.get("vessel")
    has_vessel = vessel is not None and len(vessel)
    if has_dp:
        distance = (dp["error_north"] ** 2 + dp["error_east"] ** 2) ** 0.5
        out["Position error, end"] = (distance.iloc[-1], "m")
        out["Position error, max"] = (distance.max(), "m")
    if has_vessel:
        out["Std position"] = (position_std(vessel), "m")
    if has_dp:
        out["Heading error, max"] = (dp["error_heading_deg"].abs().max(), "deg")
    if has_vessel:
        out["Std heading"] = (np.rad2deg(np.unwrap(vessel["yaw"].to_numpy())).std(ddof=1), "deg")
    dynamics = results.get("dynamics")
    if dynamics is not None and len(dynamics):
        out["Power, mean"] = (dynamics["total_power_kw"].mean(), "kW")
        out["Power, max"] = (dynamics["total_power_kw"].max(), "kW")
    allocation = results.get("allocation")
    if allocation is not None and len(allocation):
        out["Utilisation, max"] = (allocation["utilisation"].max(), "")
        out["Time saturated"] = (100.0 * allocation["saturated"].astype(bool).mean(), "%")
    environment = results.get("environment")
    if environment is not None and len(environment):
        out["Wind speed, mean"] = (total_wind_speed(environment).mean(), "m/s")
        out["Current speed, mean"] = (environment["current_speed"].mean(), "m/s")
    return out
