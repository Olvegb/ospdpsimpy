"""Example: a 1000 s simulation of RV Gunnerus.

Run from the repo folder, with the conda environment master active:

    python scripts/runsimulationexample.py

run_simulation() builds the FMUs that have changed, runs the system and saves
the results in results/<time stamp>_gunnerus_1000s/: one CSV per FMU, and a
copy of the config and the system structure the run used.

License: MIT (see LICENSE in the repository root)
Authors: Olve Grønås Birkeland (2026)
"""
from pathlib import Path

from app.build.data import load_results
from ospfmusim.run import print_summary, run_simulation

ROOT = Path(__file__).resolve().parents[1]       # the repo folder

MAT_FILE = ROOT / "data_examples" / "GunnerusL35_20201112.mat"
CONFIG_FILE = ROOT / "configs" / "gunnerus.toml"
SYSTEM_STRUCTURE = ROOT / "src" / "ospfmusim" / "OspSystemStructure.xml"
DURATION = 100.0                                # simulated time [s], instead of the config's


if __name__ == "__main__":
    run_dir = run_simulation(MAT_FILE, CONFIG_FILE, SYSTEM_STRUCTURE, duration=DURATION,
                             name="gunnerus_1000s")
    print_summary(run_dir)
    print(f"\nResults: {run_dir}")

    # The results as pandas DataFrames, one per FMU, indexed by time [s].
    vessel = load_results(run_dir)["vessel"]
    distance = (vessel["north"] ** 2 + vessel["east"] ** 2) ** 0.5
    print(f"Largest distance from the start position: {distance.max():.3f} m")
