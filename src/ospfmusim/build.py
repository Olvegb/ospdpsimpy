"""Builds the FMUs from the models in this package.

An FMU is one model script (the class that inherits Fmi2Slave) plus a copy of
the model folders of this package in its resources/. The model imports its
help functions through the package, as ospfmusim.<folder>.helpfunctions.<module>.

Every FMU gets the same copy. The FMUs share one Python interpreter when they
run, so a module name must mean the same code in all of them: three different
helpfunctions/ folders at the top of resources/ would collide.

An FMU is only built again when a file in the model folders is newer than the
FMU. Changes to the config files in configs/ and to OspSystemStructure.xml do
NOT need a rebuild.

From the repo folder, with the conda environment master active:

    python -m ospfmusim.build            # build the FMUs that have changed
    python -m ospfmusim.build --force    # build every FMU again

License: MIT (see LICENSE in the repository root)
Authors: Olve Grønås Birkeland, William Boland, Viljar Simonnes Tveit, Araz (2026)
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent        # src/ospfmusim
SYSTEM_STRUCTURE = PACKAGE / "OspSystemStructure.xml"

# The FMUs made from this package: FMU name (the Fmi2Slave class, and the name
# of the .fmu file in OspSystemStructure.xml) -> model script.
FMUS = {
    "Environment": "enviroment/environment.py",
    "Waves": "waves/waves.py",
    "VesselBlock": "hull/vessel.py",
    "ExtendedKalmanFilter": "observer/ekf/ekf.py",
    "DPController": "controller/dp_controller.py",
    "ThrustAllocation": "thruster/thrust_allocation.py",
    "ThrusterDynamics": "thruster/thruster_dynamics.py",
    "ThrusterDynamicsKtKq": "thruster_ktkq/thruster_dynamics_ktkq.py",
}

# Folders that are not copied into the FMUs.
SKIPPED_FOLDERS = {"fmus", "__pycache__", "plots"}


def model_files():
    """The files every FMU gets a copy of: the model folders and the package's __init__.py.

    The other files directly in src/ospfmusim (build.py, run.py and
    OspSystemStructure.xml) are tools, not models, and are left out.
    """
    files = [PACKAGE / "__init__.py"]
    for path in PACKAGE.rglob("*"):
        parts = path.relative_to(PACKAGE).parts
        if path.is_file() and len(parts) > 1 and not SKIPPED_FOLDERS.intersection(parts):
            files.append(path)
    return sorted(files)


def fmu_paths(system_structure):
    """The FMU files a system structure uses. source="..." is relative to the .xml file."""
    system_structure = Path(system_structure).resolve()
    root = ET.parse(system_structure).getroot()
    return [(system_structure.parent / element.get("source")).resolve()
            for element in root.iter() if element.tag.endswith("Simulator")]


def build_fmus(system_structure=SYSTEM_STRUCTURE, force=False):
    """Builds the FMUs system_structure uses from the models in this package.

    Each FMU is built where system_structure looks for it. An FMU that is newer
    than every model file is skipped, unless force is True. An FMU that is not
    made from this package must already be there.

    Returns the FMUs that were built.
    """
    files = model_files()
    newest = max(path.stat().st_mtime for path in files)
    outdated = {}
    for fmu in fmu_paths(system_structure):
        if fmu.stem not in FMUS:
            if not fmu.is_file():
                raise FileNotFoundError(f"{fmu} is used in {system_structure}, but does not exist")
        elif force or not fmu.is_file() or fmu.stat().st_mtime < newest:
            outdated[fmu] = FMUS[fmu.stem]
    if not outdated:
        return []

    with tempfile.TemporaryDirectory(prefix="ospfmusim_") as temp:
        # The copy that goes into the FMUs. pythonfmu also imports the model
        # from this copy while building, to find its variables.
        package = Path(temp) / PACKAGE.name
        for path in files:
            target = package / path.relative_to(PACKAGE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        python_path = os.pathsep.join(filter(None, [temp, os.environ.get("PYTHONPATH")]))
        env = {**os.environ, "PYTHONPATH": python_path}
        for fmu, script in outdated.items():
            print(f"Building {fmu.name} ...")
            # sys.executable is the Python of the active conda environment,
            # so its pythonfmu is used.
            subprocess.run([sys.executable, "-m", "pythonfmu", "build", "-f", package / script,
                            package, "-d", fmu], check=True, env=env)
    return list(outdated)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", default=SYSTEM_STRUCTURE,
                        help="OspSystemStructure.xml (default: the one in src/ospfmusim)")
    parser.add_argument("--force", action="store_true", help="build every FMU again")
    args = parser.parse_args(argv)
    built = build_fmus(args.system, args.force)
    print(f"Built {len(built)} FMU(s)." if built else "Every FMU is up to date.")


if __name__ == "__main__":
    main()
