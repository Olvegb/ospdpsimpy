"""Tests of the built VesselBlock FMU: runs the real FMU with fmpy and compares it with
independent calls to the four hull modules (LF, WF, WindModel, WaveLoads).

Needs fmpy and the built FMU (python -m ospfmusim.build). Skipped without them.
The vessel data is the SYNTHETIC fixture hull/helpfunctions/example_test_config.json,
NOT calibrated Gunnerus data.

License: MIT (see LICENSE in the repository root)
Authors: Araz, Olve Grønås Birkeland (2026)
"""
import json
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest

from app.build.data import default_config
from ospfmusim.build import PACKAGE
from ospfmusim.hull.helpfunctions.LF import LF
from ospfmusim.hull.helpfunctions.WaveLoads import WaveComponents, WaveLoads
from ospfmusim.hull.helpfunctions.WF import WF
from ospfmusim.hull.helpfunctions.WindModel import WindLoads
from ospfmusim.hull.vessel import POSE, VELOCITY, read_table

fmpy = pytest.importorskip("fmpy")
FMU2Slave = pytest.importorskip("fmpy.fmi2").FMU2Slave
validate_fmu = pytest.importorskip("fmpy.validation").validate_fmu

HELP = PACKAGE / "hull" / "helpfunctions"
FMU = PACKAGE / "fmus" / "VesselBlock.fmu"
FIXTURE = HELP / "example_test_config.json"
NAMES = (list(POSE) + ["lf_" + s for s in POSE] + ["wf_" + s for s in POSE]
         + ["lf_" + s for s in VELOCITY])
CASES = {
    "calm": {},
    "thrusters": {"X_thrusters": 10000, "Y_thrusters": 2000, "N_thrusters": 30000},
    "wind": {"mean_wind_speed": 15, "wind_direction_deg": 90},
    "waves": {"wave1_amplitude": .8, "wave1_omega": 1., "wave1_phase": .3,
              "wave1_direction_deg": 45},
    "combined": {"mean_wind_speed": 10, "wind_direction_deg": 120, "X_thrusters": 10000,
                 "wave1_amplitude": .8, "wave1_omega": 1., "wave1_phase": .3,
                 "wave1_direction_deg": 45, "current_north": .1, "current_east": .03},
}

pytestmark = pytest.mark.skipif(not FMU.is_file(),
                                reason=f"{FMU} is not built (python -m ospfmusim.build)")


@pytest.fixture(scope="module")
def model_description():
    return fmpy.read_model_description(str(FMU), validate=True)


@pytest.fixture(scope="module")
def unzipped(tmp_path_factory):
    return fmpy.extract(str(FMU), unzipdir=str(tmp_path_factory.mktemp("VesselBlock")))


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    """The config the simulation uses, with [vessel_fmu] pointing to the synthetic fixture.

    Returns the path of the copy and its contents.
    """
    data_file = FIXTURE.as_posix()
    text = re.sub(r"^\[vessel_fmu\]\n.*?(?=^\[)",
                  lambda _: f'[vessel_fmu]\ndata_file = "{data_file}"\n\n',
                  default_config().read_text(encoding="utf-8-sig"), count=1, flags=re.M | re.S)
    path = tmp_path_factory.mktemp("config") / "config.toml"
    path.write_text(text, encoding="utf-8")
    shared = tomllib.loads(text)
    assert shared["vessel_fmu"] == {"data_file": data_file}
    return path, shared


def test_fmu_passes_schema_validation():
    assert not validate_fmu(str(FMU))


@pytest.mark.parametrize("name", ["LF.py", "WF.py", "WindModel.py", "WaveLoads.py"])
def test_packaged_modules_match_the_source(unzipped, name):
    packaged = Path(unzipped) / "resources" / "ospfmusim" / "hull" / "helpfunctions"
    assert (packaged / name).read_bytes() == (HELP / name).read_bytes(), \
        "the FMU is older than the source: python -m ospfmusim.build"


@pytest.mark.parametrize("label", CASES)
def test_fmu_matches_the_modules(model_description, unzipped, config, label):
    inputs = CASES[label]
    config_path, shared = config
    refs = {v.name: v.valueReference for v in model_description.modelVariables}
    data = json.loads(FIXTURE.read_text())
    vessel = shared["vessel"]
    wind_settings = {k: vessel[k] for k in ("loa", "area_lateral", "area_frontal", "s_L", "s_H")}
    wind_settings.update(shared["blendermann"][vessel["type"]])
    wind_settings["air_density"] = shared["environment"]["air_density"]

    f = FMU2Slave(guid=model_description.guid, unzipDirectory=unzipped,
                  modelIdentifier=model_description.coSimulation.modelIdentifier,
                  instanceName=label)
    f.instantiate()
    f.setupExperiment(startTime=3.)
    f.enterInitializationMode()
    f.setString([refs["config_path"]], [str(config_path)])
    if inputs:
        f.setReal([refs[n] for n in inputs], list(inputs.values()))
    f.exitInitializationMode()

    # The same models, called directly.
    lf = LF(**{k: data[k] for k in ("rigid_mass", "added_mass", "damping", "restoring")})
    wind = WindLoads(wind_settings)
    wave = WaveLoads(read_table(data["force_table"], "rao"),
                     read_table(data["drift_table"], "drift"))
    wf = WF(read_table(data["motion_table"], "rao"))
    w = WaveComponents([inputs.get("wave1_amplitude", 0)], [inputs.get("wave1_omega", 0)],
                       [inputs.get("wave1_phase", 0)],
                       np.deg2rad([inputs.get("wave1_direction_deg", 0)]))
    wave.set_components(w)
    wf.set_components(w)
    state = np.zeros(12)
    t = 3.
    tau_thruster = [inputs.get("X_thrusters", 0), inputs.get("Y_thrusters", 0), 0, 0, 0,
                    inputs.get("N_thrusters", 0)]
    current_ned = [inputs.get("current_north", 0), inputs.get("current_east", 0), 0]
    try:
        for k in range(60):
            dt = (.02, .05, .12)[k % 3]
            count = int(np.ceil(dt / .05))
            sub = dt / count
            for j in range(count):
                wind_force = wind.calculate(
                    state[:6], state[6:],
                    mean_wind_speed=inputs.get("mean_wind_speed", 0),
                    wind_direction_deg=inputs.get("wind_direction_deg", 0))
                drift = wave.calculate(t + j*sub, state[:6]).tau_second_order
                state = lf.step(state, sub, tau_wind=wind_force, tau_wave_drift=drift,
                                tau_thruster=tau_thruster, current_ned=current_ned)
            f.doStep(currentCommunicationPoint=t, communicationStepSize=dt)
            t += dt
            motion = wf.displacement(t, state[:6])
            expected = np.r_[state[:6] + motion, state[:6], motion, state[6:]]
            actual = np.array(f.getReal([refs[n] for n in NAMES]))
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        if label == "calm":
            np.testing.assert_array_equal(actual, 0)
        if label == "thrusters":
            assert actual[0] > 0 and actual[1] > 0 and actual[5] > 0
        if label == "wind":
            assert actual[1] < 0
        if label == "waves":
            assert abs(actual[2]) > 0 and np.linalg.norm(state[:2]) > 0
        # Actual FMI reset and fresh calm initialization.
        f.reset()
        f.setupExperiment(startTime=0)
        f.enterInitializationMode()
        f.setString([refs["config_path"]], [str(config_path)])
        f.exitInitializationMode()
        f.doStep(currentCommunicationPoint=0, communicationStepSize=.1)
        np.testing.assert_array_equal(f.getReal([refs[n] for n in NAMES]), 0)
    finally:
        f.terminate()
        f.freeInstance()
