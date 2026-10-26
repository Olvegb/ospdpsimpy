"""Execute the real FMU and compare with independent calls to the four modules."""
import json
import re
import tempfile
import tomllib
from pathlib import Path

import numpy as np
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave
from fmpy.validation import validate_fmu

from app.build.data import default_config
from ospfmusim.build import PACKAGE
from ospfmusim.hull.helpfunctions.LF import LF
from ospfmusim.hull.helpfunctions.WF import WF
from ospfmusim.hull.helpfunctions.WindModel import WindLoads
from ospfmusim.hull.helpfunctions.WaveLoads import WaveLoads, WaveComponents
from ospfmusim.hull.vessel import read_table, POSE, VELOCITY

HELP = PACKAGE / 'hull' / 'helpfunctions'


def main():
    file = PACKAGE / 'fmus/VesselBlock.fmu'
    assert not validate_fmu(str(file)), validate_fmu(str(file))
    md = read_model_description(str(file), validate=True)
    refs = {v.name: v.valueReference for v in md.modelVariables}
    cfg = json.loads((HELP / 'example_test_config.json').read_text())
    # The config the simulation uses, with [vessel_fmu] pointing to the synthetic JSON fixture.
    data_file = (HELP / 'example_test_config.json').as_posix()
    text = re.sub(r'^\[vessel_fmu\]\n.*?(?=^\[)',
                  lambda _: f'[vessel_fmu]\ndata_file = "{data_file}"\n\n',
                  default_config().read_text(encoding='utf-8-sig'), count=1, flags=re.M | re.S)
    config = Path(tempfile.mkdtemp()) / 'config.toml'
    config.write_text(text, encoding='utf-8')
    shared = tomllib.loads(text)
    assert shared['vessel_fmu'] == {'data_file': data_file}
    vessel = shared['vessel']
    wind_settings = {k: vessel[k] for k in ('loa', 'area_lateral', 'area_frontal', 's_L', 's_H')}
    wind_settings.update(shared['blendermann'][vessel['type']])
    wind_settings['air_density'] = shared['environment']['air_density']
    checks = ['FMI schema validation passed']
    with tempfile.TemporaryDirectory() as temp:
        directory = extract(str(file), unzipdir=temp)
        packaged = Path(temp) / 'resources' / 'ospfmusim' / 'hull' / 'helpfunctions'
        for name in ('LF.py', 'WF.py', 'WindModel.py', 'WaveLoads.py'):
            assert (packaged / name).read_bytes() == (HELP / name).read_bytes()
        checks.append('Packaged calculation modules match repository source')
        cases = {
            'calm': {},
            'thrusters': {'X_thrusters': 10000, 'Y_thrusters': 2000, 'N_thrusters': 30000},
            'wind': {'mean_wind_speed': 15, 'wind_direction_deg': 90},
            'waves': {'wave1_amplitude': .8, 'wave1_omega': 1., 'wave1_phase': .3,
                      'wave1_direction_deg': 45},
            'combined': {'mean_wind_speed': 10, 'wind_direction_deg': 120, 'X_thrusters': 10000,
                         'wave1_amplitude': .8, 'wave1_omega': 1., 'wave1_phase': .3,
                         'wave1_direction_deg': 45, 'current_north': .1, 'current_east': .03},
        }
        for label, inputs in cases.items():
            f = FMU2Slave(guid=md.guid, unzipDirectory=directory,
                          modelIdentifier=md.coSimulation.modelIdentifier, instanceName=label)
            f.instantiate()
            f.setupExperiment(startTime=3.)
            f.enterInitializationMode()
            f.setString([refs['config_path']], [str(config)])
            if inputs:
                f.setReal([refs[n] for n in inputs], list(inputs.values()))
            f.exitInitializationMode()
            lf = LF(**{k: cfg[k] for k in ('rigid_mass', 'added_mass', 'damping', 'restoring')})
            wind = WindLoads(wind_settings)
            wave = WaveLoads(read_table(cfg['force_table'], 'rao'),
                             read_table(cfg['drift_table'], 'drift'))
            wf = WF(read_table(cfg['motion_table'], 'rao'))
            w = WaveComponents([inputs.get('wave1_amplitude', 0)], [inputs.get('wave1_omega', 0)],
                               [inputs.get('wave1_phase', 0)],
                               np.deg2rad([inputs.get('wave1_direction_deg', 0)]))
            wave.set_components(w)
            wf.set_components(w)
            state = np.zeros(12)
            t = 3.
            max_error = 0.
            names = (list(POSE) + ['lf_' + s for s in POSE] + ['wf_' + s for s in POSE]
                     + ['lf_' + s for s in VELOCITY])
            tau_thruster = [inputs.get('X_thrusters', 0), inputs.get('Y_thrusters', 0), 0, 0, 0,
                            inputs.get('N_thrusters', 0)]
            current_ned = [inputs.get('current_north', 0), inputs.get('current_east', 0), 0]
            try:
                for k in range(60):
                    dt = (.02, .05, .12)[k % 3]
                    count = int(np.ceil(dt / .05))
                    sub = dt / count
                    for j in range(count):
                        wind_force = wind.calculate(
                            state[:6], state[6:],
                            mean_wind_speed=inputs.get('mean_wind_speed', 0),
                            wind_direction_deg=inputs.get('wind_direction_deg', 0))
                        drift = wave.calculate(t + j*sub, state[:6]).tau_second_order
                        state = lf.step(state, sub, tau_wind=wind_force, tau_wave_drift=drift,
                                        tau_thruster=tau_thruster, current_ned=current_ned)
                    f.doStep(currentCommunicationPoint=t, communicationStepSize=dt)
                    t += dt
                    motion = wf.displacement(t, state[:6])
                    expected = np.r_[state[:6] + motion, state[:6], motion, state[6:]]
                    actual = np.array(f.getReal([refs[n] for n in names]))
                    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
                    max_error = max(max_error, float(np.max(abs(actual - expected))))
                if label == 'calm':
                    np.testing.assert_array_equal(actual, 0)
                if label == 'thrusters':
                    assert actual[0] > 0 and actual[1] > 0 and actual[5] > 0
                if label == 'wind':
                    assert actual[1] < 0
                if label == 'waves':
                    assert abs(actual[2]) > 0 and np.linalg.norm(state[:2]) > 0
                # Actual FMI reset and fresh calm initialization.
                f.reset()
                f.setupExperiment(startTime=0)
                f.enterInitializationMode()
                f.setString([refs['config_path']], [str(config)])
                f.exitInitializationMode()
                f.doStep(currentCommunicationPoint=0, communicationStepSize=.1)
                np.testing.assert_array_equal(f.getReal([refs[n] for n in names]), 0)
            finally:
                f.terminate()
                f.freeInstance()
            checks.append(f'{label}: native FMI stepping matched module calls; '
                          f'max error {max_error:.3g}; reset passed')
    report = {'fmu': str(file),
              'configuration': ('SYNTHETIC software integration fixture; '
                                'NOT calibrated Gunnerus data'),
              'checks': checks}
    (Path(__file__).parent / 'TEST_RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
    print('\n'.join(checks))


if __name__ == '__main__':
    main()
