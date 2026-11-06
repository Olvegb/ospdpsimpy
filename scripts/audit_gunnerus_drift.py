"""Read-only forensic audit of original coefficients and saved run forces.

Run from the repository: .venv/bin/python scripts/audit_gunnerus_drift.py
Writes only docs/gunnerus_drift_audit_20261103.json; never changes source data.
"""
import hashlib
import json
from pathlib import Path
import sys
import tomllib
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ospfmusim.hull.helpfunctions.Wamit import load_vessel
from ospfmusim.hull.helpfunctions.WaveLoads import WaveLoads, WaveComponents

ROOT = Path(__file__).resolve().parents[1]
MAT = ROOT/'data_examples/gunnerus2.mat'
WAMIT = ROOT.parent/'mcmodels/data/vessels/gunnerus/hydro/wamit'
RAW = WAMIT/'outputs/gunnerus.8'
RUNS = ['20261103_140110', '20261103_132738_check_yaw_waves_1',
        '20261103_133956_check_yaw_wind_1']


def audit():
    d = load_vessel(MAT)
    v = loadmat(MAT, simplify_cells=True)['vessel']
    rows = np.loadtxt(RAW, skiprows=1)
    source = Path(v['wamit']['sourcePrefix'])
    files = [RAW, source.with_suffix('.8'), MAT, source.parent/'gunnerus2.mat']
    result = {'sha256': {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'lf_approximation':d['lf_approximation'], 'damping':d['damping'].tolist(),
              'coefficients':[], 'runs':[]}
    result['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'src/ospfmusim').rglob('*.py') if '__pycache__' not in p.parts}
    for period in (4.8, 8.72, 9., 9.28):
        cache = d['drift_table'].at_frequencies([2*np.pi/period])
        for beta in (0,10,90,170,180,190,270,350):
            direct = rows[np.isclose(rows[:,0],period)&(rows[:,1]==min(beta,360-beta))]
            coeff = d['drift_table'].at_headings(cache, np.deg2rad([beta]))[:,0]
            result['coefficients'].append({'period':period, 'MSS_beta':beta,
                'raw_beta':min(beta,360-beta), 'raw_XYN':direct[:,6].tolist(),
                'MSS_XYN_per_amplitude_squared':coeff.tolist(),
                'force_H5':(6.25*coeff).tolist(),
                'along_propagation_force_H5':float(6.25*coeff[:2]@np.array([np.cos(np.deg2rad(beta)),np.sin(np.deg2rad(beta))]))})
    model = WaveLoads(d['force_table'],d['drift_table'])
    for name in RUNS:
        p = ROOT/'results'/name
        c = tomllib.loads((p/'config.toml').read_text())
        df = pd.read_csv(next(p.glob('vessel_*.csv')))
        error = []
        for _, r in df.iloc[3::100].iterrows():
            model.set_components(WaveComponents([r.wave1_amplitude], [r.wave1_omega],
                [r.wave1_phase], np.deg2rad([r.wave1_direction_deg])))
            eta = np.array([r['lf_'+s] for s in ('north','east','down','roll','pitch','yaw')])
            expected = model.calculate(r.Time,eta).tau_second_order[[0,1,5]]
            error.append(abs(expected-r[['wave_drift_X','wave_drift_Y','wave_drift_N']].to_numpy(float)))
        ns = {'o':'http://opensimulationplatform.com/MSMI/OSPSystemStructure'}
        tree = ET.parse(p/'OspSystemStructure.xml')
        connections = [[x.attrib for x in e] for e in tree.findall('.//o:VariableConnection', ns)]
        wave_connections = [pair for pair in connections if len(pair)==2
                            and pair[0].get('simulator')=='waves' and pair[1].get('simulator')=='vessel']
        assert len(wave_connections)==400
        assert all(a['name']==b['name'] for a,b in wave_connections)
        assert np.max(error,axis=0)[0]<.001 and np.max(error,axis=0)[2]<.1
        result['runs'].append({'run':name, 'wind_enabled':c['wind']['enabled'],
            'waves_enabled':c['waves']['enabled'], 'wave_FROM':c['waves']['direction_deg'],
            'period':c['waves']['tp'], 'height':c['waves']['hs'],
            'current_enabled':c['current']['enabled'], 'allocation_enabled':c['allocation']['enabled'],
            'last':df.iloc[-1][['Time','lf_north','lf_east','lf_yaw','wave_drift_X','wave_drift_Y','wave_drift_N']].to_dict(),
            'sampled_force_max_absolute_error_XYN':np.max(error,axis=0).tolist(),
            'matched_wave_connections':len(wave_connections),
            'maximum_delivered_thrust':float(df[[x+'_thrusters' for x in ('X','Y','Z','K','M','N')]].abs().to_numpy().max()),
            'maximum_ambient_wind_speed':float(df[['mean_wind_speed','slowly_varying_wind_speed','gust_wind_speed']].abs().to_numpy().max())})
    replay = ROOT/'results/forensic_replay_20261103_140110_complete'
    replay_files = list(replay.glob('vessel_*.csv'))
    if replay_files:
        cols = ['Time','lf_north','lf_east','lf_yaw','wave_drift_X','wave_drift_Y','wave_drift_N']
        before = pd.read_csv(next((ROOT/'results'/RUNS[0]).glob('vessel_*.csv')),usecols=cols)
        after = pd.read_csv(replay_files[0],usecols=cols).dropna()
        complete = bool(len(after)==len(before) and after.Time.iloc[-1]==2000.)
        if complete:
            np.testing.assert_array_equal(before.to_numpy(),after.to_numpy())
        result['replay'] = {'directory':str(replay), 'complete':complete,
                            'final_time':float(after.Time.iloc[-1]),
                            'exact_csv_match_for_pose_and_drift':bool(complete)}
    path = ROOT/'docs/gunnerus_drift_audit_20261103.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(path)
    for run in result['runs']:
        print(run['run'], run['last'], 'force error', run['sampled_force_max_absolute_error_XYN'])


if __name__ == '__main__':
    audit()
