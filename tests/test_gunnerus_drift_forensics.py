"""Real-data regression: transport correctness, NOT hydrodynamic validation.

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_gunnerus_drift_forensics.py
The original MAT and WAMIT files are read only. Set GUNNERUS_WAMIT8 to
the original .8 path if the sibling mcmodels checkout is elsewhere.
"""
import copy
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from ospfmusim.hull.helpfunctions.Wamit import load_vessel
from ospfmusim.hull.helpfunctions.WaveLoads import WaveComponents, WaveLoads
from ospfmusim.hull.helpfunctions._wamit_motion import read_response
from ospfmusim.hull.helpfunctions.LF import LF, rotation
from ospfmusim.waves.helpfunctions.sea_state import regular_wave, directions_to

ROOT = Path(__file__).resolve().parents[1]
MAT = ROOT / 'data_examples/gunnerus2.mat'
RAW = Path(os.environ.get('GUNNERUS_WAMIT8', str(
    ROOT.parent / 'mcmodels/data/vessels/gunnerus/hydro/wamit/outputs/gunnerus.8')))


@pytest.fixture(scope='module')
def data():
    assert MAT.is_file() and RAW.is_file(), 'Real Gunnerus source files are required'
    return load_vessel(MAT), loadmat(MAT, simplify_cells=True)['vessel'], np.loadtxt(RAW, skiprows=1)


def raw_coefficient(rows, period, beta):
    result = []
    for dof in (1, 2, 6):
        selected = rows[np.isclose(rows[:, 0], period, rtol=0, atol=1e-8)
                        & (rows[:, 1] == beta) & (rows[:, 2] == beta)
                        & (rows[:, 3] == dof)]
        assert len(selected) == 1
        result.append(selected[0, 6])  # Signed REAL column, never modulus.
    return np.array(result)


def model_at(d, direction, period=9., method='newman', amplitude=2.5):
    model = WaveLoads(d['force_table'], d['drift_table'], drift_method=method)
    model.set_components(WaveComponents([amplitude], [2*np.pi/period], [.37],
                                       np.deg2rad([direction])))
    return model


def test_entire_original_grid_and_mirror(data):
    d, v, rows = data
    scale = v['main']['rho'] * v['main']['g'] * np.array([1., 1., 1.])
    scale *= np.array([v['wamit']['ULEN'], v['wamit']['ULEN'], v['wamit']['ULEN']**2])
    # FPU->FSD axis signs and beta reflection cancel for X,Y,N on 0..180.
    for i, omega in enumerate(d['frequency_grid']):
        for j, beta in enumerate(range(0, 181, 10)):
            expected = raw_coefficient(rows, 2*np.pi/omega, beta)*scale
            np.testing.assert_allclose(d['drift_table'].values[:, i, j], expected, atol=1e-9)
    a = d['drift_table'].values
    np.testing.assert_allclose(a[:, :, 19:], a[:, :, 17:0:-1]*np.array([1,-1,-1])[:,None,None])
    # Exercise the legacy 19-column expansion separately from the full MAT.
    short = copy.deepcopy(v)
    short['driftfrc']['amp'] = [np.asarray(x)[:, :19] for x in v['driftfrc']['amp']]
    reconstructed = read_response(short, 'driftfrc', [1,-1,-1], np.arange(0,360,10))
    np.testing.assert_allclose(reconstructed.values, a)


@pytest.mark.parametrize('source', [0, 10, 90, 170, 180, 270, 350])
@pytest.mark.parametrize('period', [4.8, 9.])
def test_from_to_signed_regular_wave_and_lf(data, source, period):
    d, v, rows = data
    sea = regular_wave(5., period, 100)
    direction = directions_to(sea, source, True)[0]
    np.testing.assert_allclose(direction, np.deg2rad((source+180)%360), atol=1e-14)
    beta = (source+180)%360
    mirror = beta > 180
    expected = raw_coefficient(rows, period, 360-beta if mirror else beta)
    expected *= np.array([1,-1,-1]) if mirror else 1
    expected *= v['main']['rho']*v['main']['g']*2.5**2  # ULEN=1 in this fixture
    for method in ('newman', 'newman_absolute'):
        model = model_at(d, beta, period, method)
        for time in (0., 1.37, 100.):
            force = model.calculate(time, np.zeros(6)).tau_second_order
            np.testing.assert_allclose(force[[0,1,5]], expected, atol=1e-8)
        lf = LF(**{k:d[k] for k in ('rigid_mass','added_mass','damping','restoring')})
        deriv = lf.derivative(np.zeros(12), force)
        np.testing.assert_allclose(lf.M @ deriv[6:], force, atol=1e-9)
        assert np.sign(deriv[6]) == np.sign(expected[0])


@pytest.mark.parametrize('yaw', [-179., -10., 0., 10., 179.])
def test_relative_heading_and_ned_rotation(data, yaw):
    d, _, _ = data
    pose = np.zeros(6); pose[5] = np.deg2rad(yaw)
    base = model_at(d, 170).calculate(0, np.zeros(6)).tau_second_order
    rotated = model_at(d, 170+yaw).calculate(0, pose).tau_second_order
    np.testing.assert_allclose(rotated, base, atol=1e-8)
    xy = rotation(pose[3:]) @ rotated[:3]
    expected = np.array([[np.cos(pose[5]),-np.sin(pose[5])],
                         [np.sin(pose[5]), np.cos(pose[5])]]) @ base[:2]
    np.testing.assert_allclose(xy[:2], expected, atol=1e-9)


def test_source_sign_reversal_is_period_dependent(data):
    _, _, rows = data
    assert raw_coefficient(rows, 9., 170)[0] == pytest.approx(.0004455706)
    assert raw_coefficient(rows, 9., 180)[0] > 0
    assert raw_coefficient(rows, 4.8, 170)[0] < 0
    assert raw_coefficient(rows, 4.8, 180)[0] < 0


@pytest.mark.parametrize('beta', [0., 10., 170., 180., 190., 350., 360.])
def test_no_discontinuity_at_heading_nodes(data, beta):
    d, _, _ = data
    center = model_at(d, beta).calculate(0.,np.zeros(6)).tau_second_order
    for offset in (-1e-7, 1e-7):
        adjacent = model_at(d, beta+offset).calculate(0.,np.zeros(6)).tau_second_order
        np.testing.assert_allclose(adjacent,center,atol=.001,rtol=0)


@pytest.mark.parametrize('source', [0., 10., 90., 180., 270., 350.])
def test_built_vessel_fmu_matches_raw_force(data, tmp_path, source):
    import fmpy
    from fmpy.fmi2 import FMU2Slave
    d, _, _ = data
    path = ROOT/'src/ospfmusim/fmus/VesselBlock.fmu'
    md = fmpy.read_model_description(str(path))
    folder = fmpy.extract(str(path), unzipdir=str(tmp_path/'fmu'))
    refs = {x.name:x.valueReference for x in md.modelVariables}
    fmu = FMU2Slave(guid=md.guid, unzipDirectory=folder,
                   modelIdentifier=md.coSimulation.modelIdentifier, instanceName='drift_audit')
    fmu.instantiate()
    try:
        fmu.setupExperiment(startTime=0.)
        fmu.enterInitializationMode()
        fmu.setString([refs['config_path']], [str(ROOT/'results/20261103_140110/config.toml')])
        inputs = {'wave1_amplitude':2.5, 'wave1_omega':2*np.pi/9,
                  'wave1_phase':.37, 'wave1_direction_deg':(source+180)%360}
        fmu.setReal([refs[k] for k in inputs], list(inputs.values()))
        fmu.exitInitializationMode()
        actual = fmu.getReal([refs['wave_drift_'+k] for k in ('X','Y','N')])
        expected = model_at(d, (source+180)%360).calculate(0.,np.zeros(6)).tau_second_order[[0,1,5]]
        np.testing.assert_allclose(actual, expected, atol=1e-8)
        fmu.doStep(currentCommunicationPoint=0., communicationStepSize=.1)
        north = fmu.getReal([refs['lf_north']])[0]
        assert np.sign(north) == np.sign(expected[0])
        fmu.terminate()
    finally:
        fmu.freeInstance()
