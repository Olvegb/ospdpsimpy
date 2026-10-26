"""Vessel FMU: coordinates the editable For_GitHub LF/WF/wind/wave modules.

License: MIT (see LICENSE in the repository root)
Authors: Araz, Olve Grønås Birkeland (2026)
"""
import json
import tomllib
from pathlib import Path

import numpy as np
from pythonfmu import Fmi2Slave, Fmi2Causality, Fmi2Variability, Real, String

from ospfmusim.hull.helpfunctions.LF import LF
from ospfmusim.hull.helpfunctions.WF import WF
from ospfmusim.hull.helpfunctions.WindModel import WindLoads
from ospfmusim.hull.helpfunctions.WaveLoads import (WaveComponents, WaveLoads, ResponseTable,
                                                    selected_drift_method)
from ospfmusim.hull.helpfunctions.Wamit import (load_mat_vessel, FREQUENCY_COUNT, HEADING_STEP_DEG,
                                                FREQUENCY_SPACING)

POSE = ('north', 'east', 'down', 'roll', 'pitch', 'yaw')
VELOCITY = ('u', 'v', 'w', 'p', 'q', 'r')
LOADS = ('X', 'Y', 'Z', 'K', 'M', 'N')
N_WAVES = 100


def read_table(data, kind):
    # MATLAB inputs arrive as in-memory ResponseTables, not JSON dictionaries.
    if isinstance(data, ResponseTable):
        if data.kind != kind:
            raise ValueError(f'Expected {kind}, got {data.kind}')
        return data
    values = np.asarray(data['values_real'], dtype=float)
    if kind == 'rao':
        values = values + 1j*np.asarray(data['values_imag'], dtype=float)
    return ResponseTable(data['frequencies'], data['headings'], values, kind=kind)


class VesselBlock(Fmi2Slave):
    description = 'LF + WF vessel, internal wind/wave loads and delivered thruster inputs'
    version = '1.0.0'
    license = 'GPL-3.0-or-later; embedded Viljar wave implementation MIT; see resources'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config_path = ''
        self.max_internal_step_s = 0.05
        self.time = 0.0
        self.ready = False
        self.register_variable(String(
            'config_path', causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed,
            description='Path to shared config.toml. '
                        '[vessel_fmu] refers to hydrodynamic data file.'))
        self.register_variable(Real('max_internal_step_s', causality=Fmi2Causality.parameter,
                                    variability=Fmi2Variability.fixed))
        inputs = ['mean_wind_speed', 'slowly_varying_wind_speed', 'gust_wind_speed',
                  'wind_direction_deg', 'X_thrusters', 'Y_thrusters', 'Z_thrusters',
                  'K_thrusters', 'M_thrusters', 'N_thrusters',
                  'current_north', 'current_east', 'current_down']
        for name in inputs:
            self.add_real(name, Fmi2Causality.input)
        for k in range(1, N_WAVES+1):
            for quantity in ('amplitude', 'omega', 'phase', 'direction_deg'):
                self.add_real(f'wave{k}_{quantity}', Fmi2Causality.input)
        for name in POSE+VELOCITY:
            self.add_real('initial_'+name, Fmi2Causality.parameter)
        for name in POSE:
            for prefix in ('', 'lf_', 'wf_'):
                self.add_real(prefix+name, Fmi2Causality.output)
        for name in VELOCITY:
            self.add_real('lf_'+name, Fmi2Causality.output)
        for prefix in ('wind_', 'wave_first_', 'wave_drift_'):
            for name in LOADS:
                self.add_real(prefix+name, Fmi2Causality.output)

    def add_real(self, name, causality):
        setattr(self, name, 0.0)
        if causality == Fmi2Causality.parameter:
            variability = Fmi2Variability.fixed
        else:
            variability = Fmi2Variability.continuous
        self.register_variable(Real(name, causality=causality, variability=variability))

    def to_xml(self, model_options=None):
        from xml.etree.ElementTree import SubElement
        root = super().to_xml(model_options or {})
        unknowns = SubElement(root.find('ModelStructure'), 'InitialUnknowns')
        for index, variable in enumerate(self.vars.values(), 1):
            if variable.causality == Fmi2Causality.output:
                SubElement(unknowns, 'Unknown', index=str(index))
        return root

    def setup_experiment(self, start_time, stop_time=None, tolerance=None):
        if not np.isfinite(start_time):
            raise ValueError('Start time must be finite')
        self.time = float(start_time)

    def exit_initialization_mode(self):
        if not self.config_path:
            raise ValueError('Set config_path to the shared config.toml before initialization')
        path = Path(self.config_path).resolve()
        if not np.isfinite(self.max_internal_step_s) or self.max_internal_step_s <= 0:
            raise ValueError('max_internal_step_s must be positive and finite')
        with path.open('rb') as stream:
            shared = tomllib.load(stream)
        settings = shared['vessel_fmu']
        if 'mat_file' in settings:
            # Load processed MSS/WAMIT hydrodynamics directly in memory.
            # No raw WAMIT parser and no JSON conversion/preparation step.
            data_path = Path(settings['mat_file'])
            if not data_path.is_absolute():
                data_path = path.parent / data_path
            damping_matrix = settings.get('lf_additional_damping_matrix_si')
            diagonal = settings.get('lf_viscous_damping_diagonal_si')
            if damping_matrix is not None and diagonal is not None:
                raise ValueError('Specify either LF damping matrix or LF damping diagonal')
            if diagonal is not None:
                diagonal = np.asarray(diagonal, dtype=float)
                if diagonal.shape != (6,):
                    raise ValueError('lf_viscous_damping_diagonal_si needs 6 values')
                damping_matrix = np.diag(diagonal)
            cfg = load_mat_vessel(
                data_path,
                frequency_count=int(settings.get('frequency_count', FREQUENCY_COUNT)),
                heading_step_deg=int(settings.get('heading_step_deg', HEADING_STEP_DEG)),
                frequency_spacing=settings.get('frequency_spacing', FREQUENCY_SPACING),
                lf_reference_omega_rad_s=settings.get('lf_reference_omega_rad_s'),
                additional_linear_damping_matrix_si=damping_matrix,
                use_mat_viscous_damping=bool(settings.get('use_mat_viscous_damping', False)),
                drift_method=selected_drift_method(),
                require_simulation_ready=True,
            )
        else:
            # Existing standalone synthetic fixture remains for unit tests;
            # this is NOT real tanker/Gunnerus hydrodynamic data.
            data_path = Path(settings['data_file'])
            if not data_path.is_absolute():
                data_path = path.parent / data_path
            cfg = json.loads(data_path.read_text())
        if cfg.get('schema_version') != 1:
            raise ValueError('Expected hydrodynamic data schema_version 1')
        lf_keys = ('rigid_mass', 'added_mass', 'damping', 'restoring')
        self.lf_model = LF(**{key: cfg[key] for key in lf_keys})
        # Reuse the group's shared vessel geometry and Blendermann coefficients.
        vessel = shared['vessel']
        wind_keys = ('loa', 'area_lateral', 'area_frontal', 's_L', 's_H')
        wind = {key: vessel[key] for key in wind_keys}
        wind.update(shared['blendermann'][vessel['type']])
        wind['air_density'] = shared['environment']['air_density']
        self.wind_model = WindLoads(wind)
        gravity = cfg['gravity']
        # Use the selected equation in WaveLoads.py for all vessel data sources.
        drift_settings = dict(drift_method=selected_drift_method(),
                              remove_sum_frequency=True)
        if 'wave_force_file' in cfg:
            force_path = Path(cfg['wave_force_file'])
            if not force_path.is_absolute():
                force_path = data_path.parent/force_path
            self.wave_model = WaveLoads.from_shipx(
                force_path, phase_unit='deg', amplitude_mode=cfg['force_amplitude_mode'],
                gravity=gravity, **drift_settings)
        else:
            self.wave_model = WaveLoads(
                read_table(cfg['force_table'], 'rao'), read_table(cfg['drift_table'], 'drift'),
                gravity=gravity, **drift_settings)
        # Explicit full-circle motion table: never guess the Gunnerus 19-column mapping.
        self.wf_model = WF(read_table(cfg['motion_table'], 'rao'), gravity=gravity)
        self.state = np.array([getattr(self, 'initial_'+s) for s in POSE+VELOCITY])
        if not np.isfinite(self.state).all():
            raise ValueError('Initial state must be finite')
        self.ready = True
        self.accept_waves()
        self.publish()

    def accept_waves(self):
        arrays = [np.array([getattr(self, f'wave{k}_{q}') for k in range(1, N_WAVES+1)])
                  for q in ('amplitude', 'omega', 'phase', 'direction_deg')]
        waves = WaveComponents(*arrays[:3], np.deg2rad(arrays[3]))
        self.wave_model.set_components(waves)
        self.wf_model.set_components(waves)

    def wind_load(self):
        return self.wind_model.calculate(
            self.state[:6], self.state[6:],
            mean_wind_speed=self.mean_wind_speed,
            sw_wind_speed=self.slowly_varying_wind_speed,
            gust_wind_speed=self.gust_wind_speed, wind_direction_deg=self.wind_direction_deg)

    def publish(self):
        wf = self.wf_model.displacement(self.time, self.state[:6])
        for labels, values in ((POSE, self.state[:6] + wf),
                               (tuple('lf_' + s for s in POSE), self.state[:6]),
                               (tuple('wf_' + s for s in POSE), wf),
                               (tuple('lf_' + s for s in VELOCITY), self.state[6:])):
            for name, value in zip(labels, values):
                setattr(self, name, float(value))
        wave = self.wave_model.calculate(self.time, self.state[:6])
        for prefix, values in (('wind_', self.wind_load()), ('wave_first_', wave.tau_first_order),
                               ('wave_drift_', wave.tau_second_order)):
            for label, value in zip(LOADS, values):
                setattr(self, prefix+label, float(value))

    def do_step(self, current_time, step_size):
        if not self.ready:
            raise RuntimeError('Initialize the vessel first')
        if not np.isfinite([current_time, step_size]).all() or step_size <= 0:
            raise ValueError('Finite time and positive step size required')
        if not np.isclose(current_time, self.time, atol=1e-8, rtol=0):
            raise ValueError('Time mismatch; rollback is unsupported')
        self.accept_waves()
        thrust = np.array([self.X_thrusters, self.Y_thrusters, self.Z_thrusters,
                           self.K_thrusters, self.M_thrusters, self.N_thrusters])
        current = np.array([self.current_north, self.current_east, self.current_down])
        count = int(np.ceil(step_size/self.max_internal_step_s))
        dt = step_size/count
        for j in range(count):
            wave = self.wave_model.calculate(current_time+j*dt, self.state[:6])
            self.state = self.lf_model.step(self.state, dt, tau_wind=self.wind_load(),
                                            tau_wave_drift=wave.tau_second_order,
                                            tau_thruster=thrust, current_ned=current)
        self.time = current_time+step_size
        self.publish()
        return True
