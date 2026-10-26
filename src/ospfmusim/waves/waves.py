"""Waves FMU: describes the sea as a set of regular wave components.

The FMU has only ONE parameter, config_path, and reads the wave settings
itself from [waves] in the config file when the simulation starts. The physics
is in helpfunctions/; this file only moves values between FMI variables and
that code.

The FMU only describes the sea. It does NOT calculate any forces on the
vessel: the hull FMU does that from the wave components sent out here.

  - inputs:   H_s, T_p, wave_direction_deg, from the Environment FMU
  - outputs:  N_WAVES wave components: wave<k>_amplitude, wave<k>_omega,
              wave<k>_phase, wave<k>_direction_deg, k = 1 ... N_WAVES,
              and kept_energy_fraction
  - settings: [waves] in config.toml (sea type, spectrum, spreading, grid)

The hull FMU must use exactly this wave elevation (t = simulation time,
x = north, y = east, direction in radians):

    zeta = sum_k amplitude_k * cos(omega_k * t
                                   - k_k * (x cos(direction_k) + y sin(direction_k)) - phase_k)
    k_k = omega_k^2 / g    (deep water)

See docs/waves_fmu.md.
"""
import numpy as np
from pythonfmu import Fmi2Causality, Fmi2Slave, Fmi2Variability, Real, String

from ospfmusim.waves.helpfunctions.sea_state import directions_to, sea_from_settings
from ospfmusim.waves.helpfunctions.settings import load_settings

# A fixed number of wave components, so the FMI variables don't depend on the
# config (the same idea as MAX_THRUSTERS in thrust_allocation.py). The
# N_WAVES components with the most energy are sent out (Sørensen 6.2.1,
# method 1). To change it: change it here, rebuild, and redo the connections
# to the hull FMU in OspSystemStructure.xml (4 per component).
N_WAVES = 100
SLOTS = range(1, N_WAVES + 1)

# Four outputs per wave component: wave1_amplitude, wave1_omega, ...
# direction_deg is the direction the wave travels TO, NED, clockwise from north, 0-360.
COMPONENT_OUTPUTS = ("amplitude",       # wave amplitude [m] (half the wave height)
                     "omega",           # frequency [rad/s]
                     "phase",           # phase [rad]
                     "direction_deg")   # direction the wave travels TO [deg]


class Waves(Fmi2Slave):

    author = "Viljar Simonnes Tveit"
    description = ("Regular or irregular (long- or short-crested) sea as a set of regular wave "
                   "components, from H_s, T_p and the wave direction")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # PARAMETER: path to the config file. The only thing the master must set.
        self.config_path = ""
        self.register_variable(String("config_path", causality=Fmi2Causality.parameter,
                                      variability=Fmi2Variability.fixed))

        # INPUTS: the sea state, from the Environment FMU.
        # For sea_type = "regular", H_s is used as the wave height H and T_p as the period T.
        self.H_s = 0.0                  # significant wave height [m]
        self.T_p = 0.0                  # peak wave period [s]
        self.wave_direction_deg = 0.0   # where the waves come FROM, clockwise from north [deg]
        for name in ("H_s", "T_p", "wave_direction_deg"):
            self.register_variable(Real(name, causality=Fmi2Causality.input))

        # OUTPUTS: the wave components, strongest first. For a regular sea,
        # wave1 is the wave and all the others have amplitude 0.
        for k in SLOTS:
            for quantity in COMPONENT_OUTPUTS:
                name = f"wave{k}_{quantity}"
                setattr(self, name, 0.0)
                self.register_variable(Real(name, causality=Fmi2Causality.output))
        # Share of the total wave energy in the components sent out (1 = all, for logging).
        self.kept_energy_fraction = 1.0
        self.register_variable(Real("kept_energy_fraction", causality=Fmi2Causality.output))

        # CONSTANTS: read from [waves] in the config file in exit_initialization_mode.
        self.settings = None
        self.sea = None              # the current wave components
        self._built_for = None       # (H_s, T_p) the sea was built for

    def exit_initialization_mode(self):
        # Runs once, right before the simulation starts: read the config file.
        # A relative path, such as "config.toml", is read from the folder the
        # simulation is started from.
        if not self.config_path:
            raise ValueError("config_path is not set. Set it in OspSystemStructure.xml.")
        self.settings = load_settings(self.config_path)
        # Calculate the outputs once already now, so they are right from time 0.
        self.do_step(0.0, 0.0)

    def do_step(self, current_time, step_size):
        # The wave components do not depend on time (the hull FMU uses t), so the
        # sea is only built again when H_s or T_p changes, with the same seed.
        # A new wave direction only turns all the components.
        if (self.H_s, self.T_p) != self._built_for:
            self.sea, self.kept_energy_fraction = sea_from_settings(self.H_s, self.T_p,
                                                                    self.settings, N_WAVES)
            self._built_for = (self.H_s, self.T_p)

        directions_deg = np.degrees(directions_to(self.sea, self.wave_direction_deg,
                                                  direction_is_from=True))
        for k in SLOTS:
            i = k - 1                                  # the arrays start at 0
            setattr(self, f"wave{k}_amplitude", float(self.sea.amplitudes[i]))
            setattr(self, f"wave{k}_omega", float(self.sea.frequencies[i]))
            setattr(self, f"wave{k}_phase", float(self.sea.phases[i]))
            setattr(self, f"wave{k}_direction_deg", float(directions_deg[i]))
        return True
