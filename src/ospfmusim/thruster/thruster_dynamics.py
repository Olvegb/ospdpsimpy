"""ThrusterDynamics FMU: the thrusters following their commands, and the force on the vessel.

The FMU has only ONE parameter, config_path, and reads the thrusters itself
from the config file when the simulation starts. The dynamics are
helpfunctions.dynamics.ThrusterDynamics; this file only moves values
between FMI variables and that code.

The thrusters are slots, as in thrust_allocation.py: the k-th [[thrusters]]
entry in config.toml is thruster<k>_... here. Slots without a thruster
output 0.

The FMU's life cycle:
  1. __init__                   Runs when the FMU is loaded (and when it is
                                built). The variables are registered here.
                                The config file is NOT read here.
  2. the master sets values     config_path and the inputs get their values
                                from OspSystemStructure.xml.
  3. exit_initialization_mode   Runs once, right before the simulation
                                starts. The config file is read here.
  4. do_step                    Runs every time step. The thrusters move
                                step_size seconds toward their commands.

License: MIT (see LICENSE in the repository root)
Authors: William Boland, Olve Grønås Birkeland (2026)
"""
import math

from pythonfmu import Boolean, Fmi2Causality, Fmi2Slave, Fmi2Variability, Real, String

from ospfmusim.thruster.helpfunctions import dynamics
from ospfmusim.thruster.helpfunctions.layout import load_layout

# The same slots as in thrust_allocation.py.
MAX_THRUSTERS = 8
SLOTS = range(1, MAX_THRUSTERS + 1)


class ThrusterDynamics(Fmi2Slave):

    author = "William"
    description = ("Thruster dynamics: thrust lag and azimuth rate limits, and the force "
                   "(X, Y, N) on the vessel")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # PARAMETER: path to the config file. The only thing the master must set.
        self.config_path = ""
        self.register_variable(String("config_path", causality=Fmi2Causality.parameter,
                                      variability=Fmi2Variability.fixed))

        # INPUTS: per slot, from the ThrustAllocation FMU, and which thrusters work.
        for k in SLOTS:
            for name in (f"thruster{k}_thrust_command", f"thruster{k}_azimuth_command_deg"):
                setattr(self, name, 0.0)
                self.register_variable(Real(name, causality=Fmi2Causality.input))
            setattr(self, f"thruster{k}_available", True)
            self.register_variable(Boolean(f"thruster{k}_available", causality=Fmi2Causality.input,
                                           variability=Fmi2Variability.discrete))

        # INPUTS: the vessel's LF velocity in the body frame (u, v [m/s],
        # r [rad/s]), its heading (yaw [rad]) and the current (north, east
        # [m/s]), for the speed through the water of thrusters with
        # open_water. Unconnected they are 0: bollard pull.
        for name in ("u", "v", "r", "yaw", "current_north", "current_east"):
            setattr(self, name, 0.0)
            self.register_variable(Real(name, causality=Fmi2Causality.input))

        # OUTPUTS: per slot the actual thrust [N], direction [deg], power
        # [kW], shaft speed [rps] and propeller torque [Nm]; the total force
        # on the vessel in the body frame (X, Y [N], N [Nm], the same names
        # as Olve's WindLoads), and the total power. Power is 0 for a
        # thruster without power_kw or open_water in the config; shaft speed
        # and torque are 0 without open_water.
        for k in SLOTS:
            for name in (f"thruster{k}_thrust", f"thruster{k}_azimuth_deg",
                         f"thruster{k}_power_kw", f"thruster{k}_shaft_speed",
                         f"thruster{k}_torque"):
                setattr(self, name, 0.0)
                self.register_variable(Real(name, causality=Fmi2Causality.output))
        self.X = self.Y = self.N = self.total_power_kw = 0.0
        for name in ("X", "Y", "N", "total_power_kw"):
            self.register_variable(Real(name, causality=Fmi2Causality.output))

        self.model = None  # made in exit_initialization_mode

    def exit_initialization_mode(self):
        # A relative path, such as "config.toml", is read from the folder the
        # simulation is started from.
        if not self.config_path:
            raise ValueError("config_path is not set. Set it in OspSystemStructure.xml.")
        layout = load_layout(self.config_path)
        if len(layout.thrusters) > MAX_THRUSTERS:
            raise ValueError(f"{self.config_path}: {len(layout.thrusters)} thrusters, "
                             f"but the FMU has {MAX_THRUSTERS} slots")
        self.model = dynamics.ThrusterDynamics(layout)
        # The outputs at time 0: a step of 0 s changes nothing.
        self.move(0.0)

    def do_step(self, current_time, step_size):
        self.move(step_size)
        return True

    def move(self, dt):
        slots = range(1, len(self.model.thrusters) + 1)
        # The current in the body frame, subtracted from the vessel's velocity.
        cos_yaw, sin_yaw = math.cos(self.yaw), math.sin(self.yaw)
        u_c = self.current_north * cos_yaw + self.current_east * sin_yaw
        v_c = -self.current_north * sin_yaw + self.current_east * cos_yaw
        state = self.model.step(
            [getattr(self, f"thruster{k}_thrust_command") for k in slots],
            [getattr(self, f"thruster{k}_azimuth_command_deg") for k in slots],
            [getattr(self, f"thruster{k}_available") for k in slots],
            dt,
            (self.u - u_c, self.v - v_c, self.r),
        )
        power = [0.0 if math.isnan(p) else float(p) for p in state.power_kw]
        for k, thrust, angle, p, n, q in zip(slots, state.thrust, state.azimuth_deg, power,
                                             state.shaft_speed, state.torque):
            setattr(self, f"thruster{k}_thrust", float(thrust))
            setattr(self, f"thruster{k}_azimuth_deg", float(angle))
            setattr(self, f"thruster{k}_power_kw", p)
            setattr(self, f"thruster{k}_shaft_speed", float(n))
            setattr(self, f"thruster{k}_torque", float(q))
        self.X, self.Y, self.N = (float(v) for v in state.tau)
        self.total_power_kw = sum(power)
