"""ThrustAllocation FMU: the thrust and azimuth command for each thruster.

The FMU has only ONE parameter, config_path, and reads the thrusters itself
from the config file when the simulation starts. The allocation is
helpfunctions.allocation.ThrustAllocator; this file only moves values
between FMI variables and that code.

The thrusters are slots: the k-th [[thrusters]] entry in config.toml is
thruster<k>_... here. The variables therefore don't depend on the config,
so a new layout needs no rebuild. Slots without a thruster output 0.

The FMU's life cycle:
  1. __init__                   Runs when the FMU is loaded (and when it is
                                built). The variables are registered here.
                                The config file is NOT read here.
  2. the master sets values     config_path and the inputs get their values
                                from OspSystemStructure.xml.
  3. exit_initialization_mode   Runs once, right before the simulation
                                starts. The config file is read here.
  4. do_step                    Runs every time step. The allocation runs here.
"""
from pythonfmu import Boolean, Fmi2Causality, Fmi2Slave, Fmi2Variability, Real, String

from ospfmusim.thruster.helpfunctions.allocation import ThrustAllocator
from ospfmusim.thruster.helpfunctions.layout import load_layout

# ponytail: a fixed number of slots, so the FMI variables don't depend on the
# config. For a bigger layout, raise it here and in thruster_dynamics.py and rebuild.
MAX_THRUSTERS = 8
SLOTS = range(1, MAX_THRUSTERS + 1)
# The commanded force comes in as P, I and D terms per degree of freedom (as in dp_controller.py).
DOFS = ("X", "Y", "N")
TERMS = ("P", "I", "D")
TAU_TERMS = tuple(f"tau_{dof}_{term}" for dof in DOFS for term in TERMS)


class ThrustAllocation(Fmi2Slave):

    author = "William"
    description = ("Thrust allocation: thrust and azimuth command per thruster for a commanded "
                   "force tau")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # PARAMETER: path to the config file. The only thing the master must set.
        self.config_path = ""
        self.register_variable(String("config_path", causality=Fmi2Causality.parameter,
                                      variability=Fmi2Variability.fixed))

        # INPUTS: the P, I and D terms of the commanded force in the body frame
        # (from the DP controller), tau_<dof>_<term> [N, N, Nm], and which
        # thrusters can be used. The terms are added up to tau in allocate().
        for name in TAU_TERMS:
            setattr(self, name, 0.0)
            self.register_variable(Real(name, causality=Fmi2Causality.input))
        for k in SLOTS:
            setattr(self, f"thruster{k}_available", True)
            self.register_variable(Boolean(f"thruster{k}_available", causality=Fmi2Causality.input,
                                           variability=Fmi2Variability.discrete))

        # OUTPUTS: per slot, connected to the ThrusterDynamics FMU. Thrust [N]
        # along the direction [deg]: >= 0 for azimuths, signed for tunnels
        # (90 deg) and shaft lines (0 deg).
        for k in SLOTS:
            for name in (f"thruster{k}_thrust_command", f"thruster{k}_azimuth_command_deg"):
                setattr(self, name, 0.0)
                self.register_variable(Real(name, causality=Fmi2Causality.output))
        # tau_X, tau_Y [N], tau_N [Nm]: the commanded force, P + I + D (for logging).
        for dof in DOFS:
            setattr(self, f"tau_{dof}", 0.0)
            self.register_variable(Real(f"tau_{dof}", causality=Fmi2Causality.output))
        # utilisation: what the command needs as a fraction of what the
        # thrusters have; above 1 (inf if out of reach in any amount) when
        # saturated, and the commands are then the closest they can give.
        self.utilisation = 0.0
        self.saturated = False
        self.register_variable(Real("utilisation", causality=Fmi2Causality.output))
        self.register_variable(Boolean("saturated", causality=Fmi2Causality.output,
                                       variability=Fmi2Variability.discrete))

        self.allocator = None  # made in exit_initialization_mode

    def exit_initialization_mode(self):
        # A relative path, such as "config.toml", is read from the folder the
        # simulation is started from.
        if not self.config_path:
            raise ValueError("config_path is not set. Set it in OspSystemStructure.xml.")
        layout = load_layout(self.config_path)
        if len(layout.thrusters) > MAX_THRUSTERS:
            raise ValueError(f"{self.config_path}: {len(layout.thrusters)} thrusters, "
                             f"but the FMU has {MAX_THRUSTERS} slots")
        self.allocator = ThrustAllocator(layout)
        # Calculate the outputs once already now, so they are right from time 0.
        self.allocate()

    def do_step(self, current_time, step_size):
        self.allocate(step_size)
        return True

    def allocate(self, dt=0.0):
        # dt: how far the azimuths may turn, max_azimuth_rate * dt (0 at the start).
        n = len(self.allocator.layout.thrusters)
        available = [getattr(self, f"thruster{k}_available") for k in range(1, n + 1)]
        for dof in DOFS:
            setattr(self, f"tau_{dof}", sum(getattr(self, f"tau_{dof}_{term}") for term in TERMS))
        command = self.allocator.step((self.tau_X, self.tau_Y, self.tau_N), available, dt)
        for k, (thrust, angle) in enumerate(zip(command.thrust, command.azimuth_deg), start=1):
            setattr(self, f"thruster{k}_thrust_command", float(thrust))
            setattr(self, f"thruster{k}_azimuth_command_deg", float(angle))
        self.utilisation = float(command.allocation.utilisation)
        self.saturated = bool(command.saturated)
