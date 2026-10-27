"""ThrusterDynamicsKtKq FMU: ThrusterDynamics with the thrust from KT/KQ propeller curves.

The same FMU as thruster/thruster_dynamics.py (the same parameter, inputs
and outputs, and the same life cycle), with
helpfunctions.dynamics_ktkq.ThrusterDynamicsKtKq as the model: thrusters
with open_water in the config lose thrust with speed through the water,
and their power is 2 pi n Qa. It also outputs the shaft speed and torque.

To compare the two, run the simulation once with each as the "dynamics"
simulator in OspSystemStructure.xml (source="fmus/ThrusterDynamics.fmu" or
"fmus/ThrusterDynamicsKtKq.fmu"); every connection stays the same.

License: MIT (see LICENSE in the repository root)
Authors: William Boland, Olve Grønås Birkeland (2026)
"""
import math

from pythonfmu import Fmi2Causality, Real

from ospfmusim.thruster.thruster_dynamics import SLOTS, ThrusterDynamics
from ospfmusim.thruster_ktkq.helpfunctions import dynamics_ktkq


class ThrusterDynamicsKtKq(ThrusterDynamics):

    description = ("Thruster dynamics with KT/KQ open-water propeller curves: thrust lag, "
                   "azimuth rate limits, thrust loss with speed, and the force (X, Y, N) "
                   "on the vessel")
    model_class = dynamics_ktkq.ThrusterDynamicsKtKq

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # OUTPUTS: per slot the shaft speed [rps] and propeller torque [Nm];
        # 0 for a thruster without open_water. Power is 2 pi n Qa with
        # open_water, else as in ThrusterDynamics.
        for k in SLOTS:
            for name in (f"thruster{k}_shaft_speed", f"thruster{k}_torque"):
                setattr(self, name, 0.0)
                self.register_variable(Real(name, causality=Fmi2Causality.output))

    def move(self, dt):
        # The current in the body frame, subtracted from the vessel's velocity.
        cos_yaw, sin_yaw = math.cos(self.yaw), math.sin(self.yaw)
        u_c = self.current_north * cos_yaw + self.current_east * sin_yaw
        v_c = -self.current_north * sin_yaw + self.current_east * cos_yaw
        self.model.velocity = (self.u - u_c, self.v - v_c, self.r)
        state = super().move(dt)
        for k, n, q in zip(SLOTS, state.shaft_speed, state.torque):
            setattr(self, f"thruster{k}_shaft_speed", float(n))
            setattr(self, f"thruster{k}_torque", float(q))
        return state
