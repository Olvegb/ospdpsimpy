"""ThrusterDynamicsDucted FMU: ThrusterDynamicsKtKq with the curves from a ducted propeller series.

The same FMU as thruster_ktkq/thruster_dynamics_ktkq.py (the same parameter,
inputs and outputs, and the same life cycle), with
helpfunctions.dynamics_ducted.ThrusterDynamicsDucted as the model: thrusters
with rated_rpm in the config get KT and KQ from the Wageningen Ka 4-70
propeller in nozzle 19A, scaled by c_t and c_q. Other thrusters are as in
ThrusterDynamicsKtKq.

To compare, run the simulation once with each as the "dynamics" simulator in
OspSystemStructure.xml (source="fmus/ThrusterDynamics.fmu",
"fmus/ThrusterDynamicsKtKq.fmu" or "fmus/ThrusterDynamicsDucted.fmu"); every
connection stays the same.

License: MIT (see LICENSE in the repository root)
Authors: William Boland, Olve Grønås Birkeland (2026)
"""
from ospfmusim.thruster_ducted.helpfunctions import dynamics_ducted
from ospfmusim.thruster_ktkq.thruster_dynamics_ktkq import ThrusterDynamicsKtKq


class ThrusterDynamicsDucted(ThrusterDynamicsKtKq):

    description = ("Thruster dynamics with the Ka 4-70 / nozzle 19A ducted propeller series: "
                   "thrust lag, azimuth rate limits, thrust loss with speed, and the force "
                   "(X, Y, N) on the vessel")
    model_class = dynamics_ducted.ThrusterDynamicsDucted
