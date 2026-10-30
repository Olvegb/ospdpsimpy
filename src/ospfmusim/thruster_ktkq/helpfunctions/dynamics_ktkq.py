"""
Thruster dynamics with open-water propeller curves (KT, KQ), to compare with
thruster.helpfunctions.dynamics, where the thrust is the lagged command.

ThrusterDynamicsKtKq is dynamics.ThrusterDynamics (thrust lag, azimuth rate
limits, trips) with one change: for a thruster with open_water in the
config, the lagged thrust is the nominal thrust Tn of a speed-controlled
propeller after Sørensen (2018), Marine Cybernetics, ch. 8, and the thrust
on the vessel, shaft speed, torque and power come from propeller().
Thrusters without open_water are as in dynamics.ThrusterDynamics.

License: MIT (see LICENSE in the repository root)
Authors: William Boland, Olve Grønås Birkeland (2026)
"""
import math
from dataclasses import dataclass

import numpy as np

from ospfmusim.thruster.helpfunctions import dynamics

RHO = 1025.0  # density of sea water [kg/m^3]


@dataclass(frozen=True)
class State(dynamics.State):
    """dynamics.State, with power_kw = 2 pi n Qa for thrusters with open_water."""

    shaft_speed: np.ndarray  # n [rps], signed like the thrust; 0 without open_water
    torque: np.ndarray  # propeller torque Qa [Nm], signed like n; 0 without open_water


def propeller(thruster, nominal_thrust, speed, kt, kq):
    """
    Actual thrust, shaft speed and torque of a speed-controlled propeller
    (Sørensen 2018, ch. 8).

    The shaft speed n is the one that gives nominal_thrust at bollard pull,
    Tn = rho D^4 KT0 n|n| (8.11a, 8.38): the thrust allocation's view. The
    propeller advances at Va = (1 - w) * speed, signed along the thrust, so
    J = Va / (|n| D) (8.6, 8.12), and

        Ta = rho D^4 KT(J) n|n|,   Qa = rho D^5 KQ(J) n|n|   (8.3, 8.4)

    Speed through the water lowers KT, so Ta < Tn: a thrust loss the DP
    controller sees as a disturbance.

    Parameters
    ----------
    thruster : layout.Thruster
        With diameter.
    nominal_thrust : float
        Tn [N], signed.
    speed : float
        Speed of the thruster through the water along its direction of
        positive thrust [m/s].
    kt, kq : callable
        KT(J) and KQ(J), from ThrusterDynamicsKtKq.curves().

    Returns
    -------
    tuple of float
        Ta [N], n [rps], Qa [Nm], all signed like nominal_thrust.
    """
    # ponytail: ideal speed loop (n follows Tn at once, 8.9.7), one curve for
    # both directions (no KT0r/KQ0r, 8.45), no ventilation or interaction
    # losses (8.6). Add the shaft model (8.30) or reverse curves when the
    # data is there.
    d = thruster.diameter
    n = math.copysign(math.sqrt(abs(nominal_thrust) / (RHO * d**4 * kt(0.0))), nominal_thrust)
    if n == 0:
        return 0.0, 0.0, 0.0
    ja = (1 - thruster.wake_fraction) * speed / (n * d)  # Va / (|n| D), Va signed along the thrust
    n2 = n * abs(n)
    return RHO * d**4 * kt(ja) * n2, n, RHO * d**5 * kq(ja) * n2


class ThrusterDynamicsKtKq(dynamics.ThrusterDynamics):
    """
    dynamics.ThrusterDynamics, with the thrust of thrusters with open_water
    from propeller().

    The lag, rate limits and trips act on the nominal thrust Tn as before.
    propeller() then gives the thrust on the vessel Ta, shaft speed, torque
    and power (2 pi n Qa), with the speed through the water at the thruster
    from velocity.
    """

    def __init__(self, layout):
        super().__init__(layout)
        # (u_r, v_r, r): the vessel's velocity relative to the water in the
        # body frame [m/s, m/s, rad/s]. Set it before step(). At rest: bollard pull.
        self.velocity = (0.0, 0.0, 0.0)
        self._curves = [self.curves(t) for t in self.thrusters]

    def curves(self, thruster):
        """
        (KT(J), KQ(J)) of a thruster, linear in open_water with the end values
        held outside the table, or None without open_water: its thrust stays
        nominal. Subclasses take their curves from elsewhere.
        """
        if not thruster.open_water:
            return None
        j, kt, kq = zip(*thruster.open_water)
        return (lambda x: float(np.interp(x, j, kt)), lambda x: float(np.interp(x, j, kq)))

    def step(self, thrust_cmd, azimuth_cmd_deg, available=None, dt=0.0):
        """dynamics.ThrusterDynamics.step(), with velocity as it is now. Returns State."""
        nominal = super().step(thrust_cmd, azimuth_cmd_deg, available, dt)
        alpha = np.deg2rad(nominal.azimuth_deg)
        u_r, v_r, r = self.velocity
        # Speed through the water at each thruster, along its positive thrust.
        speed = (u_r - r * self._y) * np.cos(alpha) + (v_r + r * self._x) * np.sin(alpha)
        thrust, power_kw = nominal.thrust.copy(), nominal.power_kw.copy()
        shaft_speed, torque = np.zeros(len(thrust)), np.zeros(len(thrust))
        for i, (t, curves) in enumerate(zip(self.thrusters, self._curves)):
            if curves:
                thrust[i], shaft_speed[i], torque[i] = propeller(t, nominal.thrust[i], speed[i],
                                                                 *curves)
                power_kw[i] = 2 * np.pi * shaft_speed[i] * torque[i] / 1000

        fx = thrust * np.cos(alpha)
        fy = thrust * np.sin(alpha)
        tau = np.array([fx.sum(), fy.sum(), (self._x * fy - self._y * fx).sum()])
        return State(thrust, nominal.azimuth_deg, fx, fy, tau, power_kw, shaft_speed, torque)
