"""
Thruster dynamics: each thruster follows its command, and the force it gives the vessel.

The commands are those of allocation.ThrustAllocator: a thrust T [N] along
a direction alpha [deg] (layout.py), alpha fixed at 90 deg for tunnels and
0 deg for shaft lines. Every kind then pushes with f = T * (cos alpha, sin alpha).

Thrusters with open-water curves (Thruster.open_water) are speed-controlled
propellers after Sørensen (2018), Marine Cybernetics, ch. 8: see propeller().

License: MIT (see LICENSE in the repository root)
Authors: William Boland, Olve Grønås Birkeland (2026)
"""
import math
from dataclasses import dataclass

import numpy as np

from .power import thruster_power_kw

RHO = 1025.0  # density of sea water [kg/m^3]


@dataclass(frozen=True)
class State:
    """Result of ThrusterDynamics.step(). The arrays follow the order of the thrusters."""

    thrust: np.ndarray  # actual thrust [N] along azimuth_deg; signed for tunnels and shaft lines
    azimuth_deg: np.ndarray  # actual direction of positive thrust [deg]
    fx: np.ndarray  # surge force from each thruster [N]
    fy: np.ndarray  # sway force from each thruster [N], positive to starboard
    tau: np.ndarray  # (X, Y, N) on the vessel [N, N, Nm]
    # power per thruster [kW]: 2 pi n Qa with open_water, else P_B * r^1.5; nan without either
    power_kw: np.ndarray
    shaft_speed: np.ndarray  # n [rps], signed like the thrust; 0 without open_water
    torque: np.ndarray  # propeller torque Qa [Nm], signed like n; 0 without open_water


def propeller(thruster, nominal_thrust, speed):
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
        With open_water.
    nominal_thrust : float
        Tn [N], signed.
    speed : float
        Speed of the thruster through the water along its direction of
        positive thrust [m/s].

    Returns
    -------
    tuple of float
        Ta [N], n [rps], Qa [Nm], all signed like nominal_thrust.
    """
    # ponytail: ideal speed loop (n follows Tn at once, 8.9.7), one curve for
    # both directions (no KT0r/KQ0r, 8.45), np.interp holds the end values
    # outside the table, no ventilation or interaction losses (8.6). Add the
    # shaft model (8.30) or reverse curves when the data is there.
    d = thruster.diameter
    n = math.copysign(math.sqrt(abs(nominal_thrust) / (RHO * d**4 * thruster.kt0)),
                      nominal_thrust)
    if n == 0:
        return 0.0, 0.0, 0.0
    ja = (1 - thruster.wake_fraction) * speed / (n * d)  # Va / (|n| D), Va signed along the thrust
    j, kt, kq = zip(*thruster.open_water)
    n2 = n * abs(n)
    return (RHO * d**4 * float(np.interp(ja, j, kt)) * n2, n,
            RHO * d**5 * float(np.interp(ja, j, kq)) * n2)


class ThrusterDynamics:
    """
    The thrusters between their commands and the vessel, stepped in time.

    Thrust: a first-order lag toward the command, solved exactly over the
    step, so any step size gives the same curve:

        T <- T + (T_cmd - T) * (1 - exp(-dt / tau_T))

    T_cmd is first clipped to [-reverse_thrust, max_thrust] (azimuths
    [0, max_thrust]: they turn instead of reversing). Without
    thrust_time_constant, T = T_cmd.

    Azimuth: turns the short way round toward its command by at most
    max_azimuth_rate * dt (no limit without max_azimuth_rate). It may turn
    through a forbidden zone; zones are about thrust direction, which the
    allocator keeps out of them. While it turns, the thrust command is
    scaled by cos(angle left to turn), and is 0 beyond 90 deg, so it doesn't
    push the wrong way. Tunnels and shaft lines keep 90 and 0 deg and ignore
    the angle command.

    An unavailable thruster (a breaker trip) has no thrust at once and
    doesn't turn.

    Power: P = P_B * r^1.5 (power.thruster_power_kw), with r = |T| over the
    largest thrust in that direction.

    With open_water, T above is the nominal thrust Tn, and the thrust on the
    vessel, shaft speed, torque and power (2 pi n Qa) come from propeller()
    with the speed through the water at the thruster.
    """

    def __init__(self, layout):
        self.thrusters = layout.thrusters
        kinds = [t.kind for t in self.thrusters]
        self._azimuth = np.array([k == "azimuth" for k in kinds])
        self._low = np.array([0.0 if k == "azimuth" else -t.reverse_thrust
                              for k, t in zip(kinds, self.thrusters)])
        self._high = np.array([t.max_thrust for t in self.thrusters])
        self._x = np.array([t.x for t in self.thrusters])
        self._y = np.array([t.y for t in self.thrusters])
        # Start at rest, azimuths pointing ahead (as ThrustAllocator assumes).
        self.thrust = np.zeros(len(kinds))
        self.azimuth_deg = np.array([90.0 if k == "tunnel" else 0.0 for k in kinds])

    def step(self, thrust_cmd, azimuth_cmd_deg, available=None, dt=0.0,
             velocity=(0.0, 0.0, 0.0)):
        """
        Move the thrusters dt seconds toward their commands.

        Parameters
        ----------
        thrust_cmd : sequence of float
            Commanded thrust [N] per thruster, in layout order.
        azimuth_cmd_deg : sequence of float
            Commanded direction [deg] per thruster; only azimuths use it.
        available : sequence of bool, optional
            None: all available.
        dt : float
            Step size [s].
        velocity : sequence of float
            (u_r, v_r, r): the vessel's velocity relative to the water in the
            body frame [m/s, m/s, rad/s]. Only thrusters with open_water use it.

        Returns
        -------
        State
        """
        n = len(self.thrusters)
        thrust_cmd = np.asarray(thrust_cmd, dtype=float)
        azimuth_cmd_deg = np.asarray(azimuth_cmd_deg, dtype=float)
        if available is None:
            available = np.ones(n, dtype=bool)
        else:
            available = np.asarray(available, dtype=bool)
        for name, value in (("thrust_cmd", thrust_cmd), ("azimuth_cmd_deg", azimuth_cmd_deg),
                            ("available", available)):
            if value.shape != (n,):
                raise ValueError(f"{name} needs one value per thruster ({n}), "
                                 f"got shape {value.shape}")
        if dt < 0:
            raise ValueError(f"dt must not be negative, got {dt}")

        # Shortest signed arc to the command, in [-180, 180).
        error = (azimuth_cmd_deg - self.azimuth_deg + 180.0) % 360.0 - 180.0
        reach = np.array([np.inf if t.max_azimuth_rate is None else t.max_azimuth_rate * dt
                          for t in self.thrusters])
        turning = self._azimuth & available
        turned = (self.azimuth_deg + np.clip(error, -reach, reach)) % 360.0
        self.azimuth_deg = np.where(turning, turned, self.azimuth_deg)

        # An azimuth still turning pushes with cos of what is left of the turn,
        # and not at all beyond 90 deg, so it doesn't push the wrong way.
        left = (azimuth_cmd_deg - self.azimuth_deg + 180.0) % 360.0 - 180.0
        left = np.where(self._azimuth, left, 0.0)
        lag = np.array([1.0 if t.thrust_time_constant is None
                        else 1.0 - np.exp(-dt / t.thrust_time_constant)
                        for t in self.thrusters])
        target = (np.clip(thrust_cmd, self._low, self._high)
                  * np.maximum(np.cos(np.deg2rad(left)), 0.0))
        self.thrust = np.where(available, self.thrust + (target - self.thrust) * lag, 0.0)

        alpha = np.deg2rad(self.azimuth_deg)
        u_r, v_r, r = velocity
        # Speed through the water at each thruster, along its positive thrust.
        speed = (u_r - r * self._y) * np.cos(alpha) + (v_r + r * self._x) * np.sin(alpha)
        thrust, shaft_speed, torque = self.thrust.copy(), np.zeros(n), np.zeros(n)
        for i, t in enumerate(self.thrusters):
            if t.open_water:
                thrust[i], shaft_speed[i], torque[i] = propeller(t, self.thrust[i], speed[i])

        fx = thrust * np.cos(alpha)
        fy = thrust * np.sin(alpha)
        tau = np.array([fx.sum(), fy.sum(), (self._x * fy - self._y * fx).sum()])
        fraction = np.abs(self.thrust) / np.where(self.thrust >= 0, self._high, -self._low)
        power_kw = np.array([2 * np.pi * shaft_speed[i] * torque[i] / 1000 if t.open_water
                             else np.nan if t.power_kw is None
                             else float(thruster_power_kw(t, fraction[i]))
                             for i, t in enumerate(self.thrusters)])
        return State(thrust, self.azimuth_deg.copy(), fx, fy, tau, power_kw, shaft_speed, torque)
