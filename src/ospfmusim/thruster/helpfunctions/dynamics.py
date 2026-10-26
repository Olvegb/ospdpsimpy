"""
Thruster dynamics: each thruster follows its command, and the force it gives the vessel.

The commands are those of allocation.ThrustAllocator: a thrust T [N] along
a direction alpha [deg] (layout.py), alpha fixed at 90 deg for tunnels and
0 deg for shaft lines. Every kind then pushes with f = T * (cos alpha, sin alpha).
"""
from dataclasses import dataclass

import numpy as np

from .power import thruster_power_kw


@dataclass(frozen=True)
class State:
    """Result of ThrusterDynamics.step(). The arrays follow the order of the thrusters."""

    thrust: np.ndarray  # actual thrust [N] along azimuth_deg; signed for tunnels and shaft lines
    azimuth_deg: np.ndarray  # actual direction of positive thrust [deg]
    fx: np.ndarray  # surge force from each thruster [N]
    fy: np.ndarray  # sway force from each thruster [N], positive to starboard
    tau: np.ndarray  # (X, Y, N) on the vessel [N, N, Nm]
    power_kw: np.ndarray  # power per thruster [kW], P_B * r^1.5; nan without power_kw


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
    allocator keeps out of them. Tunnels and shaft lines keep 90 and 0 deg
    and ignore the angle command.

    An unavailable thruster (a breaker trip) has no thrust at once and
    doesn't turn.

    Power: P = P_B * r^1.5 (power.thruster_power_kw), with r = |T| over the
    largest thrust in that direction.
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

    def step(self, thrust_cmd, azimuth_cmd_deg, available=None, dt=0.0):
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

        lag = np.array([1.0 if t.thrust_time_constant is None
                        else 1.0 - np.exp(-dt / t.thrust_time_constant)
                        for t in self.thrusters])
        target = np.clip(thrust_cmd, self._low, self._high)
        self.thrust = np.where(available, self.thrust + (target - self.thrust) * lag, 0.0)

        # Shortest signed arc to the command, in [-180, 180).
        error = (azimuth_cmd_deg - self.azimuth_deg + 180.0) % 360.0 - 180.0
        reach = np.array([np.inf if t.max_azimuth_rate is None else t.max_azimuth_rate * dt
                          for t in self.thrusters])
        turning = self._azimuth & available
        turned = (self.azimuth_deg + np.clip(error, -reach, reach)) % 360.0
        self.azimuth_deg = np.where(turning, turned, self.azimuth_deg)

        alpha = np.deg2rad(self.azimuth_deg)
        fx = self.thrust * np.cos(alpha)
        fy = self.thrust * np.sin(alpha)
        tau = np.array([fx.sum(), fy.sum(), (self._x * fy - self._y * fx).sum()])
        fraction = np.abs(self.thrust) / np.where(self.thrust >= 0, self._high, -self._low)
        power_kw = np.array([np.nan if t.power_kw is None else float(thruster_power_kw(t, r))
                             for t, r in zip(self.thrusters, fraction)])
        return State(self.thrust.copy(), self.azimuth_deg.copy(), fx, fy, tau, power_kw)
