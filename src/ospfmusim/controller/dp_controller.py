"""DP controller FMU: a simple PID controller that holds the vessel at a set-point.

The FMU has only ONE parameter, config_path, and reads [controller] and
[vessel] from the config file when the simulation starts.

  - inputs:   the estimates from the ExtendedKalmanFilter FMU
              north_hat, east_hat [m], psi_hat [rad], u_hat, v_hat [m/s], r_hat [rad/s]
  - outputs:  the P, I and D terms of the commanded force in the body frame,
              tau_<dof>_P, tau_<dof>_I, tau_<dof>_D for dof X, Y [N] and N [Nm],
              to the ThrustAllocation FMU, which adds them up to tau_X, tau_Y, tau_N
  - settings: [controller] (set-point, natural frequency, damping ratio, integral limit)
              and the mass and damping matrices in [vessel]

Control law: PID with velocity feedback (Fossen, Handbook of Marine Craft
Hydrodynamics and Motion Control, ch. 12). The position error is taken in NED
and rotated into the body frame, because tau is in the body frame:

    e_ned = [north_hat - north_ref, east_hat - east_ref, wrap(psi_hat - psi_ref)]
    e     = R(psi_hat)^T e_ned                         (error in the body frame)
    z    += e * dt                                     (integral of the error)
    tau   = -Kp e - Ki z - Kd [u_hat, v_hat, r_hat]    (body frame)
          =  tau_P  + tau_I  + tau_D                   (sent out one by one)

The D term uses the velocity estimates directly, so nothing has to be
differentiated. The integral removes the steady offset from current, wave
drift and wind. It stops growing when |Ki z| reaches integral_force_limit
(anti-windup), for example when the thrusters cannot hold the vessel.
The integral is in the body frame, which is fine when the heading is held
(station keeping), but not exact during large heading changes.

The gains are calculated from the diagonals of M and D in [vessel], so each
degree of freedom behaves like a mass-spring-damper with natural frequency
wn and damping ratio zeta:

    Kp = M wn^2,   Kd = 2 zeta wn M - D  (at least 0),   Ki = Kp wn / 10

So the gains follow the vessel when the matrices in [vessel] change.
"""
import tomllib
from pathlib import Path

import numpy as np
from pythonfmu import Fmi2Causality, Fmi2Slave, Fmi2Variability, Real, String

INPUTS = (
    "north_hat", "east_hat", "psi_hat",      # estimated position and heading [m, m, rad]
    "u_hat", "v_hat", "r_hat",               # estimated velocity, body frame [m/s, m/s, rad/s]
)
DOFS = ("X", "Y", "N")
TERMS = ("P", "I", "D")
# The commanded force, one output per degree of freedom and term, body frame [N, N, Nm].
TAU_TERMS = tuple(f"tau_{dof}_{term}" for dof in DOFS for term in TERMS)
OUTPUTS = TAU_TERMS + (
    "error_north", "error_east",             # position error, NED [m] (for logging)
    "error_heading_deg",                     # heading error, -180 to 180 [deg] (for logging)
)


def total_tau(fmu):
    """tau = P + I + D [X, Y, N] from the controller's outputs, as the allocation adds them up."""
    return np.array([sum(getattr(fmu, f"tau_{dof}_{term}") for term in TERMS) for dof in DOFS])


def rotation(psi):
    """R(psi): rotates a body-frame vector [x, y, n] into NED. R^T rotates NED into body."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_angle(angle):
    """Wraps an angle to [-pi, pi), so a heading error of 359 deg becomes -1 deg."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def pid_gains(mass_matrix, damping_matrix, natural_frequency, damping_ratio):
    """Kp, Ki, Kd (one value per degree of freedom) from the diagonals of M and D."""
    m = np.diag(np.asarray(mass_matrix, dtype=float))
    d = np.diag(np.asarray(damping_matrix, dtype=float))
    wn = np.asarray(natural_frequency, dtype=float)
    zeta = np.asarray(damping_ratio, dtype=float)
    kp = m * wn**2
    kd = np.maximum(2.0 * zeta * wn * m - d, 0.0)   # the vessel's own damping already helps
    ki = kp * wn / 10.0
    return kp, ki, kd


class DPController(Fmi2Slave):

    author = "Viljar Simonnes Tveit"
    description = ("Simple DP controller: PID on position and heading, commanded force tau in "
                   "the body frame")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # PARAMETER: path to the config file.
        self.config_path = ""
        self.register_variable(String("config_path", causality=Fmi2Causality.parameter,
                                      variability=Fmi2Variability.fixed))

        # INPUTS: the estimates from the Kalman filter.
        for name in INPUTS:
            setattr(self, name, 0.0)
            self.register_variable(Real(name, causality=Fmi2Causality.input))

        # OUTPUTS: the P, I and D terms of the commanded force, and the errors for logging.
        for name in OUTPUTS:
            setattr(self, name, 0.0)
            self.register_variable(Real(name, causality=Fmi2Causality.output))

        # CONSTANTS and controller state: set in exit_initialization_mode.
        self.reference = None                    # [north_ref, east_ref, psi_ref] [m, m, rad]
        self.kp = self.ki = self.kd = None       # gains per degree of freedom
        self.integral_limit = None               # largest |Ki z| [N, N, Nm]
        # Integral of the body-frame error [m s, m s, rad s]
        self.z = np.zeros(3)

    def exit_initialization_mode(self):
        if not self.config_path:
            raise ValueError("config_path is not set. Set it in OspSystemStructure.xml.")
        config = tomllib.loads(Path(self.config_path).read_text(encoding="utf-8-sig"))

        controller = config["controller"]
        self.reference = np.array([float(controller["north_ref"]), float(controller["east_ref"]),
                                   np.radians(float(controller["heading_ref_deg"]))])
        self.kp, self.ki, self.kd = pid_gains(config["vessel"]["mass_matrix"],
                                              config["vessel"]["damping_matrix"],
                                              controller["natural_frequency"],
                                              controller["damping_ratio"])
        self.integral_limit = np.asarray(controller["integral_force_limit"], dtype=float)
        self.z = np.zeros(3)

        # Calculate the outputs once already now, so they are right from time 0.
        self.do_step(0.0, 0.0)

    def do_step(self, current_time, step_size):
        eta = np.array([self.north_hat, self.east_hat, self.psi_hat])
        nu = np.array([self.u_hat, self.v_hat, self.r_hat])

        # Error in NED, with the heading error wrapped to [-pi, pi), then rotated into the
        # body frame.
        e_ned = eta - self.reference
        e_ned[2] = wrap_angle(e_ned[2])
        e = rotation(self.psi_hat).T @ e_ned

        # Integral, with anti-windup: |Ki z| is kept within integral_force_limit.
        self.z = self.z + e * step_size
        z_max = np.divide(self.integral_limit, self.ki, out=np.full(3, np.inf), where=self.ki > 0)
        self.z = np.clip(self.z, -z_max, z_max)

        # PID in the body frame, one term at a time. The allocation adds them up.
        terms = {"P": -self.kp * e, "I": -self.ki * self.z, "D": -self.kd * nu}
        for term, values in terms.items():
            for dof, value in zip(DOFS, values):
                setattr(self, f"tau_{dof}_{term}", float(value))
        self.error_north, self.error_east = float(e_ned[0]), float(e_ned[1])
        self.error_heading_deg = float(np.degrees(e_ned[2]))
        return True
