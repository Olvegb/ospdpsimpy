
"""Low-frequency (LF) 6-DOF vessel dynamics.

State:
    [north, east, down, roll, pitch, yaw, u, v, w, p, q, r]

Coordinates:
    NED  : north, east, down
    Body : x forward, y starboard, z down
    Angles in radians.

Hydrodynamic matrices are supplied by Wamit.py.

Diagnostics:
    - Explain the exact numerical condition that stopped LF.
    - Distinguish the stop condition from a suspected root cause.
    - Record the first questionable motion without stopping.
    - Print numerical information only when an LF error occurs.
    - Preserve the existing equations and RK4 integration.

License: GPL-3.0-or-later
Authors: Araz, Olve Grønås Birkeland (2026)
"""

import sys
from collections import deque

import numpy as np

# Editable diagnostic defaults live in the central hull control panel.
from ospfmusim.hull.Control_Panel_Hull import (
    WATCH_ROLL_DEG, WATCH_PITCH_DEG, WATCH_SPEED, WATCH_RATE_DEG,
    ENABLE_WATCH, LOG_DYNAMICS_TERMS, LOG_INTERVAL_S,
    REPORT_TERMS_ON_FAILURE,
)


# ===============================================================
# 1. DIAGNOSTIC SETTINGS
# ===============================================================

# These are investigation thresholds, NOT Gunnerus safety limits.
# They only record the first concerning state.
# Exceeding them will NEVER stop the simulation.

# WATCH_* constants are imported from Control_Panel_Hull.py.
# No threshold stops the integrator.

STATE_NAMES = (
    "North", "East", "Down",
    "Roll", "Pitch", "Yaw",
    "Surge velocity", "Sway velocity", "Heave velocity",
    "Roll rate", "Pitch rate", "Yaw rate",
)

LOAD_NAMES = ("X", "Y", "Z", "K", "M", "N")


# ===============================================================
# 2. INPUT VALIDATION
# ===============================================================

# Verify input dimensions and numerical validity.
#
# An invalid input is rejected because the equations cannot
# reliably use NaN or infinity.
#
# Unlike the previous version, the exception identifies
# the incorrect shape or the first invalid component.
def checked(value, shape, name):
    result = np.asarray(value, dtype=float)

    if result.shape != shape:
        raise ValueError(
            f"{name}: wrong dimensions. "
            f"Received {result.shape}, expected {shape}."
        )

    invalid = np.argwhere(~np.isfinite(result))

    if invalid.size:
        index = tuple(int(i) for i in invalid[0])
        component = index[0] if len(index) == 1 else index

        raise ValueError(
            f"{name}: invalid numerical value at "
            f"index {component}: {result[index]}. "
            "The calculation requires finite numbers."
        )

    return result.copy()


# ===============================================================
# 3. COORDINATE TRANSFORMATIONS
# ===============================================================

# Rotation from body coordinates into the NED frame.
#
# Input:  [roll, pitch, yaw]
# Output: 3x3 rotation matrix R
#
#     v_NED = R @ v_body
def rotation(angles):
    phi, theta, psi = angles

    cp, ct, cs = np.cos([phi, theta, psi])
    sp, st, ss = np.sin([phi, theta, psi])

    return np.array([
        [cs*ct, cs*st*sp-ss*cp, cs*st*cp+ss*sp],
        [ss*ct, ss*st*sp+cs*cp, ss*st*cp-cs*sp],
        [-st, ct*sp, ct*cp],
    ])


# Construct the 6x6 kinematic transformation matrix.
#
#     eta_dot = J(eta) @ nu
#
# Euler angles become singular near pitch +/-90 degrees.
# This is a mathematical limitation, not a capsize detector.
def kinematics(eta):
    phi, theta, _ = eta[3:]

    if abs(np.cos(theta)) < 1e-6:
        raise ValueError(
            "Euler-angle singularity: pitch is too close "
            "to +/-90 degrees for this coordinate representation."
        )

    s, c, t = np.sin(phi), np.cos(phi), np.tan(theta)

    J = np.zeros((6, 6))
    J[:3, :3] = rotation(eta[3:])

    J[3:, 3:] = [
        [1., s*t, c*t],
        [0., c, -s],
        [0., s/np.cos(theta), c/np.cos(theta)],
    ]

    return J


# ===============================================================
# 4. CORIOLIS AND CENTRIPETAL EFFECTS
# ===============================================================

# Skew-symmetric cross-product matrix.
#
#     S(a) @ b = a x b
def skew(v):
    x, y, z = v

    return np.array([
        [0., -z, y],
        [z, 0., -x],
        [-y, x, 0.],
    ])


# Construct the existing 6x6 Coriolis/centripetal matrix.
#
# Input:
#     mass     : 6x6 matrix
#     velocity : 6-DOF body velocity
#
# Mathematical formulation remains unchanged.
def coriolis(mass, velocity):
    momentum = mass @ velocity

    p = skew(momentum[:3])
    h = skew(momentum[3:])
    matrix = np.zeros((6, 6))
    matrix[:3, 3:] = matrix[3:, :3] = -p
    matrix[3:, 3:] = -h
    return matrix


def cross(a, b):
    """Fast 3D cross product, equivalent to np.cross for 3-element vectors."""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ])


# ===============================================================
# 5. HUMAN-READABLE DIAGNOSTICS
# ===============================================================

# Display small numerical values normally.
# Extremely large values use scientific notation.
# This changes DISPLAY only, never model calculations.
def readable(value, unit=""):
    try:
        value = float(value)

        if np.isnan(value):
            return "NaN (invalid)"

        if np.isposinf(value):
            return "+INFINITY (invalid)"

        if np.isneginf(value):
            return "-INFINITY (invalid)"

        if abs(value) >= 1e6:
            return f"{value:.3e} {unit} [EXTREME]"

        if value != 0 and abs(value) < 0.001:
            return f"{value:.3e} {unit}"

        return f"{value:,.3f} {unit}"

    except (TypeError, ValueError, OverflowError):
        return "UNAVAILABLE"


# Convert angular states into degrees for reporting only.
#
# The actual model continues using radians.
def angle_text(value, rate=False):
    try:
        value = float(value)

        if not np.isfinite(value) or abs(value) > 1e6:
            return readable(value, "rad/s" if rate else "rad")

        unit = "deg/s" if rate else "deg"
        return readable(np.degrees(value), unit)

    except (TypeError, ValueError, OverflowError):
        return "UNAVAILABLE"


# Print the full 12-DOF state in four readable groups.
#
# This function does not modify any state variables.
def print_state(state, title):
    out = sys.stderr

    print(f"\n{title}", file=out)

    try:
        x = np.asarray(state, dtype=float).ravel()

        if x.size != 12:
            print(
                f"  Invalid state size: {x.size}, expected 12.",
                file=out,
            )
            return

        print("\n  POSITION (NED)", file=out)
        for i in range(3):
            print(
                f"    {STATE_NAMES[i]:<18}: {readable(x[i], 'm')}",
                file=out,
            )

        print("\n  ORIENTATION", file=out)
        for i in range(3, 6):
            print(
                f"    {STATE_NAMES[i]:<18}: {angle_text(x[i])}",
                file=out,
            )

        print("\n  LINEAR VELOCITY", file=out)
        for i in range(6, 9):
            print(
                f"    {STATE_NAMES[i]:<18}: {readable(x[i], 'm/s')}",
                file=out,
            )

        print("\n  ANGULAR VELOCITY", file=out)
        for i in range(9, 12):
            print(
                f"    {STATE_NAMES[i]:<18}: "
                f"{angle_text(x[i], rate=True)}",
                file=out,
            )

    except Exception as error:
        print(
            f"  Could not format state: {error}",
            file=out,
        )


# Print six external load components.
#
# X,Y,Z are forces.
# K,M,N are moments.
def print_load(name, values):
    out = sys.stderr

    print(f"\n  {name}", file=out)

    try:
        values = np.asarray(values, dtype=float).ravel()

        if values.size != 6:
            print(
                f"    Invalid load size: {values.size}, expected 6.",
                file=out,
            )
            return

        for i, value in enumerate(values):
            unit = "N" if i < 3 else "Nm"

            print(
                f"    {LOAD_NAMES[i]}: {readable(value, unit)}",
                file=out,
            )

    except Exception as error:
        print(
            f"    Could not format load: {error}",
            file=out,
        )


# Identify the first accepted state that exceeds an
# illustrative diagnostic threshold.
#
# These are monitoring rules only.
# They DO NOT modify the state or stop the simulator.
def motion_warning(state, thresholds=None):
    # Per-run thresholds can override central defaults without rebuilding.
    x = np.asarray(state, dtype=float)
    thresholds = thresholds or {
        "roll_deg": WATCH_ROLL_DEG,
        "pitch_deg": WATCH_PITCH_DEG,
        "speed_m_s": WATCH_SPEED,
        "rate_deg_s": WATCH_RATE_DEG,
    }
    roll = thresholds["roll_deg"]
    pitch = thresholds["pitch_deg"]
    speed = thresholds["speed_m_s"]
    rate = thresholds["rate_deg_s"]

    if x.shape != (12,) or not np.all(np.isfinite(x)):
        return "State has invalid numerical values."
    if abs(x[3]) > np.deg2rad(roll):
        return f"Roll exceeded {roll:g} deg."
    if abs(x[4]) > np.deg2rad(pitch):
        return f"Pitch exceeded {pitch:g} deg."
    for i in range(6, 9):
        if abs(x[i]) > speed:
            return f"{STATE_NAMES[i]} exceeded {speed:g} m/s."
    for i in range(9, 12):
        if abs(x[i]) > np.deg2rad(rate):
            return f"{STATE_NAMES[i]} exceeded {rate:g} deg/s."
    return None


# Explain the exact reason LF raised an exception.
#
# This function distinguishes:
#     - The immediate numerical stopping condition
#     - Possible preceding instability
#
# It must not claim that the root physical cause is known.
def explain_stop(error, stage):
    msg = str(error).lower()

    if "tau_wind" in msg:
        return (
            "The LF model received an invalid WIND-LOAD vector.\n"
            "A component has an incorrect shape, NaN or infinity.\n"
            "LF cannot continue with invalid applied forces.\n\n"
            "IMPORTANT:\n"
            "This does not mean environmental wind was enabled.\n"
            "The relative-air calculation may overflow after the\n"
            "vessel velocities have already become unrealistic.\n"
            "The underlying source must be investigated separately."
        )

    if "tau_wave_drift" in msg:
        return (
            "The wave-drift load supplied to LF is invalid.\n"
            "LF cannot integrate a non-finite wave-force input.\n"
            "Inspect WaveLoads.py and the upstream vessel state."
        )

    if "tau_thruster" in msg:
        return (
            "The thruster-load vector supplied to LF is invalid.\n"
            "Inspect the upstream thrust calculation."
        )

    if "tau_external" in msg:
        return (
            "An external-force input is invalid.\n"
            "Inspect the source of the external load."
        )

    if "current_ned" in msg:
        return (
            "The ocean-current velocity input is invalid.\n"
            "LF requires three finite NED velocity components."
        )

    if "updated lf state" in msg:
        return (
            "The RK4 integration produced a non-finite next state.\n"
            "At least one position, orientation or velocity is\n"
            "NaN or infinity. The numerical solution has broken down."
        )

    if "state" in msg:
        return (
            "A vessel state passed to LF contains invalid values.\n"
            "The state is required to contain 12 finite numbers.\n"
            "This may reflect an earlier numerical divergence."
        )

    if "euler" in msg or "singular" in msg:
        return (
            "The Euler-angle kinematic representation became\n"
            "singular or otherwise unusable near pitch +/-90 deg.\n"
            "This is a coordinate-system limitation, not proof\n"
            "that Gunnerus physically capsized."
        )

    if "time and positive step size" in msg:
        return (
            "The integration time step is invalid.\n"
            "LF requires a finite, strictly positive dt."
        )

    if isinstance(error, np.linalg.LinAlgError):
        return (
            "The matrix solver could not solve the vessel\n"
            "acceleration equations. Inspect the mass matrix\n"
            "and the numerical state."
        )

    return (
        f"The LF calculation raised an exception during {stage}.\n"
        "The original Python error is printed below.\n"
        "The exact root cause has not been established."
    )


# Print a concise terminal report after an LF exception.
#
# Important:
# A reported failure condition is NOT automatically the
# first cause of numerical instability.
def report_lf_failure(
    error,
    stage,
    time,
    dt,
    state,
    attempted_state,
    loads,
    total_load,
    history,
    first_warning,
    last_quiet,
):
    out = sys.stderr

    print("\n" + "=" * 65, file=out)
    print("                  LF SIMULATION STOPPED", file=out)
    print("=" * 65, file=out)

    print("\n1. EXACT STOP CONDITION", file=out)
    print(explain_stop(error, stage), file=out)

    print("\n2. WHERE DID IT STOP?", file=out)
    print(
        f"  Approximate LF time : {readable(time, 's')}",
        file=out,
    )
    print(
        f"  Step size           : {readable(dt, 's')}",
        file=out,
    )
    print(
        f"  Failed calculation  : {stage}",
        file=out,
    )
    print(
        f"  Original exception  : {type(error).__name__}: {error}",
        file=out,
    )

    print("\n3. FIRST INVESTIGATION WARNING", file=out)

    if first_warning is None:
        print(
            "  No monitoring threshold was recorded before failure.",
            file=out,
        )
    else:
        warning_time, warning_state, reason = first_warning

        print(
            f"  First detected at LF time "
            f"{readable(warning_time, 's')}",
            file=out,
        )
        print(f"  Reason: {reason}", file=out)

        print_state(
            warning_state,
            "  STATE AT FIRST WARNING",
        )

    print(
        "\n  These are investigation thresholds only.",
        file=out,
    )
    print(
        "  They do not stop or modify the simulation.",
        file=out,
    )

    if last_quiet is not None:
        quiet_time, quiet_state = last_quiet

        print(
            "\n4. LAST STATE BELOW THE WATCH THRESHOLDS",
            file=out,
        )
        print(
            f"  LF time: {readable(quiet_time, 's')}",
            file=out,
        )

        print_state(
            quiet_state,
            "  RECORDED STATE",
        )

    print_state(
        state,
        "5. STATE AT START OF FAILED STEP",
    )

    # Useful when the failure happened during an RK4 substep.
    if stage.startswith("RK4"):
        print_state(
            attempted_state,
            "6. STATE USED BY FAILED RK4 CALCULATION",
        )

    print("\n7. APPLIED LOADS AT FAILURE", file=out)

    for name, values in loads.items():
        print_load(name, values)

    if total_load is not None:
        print_load("TOTAL LOAD", total_load)

    print("\n8. LAST FIVE ACCEPTED LF STEPS", file=out)

    if not history:
        print("  No previous accepted state.", file=out)

    for t, x in history:
        try:
            x = np.asarray(x, dtype=float)

            # Give a compact summary instead of another huge table.
            if x.shape != (12,):
                summary = "Invalid state dimensions."

            elif not np.all(np.isfinite(x)):
                summary = "Non-finite state."

            elif np.max(np.abs(x)) >= 1e6:
                summary = "EXTREME VALUES: numerical result unreliable."

            else:
                summary = (
                    f"Roll={angle_text(x[3])}, "
                    f"Pitch={angle_text(x[4])}, "
                    f"Yaw={angle_text(x[5])}, "
                    f"u={readable(x[6], 'm/s')}, "
                    f"v={readable(x[7], 'm/s')}"
                )

            print(
                f"  t={t:.4f} s | {summary}",
                file=out,
            )

        except Exception as history_error:
            print(
                f"  Could not read history: {history_error}",
                file=out,
            )

    print("\n9. CONCLUSION", file=out)
    print(
        "  LF stopped because an existing numerical validation",
        file=out,
    )
    print(
        "  or calculation failed.",
        file=out,
    )
    print(
        "  It did NOT automatically detect physical capsizing.",
        file=out,
    )
    print(
        "  The component reporting the final error may not",
        file=out,
    )
    print(
        "  be the component that caused the instability.",
        file=out,
    )

    print("\n" + "=" * 65, file=out)
    print("                       END OF REPORT", file=out)
    print("=" * 65 + "\n", file=out, flush=True)


# ===============================================================
# 6. LOW-FREQUENCY VESSEL MODEL
# ===============================================================

class LF:
    """Integrate the low-frequency 6-DOF vessel equations."""

    # Initialize the hydrodynamic matrices.
    #
    # M_RB : rigid-body mass
    # A    : added mass
    # D    : damping
    # G    : restoring
    #
    #     M = M_RB + A
    #
    # No physical parameters are modified.
    def __init__(self, rigid_mass, added_mass, damping, restoring, *, diagnostics=None):
        self.M_RB = checked(rigid_mass, (6, 6), "rigid_mass")
        self.A = checked(added_mass, (6, 6), "added_mass")
        self.D = checked(damping, (6, 6), "damping")
        self.G = checked(restoring, (6, 6), "restoring")

        self.M = self.M_RB + self.A

        if np.linalg.matrix_rank(self.M) != 6:
            raise ValueError("Total mass matrix must be invertible")

        # Monitoring only; never used in the equations of motion.
        self._elapsed_s = 0.0
        self._recent_states = deque(maxlen=5)
        self._first_warning = None
        self._last_quiet = None
        self._diagnostics = dict(diagnostics or {})
        self._last_logged_s = float("-inf")

    # ---------------------------------------------------------------
    # 7. VESSEL EQUATIONS OF MOTION
    # ---------------------------------------------------------------

    # Compute the derivative of the 12-state vessel vector.
    #
    #     M @ nu_dot = tau - C_RB @ nu - C_A @ nu_r
    #                  - D @ nu_r - G @ eta + A @ nu_c_dot
    #
    # Inputs:
    #     state       : 12 elements [eta, nu]
    #     tau         : 6 forces/moments
    #     current_ned : 3 current components
    #
    # Output:
    #     [eta_dot, nu_dot]
    def diagnostic_terms(self, state, tau, current_ned=(0., 0., 0.)):
        """Inspect each term of M*nu_dot without affecting the integration.

        Returns terms [X,Y,Z,K,M,N] and nu_dot. Used only by diagnostics.
        """
        state = checked(state, (12,), "diagnostic state")
        tau = checked(tau, (6,), "diagnostic applied loads")
        current_ned = checked(current_ned, (3,), "diagnostic current")
        eta, nu = state[:6], state[6:]
        current_body = rotation(eta[3:]).T @ current_ned
        nu_c = np.r_[current_body, np.zeros(3)]
        nu_r = nu - nu_c
        nu_c_dot = np.r_[-np.cross(nu[3:], current_body), np.zeros(3)]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            terms = {
                "external": tau,
                "rigid Coriolis": -(coriolis(self.M_RB, nu) @ nu),
                "added Coriolis": -(coriolis(self.A, nu_r) @ nu_r),
                "damping": -(self.D @ nu_r),
                "restoring": -(self.G @ eta),
                "current acceleration": self.A @ nu_c_dot,
            }
            net = sum(terms.values())
            terms["NET force"] = net
            terms["ACCELERATION"] = np.linalg.solve(self.M, net)
        return terms

    def print_diagnostic_terms(self, state, tau, current, *, label):
        """Print contributions only; never alter vessel forces or states."""
        terms = self.diagnostic_terms(state, tau, current)
        print(f"\n[LF DYNAMICS] {label}", file=sys.stderr)
        print("  DOF order = [surge, sway, heave, roll, pitch, yaw]", file=sys.stderr)
        print("  Rows are forces [N] / moments [Nm]; ACCELERATION is SI.", file=sys.stderr)
        for term, values in terms.items():
            formatted = " ".join(f"{value:+.3e}" for value in values)
            print(f"  {term:22s}: {formatted}", file=sys.stderr)
        sys.stderr.flush()

    def derivative(self, state, tau, current_ned=(0., 0., 0.)):
        state = checked(state, (12,), "state")
        tau = checked(tau, (6,), "tau")
        current_ned = checked(current_ned, (3,), "current_ned")

        eta = state[:6]
        nu = state[6:]

        current_body = rotation(eta[3:]).T @ current_ned

        nu_c = np.concatenate((current_body, np.zeros(3)))
        nu_r = nu - nu_c

        nu_c_dot = np.concatenate((
            -cross(nu[3:], current_body),
            np.zeros(3),
        ))

        # Coriolis and centripetal contributions.
        rigid_coriolis = coriolis(self.M_RB, nu) @ nu
        added_coriolis = coriolis(self.A, nu_r) @ nu_r

        # Total effective force.
        net_load = (
            tau
            - rigid_coriolis
            - added_coriolis
            - self.D @ nu_r
            - self.G @ eta
            + self.A @ nu_c_dot
        )

        # Position and orientation derivatives.
        eta_dot = kinematics(eta) @ nu

        # Body-frame accelerations.
        nu_dot = np.linalg.solve(self.M, net_load)

        return np.concatenate((eta_dot, nu_dot))

    # ---------------------------------------------------------------
    # 8. RK4 INTEGRATION WITH DIAGNOSTICS
    # ---------------------------------------------------------------

    # Advance the LF vessel state using classical fixed-step RK4.
    #
    #     x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    #
    # Physics and integration remain unchanged.
    # Monitoring happens only after an accepted step.
    #
    # On failure:
    #     1. Print the exact stop condition.
    #     2. Print earlier monitoring information.
    #     3. Re-raise the original exception.
    def step(
        self, state, dt, *,
        tau_wind=(0.,)*6,
        tau_thruster=(0.,)*6,
        tau_wave_drift=(0.,)*6,
        current_ned=(0., 0., 0.),
        tau_external=(0.,)*6,
    ):
        stage = "Input validation"
        attempted_state = state
        total_load = None

        loads = {
            "WIND": tau_wind,
            "THRUSTERS": tau_thruster,
            "WAVE DRIFT": tau_wave_drift,
            "EXTERNAL": tau_external,
        }

        try:
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError(
                    "Finite time and positive step size required"
                )

            stage = "Validate vessel state"
            state = checked(state, (12,), "state")
            attempted_state = state

            # Validate loads separately to identify the precise input.
            stage = "Validate wind load"
            wind = checked(tau_wind, (6,), "tau_wind")

            stage = "Validate thruster load"
            thruster = checked(tau_thruster, (6,), "tau_thruster")

            stage = "Validate wave-drift load"
            wave_drift = checked(
                tau_wave_drift, (6,), "tau_wave_drift"
            )

            stage = "Validate external load"
            external = checked(tau_external, (6,), "tau_external")

            stage = "Combine external loads"
            total_load = wind + thruster + wave_drift + external

            stage = "Validate ocean current"
            current = checked(
                current_ned, (3,), "current_ned"
            )

            # Original RK4 mathematical operations.

            stage = "RK4 - k1"
            attempted_state = state
            k1 = self.derivative(state, total_load, current)

            stage = "RK4 - k2"
            attempted_state = state + dt*k1/2
            k2 = self.derivative(
                attempted_state, total_load, current
            )

            stage = "RK4 - k3"
            attempted_state = state + dt*k2/2
            k3 = self.derivative(
                attempted_state, total_load, current
            )

            stage = "RK4 - k4"
            attempted_state = state + dt*k3
            k4 = self.derivative(
                attempted_state, total_load, current
            )

            stage = "RK4 - final state"
            attempted_state = (
                state + dt*(k1 + 2*k2 + 2*k3 + k4)/6
            )

            next_state = checked(
                attempted_state,
                (12,),
                "updated LF state",
            )

            # Store accepted state; no changes to model outputs.
            self._elapsed_s += dt

            self._recent_states.append(
                (self._elapsed_s, next_state.copy())
            )

            # Record a diagnostic warning only; no physical stop condition.
            enable_watch = self._diagnostics.get("enable_watch", ENABLE_WATCH)
            reason = (
                motion_warning(next_state, self._diagnostics.get("watch"))
                if enable_watch else None
            )

            if reason is None:
                if self._first_warning is None:
                    self._last_quiet = (
                        self._elapsed_s,
                        next_state.copy(),
                    )

            elif self._first_warning is None:
                self._first_warning = (
                    self._elapsed_s,
                    next_state.copy(),
                    reason,
                )

            # Optional sampling of the actual 6DOF force contributions.
            # The diagnostic calculation never feeds back into the solver.
            if self._diagnostics.get("log_dynamics_terms", LOG_DYNAMICS_TERMS):
                sample_interval = self._diagnostics.get("log_interval_s", LOG_INTERVAL_S)
                if self._elapsed_s - self._last_logged_s >= sample_interval:
                    try:
                        self.print_diagnostic_terms(
                            next_state, total_load, current,
                            label=f"t={self._elapsed_s:.3f} s",
                        )
                    except Exception as diag_error:
                        print(f"[LF] Dynamics diagnostics unavailable: {diag_error}",
                              file=sys.stderr)
                    self._last_logged_s = self._elapsed_s

            return next_state

        except Exception as error:
            # Diagnostics must not replace the original exception.
            try:
                # Before reporting the final invalid state, reconstruct the
                # per-DOF dynamics at the last accepted finite state, if any.
                if self._diagnostics.get(
                    "report_terms_on_failure", REPORT_TERMS_ON_FAILURE
                ) and total_load is not None:
                    candidate = (
                        self._recent_states[-1][1] if self._recent_states
                        else state
                    )
                    try:
                        self.print_diagnostic_terms(
                            candidate, total_load, current_ned,
                            label=f"last accepted state, t~{self._elapsed_s:.3f} s",
                        )
                    except Exception as diag_error:
                        print(f"[LF] Term audit unavailable: {diag_error}",
                              file=sys.stderr)
                report_lf_failure(
                    error=error,
                    stage=stage,
                    time=self._elapsed_s,
                    dt=dt,
                    state=state,
                    attempted_state=attempted_state,
                    loads=loads,
                    total_load=total_load,
                    history=self._recent_states,
                    first_warning=self._first_warning,
                    last_quiet=self._last_quiet,
                )

            except Exception as diagnostic_error:
                print(
                    f"LF diagnostic reporting failed: "
                    f"{diagnostic_error}",
                    file=sys.stderr,
                    flush=True,
                )

            raise

    # ---------------------------------------------------------------
    # 9. OPTIONAL WAVE-LOAD INTERFACE
    # ---------------------------------------------------------------

    # Calculate second-order wave drift before advancing LF.
    #
    # Keep this entry point for compatibility with external callers.
    def step_with_waves(
        self, state, dt, *,
        time,
        wave_loads,
        tau_wind=(0.,)*6,
        tau_thruster=(0.,)*6,
        current_ned=(0., 0., 0.),
        tau_external=(0.,)*6,
    ):
        state = checked(state, (12,), "state")

        drift = wave_loads.calculate(
            time, state[:6]
        ).tau_second_order

        return self.step(
            state,
            dt,
            tau_wind=tau_wind,
            tau_thruster=tau_thruster,
            tau_wave_drift=drift,
            current_ned=current_ned,
            tau_external=tau_external,
        )
