"""
Thruster dynamics with a generalised ducted propeller: the Wageningen Ka 4-70
propeller in nozzle 19A (Oosterveld 1973), to compare with thruster_ktkq,
where the curves are a table in the config.

ThrusterDynamicsDucted is dynamics_ktkq.ThrusterDynamicsKtKq with one change:
a thruster with rated_rpm in the config gets KT(J) and KQ(J) from the series
polynomials instead of open_water.

  1. P/D is chosen so the unit absorbs power_kw at rated_rpm in bollard pull,
     KQ0 = P / (2 pi rho n^3 D^5). No blade geometry is needed.
  2. c_t and c_q scale KT and KQ. Fitted to vessel data, they absorb what the
     series does not represent (rim drive, stators, custom duct, scale
     effects). For a vessel without data, leave them at 1.

Thrusters without rated_rpm are as in ThrusterDynamicsKtKq. The series is
valid for 0.6 <= P/D <= 1.4 and 0 <= J <= J(KT = 0): the first quadrant.

License: MIT (see LICENSE in the repository root)
Authors: William Boland, Olve Grønås Birkeland (2026)
"""
import math

from scipy.optimize import brentq

from ospfmusim.thruster_ktkq.helpfunctions.dynamics_ktkq import RHO, ThrusterDynamicsKtKq

# Oosterveld polynomial coefficients for Ka 4-70 in nozzle 19A.
# Keys are (x, y) = (power of P/D, power of J).
KT_19A = {  # total thrust (propeller + duct)
    (0, 0): +0.030550, (0, 1): -0.148687, (0, 3): -0.391137,
    (1, 1): -0.432612,
    (2, 0): +0.667657, (2, 2): +0.285076,
    (3, 0): -0.172529,
    (6, 1): -0.017283,
}
KQ_19A = {
    (0, 0): +0.006735, (0, 2): -0.016306, (0, 4): -0.007244,
    (1, 2): -0.024012,
    (2, 2): +0.005193,
    (3, 0): +0.046605,
    (4, 0): -0.007366,
    (6, 0): -0.001730, (6, 1): -0.000337, (6, 2): +0.000861,
}
PD_RANGE = (0.6, 1.4)  # P/D the series covers


def _poly(coeffs, j, pd):
    return sum(c * pd**x * j**y for (x, y), c in coeffs.items())


def pitch_ratio(thruster):
    """P/D at which the series absorbs power_kw at rated_rpm in bollard pull (c_q = 1)."""
    n = thruster.rated_rpm / 60
    target = thruster.power_kw * 1e3 / (2 * math.pi * RHO * n**3 * thruster.diameter**5)
    low, high = (_poly(KQ_19A, 0.0, pd) for pd in PD_RANGE)
    if not low <= target <= high:
        raise ValueError(f"{thruster.name}: KQ0 = {target:.4f} from power_kw and rated_rpm is "
                         f"outside the Ka 4-70/19A series [{low:.4f}, {high:.4f}]. Check "
                         "rated_rpm and power_kw.")
    return brentq(lambda pd: _poly(KQ_19A, 0.0, pd) - target, *PD_RANGE)


class ThrusterDynamicsDucted(ThrusterDynamicsKtKq):
    """ThrusterDynamicsKtKq, with the curves of thrusters with rated_rpm from the Ka 4-70/19A series."""

    def curves(self, thruster):
        if thruster.rated_rpm is None:
            return super().curves(thruster)
        pd = pitch_ratio(thruster)
        j0 = brentq(lambda j: _poly(KT_19A, j, pd), 0.0, 2.0)  # J where KT = 0
        # ponytail: first quadrant only. J is held in [0, J0]: inflow from
        # behind gives the bollard values, beyond J0 no thrust. Reverse thrust
        # uses the same curve (dynamics_ktkq.propeller), fine for azimuths that
        # turn instead; add 4-quadrant data for tunnels or shaft lines.
        clip = lambda j: min(max(j, 0.0), j0)  # noqa: E731
        return (lambda j: thruster.c_t * _poly(KT_19A, clip(j), pd),
                lambda j: thruster.c_q * _poly(KQ_19A, clip(j), pd))
