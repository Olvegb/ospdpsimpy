"""Hull-side wind loads adapted from Olve Gronas Birkeland's supplied formula. 1234

Pure Python calculation, not an FMU. Inputs follow his updated message:
mean_wind_speed, sw_wind_speed, gust_wind_speed (m/s) and wind_direction_deg
(comes FROM, clockwise from north). LF pose/body velocity provide apparent wind.
The coefficient function below is retained verbatim from his supplied script.

DIRECTION ADAPTER: the supplied function returns positive Y at angle +90 deg.
Our body axes have +Y starboard; wind FROM starboard must push toward port.
We therefore pass minus the clockwise apparent FROM angle to the unchanged
function. This also mirrors its roll/yaw signs. Confirm this convention with
Olve before integration; simply subtracting heading gives the wrong lateral sign
for the supplied function and declared input convention.

The force model is horizontal, using heading and LF surge/sway velocity; no
vertical wind or WF velocity/rotational velocity at the aerodynamic centre.
Moments are assumed referenced consistently with the LF matrices. Small heel.

License: GPL-3.0-or-later (see LICENSE and NOTICE.md in this folder)
Authors: Araz, Olve Grønås Birkeland (2026)
"""
from math import cos, pi, radians, sin

import numpy as np


def blendermann_wind_coefficients(
    loa, area_lateral, area_frontal, s_L, s_H, relative_angle_deg,
    cd_t, cd_lAF_bow, cd_lAF_stern, delta, kappa,
):
    """Dimensionless Blendermann wind coefficients (C_X, C_Y, C_N, C_K).

    They are used on the form load = 1/2 * rho_air * A_w * C_w * V^2, where
    each coefficient has its own reference area (and length for moments):
        X = 1/2 rho V^2 * A_F       * C_X
        Y = 1/2 rho V^2 * A_L       * C_Y
        N = 1/2 rho V^2 * A_L * L   * C_N
        K = 1/2 rho V^2 * A_L * H_m * C_K,   H_m = A_L / L
    """
    is_port_side = relative_angle_deg > 180.0
    folded_angle = 360.0 - relative_angle_deg if is_port_side else relative_angle_deg
    eps = radians(folded_angle)

    cd_lAF = cd_lAF_bow if folded_angle <= 90.0 else cd_lAF_stern
    cd_l = cd_lAF * area_frontal / area_lateral
    denominator = 1.0 - (delta / 2.0) * (1.0 - cd_l / cd_t) * sin(2.0 * eps) ** 2

    cx = -cd_lAF * cos(eps) / denominator
    cy = cd_t * sin(eps) / denominator
    if is_port_side:
        cy = -cy

    h_m = area_lateral / loa
    cn = (s_L / loa - 0.18 * (eps - pi / 2.0)) * cy
    ck = kappa * (s_H / h_m) * cy

    return cx, cy, cn, ck


class WindLoads:
    """Tabulate Olve's coefficients and calculate body loads from apparent wind."""

    def __init__(self, parameters):
        """Receive air density, vessel wind areas/geometry and Blendermann constants.

        SI units; parameters are supplied by a test or future shared-config loader.
        The same ten-degree coefficient interpolation as Olve's FMU is retained.
        """
        names = ('air_density', 'loa', 'area_lateral', 'area_frontal', 's_L', 's_H',
                 'cd_t', 'cd_lAF_bow', 'cd_lAF_stern', 'delta', 'kappa')
        self.p = {k: float(parameters[k]) for k in names}
        if not all(np.isfinite(list(self.p.values()))):
            raise ValueError('Wind parameters must be finite')
        for k in ('air_density', 'loa', 'area_lateral', 'area_frontal', 'cd_t'):
            if self.p[k] <= 0:
                raise ValueError(f'{k} must be positive')
        args = {k: v for k, v in self.p.items() if k != 'air_density'}
        self.table = np.array([blendermann_wind_coefficients(**args, relative_angle_deg=a)
                               for a in range(0, 361, 10)])
        if not np.isfinite(self.table).all():
            raise ValueError('Invalid wind coefficients')

    @classmethod
    def from_toml(cls, path):
        """Read Olve's shared configuration sections without guessing vessel geometry.

        Requires environment.air_density, vessel geometry/type and the matching
        blendermann type entry. ShipX underwater dimensions do not provide the
        above-water wind areas or their centres. Supply vessel-specific values.
        """
        import tomllib
        from pathlib import Path
        config = tomllib.loads(Path(path).read_text(encoding='utf-8-sig'))
        vessel = config['vessel']
        coefficients = config['blendermann'][vessel['type']]
        return cls({'air_density': config['environment']['air_density'],
                    **vessel, **coefficients})

    def apparent_wind(self, eta_lf, nu_lf, *, mean_wind_speed,
                      sw_wind_speed=0., gust_wind_speed=0., wind_direction_deg):
        """Return horizontal apparent speed and clockwise FROM angle relative to bow.

        Combine all three speed components (negative fluctuations are allowed;
        negative total speed is rejected). Convert FROM direction to air travel
        velocity in NED, rotate to body axes, subtract LF body velocity, then
        recover the direction the apparent wind comes FROM.
        """
        eta, nu = np.asarray(eta_lf, float), np.asarray(nu_lf, float)
        if eta.shape != (6,) or nu.shape != (6,) or not np.isfinite(np.r_[eta, nu]).all():
            raise ValueError('Need six finite pose and six finite body velocity values')
        values = np.array([mean_wind_speed, sw_wind_speed, gust_wind_speed,
                           wind_direction_deg], float)
        if not np.isfinite(values).all():
            raise ValueError('Wind inputs must be finite')
        speed = sum(values[:3])
        if speed < 0:
            raise ValueError('Total wind speed cannot be negative')
        beta = np.deg2rad(wind_direction_deg) - eta[5]
        relative = -speed*np.array([np.cos(beta), np.sin(beta)]) - nu[:2]
        apparent = float(np.linalg.norm(relative))
        if apparent > 1e-12:
            angle = float(np.rad2deg(np.arctan2(-relative[1], -relative[0])) % 360)
        else:
            angle = 0.
        return apparent, angle

    def calculate(self, eta_lf, nu_lf, **wind_inputs):
        """Return [X,Y,0,K,0,N] in body axes (N and N m) for the LF solver.

        Z and M are omitted by this supplied wind model. The explicit minus-angle
        adapter reconciles its lateral signs with clockwise FROM inputs.
        """
        speed, angle = self.apparent_wind(eta_lf, nu_lf, **wind_inputs)
        pos = ((-angle) % 360) / 10
        i = min(int(pos), 35)
        fraction = pos - i
        cx, cy, cn, ck = (1 - fraction)*self.table[i] + fraction*self.table[i + 1]
        p = self.p
        q = .5*p['air_density']*speed**2
        return np.array([q*p['area_frontal']*cx, q*p['area_lateral']*cy, 0.,
                         q*p['area_lateral']*(p['area_lateral']/p['loa'])*ck,
                         0., q*p['area_lateral']*p['loa']*cn])
