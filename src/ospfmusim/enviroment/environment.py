"""Environment FMU: Changes the enviroment during simulation.

Reads the wind, wave and current parameters from the config file. All outputs are in
the earth-fixed NED frame; the FMU knows nothing about the vessel.

The wind model follows Sorensen, Marine Cybernetics lecture notes (2018), section 6.2.2.

Wind speed, split into three parts that are each given as their own output.
The total, V(t) = V_mean + V_sv(t) + V_gust(t), is calculated where it is used.

  - V_mean   mean wind speed at the height z, constant. Logarithmic profile
             from the 1-hour mean at 10 m, U10 (eq. 6.25):
                 V_mean = U10 * 5/2 sqrt(kappa) ln(z / z0),
                 z0 = 10 exp(-2 / (5 sqrt(kappa)))
             kappa is the sea surface drag coefficient. z = 10 m gives U10.
  - V_sv     slowly varying wind: a first-order Gauss-Markov process
             (eq. 6.26, Fossen, Handbook of Marine Craft Hydrodynamics and
             Motion Control), discretised exactly so it does not depend on
             the step size:
                 x[k+1] = a * x[k] + sigma * sqrt(1 - a^2) * w[k],
                 a = exp(-dt / T),  w[k] ~ N(0, 1)
             Saturated so that V_min <= V_mean + V_sv <= V_max (eq. 6.27).
  - V_gust   gusts: sum of harmonics from the NORSOK N-003 / ISO 19901-1
             (Froya) wind spectrum, with random phases (eq. 6.31-6.32):
                 V_gust(t) = sum_i sqrt(2 S(f_i) df) cos(2 pi f_i t + phi_i)
             The gusts do not change with the height (only through S).

Wind direction (where the wind comes from, clockwise from north):

    beta(t) = beta_mean + rate * t + beta_sv(t)

  - rate     constant turning rate, for a steady veer.
  - beta_sv  slow random variation, the same Gauss-Markov process as V_sv
             (eq. 6.28), without saturation.

A part is switched off by setting it to 0 in the config file: std = 0 for
V_sv, n_components = 0 for V_gust, rate_deg_per_s = 0 and std_deg = 0 for
the direction.

License: MIT (see LICENSE in the repository root)
Authors: Olve Grønås Birkeland (2026)
"""
import random
import tomllib
from math import cos, exp, inf, log, pi, radians, sin, sqrt
from pathlib import Path

from pythonfmu import Fmi2Causality, Fmi2Slave, Fmi2Variability, Real, String


def mean_wind_profile(height, drag_coefficient):
    """U(z) / U10, the logarithmic mean wind profile (Sorensen 2018, eq. 6.25).

    height = z [m], drag_coefficient = kappa, the sea surface drag coefficient.
    """
    z0 = 10.0 * exp(-2.0 / (5.0 * sqrt(drag_coefficient)))
    return 2.5 * sqrt(drag_coefficient) * log(height / z0)


def norsok_wind_spectrum(f, mean_speed, height):
    """NORSOK N-003 / ISO 19901-1 (Froya) wind spectrum S(f) [m^2/s^2 / Hz].

    f in Hz, mean_speed = 1-hour mean wind speed at 10 m [m/s], height [m].
    """
    n = 0.468
    f_tilde = 172.0 * f * (height / 10.0) ** (2.0 / 3.0) * (mean_speed / 10.0) ** -0.75
    return (320.0 * (mean_speed / 10.0) ** 2 * (height / 10.0) ** 0.45
            / (1.0 + f_tilde ** n) ** (5.0 / (3.0 * n)))


def gust_components(mean_speed, height, f_min, f_max, n_components, rng):
    """Frequencies [rad/s], amplitudes [m/s] and phases [rad] of the gust harmonics.

    The frequencies are spread logarithmically, and each one is drawn at random
    inside its band so the gust signal does not repeat itself.
    n_components = 0 gives no gusts.
    """
    if n_components == 0:
        return []
    edges = [exp(log(f_min) + (log(f_max) - log(f_min)) * i / n_components)
             for i in range(n_components + 1)]
    components = []
    for f_low, f_high in zip(edges[:-1], edges[1:]):
        f = rng.uniform(f_low, f_high)
        amplitude = sqrt(2.0 * norsok_wind_spectrum(f, mean_speed, height) * (f_high - f_low))
        components.append((2.0 * pi * f, amplitude, rng.uniform(0.0, 2.0 * pi)))
    return components


class GaussMarkov:
    """First-order Gauss-Markov process with time constant T and stationary std sigma.

    The value is saturated to [lower, upper]. The default is no limits.
    """

    def __init__(self, time_constant, std, rng, lower=-inf, upper=inf):
        self.time_constant = time_constant
        self.std = std
        self.rng = rng
        self.lower = lower
        self.upper = upper
        # Start in the stationary distribution, so there is no transient.
        self.value = self.saturate(rng.gauss(0.0, std))

    def saturate(self, value):
        return min(max(value, self.lower), self.upper)

    def step(self, dt):
        # std = 0 means switched off. Then the value stays 0, whatever the time constant is.
        if self.std == 0.0:
            return self.value
        # dt = 0 (the call from exit_initialization_mode) leaves the value unchanged.
        a = exp(-dt / self.time_constant)
        self.value = self.saturate(
            a * self.value + self.std * sqrt(1.0 - a * a) * self.rng.gauss(0.0, 1.0))
        return self.value


class Environment(Fmi2Slave):

    author = "Olve Grønås Birkeland"
    description = ("Changes the environment during simulation. Mean, slowly varying and gust "
                   "wind speed, slowly varying wind direction (NED), waves and current, "
                   "from the config file.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # PARAMETER: path to the config file.
        self.config_path = ""
        self.register_variable(String("config_path", causality=Fmi2Causality.parameter,
                                      variability=Fmi2Variability.fixed))

        # CONSTANTS: read from [wind] in the config file in exit_initialization_mode.
        self.mean_speed_10m = 0.0           # U10, 1-hour mean wind speed at 10 m [m/s]
        self.mean_wind_direction_deg = 0.0  # where the wind comes from, clockwise from north [deg]
        self.direction_rate = 0.0           # constant turning rate [deg/s]
        self.speed_sv = None                # GaussMarkov for V_sv
        self.direction_sv = None            # GaussMarkov for beta_sv
        self.gust_components = []           # (omega, amplitude, phase) per harmonic

        # CONSTANTS: read from [waves] in the config file in exit_initialization_mode.
        self.hs = 0.0                       # significant wave height [m]
        self.tp = 0.0                       # peak wave period [s]
        self.waves_from_deg = 0.0           # where the waves come from, clockwise from north [deg]

        # OUTPUTS: wind, NED frame.
        self.mean_wind_speed = 0.0              # V_mean, at the height z [m/s]
        self.slowly_varying_wind_speed = 0.0    # V_sv [m/s]
        self.gust_wind_speed = 0.0              # V_gust [m/s]
        # Where the wind comes from, clockwise from north, 0-360 [deg]
        self.wind_direction_deg = 0.0
        for name in ("mean_wind_speed", "slowly_varying_wind_speed", "gust_wind_speed",
                     "wind_direction_deg"):
            self.register_variable(Real(name, causality=Fmi2Causality.output))

        # OUTPUTS: waves, passed straight on from the config file.
        self.H_s = 0.0                  # [m]
        self.T_p = 0.0                  # [s]
        self.wave_direction_deg = 0.0   # where the waves come from, clockwise from north [deg]
        for name in ("H_s", "T_p", "wave_direction_deg"):
            self.register_variable(Real(name, causality=Fmi2Causality.output))

        # OUTPUTS: current, NED frame, passed straight on from the config file.
        self.current_speed = 0.0            # [m/s]
        # Where the current comes FROM, clockwise from north [deg]
        self.current_direction_deg = 0.0
        self.current_north = 0.0            # north component of the current velocity [m/s]
        self.current_east = 0.0             # east component of the current velocity [m/s]
        for name in ("current_speed", "current_direction_deg", "current_north", "current_east"):
            self.register_variable(Real(name, causality=Fmi2Causality.output))

    def exit_initialization_mode(self):
        if not self.config_path:
            raise ValueError("config_path is not set. Set it in OspSystemStructure.xml.")
        config = tomllib.loads(Path(self.config_path).read_text(encoding="utf-8-sig"))

        wind = config["wind"]
        self.mean_speed_10m = float(wind["mean_speed"])
        self.mean_wind_direction_deg = float(wind["direction_deg"])
        reference_height = float(wind["reference_height"])
        self.mean_wind_speed = self.mean_speed_10m * mean_wind_profile(
            reference_height, float(wind["surface_drag_coefficient"]))
        # Same seed = same wind realisation every run.
        rng = random.Random(wind["seed"])

        # The limits are on V_mean + V_sv, so V_sv is limited to the limits minus V_mean.
        slowly_varying = wind["slowly_varying"]
        min_speed = float(slowly_varying["min_speed"])
        max_speed = float(slowly_varying["max_speed"])
        if not 0.0 <= min_speed <= self.mean_wind_speed <= max_speed:
            raise ValueError(
                f"[wind.slowly_varying] needs 0 <= min_speed <= mean wind speed at "
                f"reference_height ({self.mean_wind_speed:.2f} m/s) <= max_speed.")
        self.speed_sv = GaussMarkov(float(slowly_varying["time_constant"]),
                                    float(slowly_varying["std"]), rng,
                                    min_speed - self.mean_wind_speed,
                                    max_speed - self.mean_wind_speed)

        # The spectrum takes U10, not the mean speed at the height z.
        gust = wind["gust"]
        self.gust_components = gust_components(
            self.mean_speed_10m, reference_height,
            float(gust["f_min"]), float(gust["f_max"]), int(gust["n_components"]), rng,
        )

        direction = wind["direction_variation"]
        self.direction_rate = float(direction["rate_deg_per_s"])
        self.direction_sv = GaussMarkov(float(direction["time_constant"]),
                                        float(direction["std_deg"]), rng)

        waves = config["waves"]
        self.hs = float(waves["hs"])
        self.tp = float(waves["tp"])
        self.waves_from_deg = float(waves["direction_deg"])

        # The current is constant, so the outputs are set once here.
        current = config["current"]
        self.current_speed = float(current["speed"])
        self.current_direction_deg = float(current["direction_deg"]) % 360.0
        # The velocity points the opposite way of where the current comes from.
        self.current_north = -self.current_speed * cos(radians(self.current_direction_deg))
        self.current_east = -self.current_speed * sin(radians(self.current_direction_deg))

        # Calculate the outputs once already now, so they are correct from
        # time 0 and not 0.0 until the first step.
        self.do_step(0.0, 0.0)

    def do_step(self, current_time, step_size):
        # Outputs at the end of the step.
        t = current_time + step_size

        # Wind speed parts. The total is calculated where it is used.
        self.slowly_varying_wind_speed = self.speed_sv.step(step_size)
        self.gust_wind_speed = sum((amplitude * cos(omega * t + phase)
                                    for omega, amplitude, phase in self.gust_components), 0.0)

        # Wind direction
        direction = (self.mean_wind_direction_deg + self.direction_rate * t
                     + self.direction_sv.step(step_size))
        self.wind_direction_deg = direction % 360.0

        self.H_s = self.hs
        self.T_p = self.tp
        self.wave_direction_deg = self.waves_from_deg
        return True
