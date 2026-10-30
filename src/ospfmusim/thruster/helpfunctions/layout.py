"""
Data containers for the thruster layout, and loading a layout from the TOML config file.

Coordinate system (Fossen body frame)
-------------------------------------
- Body axes: right-handed, x forward, y to starboard, z down. The origin is
  the vessel model's body-frame origin; agree on it with the vessel model.
- Forces are positive pushing the vessel forward (surge) and to starboard
  (sway). The yaw moment N is positive clockwise seen from above (bow to
  starboard). A force (fx, fy) at (x, y) gives N = x*fy - y*fx.
- Thrust angles: 0 deg pushes forward, increasing toward starboard, so
  90 deg pushes to starboard and 270 deg to port. This is atan2(fy, fx).

Public functions take angles in degrees.

License: MIT (see LICENSE in the repository root)
Authors: William Boland, Olve Grønås Birkeland (2026)
"""
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

# "azimuth" covers anything that can turn its thrust in any direction (pods too).
KINDS = ("azimuth", "tunnel", "shaft_line")


@dataclass(frozen=True)
class Thruster:
    """One thruster. SI units, except the power, which is in kW."""

    name: str  # identification, e.g. "BT1"
    kind: str  # "azimuth", "tunnel" or "shaft_line"
    x: float  # position [m]
    y: float
    # Largest thrust [N]: forward for azimuths and shaft lines, to starboard
    # (positive y) for tunnels.
    max_thrust: float
    # Largest thrust the other way [N]: astern for shaft lines, to port for
    # tunnels. None: the same as max_thrust. Azimuths turn round instead of
    # reversing, so they don't use it.
    max_reverse_thrust: float | None = None
    # Forbidden thrust directions: (start, end) thrust angles [deg], each zone
    # going from start toward starboard to end, e.g. ((80, 100),).
    forbidden_zones: tuple[tuple[float, float], ...] = ()
    # Propeller diameter [m]; only needed for the flushing sectors.
    diameter: float | None = None
    # Power at max_thrust [kW]; needed for power output and bus limits.
    power_kw: float | None = None
    # Where the thruster takes its power from: (PowerSource name, share)
    # pairs with the shares summing to 1, e.g. (("SWBD 1", 1.0),). Only
    # needed when the allocation has power sources.
    power_supply: tuple[tuple[str, float], ...] = ()
    # Dynamics (dynamics.py). None: the thruster follows its command at once.
    thrust_time_constant: float | None = None  # first-order lag of the thrust [s]
    max_azimuth_rate: float | None = None  # fastest an azimuth turns [deg/s]
    # Open-water propeller curves, only used by the ThrusterDynamicsKtKq FMU
    # (thruster_ktkq/helpfunctions/dynamics_ktkq.py): (J, KT, KQ) rows, J
    # increasing and covering J = 0, interpolated linearly. Two rows give the
    # linear model KT = KT0 - alpha_T1*J of Sørensen (2018) (8.10). Needs
    # diameter. Empty: thrust follows the command, power P_B * r^1.5.
    open_water: tuple[tuple[float, float, float], ...] = ()
    wake_fraction: float = 0.0  # w in Va = (1 - w) * speed through the water (8.12)
    # Ducted propeller series, only used by the ThrusterDynamicsDucted FMU
    # (thruster_ducted/helpfunctions/dynamics_ducted.py): KT and KQ of the
    # Wageningen Ka 4-70 propeller in nozzle 19A, with P/D chosen so the unit
    # absorbs power_kw at rated_rpm in bollard pull. Needs diameter and
    # power_kw. None: open_water as in ThrusterDynamicsKtKq.
    rated_rpm: float | None = None
    c_t: float = 1.0  # scale on the series KT, fitted to bollard pull or thrust data
    c_q: float = 1.0  # scale on the series KQ, fitted to logged shaft power

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind must be one of {KINDS}, got {self.kind!r}")
        if not self.max_thrust > 0:
            raise ValueError(f"{self.name}: max_thrust must be positive, got {self.max_thrust}")
        if self.max_reverse_thrust is not None and self.max_reverse_thrust < 0:
            raise ValueError(f"{self.name}: max_reverse_thrust must not be negative, "
                             f"got {self.max_reverse_thrust}")
        for field in ("diameter", "power_kw", "thrust_time_constant", "max_azimuth_rate",
                      "rated_rpm", "c_t", "c_q"):
            value = getattr(self, field)
            if value is not None and not value > 0:
                raise ValueError(f"{self.name}: {field} must be positive, got {value}")
        if self.rated_rpm is not None and (self.diameter is None or self.power_kw is None):
            raise ValueError(f"{self.name}: rated_rpm needs diameter and power_kw")
        if not 0 <= self.wake_fraction < 1:
            raise ValueError(f"{self.name}: wake_fraction must be in [0, 1), "
                             f"got {self.wake_fraction}")
        if self.open_water:
            if self.diameter is None:
                raise ValueError(f"{self.name}: open_water needs diameter")
            if len(self.open_water) < 2 or any(len(row) != 3 for row in self.open_water):
                raise ValueError(f"{self.name}: open_water needs at least 2 [J, KT, KQ] rows")
            j = [row[0] for row in self.open_water]
            if any(b <= a for a, b in zip(j, j[1:])):
                raise ValueError(f"{self.name}: open_water J must increase, got {j}")
            if not j[0] <= 0 <= j[-1] or not self.kt0 > 0:
                raise ValueError(f"{self.name}: open_water must cover J = 0 with KT > 0")

    @property
    def kt0(self):
        """KT at J = 0 (bollard pull) from open_water; None without open_water."""
        if not self.open_water:
            return None
        j, kt, _ = zip(*self.open_water)
        return float(np.interp(0.0, j, kt))

    @property
    def reverse_thrust(self):
        """Largest thrust the other way [N]: max_reverse_thrust, or max_thrust if that is None."""
        return self.max_thrust if self.max_reverse_thrust is None else self.max_reverse_thrust


@dataclass(frozen=True)
class PowerSource:
    """A switchboard (bus) or prime mover that supplies thrusters.

    Switchboards joined by a closed bus-tie are one source.
    """

    name: str  # e.g. "SWBD 1"
    available_kw: float  # power available [kW]
    # Share of available_kw kept for other consumers. DNV-ST-0111 [3.12.3]
    # uses 0.10 for switchboards.
    reserve_fraction: float = 0.0


@dataclass(frozen=True)
class Layout:
    """Everything the allocation needs to know about the vessel's thrusters."""

    thrusters: tuple[Thruster, ...]
    power_sources: tuple[PowerSource, ...] | None = None  # None: no power limit
    # Add the DNV-ST-0111 [3.11.3] flushing sectors to the forbidden zones.
    flushing_sectors: bool = False
    # When the command is out of reach: cost per N of error in X and Y and
    # per Nm of error in N (allocation.ThrustAllocator). A larger N weight
    # gives up sway or surge before heading.
    tau_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # False: the allocation is switched off and every thrust command is 0.
    enabled: bool = True

    def __post_init__(self):
        if len(self.tau_weights) != 3 or min(self.tau_weights) < 0:
            raise ValueError("tau_weights must be three numbers >= 0 (X, Y, N), "
                             f"got {self.tau_weights}")


def _tuples(value):
    """TOML arrays as (nested) tuples, like the dataclass defaults."""
    return tuple(map(_tuples, value)) if isinstance(value, list) else value


def _build(cls, data, where):
    """cls(**data), with lists turned into tuples and unknown or missing keys reported."""
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    try:
        return cls(**{key: _tuples(value) for key, value in data.items()})
    except TypeError as error:
        raise ValueError(f"{where}: {error}") from None


def load_layout(path):
    """
    Read a thruster layout from the TOML config file (config.toml):

        [allocation]
        enabled = true
        flushing_sectors = true

        [[power_sources]]
        name = "SWBD 1"
        available_kw = 3600.0
        reserve_fraction = 0.1

        [[thrusters]]
        name = "AZ1"
        kind = "azimuth"
        x = -40.0
        ...

    The keys of each thruster and power source are the fields of Thruster
    and PowerSource; the keys of [allocation] are the other fields of
    Layout. [allocation] and [[power_sources]] are optional. Other sections
    belong to other readers of the file and are ignored.

    Returns
    -------
    Layout
    """
    # "utf-8-sig" copes with the file being saved with a BOM (Notepad, PowerShell).
    data = tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not data.get("thrusters"):
        raise ValueError(f"{path}: no [[thrusters]]")
    thrusters = tuple(_build(Thruster, t, f"{path}, thruster {k}")
                      for k, t in enumerate(data["thrusters"]))
    names = [t.name for t in thrusters]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: thruster names must be different, got {names}")
    sources = data.get("power_sources")
    if sources is not None:
        sources = tuple(_build(PowerSource, s, f"{path}, power source {k}")
                        for k, s in enumerate(sources))
    allocation = data.get("allocation", {})
    unknown = set(allocation) - {"enabled", "flushing_sectors", "tau_weights"}
    if unknown:
        raise ValueError(f"{path}: unknown keys in [allocation]: {sorted(unknown)}")
    for key in ("enabled", "flushing_sectors"):
        if not isinstance(allocation.get(key, False), bool):
            raise ValueError(f"{path}: {key} must be true or false, "
                             f"got {allocation[key]!r}")
    try:
        settings = {key: _tuples(value) for key, value in allocation.items()}
        return Layout(thrusters, sources, **settings)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None
