"""
Thruster power at part thrust, and how the power sources share it.

Copied from the DNV-ST-0111 project (dp_capability/models/power.py). The
fixed 10% switchboard reserve of [3.12.3] became PowerSource.reserve_fraction.

License: MIT (see LICENSE in the repository root)
Authors: William Boland, Olve Grønås Birkeland (2026)
"""
import numpy as np


def thruster_power_kw(thruster, thrust_fraction):
    """
    Power a thruster uses at a fraction r of its largest thrust:

        P = P_B * r^(3/2)

    From the bollard pull relation T ~ (D * P)^(2/3) turned around (the
    nominal thrust formula of DNV-ST-0111 [3.9.2]), so the largest thrust
    uses exactly P_B = Thruster.power_kw.

    Parameters
    ----------
    thruster : layout.Thruster
        Needs power_kw.
    thrust_fraction : float or array-like
        r, 0 for no thrust and 1 for the largest thrust in that direction.

    Returns
    -------
    numpy.ndarray
        Power [kW], the shape of thrust_fraction.
    """
    if thruster.power_kw is None:
        raise ValueError(f"{thruster.name}: no power_kw")
    return thruster.power_kw * np.asarray(thrust_fraction, dtype=float) ** 1.5


def usable_power_kw(source):
    """Power a source can give the thrusters [kW]: what is available, minus the reserve."""
    return source.available_kw * (1 - source.reserve_fraction)


def supply_matrix(thrusters, sources):
    """
    Share of each thruster's power taken from each source (DNV-ST-0111
    Table A-5).

    Parameters
    ----------
    thrusters : sequence of layout.Thruster
        Each with power_supply = ((source name, share), ...), shares summing to 1.
    sources : sequence of layout.PowerSource

    Returns
    -------
    numpy.ndarray
        Shares, shape (sources, thrusters).
    """
    index = {s.name: j for j, s in enumerate(sources)}
    if len(index) != len(sources):
        raise ValueError("power sources must have different names")
    shares = np.zeros((len(sources), len(thrusters)))
    for i, t in enumerate(thrusters):
        if not t.power_supply:
            raise ValueError(f"{t.name}: no power_supply, but the allocation has power sources")
        for name, share in t.power_supply:
            if name not in index:
                raise ValueError(f"{t.name}: unknown power source {name!r}")
            if share < 0:
                raise ValueError(f"{t.name}: negative share {share} of {name!r}")
            shares[index[name], i] += share
        if not np.isclose(shares[:, i].sum(), 1.0):
            raise ValueError(f"{t.name}: power supply shares sum to {shares[:, i].sum()}, not 1")
    return shares
