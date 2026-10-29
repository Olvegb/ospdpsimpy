"""The thruster view of the results app: arrows point the way the thrust pushes the vessel,
forbidden zones are drawn where they are, and the switchboard load adds up.

License: MIT (see LICENSE in the repository root)
"""
import base64
import math

import numpy as np
import pandas as pd
import pytest

from app.build.thruster_view import animated_thrusters
from ospfmusim.thruster.helpfunctions.layout import Layout, PowerSource, Thruster

SUPPLY = (("SWBD 1", 1.0),)
LAYOUT = Layout(
    thrusters=(Thruster("AZ1", "azimuth", x=-10.0, y=-2.0, max_thrust=50e3, power_kw=500.0,
                        power_supply=SUPPLY, forbidden_zones=((80.0, 100.0),)),
               Thruster("BT1", "tunnel", x=10.0, y=0.0, max_thrust=20e3, power_kw=200.0,
                        power_supply=SUPPLY)),
    power_sources=(PowerSource("SWBD 1", available_kw=1000.0, reserve_fraction=0.5),))


def test_thruster_view():
    # Row 0: AZ1 pushes to starboard at full thrust, BT1 to port, sum to starboard. Row 1: nothing.
    index = pd.Index([0.0, 1.0], name="Time")
    dynamics = pd.DataFrame({
        "thruster1_thrust": [50e3, 0.0], "thruster1_azimuth_deg": [90.0, 90.0],
        "thruster1_power_kw": [500.0, 0.0],
        "thruster1_thrust_command": [50e3, 0.0], "thruster1_azimuth_command_deg": [90.0, 90.0],
        "thruster2_thrust": [-20e3, 0.0], "thruster2_azimuth_deg": [90.0, 90.0],
        "thruster2_power_kw": [200.0, 0.0],
        "thruster2_thrust_command": [-20e3, 0.0], "thruster2_azimuth_command_deg": [90.0, 90.0],
        "X": [0.0, 0.0], "Y": [30e3, 0.0], "N": [-100e3, 0.0]}, index=index)
    allocation = pd.DataFrame({"tau_X": [0.0, 0.0], "tau_Y": [30e3, 0.0], "tau_N": [0.0, 0.0],
                               "saturated": [True, False]}, index=index)
    figure = animated_thrusters(dynamics, LAYOUT, loa=36.0, n_frames=2, allocation=allocation)
    first, last = figure["frames"]
    static = {trace["name"]: trace for trace in figure["data"]}

    def tip(frame, name):
        """The arrow tip (screen x, screen y) of the trace called name, None if not drawn."""
        traces = {figure["data"][k]["name"]: d for k, d in zip(frame["traces"], frame["data"])}
        x, y = traces[name]["x"], traces[name]["y"]
        return (x[3], y[3]) if len(x) > 1 else None

    def box(frame):
        return frame["layout"]["annotations"][0]["text"].replace(chr(0xA0), " ")

    ring = 0.3 * 36.0                                    # the arrow at the largest max_thrust
    assert tip(first, "AZ1 thrust") == pytest.approx((-2.0 + ring, -10.0))   # at the ring
    assert tip(first, "BT1 thrust") == pytest.approx((-ring * 20 / 50, 10.0))  # to port
    assert tip(first, "Sum of thrusters") == pytest.approx((ring * 30 / 50, 0.0))
    assert "500 kW" in box(first) and "SATURATED" in box(first)
    assert tip(last, "AZ1 thrust") is None and tip(last, "Sum of thrusters") is None
    assert "SATURATED" not in box(last)

    # The 80-100 deg zone of AZ1: a wedge out to its ring, between those angles.
    zone = static["Forbidden zones"]
    for x, y in zip(zone["x"], zone["y"]):
        if x is not None and (x, y) != (-2.0, -10.0):
            assert math.hypot(x + 2.0, y + 10.0) == pytest.approx(ring)
            assert 80.0 - 1e-9 <= math.degrees(math.atan2(x + 2.0, y + 10.0)) <= 100.0 + 1e-9

    # 500 + 200 kW from a switchboard that can give 1000 * (1 - 0.5) = 500 kW.
    assert "700 / 500 kW  140%" in box(first)
    line = static["SWBD 1"]["y"]                      # plotly writes numpy arrays as base64
    assert np.frombuffer(base64.b64decode(line["bdata"]), line["dtype"])[0] == pytest.approx(140.0)
