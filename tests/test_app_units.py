"""Units the results app gives the result columns, for names where the order of data.UNITS matters.

License: MIT (see LICENSE in the repository root)
"""
from app.build.data import label, unit


def test_units():
    expected = {"current_north": "m/s", "error_north": "m", "N": "Nm", "tau_N_P": "Nm",
                "tau_X_P": "N", "b_n_hat": "Nm", "tau_thr_x": "N", "yaw": "rad", "r_hat": "rad/s",
                "wave3_omega": "rad/s", "H_s": "m", "total_power_kw": "kW",
                "error_heading_deg": "deg", "unknown_thing": ""}
    assert {name: unit(name) for name in expected} == expected
    assert label("vessel.north") == "vessel.north [m]"
    assert label("vessel.unknown_thing") == "vessel.unknown_thing"
