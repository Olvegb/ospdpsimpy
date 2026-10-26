"""Check that turning azimuths get commands they can follow, and don't push the wrong way.
Run from the repo folder:  python src/ospfmusim/thruster/tests/check_thrusters.py

1. ThrusterDynamics alone: one azimuth at rest pointing ahead (0 deg) is told
   to push 100 kN astern (180 deg). At 10 deg/s it needs 18 s to turn.
2. ThrustAllocator + ThrusterDynamics, a Gunnerus-like layout without thrust
   lag, starting at rest with an astern command: the force the dynamics give
   must be the one the allocator says it commands, every step.
Exits with code 1 if a check fails.
"""
import sys

import numpy as np

from ospfmusim.thruster.helpfunctions.allocation import ThrustAllocator
from ospfmusim.thruster.helpfunctions.dynamics import ThrusterDynamics
from ospfmusim.thruster.helpfunctions.layout import Layout, Thruster

RATE = 10.0  # [deg/s]
failures = []


def check(text, ok):
    print(f"  [{'OK' if ok else 'FAIL'}] {text}")
    if not ok:
        failures.append(text)


print("THRUSTER DYNAMICS: a turning azimuth")
thrust = 100e3


def azimuth(**kwargs):
    return ThrusterDynamics(Layout((Thruster("AZ", "azimuth", 0.0, 0.0, 200e3, **kwargs),)))


model = azimuth(max_azimuth_rate=RATE)
states = [model.step([thrust], [180.0], dt=1.0) for _ in range(18)]
check("never pushes forward while turning", all(s.fx[0] <= 1e-6 for s in states))
check("no thrust while more than 90 deg off", all(s.thrust[0] < 1e-6 for s in states[:9]))
check("full thrust astern once turned", np.isclose(states[-1].azimuth_deg[0], 180.0)
      and np.isclose(states[-1].thrust[0], thrust))
state = azimuth().step([thrust], [180.0], dt=1.0)
check("without max_azimuth_rate: full thrust astern at once", np.isclose(state.fx[0], -thrust))

print("RATE-AWARE ALLOCATION: allocator and dynamics together")
layout = Layout((
    Thruster("AZ1", "azimuth", -16.95, 2.7, 50e3, max_azimuth_rate=RATE),
    Thruster("AZ2", "azimuth", -16.95, -2.7, 50e3, max_azimuth_rate=RATE),
    Thruster("BT1", "tunnel", 14.5, 0.0, 20e3),
))
tau = np.array([-30e3, 5e3, 20e3])
dt = 0.1
allocator, model = ThrustAllocator(layout), ThrusterDynamics(layout)
angle = np.array([0.0, 0.0, 90.0])
commands, states, turns = [], [], []
for _ in range(250):
    command = allocator.step(tau, dt=dt)
    turns.append(np.abs((command.azimuth_deg - angle + 180.0) % 360.0 - 180.0).max())
    angle = command.azimuth_deg
    commands.append(command)
    states.append(model.step(command.thrust, command.azimuth_deg, dt=dt))
check(f"the angle commands turn at most {RATE} deg/s", max(turns) <= RATE * dt + 1e-9)
check("saturated at the start: the azimuths can't push astern yet", commands[0].saturated)
check("the dynamics give the commanded force every step",
      all(np.allclose(s.tau, c.tau, atol=1.0) for s, c in zip(states, commands)))
check("after 25 s: tau reached and delivered", not commands[-1].saturated
      and np.allclose(states[-1].tau, tau, atol=1.0))

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("All thruster checks passed.")
