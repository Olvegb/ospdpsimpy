"""Quick check of spectra.py.

Run from the repo folder:  python src/ospfmusim/waves/tests/check_spectra.py
"""
import numpy as np

from ospfmusim.waves.helpfunctions.spectra import jonswap, pierson_moskowitz

hs = 3.0   # significant wave height [m]
tp = 9.0   # peak period [s]

omega = np.linspace(0.01, 5.0, 5000)          # frequencies [rad/s]
S = pierson_moskowitz(omega, hs, tp)

print("CHECK FOR PIERSON-MOSKOWITZ:")

m0 = np.trapezoid(S, omega)                   # area under the spectrum
print(f"Area under spectrum m0 = {m0:.4f}")
print(f"Expected Hs^2/16       = {hs**2 / 16:.4f}")
print(f"Hs back from spectrum  = {4 * np.sqrt(m0):.3f} m  (should be {hs} m)")
print(f"Peak at omega          = {omega[np.argmax(S)]:.3f} rad/s  "
      f"(should be 2*pi/Tp = {2 * np.pi / tp:.3f})")

# -- JONSWAP ---------------------------------------------------------------------

hs = 3.0   # significant wave height [m]
tp = 9.0   # peak period [s]
omega = np.linspace(0.01, 5.0, 5000)          # frequencies [rad/s]

print('CHECK FOR JONSWAP:')
for gamma in (1.0, 3.3, 5.0):
    S = jonswap(omega, hs, tp, gamma)
    m0 = np.trapezoid(S, omega)
    print(f"gamma = {gamma:3.1f}:  Hs back = {4 * np.sqrt(m0):.3f} m,  "
          f"peak value S_max = {S.max():.3f}")

S_pm = pierson_moskowitz(omega, hs, tp)
S_j1 = jonswap(omega, hs, tp, 1.0)
print(f"gamma = 1 equals Pierson-Moskowitz: {np.allclose(S_pm, S_j1)}")
