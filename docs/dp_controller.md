# DP controller FMU

One FMU (FMI 2.0 co-simulation, built with pythonfmu 0.7.0, run in OSP) that holds the vessel at a
set-point. It is a **simple PID**, there to close the loop so the whole simulator can run. Owner: Viljar.

```mermaid
flowchart LR
    config["[controller] and [vessel]<br>in configs/*.toml"] --> DPController
    kalman_filter -->|"north_hat, east_hat, psi_hat, u_hat, v_hat, r_hat"| DPController
    DPController -->|"tau_X_P ... tau_N_D"| ThrustAllocation
```

## 1. Inputs, outputs, parameter

**Parameter**

| Name | Description |
|---|---|
| `config_path` | Path to `configs/*.toml`, set in `OspSystemStructure.xml` |

**Inputs** (from the ExtendedKalmanFilter FMU)

| Name | Unit | Description |
|---|---|---|
| `north_hat`, `east_hat` | m | Estimated position, NED |
| `psi_hat` | rad | Estimated heading, clockwise from north |
| `u_hat`, `v_hat` | m/s | Estimated velocity, body frame |
| `r_hat` | rad/s | Estimated yaw rate |

**Outputs**

| Name | Unit | To | Description |
|---|---|---|---|
| `tau_X_P`, `tau_X_I`, `tau_X_D` | N | ThrustAllocation | P, I and D term of the commanded surge force, body frame (x forward) |
| `tau_Y_P`, `tau_Y_I`, `tau_Y_D` | N | ThrustAllocation | P, I and D term of the commanded sway force (y to starboard) |
| `tau_N_P`, `tau_N_I`, `tau_N_D` | Nm | ThrustAllocation | P, I and D term of the commanded yaw moment |
| `error_north`, `error_east` | m | logging | Position minus set-point, NED |
| `error_heading_deg` | deg | logging | Heading minus set-point, −180 to 180 |

## 2. Control law

PID with velocity feedback (Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control*,
ch. 12). The error is taken in NED and rotated into the body frame, because tau is in the body frame:

```
e_ned = [north_hat − north_ref,  east_hat − east_ref,  wrap(psi_hat − psi_ref)]
e     = R(psi_hat)ᵀ e_ned                              error in the body frame
z    += e dt                                           integral
tau   = −Kp e − Ki z − Kd [u_hat, v_hat, r_hat]
      =  tau_P  + tau_I  + tau_D
```

The three terms are sent out one by one for each degree of freedom (`tau_<dof>_<term>`), so they can be
logged and plotted. ThrustAllocation adds them up to `tau_X`, `tau_Y`, `tau_N`.

- **D term on the estimated velocity**, so nothing is differentiated.
- **wrap()** keeps the heading error between −180° and 180° (359° → −1°).
- **Anti-windup:** the integral stops growing when |Ki z| reaches `integral_force_limit`.
- The integral is in the body frame. That is fine for station keeping (constant heading), but not exact
  during large heading changes.

**Gains.** Calculated from the diagonals of `mass_matrix` M and `damping_matrix` D in `[vessel]`, so each
degree of freedom behaves like a mass-spring-damper with natural frequency ωₙ and damping ratio ζ:

```
Kp = M ωₙ²,    Kd = 2 ζ ωₙ M − D   (at least 0),    Ki = Kp ωₙ / 10
```

The gains therefore follow the vessel when the matrices in `[vessel]` change. With the current
placeholder OSV matrices, ωₙ = 0.05 rad/s and ζ = 0.9:

| | Surge | Sway | Yaw |
|---|---|---|---|
| Kp | 1.65e4 N/m | 2.25e4 N/m | 1.13e7 Nm/rad |
| Ki | 82.5 N/(m s) | 112.5 N/(m s) | 5.63e4 Nm/(rad s) |
| Kd | 5.28e5 N s/m | 6.30e5 N s/m | 3.15e8 Nm s/rad |

## 3. `configs/*.toml`

| Key in `[controller]` | Description |
|---|---|
| `north_ref`, `east_ref` | Set-point [m] |
| `heading_ref_deg` | Set-point heading, clockwise from north [deg] |
| `natural_frequency` | ωₙ for [surge, sway, yaw] [rad/s]. Higher = faster and harder. 0.05 ≈ a period of 2 min |
| `damping_ratio` | ζ for [surge, sway, yaw]. 1 = critically damped, lower = more overshoot |
| `integral_force_limit` | Largest force from the I term [N, N, Nm] (anti-windup) |

Plus `mass_matrix` and `damping_matrix` in `[vessel]`, shared with the Kalman filter and the hull FMU.
**Changing `configs/*.toml` needs no rebuild.**

## 4. Files

| File | What it does |
|---|---|
| `src/ospfmusim/controller/dp_controller.py` | The FMU, everything in one file (like `ekf.py`) |
| `tests/test_dp_controller.py` | Signs: the force pushes back towards the set-point, also with another heading; wrap of the heading error; D term; anti-windup |
| `tests/test_closed_loop.py` | A very simple 3-DOF vessel (`M ν̇ + D ν = τ + Rᵀ b`) in plain Python, without OSP or a hull FMU: the vessel returns to the set-point and holds it against a constant force |

Run from the repo folder (works on Mac):

```
python -m pytest tests/test_dp_controller.py tests/test_closed_loop.py
```

## 5. Known limitations

- Until the hull FMU is connected, the Kalman filter gets no measurements, so the estimates stay at 0
  and the controller only acts if the set-point is not at the origin.
- No wind feed-forward and no use of the bias estimate (`b_*_hat`): the I term takes care of slowly
  varying forces. Both are easy to add later.
- No reference model: a new set-point gives a step in the error, and so a large force at once.
  The thrust allocation saturates it.
- The gains are tuned for the placeholder matrices. Check them again with the Gunnerus matrices.
