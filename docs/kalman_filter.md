# Extended Kalman filter FMU

One FMU (FMI 2.0 co-simulation, built with pythonfmu 0.7.0, run in OSP) that estimates the vessel's
low-frequency (LF) motion, wave-frequency (WF) motion and slowly varying disturbances from position and
heading measurements. It is a **3-DOF DP observer** (surge, sway, yaw) with a wave model, after Fossen.
Owner: Olve.

```mermaid
flowchart LR
    config["[vessel], [waves] tp and [kalman_filter]<br>in configs/*.toml"] --> ExtendedKalmanFilter
    VesselBlock -->|"north, east, yaw (LF + WF)"| ExtendedKalmanFilter
    VesselBlock -->|"wind_X, wind_Y, wind_N"| ExtendedKalmanFilter
    ThrusterDynamics -->|"X, Y, N"| ExtendedKalmanFilter
    ExtendedKalmanFilter -->|"north_hat ... r_hat (LF)"| DPController
    ExtendedKalmanFilter -->|"b_x_hat, b_y_hat, b_n_hat"| logging
    ExtendedKalmanFilter -->|"wf_north_hat, wf_east_hat, wf_psi_hat"| logging
```

The point of the WF states: the measurements contain the total motion, but the controller should only
act on the LF part. Without the wave model, the oscillating wave motion would go straight into the
estimates and make the thrusters chase every wave.

## 1. Inputs, outputs, parameter

**Parameter**

| Name | Description |
|---|---|
| `config_path` | Path to `configs/*.toml`, set in `OspSystemStructure.xml` |

**Inputs**

| Name | Unit | From | Description |
|---|---|---|---|
| `north_meas`, `east_meas` | m | VesselBlock (`north`, `east`) | Measured total position (LF + WF), NED (GNSS) |
| `psi_meas` | rad | VesselBlock (`yaw`) | Measured total heading (gyro) |
| `tau_thr_x`, `tau_thr_y` | N | ThrusterDynamics (`X`, `Y`) | Delivered thruster force, body frame |
| `tau_thr_n` | Nm | ThrusterDynamics (`N`) | Delivered thruster yaw moment |
| `tau_wind_x`, `tau_wind_y` | N | VesselBlock (`wind_X`, `wind_Y`) | Wind force (feed-forward), body frame |
| `tau_wind_n` | Nm | VesselBlock (`wind_N`) | Wind yaw moment (feed-forward) |

**Outputs**

| Name | Unit | To | Description |
|---|---|---|---|
| `north_hat`, `east_hat` | m | DPController | Estimated **LF** position, NED |
| `psi_hat` | rad | DPController | Estimated **LF** heading, −π to π |
| `u_hat`, `v_hat` | m/s | DPController | Estimated LF velocity, body frame |
| `r_hat` | rad/s | DPController | Estimated LF yaw rate |
| `b_x_hat`, `b_y_hat` | N | logging | Estimated bias force, **NED** |
| `b_n_hat` | Nm | logging | Estimated bias yaw moment |
| `wf_north_hat`, `wf_east_hat` | m | logging | Estimated WF position, NED |
| `wf_psi_hat` | rad | logging | Estimated WF heading |

The LF and WF estimates add up to the estimated total motion.

## 2. Model

Fifteen states:

```
x   = [η, ν, b, ξ₁, ξ₂]
η   = [north, east, ψ]      LF position and heading, NED          [m, m, rad]
ν   = [u, v, r]             LF velocity, body frame               [m/s, m/s, rad/s]
b   = [b_x, b_y, b_n]       bias, NED                             [N, N, Nm]
ξ₁, ξ₂                      WF states, one pair per DOF; η_w = ξ₂ [m, m, rad]
```

The bias b collects every force the model does not know about: current, wave drift, model errors.

**Process model** (Fossen's DP observer with a wave model):

```
η̇  = R(ψ) ν
ν̇  = M⁻¹ (τ_thr + τ_wind + R(ψ)ᵀ b − D ν)
ḃ  = −T_b⁻¹ b                                   (+ process noise)
ξ̇₁ = ξ₂
ξ̇₂ = −Ω² ξ₁ − 2 Λ Ω ξ₂                           (+ process noise)
```

M and D are the 3-DOF `mass_matrix` and `damping_matrix` from `[vessel]`, shared with the DP controller.

**The WF model** is a linear second-order system per DOF, a narrow-band approximation of the wave
spectrum. Ω = diag(ω₀) is the peak frequency, taken from the sea state: ω₀ = 2π / T_p with T_p from
`[waves] tp`. At zero speed (DP) the encounter frequency equals the wave frequency. Λ = diag(λ) is the
relative damping, which sets how wide the band is.

The noise on ξ₂ is not tuned directly. It follows from the expected WF motion σ (`wave_motion_std`):
with noise intensity q, the stationary variances are var(ξ₂) = q / (4λω₀) and var(ξ₁) = var(ξ₂) / ω₀², so

```
q = 4 λ ω₀ σ²
```

The WF states start in this stationary distribution.

**Measurement model:** GNSS position and gyro heading measure the total motion.

```
y = η + η_w = H x,    H = [I₃ 0 0 0 I₃]
```

**Each time step** (continuous-discrete EKF):

1. **Predict.** Integrate the state and the covariance from t to t + dt with **RK4** (fourth-order
   Runge-Kutta):

   ```
   ẋ = f(x, τ),    Ṗ = F P + P Fᵀ + Q,    F = ∂f/∂x
   ```

   τ is held constant over the step (zero-order hold, like every FMU input).
2. **Correct.** Kalman gain K = P Hᵀ (H P Hᵀ + R)⁻¹, then x ← x + K (y − H x). The covariance is updated
   in Joseph form, which keeps P symmetric and positive definite.

**Heading wrap.** ψ and the heading innovation are wrapped to [−π, π), so 359° against 1° is a 2° error,
not 358°.

## 3. `configs/*.toml`

| Key in `[kalman_filter]` | Description |
|---|---|
| `bias_time_constants` | T_b for [b_x, b_y, b_n] [s]. Large = the bias changes slowly |
| `process_noise` | Diagonal of Q for the first 9 states (η, ν, b), per second. Larger = trust the model less |
| `measurement_noise` | Diagonal of R = std². Default: GNSS 0.5 m, gyro 0.5° |
| `initial_covariance` | Diagonal of P₀ for the first 9 states. Default: the bias starts unknown (std 1e5 N, 1e6 Nm) |
| `wave_damping_ratio` | λ for [north, east, yaw]. Typically 0.05–0.2. Lower = narrower band around ω₀ |
| `wave_motion_std` | Expected std of the WF motion σ for [north, east, yaw] [m, m, rad] |

Plus `mass_matrix` and `damping_matrix` in `[vessel]`, and `tp` in `[waves]`.
**Changing `configs/*.toml` needs no rebuild.**

**Tuning in short.**
- The bias process noise (the last three entries of `process_noise`) sets how fast the bias follows a
  new disturbance. Larger = faster, but noisier estimates.
- `wave_motion_std` should match the real WF motion. Too large: the filter calls slow LF drift "waves".
  Too small: wave motion leaks into the LF estimate.
- The ratio between Q and R sets how much the filter trusts the measurements over the model.

## 4. Files

| File | What it does |
|---|---|
| `src/ospfmusim/observer/ekf/ekf.py` | The FMU, everything in one file |
| `tests/test_kalman_filter.py` | A simple 3-DOF "true" vessel with a constant disturbance and made-up WF motion, in plain Python without OSP: the LF and WF estimates follow the truth, the bias converges, and the heading wraps cleanly at ±π |

Run from the repo folder:

```
python -m pytest tests/test_kalman_filter.py
```

Result with the default config (last 300 s of 600 s):

| | North | East | Yaw |
|---|---|---|---|
| RMS WF motion (truth) | 0.32 m | 0.48 m | 0.0064 rad |
| RMS error, LF estimate | 0.027 m | 0.038 m | 0.0008 rad |
| RMS error, WF estimate | 0.044 m | 0.064 m | 0.0009 rad |

The bias estimate ends within 4 % of the true disturbance.

## 5. Known limitations

- **The WF motion of the current vessel data is almost zero.** The synthetic data file gives WF motion
  of about 0.1 mm, while `wave_motion_std` defaults to 0.5 m. In the full simulation the filter then
  explains some of the small LF residuals as waves (`wf_*_hat` with a std of a few mm). Set
  `wave_motion_std` from the real Gunnerus RAOs and sea state when they are in.
- The same ω₀ is used in all three DOF, and it is fixed during the run. A changing sea state or forward
  speed (encounter frequency) is not followed.
- The LF vessel model is linear and 3-DOF, with the **placeholder** matrices in `[vessel]`. It is not the
  6-DOF model inside VesselBlock. Update the matrices when the Gunnerus data is in.
- The measurements are perfect (no noise added). R only tells the filter how much to trust them.
- The bias is in NED, so with a constant heading it maps directly to a constant current or drift force.
