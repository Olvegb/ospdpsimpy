# Thruster FMUs: thrust allocation and thruster dynamics

Two FMUs (FMI 2.0 co-simulation, built with pythonfmu 0.7.0, run in OSP) that turn the DP
controller's force command into the force the thrusters actually put on the vessel. Owner: William.

```mermaid
flowchart LR
    controller["DP controller"] -->|"P, I, D terms of tau"| ThrustAllocation
    ThrustAllocation -->|"thrust + azimuth command<br>per thruster (8 slots)"| ThrusterDynamics
    ThrusterDynamics -->|"X, Y, N"| vessel["vessel model"]
    failure["failure injection (optional)"] -.->|"thrusterK_available"| ThrustAllocation
    failure -.->|"thrusterK_available"| ThrusterDynamics
```

- **ThrustAllocation** decides how hard each thruster pushes, and in which direction, so that together
  they give the commanded force τ = (X, Y, N).
- **ThrusterDynamics** makes each thruster follow its command the way a real one would: thrust builds
  up with a time lag, and azimuths turn at a limited rate. Its outputs are the total force and moment
  the thrusters put on the vessel, and their power.

## 1. What you need to run them

| File | What it is |
|---|---|
| `ThrustAllocation.fmu`, `ThrusterDynamics.fmu` | The two FMUs (win64 and linux64 binaries). Not in git: built into `src/ospfmusim/fmus/` by `python -m ospfmusim.build` (and by `python -m ospfmusim.run`) |
| `configs/*.toml` | The thruster layout. Both FMUs read it themselves at start-up |

- **Python must be installed.** pythonfmu FMUs do not bundle Python; they use the Python on the
  machine. That Python needs to be **3.11 with numpy and scipy** (tested with numpy 2.1.0 and
  scipy 1.14.1).
  - Run from Python with libcosimpy (tested with 0.0.6): the Python running the script is used, so
    nothing else is needed.
  - cosim CLI or OSP-GUI: not tested. The Python 3.11 DLL must be on PATH.
- **`config_path`** is a parameter on both FMUs. A relative path such as `configs/tests.toml` is read from
  the folder the simulation is started from. With OSP-GUI, use a full path.
- **Changing `configs/*.toml` needs no rebuild.** The FMUs read it every time the simulation starts.
  Only a change to the Python code needs a rebuild, which `python -m ospfmusim.run` does by itself (or `python -m ospfmusim.build`).

## 2. Frame and signs

Everything at the FMU boundary is in the **Fossen body frame**:

- x forward, y to **starboard**, z down;
- forces are positive forward (X) and to starboard (Y);
- the yaw moment N is positive clockwise seen from above (bow to starboard). A force (fx, fy) at
  (x, y) gives `N = x·fy − y·fx`;
- thrust angles are `atan2(fy, fx)` in degrees, in [0°, 360°): 0° pushes forward, 90° to starboard,
  180° aft, 270° to port.

Units are SI (N, Nm, m, s), with power in kW and angles in degrees. Variable names that hold an angle
end in `_deg`.

The thruster positions (x, y) in `configs/*.toml` are measured from the vessel model's body origin.
**This origin must be the same one the vessel model uses for N** (still to agree, §9).

`X`, `Y` and `N` from ThrusterDynamics are the thrust loads on the vessel, with the same names and
signs as Olve's WindLoads, so the vessel model can add them to the other loads.

## 3. ThrustAllocation FMU

Every time step it:

1. Reads τ and which thrusters are available.
2. Finds the thruster forces that give τ with the **lowest utilisation** u, where u is the highest
   load on any thruster as a fraction of its limit (thrust or bus power). Among the solutions with
   that u, it picks the one with the least total thrust, so thrusters don't push against each other.
   Both are linear programs (scipy HiGHS).
   - Each azimuth's thrust circle is approximated by a 36-sided polygon. Tunnels push only along y,
     shaft lines only along x.
   - Forbidden zones are respected: each thruster's own, plus optionally the DNV-ST-0111 [3.11.3]
     flushing sectors, where one thruster's race would hit another.
   - With power sources in the config, the thrusters on each switchboard stay within its usable
     power.
3. If τ is more than the thrusters can give (u > 1), it gives the **closest reachable τ** instead,
   with the errors in X, Y and N weighted by `tau_weights`, and sets `saturated`.
4. An azimuth that isn't needed keeps its last angle, so it doesn't turn for nothing.
5. **Rate limits.** Each azimuth turns toward the angle from steps 2–3 by at most
   `max_azimuth_rate · dt` (dt = the FMU step size), the shortest way round, as ThrusterDynamics does.
   While one can't get there in this step, every azimuth is held at its new angle (thrust along it
   only, 0 to `max_thrust`) and steps 2–3 run again. The other thrusters then make up what they can,
   and `saturated` shows when they can't. Every command can be followed by ThrusterDynamics, so with
   no thrust lag the delivered force is the allocated one.

| | Name | Type | Unit | Meaning |
|---|---|---|---|---|
| parameter | `config_path` | String | – | Path to `configs/*.toml` |
| input | `tau_X_P`, `tau_X_I`, `tau_X_D` | Real | N | P, I and D term of the commanded surge force, + forward |
| input | `tau_Y_P`, `tau_Y_I`, `tau_Y_D` | Real | N | P, I and D term of the commanded sway force, + to starboard |
| input | `tau_N_P`, `tau_N_I`, `tau_N_D` | Real | Nm | P, I and D term of the commanded yaw moment, + bow to starboard |
| input | `thrusterK_available` | Boolean | – | False: thruster K is not used. Default true |
| output | `thrusterK_thrust_command` | Real | N | Thrust along `thrusterK_azimuth_command_deg`: ≥ 0 for azimuths, signed for tunnels and shaft lines |
| output | `thrusterK_azimuth_command_deg` | Real | deg | Direction positive thrust pushes: the commanded angle for azimuths, always 90 for tunnels and 0 for shaft lines |
| output | `tau_X`, `tau_Y`, `tau_N` | Real | N, N, Nm | τ = P + I + D, the commanded force that is allocated |
| output | `utilisation` | Real | – | What τ needs as a fraction of what the thrusters have. Above 1 when saturated; `inf` when τ can't be given in any amount (e.g. tunnels only against a surge command) |
| output | `saturated` | Boolean | – | True when τ was out of reach and the commands give the closest reachable τ. Out of reach includes azimuths that haven't turned far enough yet |

K = 1 … 8 (§6). With these definitions every kind of thruster pushes with
`f = thrust · (cos α, sin α)`. A tunnel with a negative thrust pushes to port, a shaft line with a
negative thrust pushes astern.

## 4. ThrusterDynamics FMU

Every time step of length dt it moves each thruster toward its command:

- **Thrust: first-order lag.** `dT/dt = (T_cmd − T) / τ_T`, solved exactly over the step, so the
  result doesn't depend on the step size. T_cmd is first clipped to the thruster's limits (azimuths
  to [0, `max_thrust`]). τ_T is `thrust_time_constant`.
- **Azimuth: rate limit.** The azimuth turns along the shortest arc (350° → 10° goes through 0°) by at
  most `max_azimuth_rate · dt`. It may pass through a forbidden zone on the way. If it is still short
  of the command, the thrust command is scaled by `max(cos(angle left to turn), 0)`, so it never
  pushes more than 90° away from the command. The rate-aware allocation (§3, step 5) never sends an
  angle it can't reach, so this only matters when the two FMUs disagree (other step sizes or
  turn rates). Tunnels and shaft lines ignore the angle command.
- **Trip.** `thrusterK_available = false` sets the thrust to 0 at once, and the azimuth stops turning.
- **Forces.** `fx = T cos α`, `fy = T sin α` per thruster, and `X = Σ fx`, `Y = Σ fy`,
  `N = Σ (x·fy − y·fx)`.
- **Power.** `P = P_B · r^1.5`, with P_B = `power_kw` and r = |T| / the thruster's limit in that
  direction.

The thrusters start at rest: zero thrust, azimuths pointing ahead (0°), tunnels at 90°.

| | Name | Type | Unit | Meaning |
|---|---|---|---|---|
| parameter | `config_path` | String | – | Path to `configs/*.toml` (the same file) |
| input | `thrusterK_thrust_command` | Real | N | From ThrustAllocation |
| input | `thrusterK_azimuth_command_deg` | Real | deg | From ThrustAllocation |
| input | `thrusterK_available` | Boolean | – | False trips thruster K. Default true |
| output | `thrusterK_thrust` | Real | N | Actual thrust, same sign convention as the command |
| output | `thrusterK_azimuth_deg` | Real | deg | Actual direction, in [0°, 360°) |
| output | `thrusterK_power_kw` | Real | kW | Power of thruster K (0 if it has no `power_kw`) |
| output | `X` | Real | N | Total thruster force on the vessel, surge |
| output | `Y` | Real | N | Total thruster force on the vessel, sway |
| output | `N` | Real | Nm | Total thruster yaw moment on the vessel |
| output | `total_power_kw` | Real | kW | Sum of the thrusters' power |

## 5. Connecting them in `OspSystemStructure.xml`

Simulator and variable names on the other FMUs (`dp`, `vessel`, …) are placeholders.

```xml
<Simulator name="allocation" source="fmus/ThrustAllocation.fmu">
    <InitialValues>
        <InitialValue variable="config_path"><String value="configs/tests.toml"/></InitialValue>
    </InitialValues>
</Simulator>
<Simulator name="dynamics" source="fmus/ThrusterDynamics.fmu">
    <InitialValues>
        <InitialValue variable="config_path"><String value="configs/tests.toml"/></InitialValue>
    </InitialValues>
</Simulator>
```

**DP controller → allocation** (nine connections, one per degree of freedom and term):

```xml
<VariableConnection>
    <Variable simulator="dp" name="tau_X_P"/>
    <Variable simulator="allocation" name="tau_X_P"/>
</VariableConnection>
<!-- the same for tau_X_I, tau_X_D, tau_Y_P ... tau_N_D -->
```

**Allocation → dynamics**: 16 connections, thrust and angle for each of the 8 slots. Connect all 8
even if the layout has fewer thrusters; empty slots carry 0.

```xml
<VariableConnection>
    <Variable simulator="allocation" name="thruster1_thrust_command"/>
    <Variable simulator="dynamics" name="thruster1_thrust_command"/>
</VariableConnection>
<VariableConnection>
    <Variable simulator="allocation" name="thruster1_azimuth_command_deg"/>
    <Variable simulator="dynamics" name="thruster1_azimuth_command_deg"/>
</VariableConnection>
<!-- the same for thruster2 … thruster8 -->
```

**Dynamics → vessel model** (three connections):

```xml
<VariableConnection>
    <Variable simulator="dynamics" name="X"/>
    <Variable simulator="vessel" name="X_thrusters"/>
</VariableConnection>
<!-- the same for Y and N -->
```

**Failures (optional).** If nothing is connected, every thruster is available. To trip thruster K,
connect the same Boolean signal to **both** `allocation.thrusterK_available` and
`dynamics.thrusterK_available`. The dynamics FMU then drops its thrust at once, and the allocation
FMU shares its load among the others. Connected only to the dynamics FMU, the allocation keeps
counting on the tripped thruster and the force falls short.

`OspSystemStructure.xml` in this repo already has the two simulators and the 16 slot connections,
with τ as fixed InitialValues for testing. `dynamics.X/Y/N` is not connected yet: it goes to an FMU
another stakeholder is working on.

## 6. `configs/*.toml`

The FMUs read three sections and ignore every other one, so one config file can be shared with the
other FMUs (Olve's `[wind]`, `[vessel]`, …). Inside these three sections, an unknown key is an error
at start-up.

**The order of `[[thrusters]]` is the slot number**: the first entry is `thruster1_…`, the second
`thruster2_…`, and so on. At most 8 thrusters.

```toml
[allocation]                  # optional
flushing_sectors = true       # add the DNV-ST-0111 [3.11.3] flushing sectors (needs diameter)
tau_weights = [1.0, 1.0, 1.0] # cost per N of X and Y error, per Nm of N error, when saturated

[[power_sources]]             # optional; leave out every one for no power limit
name = "SWBD 1"
available_kw = 3600.0
reserve_fraction = 0.1        # share kept for other consumers

[[thrusters]]
name = "AZ1"
kind = "azimuth"              # "azimuth" (pods too), "tunnel" or "shaft_line"
x = -40.0                     # [m] from the body origin, + forward
y = -5.5                      # [m], + to starboard
max_thrust = 377500.0         # [N]
diameter = 3.0                # [m]
power_kw = 2000.0             # [kW] at max_thrust
power_supply = [["SWBD 1", 1.0]]
thrust_time_constant = 5.0    # [s]
max_azimuth_rate = 10.0       # [deg/s]
```

| Thruster key | Needed | Meaning |
|---|---|---|
| `name` | yes | Unique name |
| `kind` | yes | `azimuth`, `tunnel` or `shaft_line` |
| `x`, `y` | yes | Position [m] |
| `max_thrust` | yes | [N]. Forward for azimuths and shaft lines, to starboard for tunnels |
| `max_reverse_thrust` | no | [N]. Astern (shaft line) or to port (tunnel). Default: `max_thrust`. Azimuths turn round instead |
| `forbidden_zones` | no | List of `[start, end]` thrust angles [deg], each going toward starboard from start to end |
| `diameter` | with `flushing_sectors` | Propeller diameter [m] |
| `power_kw` | with `[[power_sources]]` | Power at `max_thrust` [kW]. Without it the power outputs are 0 |
| `power_supply` | with `[[power_sources]]` | `[[source name, share], …]`, shares summing to 1 |
| `thrust_time_constant` | no | Thrust lag [s]. Without it the thrust follows the command at once |
| `max_azimuth_rate` | no | [deg/s]. Without it the azimuth turns at once |

The thruster layout in `configs/tests.toml` is a made-up ~80 m PSV: two azimuths aft (AZ1 port, AZ2 starboard,
x = −40 m) and two bow tunnels (BT1, BT2), two 3600 kW switchboards with a 10% reserve. The time
constants (5 s azimuths, 3 s tunnels) and turn rates (10 °/s) are typical but made up.

## 7. Behaviour to be aware of

- **Turning azimuths.** The allocation doesn't read the actual angles back; it keeps its own and
  turns them as fast as ThrusterDynamics does. Both FMUs must therefore run with the same step size
  and read the same `max_azimuth_rate`. During a turn `saturated` can be true even though the
  thrusters are lightly loaded: the command is out of reach where the azimuths point now. The
  target angle comes from an allocation without rate limits and can jump when τ changes direction;
  the azimuths then turn toward the new one.
- **Commanded vs delivered.** The plots compare P + I + D with the delivered force. They differ
  by the thrust lag, and while `saturated` is true also by what the thrusters can't give.
- **Saturation.** When τ is out of reach, `saturated` is true and `utilisation` is above 1, or `inf`
  if τ can't be given in any amount. The commands then give the closest reachable τ. With equal
  `tau_weights`, the heading (N) is held first and Y or X gives way.
- **Switchboard power is fixed** from `configs/*.toml`; it is not an FMU input yet.
- **Step size.** Tested with fixed steps of 0.1 s. One allocation step takes about 6 ms on the
  4-thruster layout (10 ms when saturated). More thrusters with forbidden zones mean more linear
  programs per step. The dynamics are exact for any step size.
- **CSV output.** libcosim writes the Booleans (`saturated`, `thrusterK_available`) as 1/0.

## 8. Check that it runs

With `configs/tests.toml`, a constant τ = (0, 300 kN, 0), 60 s and steps of 0.1 s, the last
step should give:

| Thruster | Command [kN @ °] | Actual [kN @ °] | Power [kW] |
|---|---|---|---|
| AZ1 | 204.4 @ 22.6 | 204.4 @ 22.6 | 797.1 |
| AZ2 | 204.4 @ 157.4 | 204.4 @ 157.4 | 797.1 |
| BT1 | 73.8 @ 90.0 | 73.8 @ 90.0 | 360.3 |
| BT2 | 68.9 @ 90.0 | 68.9 @ 90.0 | 360.3 |

`utilisation` = 0.543, `saturated` = false, and (X, Y, N) ≈ (0, 300.0 kN, 26 Nm), total power
2314.7 kW. The azimuths push against each other in surge: their opposite surge forces cancel the
yaw moment from pushing sideways at x = −40 m.

## 9. Still to agree with the group

- **Variable names.** The names above are a proposal and can still change.
- **The body-frame origin** (CO or CG) for the thruster positions and N, and whether the vessel model
  wants the force in its own frame.
- **Who drives `thrusterK_available`**, and whether a power plant FMU will give the available power
  per switchboard (it could become an input).
- **The co-sim step size.** 0.1 s is tested and fast enough.
