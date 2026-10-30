# ospdpsimpy – DP co-simulation in OSP where each FMU reads the config file itself

The Python package is `ospfmusim` (in `src/ospfmusim/`). A complete DP (dynamic positioning) simulator in OSP, with one closed loop of seven FMUs:

- **Environment** gives the wind (mean, slowly varying and gust speed, slowly varying direction), the sea state (H_s, T_p, wave direction) and the current, all in the earth-fixed NED frame. See [docs/environment_fmu.md](docs/environment_fmu.md).
- **Waves** turns the sea state into 100 regular wave components (amplitude, frequency, phase, direction). Regular or irregular (long- or short-crested) sea. See [docs/waves_fmu.md](docs/waves_fmu.md).
- **VesselBlock** is the 6-DOF hull: low-frequency (LF) and wave-frequency (WF) motion, with wind loads (Blendermann), wave loads and current calculated inside. See [docs/vessel_fmu.md](docs/vessel_fmu.md).
- **ExtendedKalmanFilter** splits the measured position and heading into low-frequency (LF) and wave-frequency (WF) motion, and estimates the LF velocity and slowly varying disturbances (bias). The controller only gets the LF part. See [docs/kalman_filter.md](docs/kalman_filter.md).
- **DPController** is a simple PID that holds the vessel at the set-point in `configs/*.toml`. See [docs/dp_controller.md](docs/dp_controller.md).
- **ThrustAllocation** and **ThrusterDynamics** turn the commanded force τ = (X, Y, N) into thrust and azimuth commands per thruster, and then into the force the thrusters actually give the vessel. See [docs/thruster_fmus.md](docs/thruster_fmus.md).

The results can be read in a small web app, see [Results app](#results-app).

Every constant is in one config file in `configs/`; `src/ospfmusim/OspSystemStructure.xml` says which one is used by default (`configs/gunnerus.toml`). Each FMU reads that file itself when the simulation starts, so you can change constants and run again **without rebuilding the FMUs**.

| Config | Vessel |
|---|---|
| `configs/gunnerus.toml` (the default) | RV Gunnerus: wind geometry, and 3-DOF matrices copied from the Gunnerus `.mat` file. |
| `configs/tests.toml` | Wind geometry and 3-DOF matrices in `[vessel]` are **placeholder values** for an 88 m OSV. The hull hydrodynamics come from the Gunnerus `.mat` file. |

> **The vessel data file is not in the repository.** Both configs read the hydrodynamic data from `data_examples/GunnerusL35_20201112.mat` (`[vessel_fmu] mat_file`). Put the file there before you run. `src/ospfmusim/hull/helpfunctions/example_test_config.json` is **synthetic** test data, used only by `tests/test_vessel_fmu.py`.

> **Known issue:** with both configs the vessel drifts away after about 25 s, and the run stops at t = 41.5 s with `tau_wind must have shape (6,) and finite values` (the vessel state has become infinite). Runs of 10 s, as below, work.

## Documentation

| Doc | FMU |
|---|---|
| [docs/environment_fmu.md](docs/environment_fmu.md) | Environment: wind, sea state and current models, `configs/*.toml` keys |
| [docs/waves_fmu.md](docs/waves_fmu.md) | Waves: inputs and outputs, `configs/*.toml` keys, the wave elevation the hull must use |
| [docs/vessel_fmu.md](docs/vessel_fmu.md) | VesselBlock: the hull model and its data file |
| [docs/kalman_filter.md](docs/kalman_filter.md) | ExtendedKalmanFilter: LF + WF states, process and measurement model, RK4, tuning |
| [docs/dp_controller.md](docs/dp_controller.md) | DPController: the control law, gains, `configs/*.toml` keys |
| [docs/thruster_fmus.md](docs/thruster_fmus.md) | ThrustAllocation and ThrusterDynamics: allocation, dynamics, power, `configs/*.toml` keys |
| [docs/app.md](docs/app.md) | The results app: editing the config, running new simulations, and the result plots |

## Getting started

You need Python 3.11 or newer (the conda environment `master` has 3.12) and the `.mat` file in `data_examples/` (see above). Run from the "ospdpsimpy" folder in the VS Code terminal:

```powershell
conda activate master
pip install -e ".[test]"          # once: installs requirements.txt and pytest, and makes ospfmusim and app in src/ importable
python -m ospfmusim.run           # builds the FMUs that have changed, runs, saves to results/<time stamp>/
python -m pytest                  # runs the tests in tests/
```

`scripts/run_simulation_example.py` is a short example script that runs the simulation and reads the results as pandas DataFrames.

The same from Python, with the `.mat` file, the config and the system structure as arguments:

```python
from ospfmusim.run import run_simulation

run_dir = run_simulation("data_examples/GunnerusL35_20201112.mat", "configs/tests.toml",
                         "src/ospfmusim/OspSystemStructure.xml", duration=10.0)
```

Expected output of `python -m ospfmusim.run --config configs/tests.toml --duration 10`:

```
Vessel:  offshore_supply_vessel, simulated for 10.0 s
Wind:    14.6 m/s from 97.3 deg (NED, at the end)
Current: 0.5 m/s towards 0 deg (NED)

Vessel at the end of the simulation:
  north          =        3.734 m
  east           =       -2.801 m
  yaw            =        0.352 rad
  wind_X         =      -5727.9 N
  wind_Y         =     -79687.1 N
  wind_N         =   -1314770.0 Nm

Kalman filter estimates at the end of the simulation:
  north_hat =        2.929 m
  ...

Thrusters at the end of the simulation (utilisation 0.509, saturated False, turn-limited False):
  X              =      38251.0 N
  Y              =      14047.5 N
  N              =    2601440.0 Nm
  total_power_kw =        713.2 kW
```

The wind comes from the east, so it pushes the vessel west (east < 0) and gives a force to port (wind_Y < 0). The current pushes it north. The thrusters push back.

Every run gets its own folder, `results/<YYYYMMDD_HHMMSS>[_name]/`, with one CSV per FMU and a copy of the
config it was run with. Earlier runs are never overwritten.

**Try it** (no rebuild needed, only `python -m ospfmusim.run` again, or use the [results app](#results-app)):
- Set `duration = 20.0` in `[simulation]` and look at `vessel_*.csv` in the new run folder (longer runs stop at 41.5 s, see the known issue above).
- Switch the gusts off with `n_components = 0` in `[wind.gust]`, or the current off with `speed = 0.0` in `[current]`.
- Change `north_ref` in `[controller]` and watch the vessel move to the new set-point.

## Results app

`src/app/build/` is a small web app (Streamlit + Plotly) for running the simulation and reading the results.
In the **New run** tab you edit the config values (wind, waves, current, controller, filter, ...) and
start a run; the result tabs show the vessel's footprint and motion, the environment, the Kalman filter
estimates against the truth, and the controller and thrusters, for any run. Start it with:

```powershell
python -m app.run          # the same as: streamlit run src/app/build/app.py
```

See [docs/app.md](docs/app.md).

## Commands

All commands are run from this folder, with the `master` environment active (`conda activate master`).

| What | Command |
|---|---|
| Install the libraries and make `src/` importable (once) | `pip install -e ".[test]"` (leave out `[test]` to skip pytest and fmpy) |
| Install only the libraries | `pip install -r requirements.txt` |
| Install scipy with OpenBLAS (if missing) | `conda install -n master scipy "libblas=*=*openblas"` |
| Run the example script | `python scripts/run_simulation_example.py` |
| Run the simulation (Python) | `python -m ospfmusim.run` (options: `--config other.toml`, `--mat other.mat`, `--system other.xml`, `--duration 60`, `--name high_wind`, `--output folder`) |
| Build the FMUs that have changed | `python -m ospfmusim.build` (`--force` builds every FMU) |
| Open the results app | `python -m app.run` (or `python src/app/run.py`, or `streamlit run src/app/build/app.py`) |
| Run the simulation (cosim CLI) | `cosim run src/ospfmusim/OspSystemStructure.xml -d 10 --output-dir results` (build the FMUs first) |
| Run every test (about 40 s) | `python -m pytest` |
| Run the tests of one part | `python -m pytest tests/test_waves_fmu.py` (one file per part, see `tests/`) |
| Run the tests whose name matches | `python -m pytest -k anti_windup` |

**The FMUs are built for you.** `run_simulation()` (and `python -m ospfmusim.run`) first calls `build_fmus()`, which builds an FMU again only when a file in the model folders of `src/ospfmusim` is newer than it. Changes to `configs/*.toml` and `OspSystemStructure.xml` need no rebuild.

Every FMU gets the same copy of the model folders of `src/ospfmusim` in its `resources/`, and imports its help functions through the package, for example `from ospfmusim.waves.helpfunctions.sea_state import ...`. The FMUs share one Python interpreter when they run, so the three `helpfunctions/` folders must be reached through their full package name. See `src/ospfmusim/build.py`.

## The files

| File | What it does | When do you change it? |
|---|---|---|
| `configs/*.toml` | **Every constant**, one file per set-up: environment, vessel, controller, filter, thrusters, simulation length | When you want to change a value. No rebuild. |
| `src/ospfmusim/OspSystemStructure.xml` | Which FMUs are included, how they are connected, step size, and **where the config file is** | When you add FMUs or connections |
| `src/ospfmusim/enviroment/environment.py` | Environment FMU. Wind, sea state and current from `configs/*.toml`, NED | When the logic changes. Rebuilt by itself. |
| `src/ospfmusim/waves/waves.py` | Waves FMU. Thin wrapper around `waves/helpfunctions/` | When the logic changes. Rebuilt by itself. |
| `src/ospfmusim/hull/vessel.py` | VesselBlock FMU. Thin wrapper around `hull/helpfunctions/` | When the logic changes. Rebuilt by itself. |
| `src/ospfmusim/observer/ekf/ekf.py` | ExtendedKalmanFilter FMU, everything in one file | When the logic changes. Rebuilt by itself. |
| `src/ospfmusim/controller/dp_controller.py` | DPController FMU, everything in one file | When the logic changes. Rebuilt by itself. |
| `src/ospfmusim/thruster/thrust_allocation.py`, `src/ospfmusim/thruster/thruster_dynamics.py` | ThrustAllocation and ThrusterDynamics FMUs. Thin wrappers around `thruster/helpfunctions/` | When the logic changes. Rebuilt by itself. |
| `src/ospfmusim/thruster_ktkq/` | ThrusterDynamicsKtKq FMU: ThrusterDynamics with the thrust from KT/KQ propeller curves, to swap in and compare. See [docs/thruster_fmus.md](docs/thruster_fmus.md) §4.1 | When the logic changes. Rebuilt by itself. |
| `src/ospfmusim/thruster_ducted/` | ThrusterDynamicsDucted FMU: ThrusterDynamicsKtKq with the curves from the Ka 4-70 / nozzle 19A ducted series, set by `rated_rpm`, `c_t` and `c_q`. See [docs/thruster_fmus.md](docs/thruster_fmus.md) §4.2 | When the logic changes. Rebuilt by itself. |
| `src/ospfmusim/*/helpfunctions/` | The wave, vessel and thruster logic. Built into every FMU's `resources/` | When the logic changes. Rebuilt by itself. |
| `tests/` | The pytest tests: one `test_*.py` per part (environment, waves, vessel, Kalman filter, controller, thrusters). Most run the FMU classes directly in Python, without OSP; `test_vessel_fmu.py` runs the built VesselBlock FMU | When that logic changes |
| `src/ospfmusim/build.py` | `build_fmus()`: builds the FMUs the system structure uses into `src/ospfmusim/fmus/`, only those that have changed | When an FMU is added |
| `src/ospfmusim/run.py` | `run_simulation(mat_file, config_file, system_structure)`: builds the FMUs, runs the simulation in a new run folder in `results/`. Also the command line (`python -m ospfmusim.run`), which prints the result | When you want other output |
| `src/app/run.py` | Starts the results app (`python -m app.run`) | Rarely |
| `src/app/build/` | The results app (`streamlit run src/app/build/app.py`), and `config_tools.py`, which changes config values as text (also used by `run.py`) | When you want other plots |
| `docs/` | One document per FMU | When an FMU changes |
| `data_examples/` | The Gunnerus hydrodynamic data (`.mat`). Not in git: put the file here yourself | When the vessel data changes |
| `scripts/` | Example scripts, such as `run_simulation_example.py` | When you want another example |
| `pyproject.toml` | The package: name, license, `pip install -e .` (the dependencies are read from `requirements.txt`), the test libraries (`[test]`) and the pytest settings | Rarely |
| `requirements.txt` | The Python libraries the simulation needs, with the versions allowed | When a library is added |
| `src/ospfmusim/fmus/` | The built FMUs | Created by `build.py` |
| `results/` | One folder per run: one CSV file per FMU with every input and output for each time step, and the run's `config.toml` and `OspSystemStructure.xml` | Created by `run.py` |

## How it fits together

```mermaid
flowchart TD
    environment -->|"wind (mean, slowly varying, gust, direction), current"| vessel
    environment -->|"H_s, T_p, wave_direction"| waves
    waves -->|"wave#lt;k#gt;_* (k = 1..100)"| vessel
    dynamics -->|"X, Y, N (delivered thrust, body)"| vessel
    dynamics -->|"tau_thr_x/y/n"| kalman_filter
    vessel -->|"north, east, yaw"| kalman_filter
    vessel -->|"wind_X/Y/N to tau_wind_x/y/n"| kalman_filter
    kalman_filter -->|"north_hat ... r_hat"| dp_controller
    dp_controller -->|"tau_X/Y/N_P, _I, _D (summed in allocation)"| allocation
    allocation -->|"thrust + azimuth command (8 thruster slots)"| dynamics
```

Step by step, when you run `run_simulation()` or `python -m ospfmusim.run`:

1. **The FMUs are built** (only those whose model code has changed), into `src/ospfmusim/fmus/`, where `OspSystemStructure.xml` looks for them.
2. **The run folder is made.** `run.py` makes `results/<time stamp>/` and saves a copy of the config there (as `config.toml`), with the `.mat` file and the duration written into it, and a copy of `OspSystemStructure.xml`. On the command line the config is the one `OspSystemStructure.xml` points to, or the one given with `--config`.
3. **The system is loaded.** libcosim reads `OspSystemStructure.xml` and loads every FMU. `config_path` on each FMU is pointed to the run's copy of the config.
4. **The FMUs read the config file.** Right before the simulation starts, each FMU runs `exit_initialization_mode()` and takes what it needs from the config (see the table at the top of the config file).
5. **The simulation runs.** Every time step (0.1 s), each FMU runs `do_step()`, and libcosim passes the outputs to the inputs through the `<VariableConnection>`s.
6. **The results are saved** in the run folder (one CSV per FMU). `run_simulation()` returns the run folder, and `python -m ospfmusim.run` prints the last values.

### Frames and angle conventions

- **NED** (north, east, down) for the environment and the vessel position. **Body frame** (x forward, y to starboard, z down) for forces, velocities and thruster positions.
- Angles in `configs/*.toml` and on the environment are in **degrees, clockwise from north**. The vessel's `yaw` and the filter's `psi` are in **radians**.

| Angle | 0° means | 90° means |
|---|---|---|
| Wind direction (`[wind] direction_deg`) | wind comes **from** north | wind comes from east |
| Wave direction (`[waves] direction_deg`) | waves come **from** north | waves come from east |
| Current direction (`[current] direction_deg`) | current flows **towards** north | current flows towards east |
| Heading (vessel `yaw`) | bow points north | bow points east |

The relative wind (wind seen from the vessel) is calculated inside VesselBlock, from the wind direction, the heading and the vessel's own velocity.

## Key concepts in the FMU

In every FMU the variables fall into four kinds. Examples from `src/ospfmusim/enviroment/environment.py`:

| Kind | Example | Where does the value come from? |
|---|---|---|
| **Parameter** | `config_path` | Set once before the start, here from the XML |
| **Input** | (VesselBlock: `mean_wind_speed`) | Can change every time step. Comes from another FMU through a connection, or a fixed value from the XML |
| **Constant** | `mean_wind_direction_deg`, the gust components | Read from `configs/*.toml`. Plain Python variables inside the FMU, not FMI variables. |
| **Output** | `mean_wind_speed`, `wind_direction_deg` | Calculated in `do_step()` and can be connected to other FMUs |

**Why isn't the config file read in `__init__`?** `__init__` also runs when the FMU is *built* (PythonFMU creates an instance to find the variables), and `config_path` is not set at that point. So the file is read in `exit_initialization_mode()`, which only runs when the simulation actually starts.

## Adding more

**A new constant in an existing FMU:**
1. Add it to `configs/*.toml`, in the section that FMU reads.
2. Read it in `exit_initialization_mode()` and store it as `self.name`.
3. Use it in `do_step()`.
4. Run again. The FMU is rebuilt by itself, because the `.py` file changed.

**A new FMU that uses the same config:**
1. Write the model in a folder in `src/ospfmusim/` the same way: a `config_path` parameter, and read the file in `exit_initialization_mode()`. Import help functions through the package (`from ospfmusim.<folder>.helpfunctions.<module> import ...`).
2. Add it to `FMUS` in `src/ospfmusim/build.py` (the class name and the script).
3. Add it to `src/ospfmusim/OspSystemStructure.xml` with `source="fmus/<class name>.fmu"` and its own `<InitialValue variable="config_path">`, and connect inputs and outputs with `<VariableConnection>`.
4. Add a document in `docs/` and a line in the tables above.

## Good to know

- **Relative paths.** `run_simulation()` reads relative paths from the folder Python is started from, and points `config_path` on every FMU to the run's copy of the config, so the FMUs do not depend on that folder. The FMU paths in the XML (`source="fmus/..."`) are relative to the XML file. With cosim CLI you must stand in the repo folder, because `config_path = "configs/tests.toml"` in the XML is read from the folder the simulation is started from (cosim CLI uses the config in the XML directly and writes loose CSV files to the output folder). With OSP-GUI it is safest to write a full path, such as `C:\Users\...\ospdpsimpy\configs\tests.toml`, in the XML.
- **File paths inside a config** (such as `[vessel_fmu] mat_file`) are relative to the config file, so data in the repo folder is written as `../data_examples/...`. `run_simulation()` replaces `mat_file` with the `.mat` file it is given.
- **An error in an FMU at the start stops Python.** If an FMU raises an exception in `exit_initialization_mode()` (for example a missing key in the config), libcosim prints `Fatal py exception encountered` with the Python error, and the process ends with exit code 127, without a Python traceback. An error later in the run (in `do_step()`) gives the same message, and then `run_simulation()` raises a `RuntimeError`.
- **Save the config file as UTF-8.** VS Code does this automatically.
- **Python must be available when the FMU runs.** FMUs made with PythonFMU use the Python on the machine. So run with the `master` environment activated, also from the CLI. That Python needs the libraries in `requirements.txt`, including **scipy** for ThrustAllocation.
- **numpy must use OpenBLAS, not MKL.** On the NTNU virtual PC, `numpy.linalg` with MKL crashes Python without an error message. Fix: `conda install -n master "libblas=*=*openblas"`.
- **Filenames on Windows.** Git cannot check out a file whose name ends with a dot (such as `README.md.`). If `git pull` fails with `invalid path`, rename the file on GitHub.
- **The CSV files have 6 significant digits**, for example `1.60439e+06`. That is how libcosim writes them.

## Code style

The code follows [PEP 8](https://peps.python.org/pep-0008/), with lines up to 99 characters (PEP 8 allows this). Two exceptions on purpose: matrices and vectors keep the symbols from the literature (`M`, `D`, `R`, `P`, `K`, ...), and the module names in `hull/helpfunctions/` (`LF.py`, `WaveLoads.py`, ...) are kept. To check:

```powershell
pip install pycodestyle
pycodestyle --max-line-length=99 src scripts tests
```

## License

MIT, see [LICENSE](LICENSE). The hull calculation modules in `src/ospfmusim/hull/helpfunctions/` are based on the NTNU mcsimpy toolbox and are under GPL-3.0-or-later; see [NOTICE.md](src/ospfmusim/hull/helpfunctions/NOTICE.md) and the [LICENSE](src/ospfmusim/hull/helpfunctions/LICENSE) there.
