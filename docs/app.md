# Results app

A small web app (Streamlit + Plotly) for running the simulation and reading the results. You can edit
the config values, start a new run, and look at any earlier run. It runs locally in the browser;
nothing is sent anywhere.

## Start it

From the repo folder, with the `master` environment active:

```powershell
conda activate master
pip install -r requirements.txt     # once: includes streamlit and plotly
pip install -e .                    # once: makes src/ importable
python -m app.run                   # the same as: streamlit run src/app/build/app.py
```

The app opens at http://localhost:8501. Stop it with Ctrl+C in the terminal.

## Runs

Every run gets its own folder, `results/<YYYYMMDD_HHMMSS>[_name]/`, with one CSV per FMU and a copy of
the config it was run with (`config.toml`). Nothing is overwritten, so earlier runs stay available.
This is the same whether the run is started from the app or with `python -m ospfmusim.run`.

The run's copy of the config is what the FMUs read: `src/ospfmusim/run.py` points `config_path` on every
FMU to it, so `OspSystemStructure.xml` is not changed. Relative file paths in the copy (such as
`[vessel_fmu] data_file`) are made absolute, because the copy lies in another folder.

## New run tab

1. **Start from:** a config file in `configs/` (the default is the one `OspSystemStructure.xml` points to),
   or the config of an earlier run (to repeat or vary it).
2. **Edit the values.** One input per value, grouped as in the config file: simulation, wind, waves,
   current, vessel, DP controller, Kalman filter and thrust allocation. Hover the ⓘ to see the comment
   from the config file. Lists with up to three numbers get one input per number; longer lists are
   comma-separated.
3. **Run name** (optional) is added to the folder name.
4. **Run simulation.** The app writes a copy of the config with your changes, runs
   `python -m ospfmusim.run` with it, and selects the new run in the sidebar. The output of the script is
   shown at the top of the tab.

**Save as default** writes the values in the form into the file chosen in **Save as default in** (a file in
`configs/`), keeping
the comments. Only the changed lines are touched.

Not editable in the app: the multi-line vessel matrices (`mass_matrix`, `damping_matrix`), the
`[blendermann]` table, the `[[power_sources]]` and `[[thrusters]]` layout, and file paths. Edit those
in the config file in `configs/`.

## Result tabs

| Tab | Content |
|---|---|
| **Overview** | Key numbers (position and heading error, power, utilisation, time saturated, time turn-limited, mean wind), the footprint (north vs east, total and LF, with the set-point), a summary of the run's config, and an animated footprint: a time slider (and ▶ Play) moves through the run and shows the track so far, earlier headings, the vessel's position and heading now, and arrows for wind, waves and current with their values at that time (see below) |
| **Vessel** | North, east and heading (total and LF), the WF motion, the LF velocity, and the loads per DOF: wind, wave drift and thrusters |
| **Environment** | Wind speed (total and each part), wind, wave and current directions, current components, sea state |
| **Kalman filter** | Measured total, LF estimate and true LF motion; WF estimate and true WF motion; the bias estimate |
| **Control and thrusters** | Position and heading error, commanded vs delivered force, thrust, azimuth and power per thruster (named from `[[thrusters]]`), utilisation, saturation and turn limit |
| **Thrusters** | The vessel seen from above (bow up, starboard right). At each thruster a pointed arrow for its thrust on the vessel (the way it pushes, length = thrust), a ring at its `max_thrust`, a dotted outline for the command from the allocation, and red wedges for the directions it may not push in (its `forbidden_zones`, and the DNV-ST-0111 flushing sectors with `[allocation] flushing_sectors = true`, drawn with every thruster working). From the middle the sum of the thrusters (X, Y), delivered and commanded by the controller; all arrows have the same scale. A box gives thrust, angle and power per thruster, the sum with N, the load on each switchboard, and the saturated / turn-limited flags. Beside it thrust, angle (-180 to 180°), power and, with `[[power_sources]]`, the switchboard load in % of usable power (available minus reserve) over time, with a line at the slider time. The load comes from the power the thrusters report; the allocation keeps its own estimate (P_B·r^1.5) under the limit, so with ThrusterDynamicsKtKq the two can differ. A time slider (and ▶ Play) moves through the run. The hull outline is drawn from `[vessel] loa` with the Gunnerus beam/length ratio. Needs an internet connection, like the animated footprint |
| **Data** | Any signals from any FMUs, with units, in one plot; table and download as CSV of the chosen signals. The units come from the column names (`UNITS` in `data.py`), since the FMUs do not write them |

**Sidebar**

- **Run.** Every run folder in `results/`, newest first. CSV files directly in `results/` (from before
  run folders, or from cosim CLI) show up as "results/ (loose files)".
- **Other folder.** Show a run that lies somewhere else.
- **Reload.** Reads the files again (also happens by itself when a CSV file changes).

**The animated footprint.** The time slider and ▶ Play run in the browser, so moving through the run
is quick. *Number of heading arrows* sets how many earlier headings are drawn, *Time steps on the
slider* how many steps the slider has (more = smoother, but a slower page). Wind, waves and current
start on the side they come from and point the way they go; the length shows the size (full length at
20 m/s wind, 5 m H_s, 1 m/s current, and at least a quarter of that so small values still show their
direction), and the label gives the value. This plot loads plotly.js from its CDN, so it needs an
internet connection.

Angles are shown in degrees, also where the FMU uses radians (vessel `yaw`, filter `psi`).

## Files

| File | What it does |
|---|---|
| `src/app/build/app.py` | The Streamlit app: sidebar and tabs |
| `src/app/build/footprint.py` | The animated footprint in the Overview tab |
| `src/app/build/thruster_view.py` | The animated thruster view in the Thrusters tab |
| `src/app/build/editor.py` | The New run tab: the inputs, running the simulation, saving as default |
| `src/app/build/data.py` | Reads the run folders, the result CSVs and the config, key numbers. Plain Python and pandas, no Streamlit |
| `src/app/build/config_tools.py` | Changes values in the config as text, so the comments are kept. Also used by `src/ospfmusim/run.py` |

## Known limitations

- A run blocks the app until it is done. Long runs (several minutes of simulated time) take a while.
- Runs are not compared side by side yet; open two browser tabs and pick a run in each.
- Runs from before run folders (loose files in `results/`) have no config copy, so their summary uses
  the default config in `configs/` as it is now.
- Old run folders are never deleted by the app. Delete them in the file explorer when they are not
  needed.
