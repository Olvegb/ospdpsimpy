"""The "New run" tab of the results app: edit config values and run a new simulation.

The edited values are written into a copy of the config (config_tools.set_values,
which keeps the comments), and ospfmusim/run.py runs with that copy. The run
gets its own folder in results/ with the copy saved as config.toml.
The files in configs/ only change when "Save as default" is pressed.

License: MIT (see LICENSE in the repository root)
Authors: Olve Grønås Birkeland (2026)
"""
import re
import subprocess
import sys
import tempfile
import tomllib
from math import floor, log10
from pathlib import Path

import streamlit as st

import config_tools
import data

# The sections that can be edited, grouped as in the config file. Multi-line arrays
# (mass_matrix, damping_matrix), [blendermann] and the [[thrusters]] layout are
# edited in the config file itself.
GROUPS = [
    ("Simulation", ["simulation"]),
    ("Wind", ["environment", "wind", "wind.slowly_varying", "wind.gust",
              "wind.direction_variation"]),
    ("Waves", ["waves", "waves.spectrum", "waves.spreading", "waves.discretization"]),
    ("Current", ["current"]),
    ("Vessel", ["vessel"]),
    ("DP controller", ["controller"]),
    ("Kalman filter", ["kalman_filter"]),
    ("Thrust allocation", ["allocation"]),
]

# The parts that can be switched off with one toggle: (title, section, what off means).
# Each is enabled = true/false in its section.
SWITCHES = [
    ("Wind", "wind", "Off: no wind, all wind speeds are 0."),
    ("Waves", "waves", "Off: calm sea, H_s = 0."),
    ("Current", "current", "Off: no current, speed 0."),
    ("Thrust allocation", "allocation",
     "Off: every thrust command is 0, so the vessel drifts. The DP controller and the "
     "Kalman filter still run."),
]
# Configs without the key get it, switched on, so the toggle can be shown and saved.
SWITCH_DEFAULTS = {(section, "enabled"): True for _, section, _ in SWITCHES}

CHOICES = {
    ("waves", "sea_type"): ["regular", "irregular"],
    ("waves.spectrum", "type"): ["jonswap", "pierson_moskowitz"],
    ("waves.spreading", "type"): ["none", "cos2s"],
}


def lookup(config, section, key):
    table = config
    for part in section.split("."):
        table = table[part]
    return table[key]


def editable_keys(text):
    """[(section, key)] of the one-line values in the edited sections, in file order."""
    sections = {s for _, group in GROUPS for s in group}
    return [(section, key) for _, section, key, *_ in config_tools.lines_with_keys(text)
            if section in sections and not key.endswith("_file")]


def nice_step(value):
    """A step for a number input: a tenth of the order of magnitude."""
    value = abs(float(value))
    return 10.0 ** (floor(log10(value)) - 1) if value else 0.1


def widget(section, key, value, help_text, prefix, choices):
    """Draws one input and returns its value, with the same type as value."""
    label, wkey = key, f"{prefix}|{section}|{key}"
    if isinstance(value, bool):
        return st.checkbox(label, value=value, help=help_text, key=wkey)
    if isinstance(value, str):
        options = choices.get((section, key))
        if options and value in options:
            return st.selectbox(label, options, index=options.index(value), help=help_text,
                                key=wkey)
        return st.text_input(label, value=value, help=help_text, key=wkey)
    if isinstance(value, int):
        return int(st.number_input(label, value=value, step=1, help=help_text, key=wkey))
    if isinstance(value, float):
        return float(st.number_input(label, value=value, step=nice_step(value), format="%g",
                                     help=help_text, key=wkey))
    short_number_list = (isinstance(value, list) and value and len(value) <= 3
                         and all(isinstance(v, (int, float)) for v in value))
    if short_number_list:
        st.markdown(f"<small>{label}</small>", unsafe_allow_html=True, help=help_text)
        columns = st.columns(len(value))
        out = []
        for i, (column, v) in enumerate(zip(columns, value)):
            with column:
                out.append(widget(section, f"{key}[{i}]", v, None, prefix, choices))
        return out
    if isinstance(value, list):
        text = st.text_input(label, value=", ".join(config_tools.format_value(v) for v in value),
                             help=(help_text or "") + "  (comma-separated)", key=wkey)
        items = text.split(",")
        if len(items) != len(value):
            return ValueError(f"{key}: give {len(value)} comma-separated values")
        try:
            return [type(v)(tomllib.loads(f"x = {item.strip()}")["x"])
                    for v, item in zip(value, items)]
        except (tomllib.TOMLDecodeError, TypeError, ValueError) as error:
            return ValueError(f"{key}: {error}")    # shown to the user below the form
    st.text(f"{label} = {value!r} (edit in the config file)")
    return value


def run_simulation(config_text, name):
    """Runs ospfmusim/run.py with config_text. Returns (ok, output, run folder or None)."""
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as file:
        file.write(config_text)
    command = [sys.executable, "-m", "ospfmusim.run", "--config", file.name]
    if name:
        command += ["--name", name]
    run = subprocess.run(command, cwd=data.ROOT, capture_output=True, text=True)
    Path(file.name).unlink(missing_ok=True)
    match = re.search(r"^Results: (.+)$", run.stdout, re.MULTILINE)
    return run.returncode == 0, run.stdout + run.stderr, Path(match[1].strip()) if match else None


def render():
    st.header("New run")
    last = st.session_state.pop("last_run", None)
    if last:
        ok, output, folder = last
        (st.success if ok else st.error)(
            f"Run finished: {folder.name}. It is selected in the sidebar." if ok
            else "The simulation failed. See the output below.")
        with st.expander("Output of python -m ospfmusim.run", expanded=not ok):
            st.code(output)

    # Where the values start from: a file in configs/, or the config of an earlier run.
    configs = [(f"configs/{path.name}", path) for path in data.list_configs()]
    if not configs:
        st.error(f"No config file (*.toml) in {data.CONFIGS}.")
        return
    sources = configs + [
        (f"Run {label}", data.run_config_path(folder)) for label, folder in data.list_runs()
        if data.run_config_path(folder)]
    labels = [label for label, _ in sources]
    default = data.default_config()
    start = next((i for i, (_, path) in enumerate(configs) if path == default), 0)
    source_label = st.selectbox("Start from", labels, index=start,
                                help="The values below start from this config. The default is "
                                     "the one OspSystemStructure.xml points to.")
    source = dict(sources)[source_label]
    # Save as default writes to the chosen file in configs/, or the default one for a run's config.
    target_index = labels.index(source_label) if source_label in dict(configs) else start
    base_text = config_tools.add_missing(source.read_text(encoding="utf-8-sig"),
                                         SWITCH_DEFAULTS)
    base = tomllib.loads(base_text)
    help_texts = config_tools.comments(base_text)
    choices = {**CHOICES, ("vessel", "type"): sorted(base.get("blendermann", {}))}
    keys = editable_keys(base_text)
    prefix = source_label                        # new widgets when the source changes

    st.caption("Change values and press **Run simulation**. The run gets its own folder in "
               "results/ with the config it used. The files in configs/ are only changed by "
               "**Save as default**. The vessel matrices, the Blendermann table and the thruster "
               "layout are edited in the file itself.")

    values = {}
    # enter_to_submit=False: Enter in a field does nothing; only a click on a button runs or
    # saves.
    with st.form("new_run", enter_to_submit=False):
        name = st.text_input("Run name (optional)", placeholder="for example high_wind",
                             help="Added to the folder name: results/<time stamp>_<name>")
        columns = st.columns(len(SWITCHES))
        for column, (title, section, help_text) in zip(columns, SWITCHES):
            if (section, "enabled") in keys:
                with column:
                    values[(section, "enabled")] = st.toggle(
                        title, value=lookup(base, section, "enabled"), help=help_text,
                        key=f"{prefix}|{section}|enabled")
        for title, sections in GROUPS:
            with st.expander(title, expanded=title in ("Simulation", "Wind")):
                for section in sections:
                    # enabled is the toggle above the groups.
                    section_keys = [(s, k) for s, k in keys if s == section and k != "enabled"]
                    if not section_keys:
                        continue
                    st.markdown(f"**[{section}]**")
                    columns = st.columns(3)
                    for i, (s, k) in enumerate(section_keys):
                        with columns[i % 3]:
                            values[(s, k)] = widget(s, k, lookup(base, s, k),
                                                    help_texts.get((s, k)), prefix, choices)
        run_clicked = st.form_submit_button("Run simulation", type="primary")
        target_label = st.selectbox("Save as default in", [label for label, _ in configs],
                                    index=target_index, key=f"{prefix}|save_target",
                                    help="The file in configs/ that **Save as default** "
                                         "writes to.")
        save_clicked = st.form_submit_button("Save as default")

    if not (run_clicked or save_clicked):
        return
    errors = [v for v in values.values() if isinstance(v, ValueError)]
    if errors:
        for error in errors:
            st.error(str(error))
        return
    changes = {k: v for k, v in values.items() if v != lookup(base, *k)}

    if save_clicked:
        target = dict(configs)[target_label]
        default_text = config_tools.add_missing(target.read_text(encoding="utf-8-sig"),
                                                SWITCH_DEFAULTS)
        default = tomllib.loads(default_text)
        to_save = {k: v for k, v in values.items()
                   if k in editable_keys(default_text) and v != lookup(default, *k)}
        if to_save:
            target.write_text(config_tools.set_values(default_text, to_save), encoding="utf-8")
            st.success(f"Saved {len(to_save)} value(s) in {target_label}: "
                       + ", ".join(f"[{s}] {k}" for s, k in to_save))
        else:
            st.info(f"{target_label} already has these values.")
        return

    st.write(f"{len(changes)} value(s) changed from {source_label}:" if changes
             else f"No changes from {source_label}.")
    for (s, k), v in changes.items():
        st.markdown(f"- [{s}] **{k}**: {lookup(base, s, k)} → {v}")
    absolute_text = config_tools.make_paths_absolute(base_text, source.parent)
    config_text = config_tools.set_values(absolute_text, changes)
    duration = changes.get(("simulation", "duration"), lookup(base, "simulation", "duration"))
    with st.spinner(f"Running the simulation ({duration} s simulated) ..."):
        ok, output, folder = run_simulation(config_text, name)
    st.session_state["last_run"] = (ok, output, folder)
    if ok and folder:
        st.session_state["pending_run"] = folder.name    # selected in the sidebar after the rerun
    st.cache_data.clear()
    st.rerun()
