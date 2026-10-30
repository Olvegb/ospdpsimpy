"""Results app: edit the config, run new simulations, and show the results as plots.

Run from the repo folder, with the master environment active:

    streamlit run src/app/build/app.py

It opens in the browser. See docs/app.md.

License: MIT (see LICENSE in the repository root)
Authors: Olve Grønås Birkeland (2026)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from ospfmusim.thruster.helpfunctions.layout import load_layout

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import footprint  # noqa: E402
import thruster_view  # noqa: E402
import editor  # noqa: E402

st.set_page_config(page_title="Gunnerus DP results", page_icon="⚓", layout="wide")


# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------

@st.cache_data(show_spinner="Reading results ...")
def cached_results(folder, stamp):           # stamp is only there to refresh the cache
    return data.load_results(folder)


def figure_html(figure, dark):
    """An animated Plotly figure dict as HTML, in the light or dark theme.

    Shown with st.iframe instead of st.plotly_chart, because st.plotly_chart
    validates every animation frame on every rerun of the app, which takes seconds.
    plotly.js is loaded from its CDN, so these plots need an internet connection.
    """
    template = pio.templates["plotly_dark" if dark else "plotly_white"]
    figure["layout"]["template"] = template.to_plotly_json()
    figure["layout"]["paper_bgcolor"] = "rgba(0,0,0,0)"
    return pio.to_html(figure, include_plotlyjs="cdn", full_html=False, auto_play=False,
                       validate=False, config={"displaylogo": False})


@st.cache_data(show_spinner="Making the footprint animation ...")
def footprint_html(folder, stamp, set_point, n_arrows, n_frames, with_environment, dark):
    """The animated footprint as HTML, cached. stamp is only there to refresh the cache."""
    results = data.load_results(folder)
    environment = results.get("environment") if with_environment else None
    return figure_html(footprint.animated_footprint(results["vessel"], set_point, n_arrows,
                                                    n_frames, environment), dark)


@st.cache_data(show_spinner="Making the thruster animation ...")
def thrusters_html(folder, stamp, config_file, loa, n_frames, dark):
    """The animated thruster view as HTML, cached. stamp is only there to refresh the cache.

    The thrusters, zones and switchboards come from the layout in config_file.
    """
    results = data.load_results(folder)
    return figure_html(thruster_view.animated_thrusters(results["dynamics"],
                                                        load_layout(config_file), loa, n_frames,
                                                        results.get("allocation")), dark)


def line_plot(series, title, y_label, height=320):
    """One plot with one line per (name, pandas Series) in series."""
    fig = go.Figure()
    for name, s in series:
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines"))
    fig.update_layout(title=title, xaxis_title="Time [s]", yaxis_title=y_label, height=height,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25),
                      hovermode="x unified")
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme="streamlit")


def missing(*names):
    st.info("No results for " + ", ".join(names) + " in this run.")


def deg(s):
    return np.rad2deg(s)


# ---------------------------------------------------------------------
#  Sidebar: which run to show
# ---------------------------------------------------------------------

st.sidebar.title("⚓ Gunnerus DP")
runs = data.list_runs()
labels = [label for label, _ in runs]
if "pending_run" in st.session_state:          # a run that just finished in the New run tab
    st.session_state["run_select"] = st.session_state.pop("pending_run")
if st.session_state.get("run_select") not in labels:
    st.session_state.pop("run_select", None)
label = st.sidebar.selectbox(
    "Run", labels, key="run_select",
    help="Every run has its own folder in results/, newest first.") if runs else None
with st.sidebar.expander("Other folder"):
    other = st.text_input("Results folder", "", placeholder=r"C:\...\my_runs\run1",
                          help="Show a run that is not in results/.")
folder = Path(other) if other else dict(runs).get(label)

if st.sidebar.button("Reload"):
    st.cache_data.clear()

results = {}
if folder is not None and folder.is_dir():
    results = cached_results(str(folder), data.folder_stamp(folder))
elif folder is not None:
    st.sidebar.error(f"The folder {folder} does not exist.")

config_path = data.run_config_path(folder) if folder is not None else None
config = data.load_config(config_path or data.default_config())
if results:
    st.sidebar.caption("Files: " + ", ".join(sorted(results)))
    st.sidebar.caption("Config: the run's own copy." if config_path else
                       f"Config: configs/{data.default_config().name} now (this run has no copy), "
                       "which may differ from the config the results were made with.")

R = results.get                                 # short name: R("vessel") -> DataFrame or None


# ---------------------------------------------------------------------
#  Pages
# ---------------------------------------------------------------------

tabs = st.tabs(["New run", "Overview", "Vessel", "Environment", "Kalman filter",
               "Control and thrusters", "Thrusters", "Data"])

with tabs[0]:
    editor.render()

if not results:
    with tabs[1]:
        st.warning("No results to show yet. Make a run in the **New run** tab "
                   "(or `python -m ospfmusim.run`).")
    st.stop()

# --- Overview --------------------------------------------------------
with tabs[1]:
    st.header("Overview")
    numbers = data.summary(results, config)
    columns = st.columns(4)
    for i, (label, (value, unit)) in enumerate(numbers.items()):
        columns[i % 4].metric(label, f"{value:,.2f} {unit}".strip())

    left, right = st.columns([3, 2])
    with left:
        vessel = R("vessel")
        if vessel is not None:
            ctrl = config.get("controller", {})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vessel["east"], y=vessel["north"], mode="lines",
                                     name="Total (LF + WF)"))
            fig.add_trace(go.Scatter(x=vessel["lf_east"], y=vessel["lf_north"], mode="lines",
                                     name="LF"))
            fig.add_trace(go.Scatter(x=[ctrl.get("east_ref", 0.0)], y=[ctrl.get("north_ref", 0.0)],
                                     mode="markers", name="Set-point",
                                     marker=dict(symbol="x", size=12)))
            fig.add_trace(go.Scatter(x=vessel["east"].iloc[[-1]], y=vessel["north"].iloc[[-1]],
                                     mode="markers", name="End", marker=dict(size=9)))
            fig.update_layout(title="Position (footprint)", xaxis_title="East [m]",
                              yaxis_title="North [m]", height=480,
                              margin=dict(l=10, r=10, t=40, b=10),
                              yaxis=dict(scaleanchor="x", scaleratio=1))
            show(fig)
        else:
            missing("vessel")
    with right:
        st.subheader("Config of this run")
        wind = config.get("wind", {})
        waves = config.get("waves", {})
        current = config.get("current", {})
        thrusters = ", ".join(name for _, name in data.thruster_slots(config)) or "none"
        st.markdown(
            f"- **Vessel:** {config.get('vessel', {}).get('type', '?')}\n"
            f"- **Wind:** {wind.get('mean_speed', '?')} m/s "
            f"from {wind.get('direction_deg', '?')}°\n"
            f"- **Waves:** H_s {waves.get('hs', '?')} m, T_p {waves.get('tp', '?')} s, "
            f"from {waves.get('direction_deg', '?')}° ({waves.get('sea_type', '?')})\n"
            f"- **Current:** {current.get('speed', '?')} m/s "
            f"from {current.get('direction_deg', '?')}°\n"
            f"- **Duration:** {config.get('simulation', {}).get('duration', '?')} s\n"
            f"- **Thrusters:** {thrusters}"
        )

    vessel = R("vessel")
    if vessel is not None and len(vessel) > 1:
        left, right = st.columns([3, 2])
        with right:
            st.subheader("Heading and environment over time")
            n_arrows = st.slider("Number of heading arrows", 5, 100, 25, step=5)
            n_frames = st.slider("Time steps on the slider", 20, 400, 150, step=10,
                                 help="More steps = smoother, but a slower page.")
            show_environment = st.checkbox("Show wind, waves and current", value=True)
            st.caption("Move the time slider under the plot, or press ▶ Play. At each time the "
                       "plot shows the track so far, earlier headings (thin arrows, dots coloured "
                       "by time) and the vessel now (large dot and thick arrow). Position and "
                       "heading are the total motion (LF + WF).")
            if show_environment:
                reference = ", ".join(
                    f"{name.lower()} {value:g} {'m H_s' if name == 'Waves' else 'm/s'}"
                    for name, value in data.ENVIRONMENT_REFERENCE.items())
                st.caption("Wind, waves and current start on the side they come from and point "
                           "the way they go, with their values at that time. The length shows the "
                           f"size: full length at {reference}, and at least a quarter of that so "
                           "small values still show their direction.")
        with left:
            ctrl = config.get("controller", {})
            set_point = (ctrl.get("north_ref", 0.0), ctrl.get("east_ref", 0.0))
            dark = getattr(st.context.theme, "type", None) == "dark"
            html = footprint_html(str(folder), data.folder_stamp(folder), set_point, n_arrows,
                                  n_frames, show_environment, dark)
            st.iframe(html, height=640)              # our own HTML, made just above

# --- Vessel ----------------------------------------------------------
with tabs[2]:
    st.header("Vessel")
    vessel = R("vessel")
    if vessel is None:
        missing("vessel")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Std north", f"{vessel['north'].std():,.2f} m")
        c2.metric("Std east", f"{vessel['east'].std():,.2f} m")
        c3.metric("Std position", f"{data.position_std(vessel):,.2f} m")
        c1, c2 = st.columns(2)
        with c1:
            show(line_plot([("Total", vessel["north"]), ("LF", vessel["lf_north"])],
                           "North", "[m]"))
            show(line_plot([("Total", deg(vessel["yaw"])), ("LF", deg(vessel["lf_yaw"]))],
                           "Heading", "[deg]"))
        with c2:
            show(line_plot([("Total", vessel["east"]), ("LF", vessel["lf_east"])],
                           "East", "[m]"))
            show(line_plot([("North [m]", vessel["wf_north"]), ("East [m]", vessel["wf_east"]),
                            ("Yaw [deg]", deg(vessel["wf_yaw"]))],
                           "Wave-frequency (WF) motion", ""))
        show(line_plot([("u", vessel["lf_u"]), ("v", vessel["lf_v"])],
                       "LF velocity, body frame", "[m/s]"))

        st.subheader("Heave, roll and pitch")
        c1, c2, c3 = st.columns(3)
        with c1:
            # down is positive downwards (NED), so heave up is negative.
            show(line_plot([("Total", vessel["down"]), ("LF", vessel["lf_down"])],
                           "Heave (down)", "[m]"))
        with c2:
            show(line_plot([("Total", deg(vessel["roll"])), ("LF", deg(vessel["lf_roll"]))],
                           "Roll", "[deg]"))
        with c3:
            show(line_plot([("Total", deg(vessel["pitch"])), ("LF", deg(vessel["lf_pitch"]))],
                           "Pitch", "[deg]"))

        st.subheader("Loads on the vessel, body frame")
        dynamics = R("dynamics")
        c1, c2, c3 = st.columns(3)
        for column, dof, unit in ((c1, "X", "N"), (c2, "Y", "N"), (c3, "N", "Nm")):
            series = [("Wind", vessel[f"wind_{dof}"]),
                      ("Wave drift", vessel[f"wave_drift_{dof}"])]
            if dynamics is not None:
                series.append(("Thrusters", dynamics[dof]))
            with column:
                show(line_plot(series, dof, f"[{unit}]"))

# --- Environment -----------------------------------------------------
with tabs[3]:
    st.header("Environment (NED)")
    env = R("environment")
    if env is None:
        missing("environment")
    else:
        show(line_plot([("Total", data.total_wind_speed(env)), ("Mean", env["mean_wind_speed"]),
                        ("Slowly varying", env["slowly_varying_wind_speed"]),
                        ("Gust", env["gust_wind_speed"])],
                       "Wind speed", "[m/s]", height=380))
        c1, c2 = st.columns(2)
        with c1:
            show(line_plot([("Wind (from)", env["wind_direction_deg"]),
                            ("Waves (from)", env["wave_direction_deg"]),
                            ("Current (from)", env["current_direction_deg"])],
                           "Directions", "[deg]"))
        with c2:
            show(line_plot([("North", env["current_north"]), ("East", env["current_east"])],
                           "Current", "[m/s]"))
        waves = R("waves")
        sea_state = (f"**Sea state:** H_s = {env['H_s'].iloc[-1]:.2f} m, "
                     f"T_p = {env['T_p'].iloc[-1]:.2f} s")
        if waves is not None and "kept_energy_fraction" in waves:
            sea_state += (", kept energy in the wave components = "
                          f"{waves['kept_energy_fraction'].iloc[-1]:.1%}")
        st.markdown(sea_state)

# --- Kalman filter ---------------------------------------------------
with tabs[4]:
    st.header("Kalman filter")
    kf, vessel = R("kalman_filter"), R("vessel")
    if kf is None:
        missing("kalman_filter")
    else:
        st.caption("Estimate vs the vessel's true LF and WF motion. "
                   "The filter only sees the measured total.")
        c1, c2, c3 = st.columns(3)
        for column, dof, true, wf_true, label, f in (
                (c1, "north", "lf_north", "wf_north", "North [m]", lambda s: s),
                (c2, "east", "lf_east", "wf_east", "East [m]", lambda s: s),
                (c3, "psi", "lf_yaw", "wf_yaw", "Heading [deg]", deg)):
            hat, wf_hat = f"{dof}_hat", f"wf_{dof}_hat"
            with column:
                series = [("Measured total", f(kf[f"{dof}_meas"])), ("LF estimate", f(kf[hat]))]
                if vessel is not None:
                    series.append(("LF true", f(vessel[true])))
                show(line_plot(series, f"LF: {label}", ""))
                series = [("WF estimate", f(kf[wf_hat]))] if wf_hat in kf else []
                if vessel is not None:
                    series.append(("WF true", f(vessel[wf_true])))
                show(line_plot(series, f"WF: {label}", ""))
        show(line_plot([("b_x [N]", kf["b_x_hat"]), ("b_y [N]", kf["b_y_hat"]),
                        ("b_n [Nm]", kf["b_n_hat"])],
                       "Bias estimate (NED)", ""))

# --- Control and thrusters --------------------------------------------
with tabs[5]:
    st.header("Control and thrusters")
    dp, dynamics, allocation = R("dp_controller"), R("dynamics"), R("allocation")
    if dp is None:
        missing("dp_controller")
    else:
        c1, c2 = st.columns(2)
        with c1:
            show(line_plot([("North [m]", dp["error_north"]), ("East [m]", dp["error_east"])],
                           "Position error", "[m]"))
        with c2:
            show(line_plot([("Heading", dp["error_heading_deg"])], "Heading error", "[deg]"))
        c1, c2, c3 = st.columns(3)
        for column, dof, unit in ((c1, "X", "N"), (c2, "Y", "N"), (c3, "N", "Nm")):
            # Runs from before the controller sent out P, I and D only have tau_<dof>.
            terms = [(term, dp[f"tau_{dof}_{term}"]) for term in ("P", "I", "D")
                     if f"tau_{dof}_{term}" in dp]
            commanded = sum(values for _, values in terms) if terms else dp[f"tau_{dof}"]
            series = [("Commanded (P + I + D)", commanded)]
            if dynamics is not None:
                series.append(("Delivered", dynamics[dof]))
            with column:
                show(line_plot(series, f"Force {dof}", f"[{unit}]"))
                if terms:
                    show(line_plot(terms, f"PID terms {dof}", f"[{unit}]"))

    if dynamics is not None:
        slots = data.thruster_slots(config)
        st.subheader("Thrusters")
        c1, c2, c3 = st.columns(3)
        with c1:
            show(line_plot([(name, dynamics[f"thruster{k}_thrust"] / 1e3) for k, name in slots],
                           "Thrust", "[kN]"))
        with c2:
            show(line_plot([(name, dynamics[f"thruster{k}_azimuth_deg"]) for k, name in slots],
                           "Azimuth", "[deg]"))
        with c3:
            show(line_plot([(name, dynamics[f"thruster{k}_power_kw"]) for k, name in slots]
                           + [("Total", dynamics["total_power_kw"])], "Power", "[kW]"))
        if allocation is not None:
            flags = [(name, allocation[col].astype(float))
                     for name, col in [("Saturated", "saturated"), ("Turn-limited", "turn_limited")]
                     if col in allocation]
            show(line_plot([("Utilisation", allocation["utilisation"])] + flags,
                           "Allocation", "[-]", height=260))

# --- Thrusters --------------------------------------------------------
with tabs[6]:
    st.header("Thrusters")
    dynamics, thrusters = R("dynamics"), config.get("thrusters", [])
    if dynamics is None or not thrusters:
        missing("dynamics")
    else:
        n_frames = st.slider("Time steps on the slider", 20, 400, 150, step=10,
                             key="thruster_frames",
                             help="More steps = smoother, but a slower page.")
        st.caption("The vessel seen from above, bow up and starboard to the right. Each arrow is "
                   "the thrust on the vessel: it points the way the thruster pushes, and its "
                   "length is the thrust. The ring around a thruster is its max_thrust, so an "
                   "arrow at the ring is full thrust. Red wedges are directions the thruster "
                   "may not push in (forbidden_zones, and the DNV-ST-0111 flushing sectors with "
                   "flushing_sectors = true). Dotted outlines are what the allocation "
                   "asked for. The orange arrow from the middle is the sum of the thrusters "
                   "(X, Y; dotted: commanded by the controller); all arrows have the same scale. "
                   "The box gives thrust, angle (0° forward, 90° to starboard, -90° to port) "
                   "and power, the sum with the yaw moment N, the load on each switchboard, and "
                   "the allocation flags. The plots on the right show the whole run, with a line "
                   "at the slider time; the bottom one is the switchboard load in % of what each "
                   "[[power_sources]] can give (available minus reserve). Move "
                   "the time slider under the plot, or press ▶ Play. Click a name in the legend "
                   "to hide it.")
        loa = config.get("vessel", {}).get("loa") or 2.2 * max(abs(t["x"]) for t in thrusters)
        dark = getattr(st.context.theme, "type", None) == "dark"
        try:
            html = thrusters_html(str(folder), data.folder_stamp(folder),
                                  str(config_path or data.default_config()), loa, n_frames, dark)
            st.iframe(html, height=800)              # our own HTML, made just above
        except ValueError as error:                  # a config the thruster layout rejects
            st.error(f"Could not read the thrusters from the config: {error}")

# --- Data ------------------------------------------------------------
with tabs[7]:
    st.header("Data")
    show_waves = st.checkbox("Show the wave component columns (wave1_ ... wave100_)", value=False)
    # One list over all FMUs, so the chosen signals are kept (a separate FMU picker would change
    # the options, and Streamlit then clears what is chosen).
    options = [f"{simulator}.{c}" for simulator in sorted(results) for c in results[simulator]
               if show_waves or not data.WAVE_COLUMN.match(c)]
    chosen = st.multiselect("Signals to plot", options, default=options[:1],
                            format_func=data.label, help="Type to search, e.g. thrust.")
    if not chosen:
        st.info("Choose one or more signals to plot.")
    else:
        # Signals from different FMUs on one time axis (NaN where an FMU has no value).
        table = pd.DataFrame({name: results[name.split(".")[0]][name.split(".", 1)[1]]
                              for name in chosen}).rename(columns=data.label)
        units = {data.unit(name) for name in chosen}
        y_label = f"[{units.pop()}]" if len(units) == 1 and "" not in units else ""
        show(line_plot(table.items(), "", y_label, height=420))
        st.dataframe(table, width="stretch", height=300)
        st.download_button("Download as CSV", table.to_csv().encode("utf-8"),
                           file_name="signals.csv", mime="text/csv")
