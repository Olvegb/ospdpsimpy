"""The animated footprint plot in the Overview tab of the results app.

A time slider (and a play button) moves through the run. At each time step the
plot shows:
  - the track so far, as a line behind the vessel,
  - heading arrows at evenly spaced earlier times, and the vessel's position
    and heading now,
  - wind, wave and current arrows with their values now.

It is a Plotly animation (frames), so moving the slider runs in the browser
without asking Streamlit to redraw the page.
"""
import numpy as np
import plotly.graph_objects as go

import data

HEADING_COLOUR = "#e4572e"
ENVIRONMENT_COLOURS = {"Wind": "#3b82c4", "Waves": "#8e5cc7", "Current": "#1f9e89"}
MAX_TRAIL_POINTS = 800      # the track line is thinned to at most this many points


def unit(going_to_deg):
    """Unit vector (east, north) for a direction clockwise from north."""
    angle = np.deg2rad(going_to_deg)
    return np.array([np.sin(angle), np.cos(angle)])


def environment_series(environment):
    """[(name, going-to direction [deg] per row, size per row, unit)] for wind, waves and current.

    Wind and waves are given as where they come from (+180), the current as where it flows to.
    """
    return [
        ("Wind", (environment["wind_direction_deg"].to_numpy() + 180.0) % 360.0,
         data.total_wind_speed(environment).to_numpy(), "m/s"),
        ("Waves", (environment["wave_direction_deg"].to_numpy() + 180.0) % 360.0,
         environment["H_s"].to_numpy(), "m H_s"),
        ("Current", environment["current_direction_deg"].to_numpy() % 360.0,
         environment["current_speed"].to_numpy(), "m/s"),
    ]


def environment_arrows(series, i, centre, span):
    """[(name, tail, head, label position, label text)] for wind, waves and current at row i.

    Each arrow starts on the side the flow comes from, outside the track, and
    points the way it goes. Its length is the size relative to
    data.ENVIRONMENT_REFERENCE (full length at the reference value), but at
    least a quarter of full length so the direction is always visible. Arrows
    from the same side are moved apart sideways.
    """
    radius, full = 1.1 * span, 0.6 * span
    arrows = []
    for k, (name, going_to_all, size_all, unit_text) in enumerate(series):
        going_to, size = going_to_all[i], size_all[i]
        u = unit(going_to)
        side = np.array([u[1], -u[0]]) * (k - 1) * 0.18 * span
        relative = size / data.ENVIRONMENT_REFERENCE[name]
        length = full * min(max(relative, 0.25), 1.5) if size > 0 else 0.0
        tail = np.array(centre) - radius * u + side
        arrows.append((name, tail, tail + length * u, tail - 0.15 * span * u,
                       f"{name}<br>{size:.2f} {unit_text}"))
    return arrows


def arrow(head, tail, colour, width, opacity=1.0):
    """A Plotly annotation that draws an arrow from tail to head in data coordinates."""
    return dict(x=float(head[0]), y=float(head[1]), ax=float(tail[0]), ay=float(tail[1]),
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=width, arrowcolor=colour,
                opacity=opacity, text="")


def animated_footprint(vessel, set_point, n_arrows, n_frames, environment=None):
    """The footprint figure with a time slider, as a Plotly figure dict.

    set_point = (north, east).
    """
    north = vessel["north"].to_numpy()
    east = vessel["east"].to_numpy()
    yaw = vessel["yaw"].to_numpy()
    time = vessel.index.to_numpy()
    span = max(np.ptp(east), np.ptp(north), 1.0)
    centre = ((east.max() + east.min()) / 2.0, (north.max() + north.min()) / 2.0)
    short, long = 0.08 * span, 0.16 * span          # earlier heading arrows, and the one now

    series = None
    if environment is not None and len(environment):
        series = environment_series(environment.reindex(vessel.index, method="nearest"))

    frame_rows = np.unique(np.linspace(0, len(vessel) - 1, n_frames).round().astype(int))
    arrow_rows = np.unique(np.linspace(0, len(vessel) - 1, n_arrows).round().astype(int))
    trail_rows = np.arange(0, len(vessel), max(1, len(vessel) // MAX_TRAIL_POINTS))

    def frame_content(i):
        """Trace data and annotations at row i."""
        trail = np.append(trail_rows[trail_rows < i], i)
        past = arrow_rows[arrow_rows <= i]
        annotations = [arrow((east[j] + short * np.sin(yaw[j]),
                              north[j] + short * np.cos(yaw[j])),
                             (east[j], north[j]), HEADING_COLOUR, 1.5, 0.55) for j in past]
        annotations.append(arrow((east[i] + long * np.sin(yaw[i]),
                                  north[i] + long * np.cos(yaw[i])),
                                 (east[i], north[i]), HEADING_COLOUR, 3.5))
        labels = ([], [], [], [])
        if series is not None:
            for name, tail, head, label, text in environment_arrows(series, i, centre, span):
                if not np.allclose(tail, head):
                    annotations.append(arrow(head, tail, ENVIRONMENT_COLOURS[name], 4, 0.85))
                labels[0].append(label[0])
                labels[1].append(label[1])
                labels[2].append(text)
                labels[3].append(ENVIRONMENT_COLOURS[name])
        # Plain dicts and lists, not go objects: Plotly validates every go object,
        # which is slow for hundreds of frames.
        traces = [
            dict(x=east[trail].tolist(), y=north[trail].tolist()),
            dict(x=east[past].tolist(), y=north[past].tolist(),
                 marker=dict(color=time[past].tolist())),
            dict(x=[float(east[i])], y=[float(north[i])],
                 customdata=[[float(time[i]), float(np.rad2deg(yaw[i]) % 360.0)]]),
            dict(x=[float(v) for v in labels[0]], y=[float(v) for v in labels[1]], text=labels[2],
                 textfont=dict(color=labels[3])),
        ]
        title = (f"Footprint with heading, t = {time[i]:.1f} s, "
                 f"heading {np.rad2deg(yaw[i]) % 360.0:.1f}°")
        return traces, annotations, title

    # The axes stay fixed while the slider moves: they cover the track and every
    # environment arrow and label in every frame. Square, so north and east have
    # the same scale.
    xs, ys = [east.min(), east.max()], [north.min(), north.max()]
    if series is not None:
        for i in frame_rows:
            for _, tail, head, label, _ in environment_arrows(series, i, centre, span):
                xs += [tail[0], head[0], label[0]]
                ys += [tail[1], head[1], label[1]]
    half = 0.55 * max(max(xs) - min(xs), max(ys) - min(ys)) + 0.1 * span
    mid = ((max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0)

    last = frame_rows[-1]
    traces, annotations, title = frame_content(last)
    trail, past, now, labels = traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trail["x"], y=trail["y"], mode="lines", name="Track (total)",
                             line=dict(width=2), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=past["x"], y=past["y"], mode="markers", name="Earlier heading",
        marker=dict(size=7, color=past["marker"]["color"], colorscale="Viridis",
                    cmin=time[0], cmax=time[-1],
                    showscale=True, colorbar=dict(title="Time [s]", thickness=12)),
        hovertemplate="east = %{x:.2f} m<br>north = %{y:.2f} m<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=now["x"], y=now["y"], mode="markers", name="Vessel now", customdata=now["customdata"],
        marker=dict(size=13, color=HEADING_COLOUR, line=dict(width=1.5, color="white")),
        hovertemplate="t = %{customdata[0]:.1f} s<br>heading = %{customdata[1]:.1f}°"
                      "<br>east = %{x:.2f} m<br>north = %{y:.2f} m<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels["x"], y=labels["y"], text=labels["text"], mode="text",
                             showlegend=False, textfont=labels["textfont"],
                             textposition="middle center", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[set_point[1]], y=[set_point[0]], mode="markers", name="Set-point",
                             marker=dict(symbol="x", size=12)))

    frames = []
    for i in frame_rows:
        traces, annotations_i, title_i = frame_content(i)
        frames.append(dict(name=f"{time[i]:.1f}", data=traces, traces=[0, 1, 2, 3],
                           layout=dict(annotations=annotations_i, title=dict(text=title_i))))

    play = dict(frame=dict(duration=60, redraw=True), transition=dict(duration=0),
                fromcurrent=True, mode="immediate")
    pause = dict(frame=dict(duration=0, redraw=False), mode="immediate")
    jump = dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0),
                mode="immediate")
    fig.update_layout(
        title=title, annotations=annotations, height=620, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="East [m]", range=[mid[0] - half, mid[0] + half], autorange=False),
        yaxis=dict(title="North [m]", range=[mid[1] - half, mid[1] + half], autorange=False,
                   scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", y=-0.28),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.1, xanchor="left",
                          yanchor="top", pad=dict(t=0, r=10), showactive=False,
                          buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], pause])])],
        sliders=[dict(active=len(frames) - 1, x=0.17, len=0.83, y=-0.07, yanchor="top",
                      currentvalue=dict(prefix="t = ", suffix=" s"), pad=dict(t=0),
                      steps=[dict(label=f["name"], method="animate", args=[[f["name"]], jump])
                             for f in frames])],
    )
    # The frames are added to the plain figure dict, which st.plotly_chart takes as it is.
    figure = fig.to_dict()
    figure["frames"] = frames
    return figure
