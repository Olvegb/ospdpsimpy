"""The animated thruster view in the Thrusters tab of the results app.

Left: the vessel seen from above, bow up and starboard to the right.
  - At each thruster a pointed arrow for its thrust on the vessel: it points
    the way the thruster pushes, and its length is the thrust. All arrows
    have the same scale, so they add up to the sum arrow.
  - A ring around each thruster at its max_thrust, so the arrow reaching
    the ring means the thruster is at full thrust.
  - Dotted outlines for what the allocation asked for (thrust and angle
    command), so a thruster that lags or is still turning shows up.
  - The sum of the thrusters (X, Y) from the body origin: filled is what the
    thrusters deliver, dotted what the controller commanded.
  - Red wedges inside the rings: the forbidden thrust directions, the
    thruster's own forbidden_zones and, with [allocation] flushing_sectors,
    the DNV-ST-0111 flushing sectors (zones.forbidden_zones).
  - A box with thrust, angle and power per thruster, the sum, the load on
    each switchboard, and the allocation flags (saturated, turn-limited).
Right: thrust, angle and power over time, and with [[power_sources]] the load
on each switchboard in % of what it can give (available minus reserve), with
a line at the slider time.

A time slider (and a play button) moves through the run. Like footprint.py
it is a Plotly animation (frames), so moving the slider runs in the browser
without asking Streamlit to redraw the page.

License: MIT (see LICENSE in the repository root)
Authors: William Boland (2026)
"""
import numpy as np
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ospfmusim.thruster.helpfunctions.power import supply_matrix, usable_power_kw
from ospfmusim.thruster.helpfunctions.zones import forbidden_zones

SUM_COLOUR = "#e4572e"                 # the heading colour in footprint.py
ZONE_COLOUR = "rgba(220,50,50,0.25)"
# One colour per thruster, leaving out the red that looks like SUM_COLOUR.
COLOURS = [c for c in plotly.colors.qualitative.Plotly if c != "#EF553B"]
# ponytail: Gunnerus beam / loa (9.6 m / 36.25 m), only for the outline; add a
# [vessel] beam to the config if the outline must fit other vessels.
BEAM_PER_LOA = 0.265
RING = 0.3                 # ring radius at the largest max_thrust, as a share of loa
MAX_LINE_POINTS = 600      # the time series are thinned to at most this many points


def hull_outline(loa):
    """(screen x, screen y) of a simple hull: flat stern, straight sides, pointed bow.

    Screen x is the body y (to starboard), screen y the body x (forward).
    """
    half_beam = BEAM_PER_LOA * loa / 2.0
    forward = [-loa / 2, loa / 6, loa / 2, loa / 6, -loa / 2, -loa / 2]
    starboard = [-half_beam, -half_beam, 0.0, half_beam, half_beam, -half_beam]
    return starboard, forward


def on_screen(forward, starboard):
    """(screen x, screen y) of a body-frame vector."""
    return np.array([starboard, forward], dtype=float)


def along(thrust, angle_deg):
    """(screen x, screen y) of a thrust along angle_deg: 0 forward, 90 to starboard (layout.py).

    Tunnels have a signed thrust at 90 deg, so a negative thrust points to port.
    """
    alpha = np.deg2rad(angle_deg)
    return on_screen(thrust * np.cos(alpha), thrust * np.sin(alpha))


def needle(tail, head, head_length, min_length):
    """dict(x, y) of a closed, pointed arrow from tail to its tip at head.

    The head is head_length long (shorter on short arrows) and half as wide. The
    shaft keeps a visible width on short arrows.
    Only the tail point when the arrow is shorter than min_length, so ~0 thrust
    draws nothing (an empty trace would also drop out of the legend).
    """
    tail, head = np.asarray(tail, dtype=float), np.asarray(head, dtype=float)
    length = np.hypot(*(head - tail))
    if length <= min_length:
        return dict(x=[float(tail[0])], y=[float(tail[1])])
    u = (head - tail) / length
    n = np.array([-u[1], u[0]])
    h = min(0.6 * length, head_length)
    neck = head - h * u
    barb = 0.25 * h * n
    shaft = min(0.07 * head_length, 0.4 * 0.25 * h) * n
    points = [tail + shaft, neck + shaft, neck + barb, head, neck - barb, neck - shaft,
              tail - shaft, tail + shaft]
    return dict(x=[float(p[0]) for p in points], y=[float(p[1]) for p in points])


def wedges(tail, radius, zones):
    """dict(x, y) of filled wedges from tail out to radius, one per (start, end) zone [deg].

    Thrust angles as in along(): 0 forward, 90 to starboard.
    """
    x, y = [], []
    for start, end in zones:
        a = np.deg2rad(np.linspace(start, end, int(np.ceil((end - start) / 5.0)) + 1))
        x += [float(tail[0]), *(tail[0] + radius * np.sin(a)).tolist(), float(tail[0]), None]
        y += [float(tail[1]), *(tail[1] + radius * np.cos(a)).tolist(), float(tail[1]), None]
    return dict(x=x, y=y)


def padded(text):
    """text with non-breaking spaces, so the monospace box lines up.

    Plotly's SVG text joins runs of normal spaces into one.
    """
    return text.replace(" ", "\u00a0")


def animated_thrusters(dynamics, layout, loa, n_frames, allocation=None):
    """The thruster view with a time slider, as a Plotly figure dict.

    dynamics is the dynamics FMU's result DataFrame, layout the thruster
    layout.Layout of the run's config: the k-th thruster is slot k
    (thruster{k}_thrust, ...). allocation is the allocation FMU's DataFrame,
    for the commanded force and the flags; without it they are left out.
    """
    thrusters, sources = layout.thrusters, layout.power_sources or ()
    slots = range(1, len(thrusters) + 1)

    def columns(name):
        """(rows, thrusters) array of thruster{k}_<name>."""
        return np.column_stack([dynamics[f"thruster{k}_{name}"].to_numpy() for k in slots])

    thrust, angle, power = columns("thrust"), columns("azimuth_deg"), columns("power_kw")
    signed = (angle + 180.0) % 360.0 - 180.0     # -90 port, 90 starboard: no 0/360 jumps
    thrust_cmd, angle_cmd = columns("thrust_command"), columns("azimuth_command_deg")
    delivered = dynamics[["X", "Y", "N"]].to_numpy()
    commanded, flags = None, []
    if allocation is not None and len(allocation):
        allocation = allocation.reindex(dynamics.index, method="nearest")
        commanded = allocation[["tau_X", "tau_Y", "tau_N"]].to_numpy()
        flags = [(text, allocation[col].astype(bool).to_numpy())
                 for text, col in (("SATURATED", "saturated"), ("TURN-LIMITED", "turn_limited"))
                 if col in allocation]           # turn_limited is not in older runs
    # ponytail: the zones with every thruster working; the allocation drops a failed
    # thruster's flushing sectors on the others, this drawing does not.
    zones = forbidden_zones(thrusters, layout.flushing_sectors)
    if sources:
        # The power the thrusters really draw (dynamics), shared out as in the allocation.
        usable = np.array([usable_power_kw(s) for s in sources])
        load = power @ supply_matrix(thrusters, sources).T          # (rows, sources) [kW]
        percent = 100.0 * load / usable
    time = dynamics.index.to_numpy()
    names = [t.name for t in thrusters]
    colours = [COLOURS[j % len(COLOURS)] for j in range(len(thrusters) + len(sources))]
    tails = [on_screen(t.x, t.y) for t in thrusters]
    scale = RING * loa / max(t.max_thrust for t in thrusters)      # screen m per N
    rings = [scale * t.max_thrust for t in thrusters]
    head_length, min_length = 0.07 * loa, 0.01 * RING * loa
    origin = on_screen(0.0, 0.0)
    width = max(len(name) for name in names + [s.name for s in sources] + ["Cmd"])
    n_rows = 3 + bool(sources)                   # thrust, angle, power (, switchboards)
    time_axes = [(f"x{k}", f"y{k}") for k in range(2, n_rows + 2)]
    frame_rows = np.unique(np.linspace(0, len(dynamics) - 1, n_frames).round().astype(int))
    line_rows = np.arange(0, len(dynamics), max(1, len(dynamics) // MAX_LINE_POINTS))

    def frame_content(i):
        """Data of the animated traces, the box, the time lines and the title at row i."""
        data = [needle(tail, tail + scale * along(f, a), head_length, min_length)
                for tail, f, a in zip(tails, thrust[i], angle[i])]
        data += [needle(tail, tail + scale * along(f, a), head_length, min_length)
                 for tail, f, a in zip(tails, thrust_cmd[i], angle_cmd[i])]
        data.append(needle(origin, origin + scale * on_screen(*delivered[i, :2]),
                           head_length, min_length))
        if commanded is not None:
            data.append(needle(origin, origin + scale * on_screen(*commanded[i, :2]),
                               head_length, min_length))

        rows = [padded(f"<b>{name:<{width}}</b> {f / 1e3:6.1f} kN {a:4.0f}° {p:5.0f} kW")
                for name, f, a, p in zip(names, thrust[i], signed[i], power[i])]
        for label, (x, y, n) in [("Sum", delivered[i])] + (
                [("Cmd", commanded[i])] if commanded is not None else []):
            rows.append(padded(f"<b>{label:<{width}}</b> X {x / 1e3:6.1f} Y {y / 1e3:6.1f} kN "
                               f"N {n / 1e3:6.1f} kNm"))
        for k, source in enumerate(sources):
            rows.append(padded(f"<b>{source.name:<{width}}</b> {load[i, k]:5.0f} / "
                               f"{usable[k]:.0f} kW {percent[i, k]:4.0f}%"))
        active = [text for text, on in flags if on[i]]
        if active:
            rows.append(f"<b>{', '.join(active)}</b>")
        box = dict(x=0.0, y=1.0, xref="paper", yref="paper", xanchor="left", yanchor="top",
                   showarrow=False, align="left", text="<br>".join(rows),
                   font=dict(family="monospace", size=11), bgcolor="rgba(128,128,128,0.15)")
        cursor = [dict(type="line", xref=x, yref=f"{y} domain", x0=time[i], x1=time[i], y0=0,
                       y1=1, line=dict(color="grey", width=1.5, dash="dot"))
                  for x, y in time_axes]
        title = f"Thrusters, t = {time[i]:.1f} s, total power {power[i].sum():.0f} kW"
        return data, [box], cursor, title

    fig = make_subplots(rows=n_rows, cols=2, column_widths=[0.55, 0.45], shared_xaxes=True,
                        horizontal_spacing=0.08, vertical_spacing=0.05,
                        specs=[[{"rowspan": n_rows}, {}]] + [[None, {}]] * (n_rows - 1))
    plan = dict(row=1, col=1)
    hull_x, hull_y = hull_outline(loa)
    fig.add_trace(go.Scatter(x=hull_x, y=hull_y, mode="lines", fill="toself", name="Hull",
                             line=dict(width=2, color="grey"), fillcolor="rgba(128,128,128,0.15)",
                             hoverinfo="skip", showlegend=False), **plan)
    circle = np.linspace(0.0, 2.0 * np.pi, 73)
    for tail, radius, name, colour in zip(tails, rings, names, colours):
        fig.add_trace(go.Scatter(x=tail[0] + radius * np.sin(circle),
                                 y=tail[1] + radius * np.cos(circle), mode="lines",
                                 name=f"{name} max thrust", legendgroup=name, showlegend=False,
                                 line=dict(width=1, color=colour), opacity=0.6,
                                 hoverinfo="skip"), **plan)

    zoned = [(tail, radius, z) for tail, radius, z in zip(tails, rings, zones) if z]
    for j, (tail, radius, z) in enumerate(zoned):
        fig.add_trace(go.Scatter(**wedges(tail, radius, z), mode="lines", fill="toself",
                                 name="Forbidden zones", legendgroup="zones", showlegend=j == 0,
                                 fillcolor=ZONE_COLOUR, line=dict(width=0), hoverinfo="skip"),
                      **plan)

    # The animated traces, in the order frame_content gives their data.
    data, annotations, shapes, title = frame_content(frame_rows[-1])
    animated = list(range(len(fig.data), len(fig.data) + len(data)))
    for d, name, colour in zip(data, names, colours):
        fig.add_trace(go.Scatter(**d, mode="lines", fill="toself", name=f"{name} thrust",
                                 legendgroup=name, showlegend=False, fillcolor=colour,
                                 line=dict(width=1, color=colour), hoverinfo="skip"), **plan)
    for j, (d, name, colour) in enumerate(zip(data[len(names):], names, colours)):
        fig.add_trace(go.Scatter(**d, mode="lines", name="Commanded (dotted)",
                                 legendgroup="commanded", showlegend=j == 0,
                                 line=dict(width=1.5, color=colour, dash="dot"),
                                 hoverinfo="skip"), **plan)
    fig.add_trace(go.Scatter(**data[2 * len(names)], mode="lines", fill="toself",
                             name="Sum of thrusters", fillcolor=SUM_COLOUR,
                             line=dict(width=1, color=SUM_COLOUR), hoverinfo="skip"), **plan)
    if commanded is not None:
        fig.add_trace(go.Scatter(**data[-1], mode="lines", name="Commanded sum",
                                 legendgroup="commanded", showlegend=False,
                                 line=dict(width=2, color=SUM_COLOUR, dash="dot"),
                                 hoverinfo="skip"), **plan)

    xs, ys = [float(t[0]) for t in tails], [float(t[1]) for t in tails]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=names, name="Thrusters", showlegend=False,
        # Names away from the middle; above on the centreline, clear of a tunnel's arrow.
        textposition=["middle left" if x < 0 else "middle right" if x > 0 else "top center"
                      for x in xs],
        marker=dict(size=9, color=colours, line=dict(width=1.5, color="white")),
        hovertemplate="%{text}<br>x = %{y:.2f} m<br>y = %{x:.2f} m<extra></extra>"), **plan)

    for j, (name, colour) in enumerate(zip(names, colours)):
        for row, values in ((1, thrust[line_rows, j] / 1e3), (2, signed[line_rows, j]),
                            (3, power[line_rows, j])):
            fig.add_trace(go.Scatter(x=time[line_rows], y=values, mode="lines", name=name,
                                     legendgroup=name, showlegend=row == 1,
                                     line=dict(width=1.5, color=colour),
                                     hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>"),
                          row=row, col=2)
    for k, (source, colour) in enumerate(zip(sources, colours[len(thrusters):])):
        fig.add_trace(go.Scatter(x=time[line_rows], y=percent[line_rows, k], mode="lines",
                                 name=source.name, line=dict(width=1.5, color=colour, dash="dash"),
                                 hovertemplate=f"{source.name}: %{{y:.1f}} %<extra></extra>"),
                      row=4, col=2)

    # Only the arrows, the box, the time lines and the title change between frames.
    frames = []
    for i in frame_rows:
        data_i, annotations_i, shapes_i, title_i = frame_content(i)
        frames.append(dict(name=f"{time[i]:.1f}", data=data_i, traces=animated,
                           layout=dict(annotations=annotations_i, shapes=shapes_i,
                                       title=dict(text=title_i))))

    # Fixed square plan axes that fit the hull, every ring and the largest sum arrow.
    half = 1.05 * max(loa / 2, *(np.abs(t).max() + r for t, r in zip(tails, rings)),
                      scale * sum(t.max_thrust for t in thrusters))
    fig.update_xaxes(title="y, to starboard [m]", range=[-half, half], autorange=False, **plan)
    fig.update_yaxes(title="x, forward [m]", range=[-half, half], autorange=False,
                     scaleanchor="x", scaleratio=1, **plan)
    for row, title_text in enumerate(["Thrust [kN]", "Angle [deg]", "Power [kW]",
                                      "Load [% usable]"][:n_rows], start=1):
        fig.update_yaxes(title=title_text, row=row, col=2)
    fig.update_xaxes(title="Time [s]", row=n_rows, col=2)

    play = dict(frame=dict(duration=60, redraw=True), transition=dict(duration=0),
                fromcurrent=True, mode="immediate")
    pause = dict(frame=dict(duration=0, redraw=False), mode="immediate")
    jump = dict(frame=dict(duration=0, redraw=True), transition=dict(duration=0),
                mode="immediate")
    fig.update_layout(
        title=title, annotations=annotations, shapes=shapes, height=780,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.3),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.13, xanchor="left",
                          yanchor="top", pad=dict(t=0, r=10), showactive=False,
                          buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], pause])])],
        sliders=[dict(active=len(frames) - 1, x=0.17, len=0.83, y=-0.1, yanchor="top",
                      currentvalue=dict(prefix="t = ", suffix=" s"), pad=dict(t=0),
                      steps=[dict(label=f["name"], method="animate", args=[[f["name"]], jump])
                             for f in frames])],
    )
    figure = fig.to_dict()
    figure["frames"] = frames
    return figure
