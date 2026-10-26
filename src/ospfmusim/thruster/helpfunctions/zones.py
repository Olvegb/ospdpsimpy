"""
Forbidden thrust directions and the allowed arcs between them.

Angles are thrust angles in degrees: 0 deg pushes forward, increasing toward
starboard (layout.py). Copied from the DNV-ST-0111 project
(dp_capability/models/forbidden_zones.py); the logic does not depend on the
frame, only the angle convention in the docstrings changed.
"""
import math

# A thruster may not flush another non-tunnel thruster closer than this many
# of its own diameters, DNV-ST-0111 [3.11.3].
FLUSHING_DISTANCE_D = 15.0


def _normalise(zone):
    """(start, end) with start in [0, 360) and end = start + width, width in [0, 360]."""
    start, end = zone
    width = (end - start) % 360.0
    if width == 0.0 and end != start:
        width = 360.0  # e.g. (0, 360): the whole circle
    return (start % 360.0, start % 360.0 + width)


def flushing_sectors(thrusters):
    """
    Forbidden thrust sectors from flushing other thrusters, the rule of
    DNV-ST-0111 [3.11.3].

    Thruster i flushes thruster j if the angle between i's thrust direction
    and the vector from j to i is less than

        arctan(0.1 + 1.0 * D / s)

    with D the diameter of i and s the horizontal distance between them.
    Pushing along that vector sends i's race towards j. This is not allowed
    closer than 15 D, unless j is a tunnel thruster.

    Parameters
    ----------
    thrusters : sequence of layout.Thruster
        The working thrusters. Each needs its diameter.

    Returns
    -------
    list of list of (float, float)
        Per thruster, the (start, end) thrust angles [deg] of its sectors.
    """
    sectors = []
    for i, flushing in enumerate(thrusters):
        if flushing.diameter is None:
            raise ValueError(f"{flushing.name}: the flushing sectors need the diameter")
        own = []
        for j, flushed in enumerate(thrusters):
            if i == j or flushed.kind == "tunnel":
                continue
            dx, dy = flushing.x - flushed.x, flushing.y - flushed.y
            s = math.hypot(dx, dy)
            if s == 0.0:
                raise ValueError(f"{flushing.name} and {flushed.name} are at the same "
                                 "horizontal position")
            if s < FLUSHING_DISTANCE_D * flushing.diameter:
                centre = math.degrees(math.atan2(dy, dx))
                half = math.degrees(math.atan(0.1 + flushing.diameter / s))
                own.append((centre - half, centre + half))
        sectors.append(own)
    return sectors


def merge_zones(zones):
    """
    Zones normalised to start in [0, 360) and merged where they overlap or
    touch, sorted by start. A zone reaching past 360 deg keeps an end above
    360, e.g. (350, 370) for 350-10 deg.
    """
    # Zero-width zones forbid nothing.
    spans = sorted(z for z in map(_normalise, zones) if z[1] > z[0])
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    # A zone wrapping past 360 deg may cover the first ones.
    while len(merged) > 1 and merged[-1][1] - 360.0 >= merged[0][0]:
        first = merged.pop(0)
        merged[-1] = (merged[-1][0], max(merged[-1][1], first[1] + 360.0))
    if len(merged) == 1 and merged[0][1] - merged[0][0] >= 360.0:
        merged = [(0.0, 360.0)]
    return merged


def forbidden_zones(thrusters, flushing=False):
    """
    All forbidden zones per thruster: its own Thruster.forbidden_zones and,
    with flushing=True, the flushing sectors among the given thrusters
    (flushing_sectors()), merged.

    Returns
    -------
    list of list of (float, float)
        Per thruster, (start, end) thrust angles [deg], start in [0, 360),
        end > start (above 360 for a zone through 0 deg).
    """
    extra = flushing_sectors(thrusters) if flushing else [[] for _ in thrusters]
    return [merge_zones(list(t.forbidden_zones) + sectors) for t, sectors in zip(thrusters, extra)]


def allowed_arcs(zones):
    """
    The allowed thrust directions outside the zones, as arcs going toward
    starboard. The thrust allocation splits them further into convex pieces.

    Parameters
    ----------
    zones : list of (float, float)
        Forbidden zones [deg], e.g. from forbidden_zones().

    Returns
    -------
    list
        [None] when there are no zones (the full circle). Otherwise the
        (start, end) arcs [deg] between the zones, end > start (above 360
        for an arc through 0 deg); [] if everything is forbidden.
    """
    merged = merge_zones(zones)
    if not merged:
        return [None]
    arcs = []
    for k, (_, end) in enumerate(merged):
        # The allowed arc runs from this zone's end to the next zone's start.
        width = (merged[(k + 1) % len(merged)][0] - end) % 360.0
        if width > 0.0:
            arcs.append((end, end + width))
    return arcs
