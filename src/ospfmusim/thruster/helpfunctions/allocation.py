"""
Thrust allocation: thruster forces that give a commanded force tau.

allocate() allocates one tau (static). ThrustAllocator does it once per time
step: it gives the closest reachable tau when the command is out of reach,
leaves out unavailable thrusters, and holds idle azimuths' angles.

Copied from the DNV-ST-0111 project (dp_capability/models/thruster_allocation.py).
The polygon and LP machinery is unchanged: it only assumes a right-handed
frame, so it works as it is in the Fossen frame (layout.py). What changed:
- the thrusters give tau itself (the DNV version balanced an environmental
  load, i.e. gave -load);
- the capacity is Thruster.max_thrust / reverse_thrust instead of the DNV
  thrust formula and loss factors;
- the skeg, dead flushing and rudder branches are left out.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import linprog

from .power import supply_matrix, thruster_power_kw, usable_power_kw
from .zones import allowed_arcs, forbidden_zones

# LP tolerance, in units of utilisation: a tau that needs up to 1 + TOLERANCE
# of the largest thrust still counts as feasible.
TOLERANCE = 1e-6
# Spacing of the thrust fractions r at which the power curve P_B * r^1.5 is
# drawn as chords. The curve is convex, so the chords lie above it and the
# power is overestimated, by at most about 0.002 P_B (between r = 0 and 0.05).
POWER_STEP = 0.05
# Cost per unit of total thrust in the saturation pass, next to the tau
# error weights (Layout.tau_weights): small, so it only picks among forces
# with the same error.
THRUST_COST = 1e-6


@dataclass(frozen=True)
class Allocation:
    """Result of allocate() for one tau. The arrays follow the order of the thrusters."""

    fx: np.ndarray  # surge force from each thruster [N]
    fy: np.ndarray  # sway force from each thruster [N], positive to starboard
    # The highest demand as a fraction of what is available: the largest
    # thrust_fraction and, with power sources, the largest source power as a
    # fraction of its usable power. inf when the thrusters cannot give tau
    # at all.
    utilisation: float
    # Thrust as a fraction of the thruster's largest thrust in that direction
    # (max_thrust, or reverse_thrust backwards). For azimuths measured
    # against the capacity polygon.
    thrust_fraction: np.ndarray
    power_kw: np.ndarray  # power per thruster [kW], P_B * r^1.5; nan without power_kw
    source_power_kw: np.ndarray  # power taken from each power source [kW]; empty without sources

    @property
    def feasible(self):
        """True if the thrusters give tau within their largest thrust and the usable power."""
        return self.utilisation <= 1 + TOLERANCE

    @property
    def angle_deg(self):
        """
        Thrust angle of each thruster [deg]: 0 deg pushes forward, increasing
        toward starboard, so 90 deg pushes to starboard. nan for a thruster
        that gives no force.
        """
        angle = np.mod(np.rad2deg(np.arctan2(self.fy, self.fx)), 360.0)
        return np.where((self.fx != 0) | (self.fy != 0), angle, np.nan)


def _inside(angle_deg, zones):
    """True if the direction lies strictly inside one of the zones [deg]."""
    for start, end in zones:
        if end - start >= 360.0 or 0.0 < (angle_deg - start) % 360.0 < end - start:
            return True
    return False


def _polygon_angles(arc, n_sides):
    """
    Corner angles [deg] of an azimuth's capacity polygon over one allowed arc
    (None: the full circle, returned without repeating 360): every
    360/n_sides deg from 0, and the arc ends.
    """
    candidates = np.arange(n_sides) * 360.0 / n_sides
    if arc is None:
        return candidates
    start, end = arc
    # Unwrap into the arc's range, which may run past 360 deg.
    shifted = start + np.mod(candidates - start, 360.0)
    return np.unique(np.round(np.r_[start, shifted[shifted < end], end], 9))


def _convex_fans(angles, radii, closed):
    """
    Split a star-shaped polygon (corners at the given angles [deg] and radii,
    seen from the origin) into convex pieces: fans of consecutive corners
    that, together with the origin, form a convex polygon. A fan grows while
    each corner turns the same way and it spans at most 180 deg; a corner
    that turns the other way starts a new one.

    Returns a list of (angles, radii, closed) per piece; closed is True only
    for a whole closed ring that is convex as it is.
    """
    theta = np.deg2rad(angles)
    points = np.column_stack([radii * np.cos(theta), radii * np.sin(theta)])
    scale = max(radii.max(), 1e-300)

    def turns_left(p0, p1, p2):
        e1, e2 = p1 - p0, p2 - p1
        return e1[0] * e2[1] - e1[1] * e2[0] >= -1e-9 * scale ** 2

    m = len(angles)
    if closed:
        reflex = [k for k in range(m)
                  if not turns_left(points[k - 1], points[k], points[(k + 1) % m])]
        if not reflex:
            return [(angles, radii, True)]
        # Open the ring at a reflex corner, which a fan boundary has to pass
        # through anyway.
        order = np.r_[np.arange(reflex[0], m), np.arange(reflex[0] + 1)]
        angles = np.r_[angles[reflex[0]:], angles[:reflex[0] + 1] + 360.0]
        radii, points = radii[order], points[order]
        m = len(angles)
    fans, i = [], 0
    while i < m - 1:
        j = i + 1
        while (j + 1 < m and angles[j + 1] - angles[i] <= 180.0 + 1e-9
               and turns_left(points[j - 1], points[j], points[j + 1])):
            j += 1
        fans.append((angles[i:j + 1], radii[i:j + 1], False))
        i = j
    return fans


def _fan_rows(angles, radii, closed):
    """
    Rows a . (fx, fy) <= limit of one convex piece: one per outer edge, with
    its unit outward normal and the distance of the edge from the origin
    (this limit scales with u), and for an open fan the two boundary rays
    (limit 0, fixed).
    """
    theta = np.deg2rad(angles)
    points = np.column_stack([radii * np.cos(theta), radii * np.sin(theta)])
    ends = np.roll(points, -1, axis=0) if closed else points[1:]
    starts = points if closed else points[:-1]
    edges = ends - starts
    length = np.hypot(edges[:, 0], edges[:, 1])
    keep = length > 1e-12 * max(radii.max(), 1e-300)
    normals = np.column_stack([edges[keep, 1], -edges[keep, 0]]) / length[keep, None]
    limits = np.maximum(np.einsum("ij,ij->i", normals, starts[keep]), 0.0)
    if closed:
        return normals, limits, np.ones(len(limits), dtype=bool)
    a, b = theta[0], theta[-1]
    # On the polygon's side of the ray at the first corner and of the ray at the last.
    rays = np.array([[np.sin(a), -np.cos(a)], [-np.sin(b), np.cos(b)]])
    scales = np.r_[np.ones(len(limits), dtype=bool), False, False]
    return np.vstack([normals, rays]), np.r_[limits, 0.0, 0.0], scales


def _no_force():
    """The piece of a thruster that can give no force at all."""
    directions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    return (directions, np.zeros(4), np.ones(4, dtype=bool))


def _thruster_pieces(thruster, n_sides, zones):
    """
    The convex pieces one thruster's force can be chosen from, each as rows
    (directions, limits, scales): directions . f <= limits, times u where
    scales is True. An azimuth's capacity is the polygon with radius
    max_thrust over its allowed directions, split into convex fans; tunnel
    thrusters and shaft lines have one piece.
    """
    if thruster.kind == "azimuth":
        arcs = allowed_arcs(zones)
        if not arcs:
            # Every direction forbidden: no force at all.
            return [_no_force()]
        pieces = []
        for arc in arcs:
            angles = _polygon_angles(arc, n_sides)
            radii = np.full(len(angles), thruster.max_thrust)
            pieces += [_fan_rows(*fan) for fan in _convex_fans(angles, radii, arc is None)]
        return pieces
    if thruster.kind == "tunnel":
        directions = np.array([[0.0, 1.0], [0.0, -1.0]])
        angles = np.array([90.0, 270.0])
    else:  # shaft line
        directions = np.array([[1.0, 0.0], [-1.0, 0.0]])
        angles = np.array([0.0, 180.0])
    limits = np.array([thruster.max_thrust, thruster.reverse_thrust])
    # A direction inside a forbidden zone is not available at all.
    limits = np.where([_inside(a, zones) for a in angles], 0.0, limits)
    return [(directions, limits, np.ones(2, dtype=bool))]


def _size_rows(thruster, n_sides):
    """
    Rows that measure the size of a thruster's force for pass 2: t >= a . f
    for each row a. For azimuths a regular n_sides-gon of unit size, scaled
    so it equals |f| at its corners.
    """
    if thruster.kind == "azimuth":
        phi = (2 * np.arange(n_sides) + 1) * np.pi / n_sides
        return np.column_stack([np.cos(phi), np.sin(phi)]) / np.cos(np.pi / n_sides)
    if thruster.kind == "tunnel":
        return np.array([[0.0, 1.0], [0.0, -1.0]])
    return np.array([[1.0, 0.0], [-1.0, 0.0]])


def _solve(c, a_ub, b_ub, a_eq, b_eq, bounds):
    """linprog with HiGHS. Returns the solution, or None if the constraints cannot be met."""
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if result.status == 2:
        return None
    if result.status != 0:
        raise RuntimeError(f"thrust allocation LP failed: {result.message}")
    return result.x


def _place(blocks, n):
    """Stack per-thruster rows into rows over all thrusters.

    Each block is k x 2 over (fx_i, fy_i); the result is over (fx_1..fx_n, fy_1..fy_n).
    """
    rows = []
    for i, block in enumerate(blocks):
        full = np.zeros((len(block), 2 * n))
        full[:, i] = block[:, 0]
        full[:, n + i] = block[:, 1]
        rows.append(full)
    return np.vstack(rows)


def _power_chords():
    """(slope, intercept) of each chord of r^1.5 between r = 0, POWER_STEP, ..., 1."""
    r = np.linspace(0.0, 1.0, round(1 / POWER_STEP) + 1)
    g = r ** 1.5
    slope = np.diff(g) / np.diff(r)
    return slope, g[:-1] - slope * r[:-1]


@dataclass(frozen=True)
class _Power:
    """The power sources, in units of p_ref [kW]."""

    p_b: np.ndarray  # P_B per thruster
    shares: np.ndarray  # (sources, thrusters)
    usable: np.ndarray  # usable power per source


@dataclass(frozen=True)
class _Problem:
    """One convex allocation problem: every thruster's force in one convex piece."""

    n: int  # number of thrusters
    a_cap: np.ndarray  # capacity rows over the forces
    limits: np.ndarray  # in units of t_ref
    scales: np.ndarray  # True where the limit is multiplied by the thruster's fraction r
    cap_owner: np.ndarray  # thruster of each capacity row
    a_size: np.ndarray  # size rows over the forces, for pass 2
    size_owner: np.ndarray  # thruster of each size row
    a_eq: np.ndarray
    b_eq: np.ndarray
    f_bounds: list
    t_ref: float
    power: _Power | None


@dataclass(frozen=True)
class _Setup:
    """The parts of the allocation problem that don't change with tau."""

    options: list  # the convex pieces of each thruster
    a_size: np.ndarray  # size rows over the forces, for pass 2
    size_owner: np.ndarray  # thruster of each size row
    t_ref: float
    power: _Power | None
    shares: np.ndarray  # (sources, thrusters); no rows without power sources


def _setup(thrusters, n_sides, flushing, power_sources, available):
    n = len(thrusters)
    if n == 0:
        raise ValueError("no thrusters")
    on = [i for i in range(n) if available[i]]
    zones = dict(zip(on, forbidden_zones([thrusters[i] for i in on], flushing)))
    # An unavailable thruster gives no force, and its flushing sectors on the
    # others are gone.
    options = [_thruster_pieces(t, n_sides, zones[i]) if i in zones else [_no_force()]
               for i, t in enumerate(thrusters)]
    sizes = [_size_rows(t, n_sides) for t in thrusters]
    t_ref = max(piece[1].max() for pieces in options for piece in pieces) or 1.0
    power, shares = None, np.zeros((0, n))
    if power_sources is not None:
        missing = [t.name for t in thrusters if t.power_kw is None]
        if missing:
            raise ValueError(f"{missing}: no power_kw, but the allocation has power sources")
        shares = supply_matrix(thrusters, power_sources)
        p_ref = max(t.power_kw for t in thrusters)
        power = _Power(
            p_b=np.array([t.power_kw for t in thrusters]) / p_ref,
            shares=shares,
            usable=np.array([usable_power_kw(s) for s in power_sources]) / p_ref,
        )
    size_owner = np.concatenate([[i] * len(s) for i, s in enumerate(sizes)])
    return _Setup(options, _place(sizes, n), size_owner, t_ref, power, shares)


def _problem(thrusters, tau, pieces, s):
    n = len(thrusters)
    a_eq = np.zeros((3, 2 * n))
    a_eq[0, :n] = 1.0
    a_eq[1, n:] = 1.0
    a_eq[2, :n] = [-t.y for t in thrusters]
    a_eq[2, n:] = [t.x for t in thrusters]
    # Tunnel thrusters give no surge force, shaft lines no sway force.
    f_bounds = [(0, 0) if t.kind == "tunnel" else (None, None) for t in thrusters]
    f_bounds += [(0, 0) if t.kind == "shaft_line" else (None, None) for t in thrusters]
    return _Problem(
        n=n,
        a_cap=_place([p[0] for p in pieces], n),
        # Forces in units of t_ref, so the LP numbers are of order 1. The
        # moment row then has the lever arms in m.
        limits=np.concatenate([p[1] for p in pieces]) / s.t_ref,
        scales=np.concatenate([p[2] for p in pieces]),
        cap_owner=np.concatenate([[i] * len(p[1]) for i, p in enumerate(pieces)]).astype(int),
        a_size=s.a_size,
        size_owner=s.size_owner,
        a_eq=a_eq,
        b_eq=tau / s.t_ref,
        f_bounds=f_bounds,
        t_ref=s.t_ref,
        power=s.power,
    )


def _inequalities(p):
    """
    The rows a . (f, r, q) + c_u * U <= b shared by both passes, over the
    forces f (2n), each thruster's fraction r of its capacity (n), and with
    power sources its power q in units of p_ref (n). Returns (a, c_u, b):
    - capacity: every force inside r times its piece;
    - r_i <= U;
    - with power: q_i >= P_B,i * chord(r_i) for every chord of r^1.5, and
      the power from each source <= U times its usable power.
    """
    n, m = p.n, len(p.limits)
    n_q = n if p.power else 0
    cap_r = np.zeros((m, n))
    cap_r[np.arange(m), p.cap_owner] = -np.where(p.scales, p.limits, 0.0)
    blocks = [np.column_stack([p.a_cap, cap_r, np.zeros((m, n_q))])]
    c_u = [np.zeros(m)]
    b = [np.where(p.scales, 0.0, p.limits)]
    blocks.append(np.column_stack([np.zeros((n, 2 * n)), np.eye(n), np.zeros((n, n_q))]))
    c_u.append(-np.ones(n))
    b.append(np.zeros(n))
    if p.power:
        slope, intercept = _power_chords()
        k = len(slope)
        rows = np.zeros((k * n, 3 * n + n_q))
        for i, p_b in enumerate(p.power.p_b):
            rows[i * k:(i + 1) * k, 2 * n + i] = p_b * slope
            rows[i * k:(i + 1) * k, 3 * n + i] = -1.0
        blocks.append(rows)
        c_u.append(np.zeros(k * n))
        b.append(np.concatenate([-p_b * intercept for p_b in p.power.p_b]))
        s = len(p.power.usable)
        blocks.append(np.column_stack([np.zeros((s, 3 * n)), p.power.shares]))
        c_u.append(-p.power.usable)
        b.append(np.zeros(s))
    return np.vstack(blocks), np.concatenate(c_u), np.concatenate(b)


def _min_utilisation(p):
    """Pass 1, variables (f, r, q, U): the lowest U. None if tau cannot be given at all."""
    a, c_u, b = _inequalities(p)
    extra = a.shape[1] - 2 * p.n  # r and q
    z = _solve(
        c=np.r_[np.zeros(a.shape[1]), 1.0],
        a_ub=np.column_stack([a, c_u]),
        b_ub=b,
        a_eq=np.column_stack([p.a_eq, np.zeros((3, extra + 1))]),
        b_eq=p.b_eq,
        bounds=p.f_bounds + [(0, None)] * (extra + 1),
    )
    return None if z is None else max(z[-1], 0.0)


def _least_total_thrust(p, utilisation):
    """Pass 2, variables (f, r, q, t): at that U, the lowest total thrust.

    That is sum(t_i), with t_i >= |f_i|.
    """
    a, c_u, b = _inequalities(p)
    n, n_var = p.n, a.shape[1]
    k = len(p.size_owner)
    size_t = np.zeros((k, n))
    size_t[np.arange(k), p.size_owner] = -1.0
    z = _solve(
        c=np.r_[np.zeros(n_var), np.ones(n)],
        a_ub=np.vstack([
            np.column_stack([a, np.zeros((len(b), n))]),
            np.column_stack([p.a_size, np.zeros((k, n_var - 2 * n)), size_t]),
        ]),
        b_ub=np.r_[b - c_u * (utilisation + TOLERANCE), np.zeros(k)],
        a_eq=np.column_stack([p.a_eq, np.zeros((3, n_var - 2 * n + n))]),
        b_eq=p.b_eq,
        bounds=p.f_bounds + [(0, None)] * (n_var - 2 * n + n),
    )
    if z is None:
        raise RuntimeError("thrust allocation LP failed: second pass found no solution")
    # Drop solver noise, so an idle thruster gets exactly zero force.
    f = np.where(np.abs(z[:2 * n]) < 1e-9, 0.0, z[:2 * n]) * p.t_ref
    return f[:n], f[n:]


def _closest_tau(p, weights):
    """
    Saturation pass, variables (f, r, q, s+, s-, t): with U fixed at 1, the
    forces that come closest to tau. sum f + s+ - s- = tau, and the cost is
    sum(w * (s+ + s-)) plus THRUST_COST * sum(t_i), t_i >= |f_i|. No force
    at all always fits, so there is always a solution. Returns fx, fy and
    the cost.
    """
    a, c_u, b = _inequalities(p)
    n, n_var = p.n, a.shape[1]
    k = len(p.size_owner)
    size_t = np.zeros((k, n))
    size_t[np.arange(k), p.size_owner] = -1.0
    c = np.r_[np.zeros(n_var), weights, weights, np.full(n, THRUST_COST)]
    z = _solve(
        c=c,
        a_ub=np.vstack([
            np.column_stack([a, np.zeros((len(b), 6 + n))]),
            np.column_stack([p.a_size, np.zeros((k, n_var - 2 * n + 6)), size_t]),
        ]),
        b_ub=np.r_[b - c_u, np.zeros(k)],
        a_eq=np.column_stack([p.a_eq, np.zeros((3, n_var - 2 * n)), np.eye(3), -np.eye(3),
                              np.zeros((3, n))]),
        b_eq=p.b_eq,
        bounds=p.f_bounds + [(0, None)] * (n_var - 2 * n + 6 + n),
    )
    if z is None:
        raise RuntimeError("thrust allocation LP failed: saturation pass found no solution")
    f = np.where(np.abs(z[:2 * n]) < 1e-9, 0.0, z[:2 * n]) * p.t_ref
    return f[:n], f[n:], c @ z


def _thrust_fractions(p, fx, fy):
    """Each thruster's fraction r of its capacity piece, from the final forces.

    That is the largest a . f / limit.
    """
    values = p.a_cap @ (np.r_[fx, fy] / p.t_ref)
    use = p.scales & (p.limits > 1e-12)
    r = np.zeros(p.n)
    np.maximum.at(r, p.cap_owner[use], values[use] / p.limits[use])
    return r


def _best(thrusters, tau, s):
    """Pass 1 in every combination of pieces: the problem with the lowest u, and u.

    (None, inf) if tau can't be given at all.
    """
    best, best_u = None, np.inf
    for pieces in product(*s.options):
        p = _problem(thrusters, tau, pieces, s)
        u = _min_utilisation(p)
        if u is not None and u < best_u:
            best, best_u = p, u
    return best, best_u


def _allocation(thrusters, p, utilisation, fx, fy, shares):
    """The Allocation for the forces fx, fy found in problem p."""
    r = _thrust_fractions(p, fx, fy)
    power_kw = np.array([np.nan if t.power_kw is None else thruster_power_kw(t, ri)
                         for t, ri in zip(thrusters, r)])
    return Allocation(fx, fy, utilisation, r, power_kw, shares @ power_kw)


def allocate(thrusters, tau, n_sides=36, flushing=False, power_sources=None):
    """
    Thruster forces that together give the commanded force tau:

        sum fx_i                      = X
        sum fy_i                      = Y
        sum (x_i * fy_i - y_i * fx_i) = N

    Two linear programs:
    1. Minimise the utilisation u: the highest force any thruster needs, as
       a fraction of its largest thrust in that direction. tau is feasible
       if u <= 1.
    2. Keep u and minimise the total thrust, so that thrusters with room to
       spare do not push against each other.

    With power sources each source must also supply its thrusters:
    sum(share * P) <= u * usable power, with P = P_B * r^1.5 per thruster
    (power.thruster_power_kw, drawn as chords above the curve, so
    conservative). u is then the higher of the thrust and the power demand,
    as a fraction of what is available.

    Azimuths push in any direction with max_thrust. This capacity is drawn
    as a polygon inside the circle, with corners every 360/n_sides deg, so
    it is conservative by at most 1 - cos(pi / n_sides). Tunnel thrusters
    push along y and shaft lines along x, with reverse_thrust in the
    negative direction.

    Forbidden zones take directions away and make an azimuth's capacity
    non-convex. It is split into convex pieces, pass 1 is solved for every
    combination of pieces and the lowest u is kept; pass 2 is solved in the
    first combination that reaches it. A tunnel or shaft line direction
    inside a zone is not available.

    When u > 1 the forces still give tau, but exceed the thrusters' limits.
    One call allocates one tau.

    Parameters
    ----------
    thrusters : sequence of layout.Thruster
        The working thrusters.
    tau : (X, Y, N)
        Commanded force [N] and moment [Nm] in the body frame (layout.py).
    n_sides : int, optional
        Corners of the azimuth polygon. The default of 36 puts corners every
        10 deg, so pure surge and pure sway get the full max_thrust.
    flushing : bool, optional
        True adds the DNV-ST-0111 [3.11.3] flushing sectors among these
        thrusters to their forbidden zones (zones.flushing_sectors(); every
        thruster then needs its diameter).
    power_sources : sequence of layout.PowerSource, optional
        Every thruster then needs power_kw and power_supply. None: no power
        limit.

    Returns
    -------
    Allocation
        Force from each thruster, the utilisation, and each thruster's thrust
        fraction and power and each source's power. Forces are nan and the
        utilisation inf when the thrusters cannot give tau in any amount
        (e.g. tunnels only against a surge command).
    """
    tau = np.asarray(tau, dtype=float)
    if tau.shape != (3,):
        raise ValueError(f"tau must be (X, Y, N), got shape {tau.shape}")
    s = _setup(thrusters, n_sides, flushing, power_sources, [True] * len(thrusters))
    best, best_u = _best(thrusters, tau, s)
    if best is None:
        nan = np.full(len(thrusters), np.nan)
        return Allocation(nan, nan, np.inf, nan, nan, np.full(len(s.shares), np.nan))
    fx, fy = _least_total_thrust(best, best_u)
    return _allocation(thrusters, best, best_u, fx, fy, s.shares)


@dataclass(frozen=True)
class Command:
    """Result of ThrustAllocator.step(). The arrays follow the order of the thrusters."""

    # Thrust [N] along azimuth_deg: >= 0 for azimuths, signed for tunnels and
    # shaft lines (negative pushes to port or astern).
    thrust: np.ndarray
    # Direction positive thrust pushes [deg] (layout.py): the azimuth's
    # angle, held while it is idle; 90 for tunnels, 0 for shaft lines.
    azimuth_deg: np.ndarray
    # (X, Y, N) the forces give: the command, or the closest reachable when saturated
    tau: np.ndarray
    saturated: bool  # True if the command was out of reach
    # The forces, thrust fractions and power. Its utilisation is that of the
    # command: above 1 (inf if out of reach in any amount) when saturated.
    allocation: Allocation


class ThrustAllocator:
    """
    Thrust allocation inside a time loop: call step() once per time step.

    The same two passes as allocate(), with three differences:
    - Saturation: when the command needs u > 1, or can't be given at all,
      u is fixed at 1 and the forces come as close to tau as they can.
      The error is weighted by Layout.tau_weights:

          minimise  sum_k w_k * |tau_k - tau_achieved,k|  (+ a little total thrust)

      over every combination of convex pieces, keeping the lowest cost.
    - Thrusters can be unavailable. They give no force, and their flushing
      sectors on the others are gone.
    - An idle azimuth keeps its last angle, so it doesn't turn for nothing.

    The convex pieces of each set of available thrusters are worked out once
    and kept.
    """

    def __init__(self, layout, n_sides=36):
        self.layout = layout
        self.n_sides = n_sides
        self._setups = {}
        # Until told otherwise, azimuths point ahead.
        self._angle = np.array([90.0 if t.kind == "tunnel" else 0.0 for t in layout.thrusters])

    def step(self, tau, available=None):
        """
        Parameters
        ----------
        tau : (X, Y, N)
            Commanded force [N] and moment [Nm] in the body frame (layout.py).
        available : sequence of bool, optional
            One per thruster in layout order. None: all available.

        Returns
        -------
        Command
        """
        thrusters = self.layout.thrusters
        n = len(thrusters)
        tau = np.asarray(tau, dtype=float)
        if tau.shape != (3,):
            raise ValueError(f"tau must be (X, Y, N), got shape {tau.shape}")
        available = (True,) * n if available is None else tuple(bool(a) for a in available)
        if len(available) != n:
            raise ValueError(f"available needs one value per thruster ({n}), got {len(available)}")
        if available not in self._setups:
            self._setups[available] = _setup(
                thrusters, self.n_sides, self.layout.flushing_sectors, self.layout.power_sources,
                available)
        s = self._setups[available]

        best, best_u = _best(thrusters, tau, s)
        saturated = best_u > 1 + TOLERANCE
        if saturated:
            weights = np.asarray(self.layout.tau_weights, dtype=float)
            cost = np.inf
            for pieces in product(*s.options):
                p = _problem(thrusters, tau, pieces, s)
                fx_p, fy_p, cost_p = _closest_tau(p, weights)
                if cost_p < cost:
                    best, fx, fy, cost = p, fx_p, fy_p, cost_p
        else:
            fx, fy = _least_total_thrust(best, best_u)
        allocation = _allocation(thrusters, best, best_u, fx, fy, s.shares)

        x = np.array([t.x for t in thrusters])
        y = np.array([t.y for t in thrusters])
        kinds = [t.kind for t in thrusters]
        thrust = np.where([k == "tunnel" for k in kinds], fy,
                          np.where([k == "shaft_line" for k in kinds], fx, np.hypot(fx, fy)))
        turning = np.array([k == "azimuth" for k in kinds]) & ~np.isnan(allocation.angle_deg)
        self._angle = np.where(turning, allocation.angle_deg, self._angle)
        achieved = np.array([fx.sum(), fy.sum(), (x * fy - y * fx).sum()])
        return Command(thrust, self._angle.copy(), achieved, saturated, allocation)
