# -*- coding: utf-8 -*-
"""
Created on Thursday Aug 27 17:00:00 2026
Modified: 5000 x 10000 um chip. Stepped-impedance (meander Bragg) filter
built as a single serpentine, launched from both ends. Whole assembly
centered on the chip.

@author: Aryan Soni
"""

import numpy as np
import gdspy
import gds_maker as GM  # Custom module -- used for the launchers only

# --------------------------------------------------------------------- #
#                          Design Parameters                            #
# --------------------------------------------------------------------- #
filename = 'AS_bragg_two_launchers.gds'
write_gaps = False          # True = write shapes; False = subtract from ground

# --- Chip envelope ---
chip_height = 5000          # um
chip_width = 10000          # um
edge_margin = 150.0         # um, keep-out from top/bottom edges

# --- Bragg mirror cross-sections (set impedance; NEVER scaled) ---
gap1, width1 = 95, 4        # high-impedance section
gap2, width2 = 52, 90       # low-impedance section

# --- Launcher CPW cross-section ---
launcher_gap, launcher_width = 52, 90

# --- Launcher (bond pad + taper; physical dimensions, NEVER scaled) ---
straightLauncher = 500
taperLauncher = 500
launcherHeight = 500
gapLauncher = 125

# --- Serpentine packing ---
bend_radius = 400.0         # um, 180-deg turn radius
jog_margin = 20.0           # um of straight inside each 180-deg turn
lead_run = 400.0            # um of horizontal run at each end of the mirror
arc_step_deg = 1.0          # centerline arc discretisation
turns_min, turns_max = 3, 14

# --- Electrical ---
frequency = 7.2e9
epsilon = 11.7
c = 3e8
no_of_quarterwave_resonators = 6
layer = 1

quarterwave_resonator_length = 1e6 * c / (np.sqrt((epsilon + 1) / 2) * frequency * 4)

# Per-section electrical lengths. Both entries default to the same value,
# which is what the original script assumed. That is only correct if the two
# sections share an eps_eff -- they do not (4/95 vs 90/52 CPW have different
# filling factors), so each section is only approximately lambda/4 here.
# Replace with per-section values once you have eps_eff from a solver.
section_lengths = [quarterwave_resonator_length] * no_of_quarterwave_resonators
section_xsec = [(gap1, width1) if k % 2 == 0 else (gap2, width2)
                for k in range(no_of_quarterwave_resonators)]
total_length = float(np.sum(section_lengths))

print(f"Quarter-wave length (frequency-locked) = "
      f"{quarterwave_resonator_length:.1f} um")
print(f"Total mirror path length = {total_length:.1f} um "
      f"({no_of_quarterwave_resonators} sections)")


# --------------------------------------------------------------------- #
#                     Centerline geometry helpers                       #
# --------------------------------------------------------------------- #
def _unit(v):
    n = np.hypot(v[0], v[1])
    return v / n if n else v


def fillet_polyline(pts, radius, arc_step_deg=1.0):
    """
    Replace every interior vertex of a polyline with a circular arc of the
    given radius, returning a densely sampled point list. Because the arcs
    are baked into the point list, downstream FlexPath calls need no corner
    processing and section boundaries can be cut at any arc length.
    """
    pts = [np.asarray(p, dtype=float) for p in pts]
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        v1, v2 = _unit(a - b), _unit(c - b)
        phi = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        if phi > np.pi - 1e-9:      # collinear, nothing to fillet
            continue
        t = radius / np.tan(phi / 2.0)
        if t > np.hypot(*(a - b)) + 1e-9 or t > np.hypot(*(c - b)) + 1e-9:
            raise ValueError(
                f"vertex {i}: fillet tangent {t:.1f} um exceeds an adjoining "
                f"segment -- radius too large for this packing."
            )
        p_s, p_e = b + v1 * t, b + v2 * t
        centre = b + _unit(v1 + v2) * (radius / np.sin(phi / 2.0))
        a0 = np.arctan2(*(p_s - centre)[::-1])
        a1 = np.arctan2(*(p_e - centre)[::-1])
        sweep = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        n = max(2, int(abs(np.degrees(sweep)) / arc_step_deg) + 1)
        out.append(p_s)
        for k in range(1, n):
            th = a0 + sweep * k / n
            out.append(centre + radius * np.array([np.cos(th), np.sin(th)]))
        out.append(p_e)
    out.append(pts[-1])

    ded = [out[0]]
    for p in out[1:]:
        if np.hypot(*(p - ded[-1])) > 1e-9:
            ded.append(p)
    return ded


def arclen(pts):
    d = np.diff(np.asarray(pts, dtype=float), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def split_by_lengths(pts, lengths):
    """
    Cut a dense polyline into consecutive runs of the given arc lengths.
    Boundary points are interpolated exactly, so each run has the requested
    electrical length and adjacent runs share an endpoint.
    Returns (list_of_runs, list_of_(boundary_point, segment_index)).
    """
    pts = [np.asarray(p, dtype=float) for p in pts]
    cuts = np.cumsum(lengths)[:-1]
    runs, bounds = [], []
    cur = [pts[0]]
    s, ci = 0.0, 0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = np.hypot(*(b - a))
        while ci < len(cuts) and s + seg >= cuts[ci] - 1e-9:
            f = (cuts[ci] - s) / seg
            q = a + (b - a) * f
            if np.hypot(*(q - cur[-1])) > 1e-9:
                cur.append(q)
            runs.append(cur)
            bounds.append((q, i))
            cur = [q]
            ci += 1
        s += seg
        if np.hypot(*(b - cur[-1])) > 1e-9:
            cur.append(b)
    runs.append(cur)
    return runs, bounds


def cpw_gaps(pts, width, gap, layer):
    """CPW gap regions for a dense centerline: outer trace minus inner trace."""
    kw = dict(tolerance=0.01, precision=1e-4, max_points=0)
    outer = gdspy.FlexPath(pts, width + 2.0 * gap, **kw).to_polygonset()
    inner = gdspy.FlexPath(pts, width, **kw).to_polygonset()
    return gdspy.boolean(outer, inner, 'not', layer=layer,
                         precision=1e-4, max_points=0)


# --------------------------------------------------------------------- #
#                    Serpentine mirror centerline                       #
# --------------------------------------------------------------------- #
def serpentine_vertices(x0, y_c, n_turns, leg_height, pitch, lead):
    """
    Boustrophedon centerline, entering and leaving horizontally on y_c.

    Layout: lead-in, half leg up, then n_turns 180-deg turns separated by
    full-height legs, then a half leg back to y_c and a lead-out. The two
    half legs at the ends put the entry and exit on the chip centerline and
    keep the vertical extent symmetric about it, so a launcher can butt onto
    either end with no connecting run.

    Filleted path length is
        n_turns*leg_height + 2*lead + n_turns*pitch - (2*n_turns+2)*delta
    with delta = (2 - pi/2)*radius the shortening from one 90-deg fillet.
    leg_height_for() below inverts that.
    """
    pts = [(x0, y_c)]
    x = x0 + lead
    pts.append((x, y_c))
    y = y_c + leg_height / 2.0
    pts.append((x, y))
    for j in range(n_turns):
        x += pitch
        pts.append((x, y))
        if j < n_turns - 1:
            y = y_c - leg_height / 2.0 if y > y_c else y_c + leg_height / 2.0
        else:
            y = y_c
        pts.append((x, y))
    pts.append((x + lead, y_c))
    return pts


def leg_height_for(total, n_turns, pitch, lead, radius):
    """Leg height that makes the filleted serpentine exactly `total` long."""
    delta = (2.0 - np.pi / 2.0) * radius
    return (total - 2.0 * lead - n_turns * pitch
            + (2 * n_turns + 2) * delta) / n_turns


# --------------------------------------------------------------------- #
#                     Launcher placement helpers                        #
# --------------------------------------------------------------------- #
def _as_polygonset(polys, layer):
    pts = []
    for p in polys:
        pts.extend(p.polygons)
    return gdspy.PolygonSet(pts, layer=layer)


def _span_at(ps, x, tol):
    """Vertical extent of the polygon set in a thin slice around x."""
    ys = []
    for poly in ps.polygons:
        x_lo, x_hi = poly[:, 0].min(), poly[:, 0].max()
        if x_lo <= x + tol and x_hi >= x - tol:
            ys += [poly[:, 1].min(), poly[:, 1].max()]
    return (max(ys) - min(ys)) if ys else 0.0


def build_launcher_template(layer):
    """
    Build one launcher and normalise it so its narrow (CPW) end is at x = 0
    with the centre conductor on y = 0, and the pad extends toward +x.

    Which end createLauncherRight puts the pad on is detected from the
    geometry rather than assumed, so this does not depend on the module's
    positional convention.
    """
    raw, _ = GM.createLauncherRight(
        (0.0, 0.0), gap=launcher_gap, width=launcher_width,
        straight=straightLauncher, height=launcherHeight,
        taper=taperLauncher, gapL=gapLauncher, notch=True
    )
    ps = _as_polygonset(raw, layer)
    (x_lo, y_lo), (x_hi, y_hi) = ps.get_bounding_box()
    tol = 0.02 * (x_hi - x_lo)
    if _span_at(ps, x_lo + tol, tol) > _span_at(ps, x_hi - tol, tol):
        ps.mirror((0, 1))                       # pad was on the left; flip
        (x_lo, y_lo), (x_hi, y_hi) = ps.get_bounding_box()
    ps.translate(-x_lo, -0.5 * (y_lo + y_hi))
    length = x_hi - x_lo
    print(f"Launcher template: {length:.1f} um long, "
          f"{y_hi - y_lo:.1f} um tall, narrow end normalised to x=0")
    return ps, length


def place_launcher(template, attach_pt, side, layer):
    """Copy the template so its narrow end sits on attach_pt, pad facing out."""
    ps = gdspy.PolygonSet([p.copy() for p in template.polygons], layer=layer)
    if side == 'left':
        ps.mirror((0, 1))                       # pad now extends toward -x
    ps.translate(attach_pt[0], attach_pt[1])
    return ps


# --------------------------------------------------------------------- #
#        Pick the fewest turns (= narrowest mirror) that fits           #
# --------------------------------------------------------------------- #
pitch = 2.0 * bend_radius + jog_margin
cpw_halfspan = max(w / 2.0 + g for g, w in section_xsec)
y_c = chip_height / 2.0

launcher_template, launcher_len = build_launcher_template(layer)

print("\nturns   leg height   mirror width   total width   vertical span")
chosen = None
for n_turns in range(turns_min, turns_max + 1):
    h = leg_height_for(total_length, n_turns, pitch, lead_run, bend_radius)
    span = h + 2.0 * cpw_halfspan
    w_mirror = 2.0 * lead_run + n_turns * pitch
    w_total = w_mirror + 2.0 * launcher_len
    ok = (h >= 4.0 * bend_radius        # half legs are h/2, fillet at each end
          and span <= chip_height - 2.0 * edge_margin
          and w_total <= chip_width)
    print(f"{n_turns:5d}   {h:10.1f}   {w_mirror:12.1f}   {w_total:11.1f}"
          f"   {span:13.1f}   {'fits' if ok else ''}")
    if ok and chosen is None:
        x0 = 0.5 * (chip_width - w_total) + launcher_len
        verts = serpentine_vertices(x0, y_c, n_turns, h, pitch, lead_run)
        try:
            chosen = (n_turns, h, w_total, verts,
                      fillet_polyline(verts, bend_radius, arc_step_deg))
        except ValueError:
            chosen = None

if chosen is None:
    raise RuntimeError(
        "No turn count fits. Height is the binder: fewer turns means taller "
        "legs. Either shrink bend_radius (narrower turns let you keep the "
        "turn count up without widening the mirror), relax edge_margin, or "
        "split the mirror into two stacked banks."
    )

n_turns, leg_height, w_total, verts, dense = chosen
actual_len = arclen(dense)
print(f"\nSerpentine: {n_turns} turns, leg height {leg_height:.1f} um, "
      f"pitch {pitch:.1f} um, radius {bend_radius:.1f} um")
print(f"Centerline length {actual_len:.1f} um "
      f"(target {total_length:.1f}, error {actual_len - total_length:+.3f})")

# --------------------------------------------------------------------- #
#          Cut into quarter-wave sections and lay down the CPW          #
# --------------------------------------------------------------------- #
runs, bounds = split_by_lengths(dense, section_lengths)
assert len(runs) == no_of_quarterwave_resonators

mirror_polys = []
for k, (run, (g_k, w_k)) in enumerate(zip(runs, section_xsec)):
    mirror_polys.append(cpw_gaps(run, w_k, g_k, layer))
    print(f"  section {k + 1}: len {arclen(run):8.1f} um  "
          f"width {w_k:5.1f}  gap {g_k:5.1f}")

# Flag any impedance step that lands inside a bend rather than on a straight.
for k, (q, i) in enumerate(bounds):
    a = np.asarray(dense[max(i - 1, 0)])
    b = np.asarray(dense[i])
    cpt = np.asarray(dense[min(i + 2, len(dense) - 1)])
    d1, d2 = _unit(b - a), _unit(cpt - b)
    if abs(np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1, 1)))) > 0.5:
        print(f"  WARNING: step {k + 1}->{k + 2} at ({q[0]:.1f}, {q[1]:.1f}) "
              f"falls inside a bend, not on a straight run.")

entry_pt = (float(dense[0][0]), float(dense[0][1]))
exit_pt = (float(dense[-1][0]), float(dense[-1][1]))

# --------------------------------------------------------------------- #
#              Launchers butted onto both mirror ends                   #
# --------------------------------------------------------------------- #
launcher_l = place_launcher(launcher_template, entry_pt, 'left', layer)
launcher_r = place_launcher(launcher_template, exit_pt, 'right', layer)

for name, pt, (g_k, w_k) in (("left", entry_pt, section_xsec[0]),
                             ("right", exit_pt, section_xsec[-1])):
    step = abs((w_k / 2.0 + g_k) - (launcher_width / 2.0 + launcher_gap))
    print(f"{name} launcher butts onto section cross-section "
          f"{w_k:.0f}/{g_k:.0f} at x={pt[0]:.1f}; outer half-span step "
          f"{step:.1f} um")

total_structure = mirror_polys + [launcher_l, launcher_r]

# --------------------------------------------------------------------- #
#                          Fit report                                   #
# --------------------------------------------------------------------- #
def bbox_of(polys):
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    for p in polys:
        bb = p.get_bounding_box()
        if bb is None:
            continue
        lo = np.minimum(lo, bb[0])
        hi = np.maximum(hi, bb[1])
    return lo, hi


lo, hi = bbox_of(total_structure)
print(f"\nAssembly bbox: x {lo[0]:.1f} to {hi[0]:.1f} ({hi[0]-lo[0]:.1f} wide)")
print(f"               y {lo[1]:.1f} to {hi[1]:.1f} ({hi[1]-lo[1]:.1f} tall)")
print(f"Slack: left {lo[0]:.1f}, right {chip_width-hi[0]:.1f}, "
      f"bottom {lo[1]:.1f}, top {chip_height-hi[1]:.1f} um")
if min(lo[0], lo[1], chip_width - hi[0], chip_height - hi[1]) < 0:
    print("ERROR: geometry runs off the chip.")
if pitch - 2.0 * cpw_halfspan < 100.0:
    print(f"WARNING: only {pitch - 2*cpw_halfspan:.1f} um of ground between "
          f"adjacent legs. Check for leg-to-leg coupling and slotline modes; "
          f"add airbridges or widen the pitch.")

# --------------------------------------------------------------------- #
#                   Boolean merge and background                        #
# --------------------------------------------------------------------- #
gdspy.current_library.cells.clear()
merged_cell = gdspy.Cell('merge')
merged_path = gdspy.fast_boolean(total_structure, None, 'or', max_points=0)

if write_gaps:
    final_merged = merged_path
else:
    chip_rect = gdspy.Rectangle((0, 0), (chip_width, chip_height))
    subtracted_path = gdspy.fast_boolean(chip_rect, merged_path, 'not', layer=2)
    final_merged = gdspy.fast_boolean(subtracted_path, None, 'or', max_points=0)

merged_cell.add(final_merged)

gdspy.write_gds(filename + ".gds", unit=1.0e-6, precision=1.0e-9)
gdspy.LayoutViewer(cells=merged_cell)

print('Finished!')
