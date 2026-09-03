# -*- coding: utf-8 -*-
"""
Created on Thursday Aug 27 17:00:00 2026
Modified: 5000 x 10000 um chip. Left launcher replaced with an IDC
quarter-wave resonator, galvanically connected directly to section 1 of
the stepped-impedance (meander Bragg) filter. Assembly centered on chip.

@author: Aryan Soni
"""

import numpy as np
import gdspy
import gds_maker as GM  # Used for right launcher only

# --------------------------------------------------------------------- #
#                         Design Parameters                             #
# --------------------------------------------------------------------- #
filename = 'AS_bragg_idc_launcher'
write_gaps = False          # True = write shapes; False = subtract from ground

# --- Chip envelope ---
chip_height = 5000.0        # um
chip_width = 10000.0        # um
edge_margin = 250.0         # um, keep-out from top/bottom edges

# --- IDC quarter-wave resonator parameters ---
idc_l0 = 240.0              # total resonator length, um
idc_w = 2.0                 # inner trunk track width, um
idc_s = 2.0                 # finger width, um
idc_h = 46.0                # finger length, um
idc_g = 2.0                 # finger-tip-to-opposite-conductor clearance, um

# --- Bragg mirror cross-sections (set impedance) ---
gap1, width1 = 95.0, 4.0    # high-impedance section (section 1 connects to IDC)
gap2, width2 = 52.0, 90.0   # low-impedance section

# --- Right Launcher CPW cross-section ---
# Matches section 6 (low-Z)
launcher_r_gap, launcher_r_width = 52.0, 90.0

# --- Right Launcher physical dimensions ---
straightLauncher = 600.0
taperLauncher = 600.0
launcherHeight = 1075.0
gapLauncher = 225.0

# --- Serpentine packing ---
bend_radius = 400.0         # um, 180-deg turn radius
jog_margin = 20.0           # um of straight inside each 180-deg turn
lead_run = 500.0            # um of horizontal run at each end of mirror
arc_step_deg = 1.0          # centerline arc discretisation
turns_min, turns_max = 3, 14

# --- Electrical parameters ---
frequency = 7.2e9           # Hz
epsilon = 11.7              # relative permittivity
c = 3.0e8                   # m/s
no_of_quarterwave_resonators = 6
layer = 1

quarterwave_resonator_length = 1e6 * c / (np.sqrt((epsilon + 1) / 2) * frequency * 4)

section_lengths = [quarterwave_resonator_length] * no_of_quarterwave_resonators
section_xsec = [(gap1, width1) if k % 2 == 0 else (gap2, width2)
                for k in range(no_of_quarterwave_resonators)]
total_length = float(np.sum(section_lengths))

print(f"IDC total length = {idc_l0:.1f} um")
print(f"Meander quarter-wave length = {quarterwave_resonator_length:.1f} um")
print(f"Total mirror path length = {total_length:.1f} um ({no_of_quarterwave_resonators} sections)")


# --------------------------------------------------------------------- #
#                      Centerline geometry helpers                      #
# --------------------------------------------------------------------- #
def _unit(v):
    n = np.hypot(v[0], v[1])
    return v / n if n else v


def fillet_polyline(pts, radius, arc_step_deg=1.0):
    pts = [np.asarray(p, dtype=float) for p in pts]
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        v1, v2 = _unit(a - b), _unit(c - b)
        phi = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        if phi > np.pi - 1e-9:
            continue
        t = radius / np.tan(phi / 2.0)
        if t > np.hypot(*(a - b)) + 1e-9 or t > np.hypot(*(c - b)) + 1e-9:
            raise ValueError(f"vertex {i}: fillet tangent {t:.1f} um exceeds segment.")
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
    kw = dict(tolerance=0.01, precision=1e-4, max_points=0)
    outer = gdspy.FlexPath(pts, width + 2.0 * gap, **kw).to_polygonset()
    inner = gdspy.FlexPath(pts, width, **kw).to_polygonset()
    return gdspy.boolean(outer, inner, 'not', layer=layer, precision=1e-4, max_points=0)


# --------------------------------------------------------------------- #
#                      IDC Builder Function                             #
# --------------------------------------------------------------------- #
def build_IDC(x0, y_center, l0, w, s, h, g, layer):
    """
    Constructs the Interdigital Capacitor geometry at x0, y_center.
    Returns (polygon_set_gaps, (end_x, end_y)).
    """
    polys = []
    seg = s
    n_segments = int(l0 // seg)

    for j in range(n_segments):
        x_lo = x0 + j * seg
        x_hi = x_lo + seg
        if j % 2 == 1:
            polys.append(gdspy.Rectangle((x_lo, y_center + w / 2), (x_hi, y_center + w / 2 + h + g), layer=layer))
            polys.append(gdspy.Rectangle((x_lo, y_center - w / 2 - h - g), (x_hi, y_center - w / 2), layer=layer))
        else:
            finger_pair_index = j // 2
            if finger_pair_index % 2 == 0:
                polys.append(gdspy.Rectangle((x_lo, y_center + w / 2 + h), (x_hi, y_center + w / 2 + h + g), layer=layer))
                polys.append(gdspy.Rectangle((x_lo, y_center - w / 2 - h - g), (x_hi, y_center - w / 2 - h), layer=layer))
            else:
                polys.append(gdspy.Rectangle((x_lo, y_center + w / 2), (x_hi, y_center + w / 2 + g), layer=layer))
                polys.append(gdspy.Rectangle((x_lo, y_center - w / 2 - g), (x_hi, y_center - w / 2), layer=layer))

    actual_len = n_segments * seg
    ps = gdspy.boolean(polys, None, 'or', layer=layer)
    return ps, (x0 + actual_len, y_center), actual_len


# --------------------------------------------------------------------- #
#                   Serpentine Mirror Centerline                        #
# --------------------------------------------------------------------- #
def serpentine_vertices(x0, y_c, n_turns, leg_height, pitch, lead):
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
    delta = (2.0 - np.pi / 2.0) * radius
    return (total - 2.0 * lead - n_turns * pitch + (2 * n_turns + 2) * delta) / n_turns


# --------------------------------------------------------------------- #
#                       Right Launcher Helpers                          #
# --------------------------------------------------------------------- #
def _as_polygonset(polys, layer):
    pts = []
    for p in polys:
        pts.extend(p.polygons)
    return gdspy.PolygonSet(pts, layer=layer)


def _span_at(ps, x, tol):
    ys = []
    for poly in ps.polygons:
        x_lo, x_hi = poly[:, 0].min(), poly[:, 0].max()
        if x_lo <= x + tol and x_hi >= x - tol:
            ys += [poly[:, 1].min(), poly[:, 1].max()]
    return (max(ys) - min(ys)) if ys else 0.0


def build_launcher_template(gap, width, label, layer):
    raw, _ = GM.createLauncherRight(
        (0.0, 0.0), gap=gap, width=width,
        straight=straightLauncher, height=launcherHeight,
        taper=taperLauncher, gapL=gapLauncher, notch=True
    )
    ps = _as_polygonset(raw, layer)
    (x_lo, y_lo), (x_hi, y_hi) = ps.get_bounding_box()
    tol = 0.02 * (x_hi - x_lo)
    if _span_at(ps, x_lo + tol, tol) > _span_at(ps, x_hi - tol, tol):
        ps.mirror((0, 1))
        (x_lo, y_lo), (x_hi, y_hi) = ps.get_bounding_box()
    ps.translate(-x_lo, -0.5 * (y_lo + y_hi))
    length = x_hi - x_lo
    return ps, length


def place_launcher(template, attach_pt, layer):
    ps = gdspy.PolygonSet([p.copy() for p in template.polygons], layer=layer)
    ps.translate(attach_pt[0], attach_pt[1])
    return ps


# --------------------------------------------------------------------- #
#                Optimization & Centered Placement                      #
# --------------------------------------------------------------------- #
pitch = 2.0 * bend_radius + jog_margin
cpw_halfspan = max(w / 2.0 + g for g, w in section_xsec)
y_c = chip_height / 2.0

tmpl_r, len_r = build_launcher_template(launcher_r_gap, launcher_r_width, "Right", layer)

chosen = None
for n_turns in range(turns_min, turns_max + 1):
    h = leg_height_for(total_length, n_turns, pitch, lead_run, bend_radius)
    span = h + 2.0 * cpw_halfspan
    w_mirror = 2.0 * lead_run + n_turns * pitch
    w_total = idc_l0 + w_mirror + len_r
    ok = (h >= 4.0 * bend_radius
          and span <= chip_height - 2.0 * edge_margin
          and w_total <= chip_width)
    if ok and chosen is None:
        # Calculate x0 so whole assembly (IDC + Mirror + Right Launcher) is horizontally centered
        x0_idc = 0.5 * (chip_width - w_total)
        x0_mirror = x0_idc + idc_l0
        verts = serpentine_vertices(x0_mirror, y_c, n_turns, h, pitch, lead_run)
        try:
            chosen = (n_turns, h, w_total, x0_idc, x0_mirror, verts,
                      fillet_polyline(verts, bend_radius, arc_step_deg))
        except ValueError:
            chosen = None

if chosen is None:
    raise RuntimeError("No turn count fits within chip bounds.")

n_turns, leg_height, w_total, x0_idc, x0_mirror, verts, dense = chosen

# Build IDC
idc_poly, idc_attach_end, actual_idc_len = build_IDC(
    x0_idc, y_c, idc_l0, idc_w, idc_s, idc_h, idc_g, layer
)

# Build Mirror
runs, bounds = split_by_lengths(dense, section_lengths)
mirror_polys = []
for k, (run, (g_k, w_k)) in enumerate(zip(runs, section_xsec)):
    mirror_polys.append(cpw_gaps(run, w_k, g_k, layer))

exit_pt = (float(dense[-1][0]), float(dense[-1][1]))

# Build Right Launcher
launcher_r = place_launcher(tmpl_r, exit_pt, layer)

total_structure = [idc_poly] + mirror_polys + [launcher_r]

# --------------------------------------------------------------------- #
#                             Fit Report                                #
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

# --------------------------------------------------------------------- #
#                      Boolean Merge & GDS Export                       #
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