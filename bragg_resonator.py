# -*- coding: utf-8 -*-
"""Half-wave CPW resonator between two Bragg mirrors, with top launchers.

Launcher -> arc -> left mirror -> cavity -> right mirror -> arc -> launcher.
The cavity is a straight line or uses the Bragg unit-cell meanders with a
flat central run. Defaults reproduce the original top-launcher design.

@author: Aryan Soni
"""

from dataclasses import dataclass, replace
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import gdspy
import gds_maker as GM

C = 3e8            # speed of light, m/s
ARC_STEP_DEG = 1.0  # meander arc discretisation


# ======================================================================= #
#                               Parameters                                #
# ======================================================================= #
@dataclass(frozen=True)
class Params:
    # chip (chip_width=None -> set by launcher positions)
    chip_height: float = 6000
    chip_width: Optional[float] = None
    edge_margin: float = 500    # clear ground on all four sides, launchers included

    # mirror cross-sections (um)
    gap1: float = 6
    width1: float = 30
    gap2: float = 16.5
    width2: float = 9
    high_z_first: bool = False   # True: high-Z next to each feedline, False: gap1/width1

    # mirror periods (two per unit-cell structure)
    left_periods: int = 17
    right_periods: Optional[int] = 9   # None -> same as left
    radius: float = 70                 # unit-cell and feedline arc radius
    length1: float = 337
    addlength: float = 150

    # 50-ohm feedline (0 length -> omitted)
    feed_gap: float = 10.9
    feed_width: float = 20.2
    feed_left: float = 600
    feed_right: float = 600

    # launchers
    launchers: bool = True
    launcher_straight: float = 425
    launcher_taper: float = 425
    launcher_height: float = 622
    launcher_gap: float = 161

    # electrical
    frequency: float = 6e9
    epsilon: float = 11.9

    # cavity (None -> feedline cross-section / half wavelength)
    cavity_gap: Optional[float] = None
    cavity_width: Optional[float] = None
    cavity_length: Optional[float] = None

    # meander cavity: left unit cell -> flat central run -> right unit cell
    cavity_meander: bool = False
    cavity_flat: float = 1000      # straight run in the centre

    # layers
    layer: int = 1        # ground
    trace_layer: int = 3
    gap_layer: int = 2

    @property
    def qw(self) -> float:
        """Quarter-wave length in um."""
        return 1e6 * C / (np.sqrt((self.epsilon + 1) / 2) * self.frequency * 4)

    @property
    def left_cells(self) -> int:
        return self.left_periods // 2

    @property
    def right_cells(self) -> int:
        n = self.left_periods if self.right_periods is None else self.right_periods
        return n // 2

    @property
    def cavity_len(self) -> float:
        return 2 * self.qw if self.cavity_length is None else self.cavity_length

    @property
    def cavity_xsec(self):
        return (self.feed_gap if self.cavity_gap is None else self.cavity_gap,
                self.feed_width if self.cavity_width is None else self.cavity_width)

    @property
    def mirror_xsecs(self):
        """((left first, left second), (right first, right second)), cavity side first on the right."""
        a, b = (self.gap1, self.width1), (self.gap2, self.width2)
        if self.high_z_first:
            hi, lo = (a, b) if self.width1 / self.gap1 <= self.width2 / self.gap2 else (b, a)
            return (hi, lo), (lo, hi)
        return (a, b), (a, b)

    @property
    def cell_lengths(self) -> dict:
        r, qw = self.radius, self.qw
        l1, add = self.length1, self.addlength
        v1 = (qw - np.pi * r * 6 / 2 - 4 * r - 2 * l1 - 2 * add) / 2
        v2 = (qw - np.pi * r) / 2
        return {
            "vertical_length1": v1,
            "vertical_length2": v2,
            "vertical_length3": 2 * v1 - v2,
            "vertical_length4": qw - np.pi * r - 2 * v1,
            "vertical_length5": (v1 + v2) / 2 - r,
            "downlength": abs((v1 + v2 + 2 * add + 2 * l1 + 6 * r) / 2
                              - v1 - add - l1 - 3 * r) - r,
        }

    def check(self):
        if self.left_cells < 1 or self.right_cells < 1:
            raise ValueError("Each mirror needs at least 2 periods (one unit-cell structure).")
        for name, v in self.cell_lengths.items():
            if v < 0:
                raise ValueError(f"{name} is negative ({v:.2f} um); check radius, "
                                 f"length1, addlength, frequency, epsilon.")

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


# ======================================================================= #
#                               Geometry                                  #
# ======================================================================= #
def end_cap(p: Params, tip):
    """Open end above a downward line tip (no launcher)."""
    half = p.feed_width / 2.0 + p.feed_gap
    return gdspy.Rectangle((tip[0] - half, tip[1]), (tip[0] + half, tip[1] + p.feed_gap),
                           layer=p.gap_layer)


def build(p: Params):
    """Chain the gds_maker pieces. Returns (gap polygons, end caps, info)."""
    p.check()
    r, fg, fw, layer = p.radius, p.feed_gap, p.feed_width, p.gap_layer
    L = p.cell_lengths
    (lf, ls), (rf, rs) = p.mirror_xsecs

    y_feed = p.chip_height / 2.0
    y_start = p.chip_height - p.edge_margin - p.launcher_gap   # pad gap reaches y_start + launcher_gap
    y_tip = y_start - (p.launcher_straight + p.launcher_taper)
    x_left = p.launcher_height / 2.0 + p.edge_margin
    x_feed_end = p.launcher_gap + p.launcher_straight + p.launcher_taper + p.feed_left

    def cell_kw(a, b):
        return dict(addlength=p.addlength,
                    vertical_length1=L["vertical_length1"], vertical_length2=L["vertical_length2"],
                    vertical_length3=L["vertical_length3"], vertical_length4=L["vertical_length4"],
                    length1=p.length1, radius=r,
                    gap1=a[0], width1=a[1], gap2=b[0], width2=b[1], layer=layer)

    def launcher(x):
        return GM.Createlauncherdown(
            [x, y_start], straight=p.launcher_straight, height=p.launcher_height,
            taper=p.launcher_taper, gapL=p.launcher_gap, gap=fg, width=fw,
            layer=layer, notch=True)

    left_kw, right_kw = cell_kw(lf, ls), cell_kw(rf, rs)
    polys, caps = [], []

    # upper-left launcher, CPW down, arc to horizontal
    if p.launchers:
        pl, pos = launcher(x_left)
        polys += pl
    else:
        pos = (x_left, y_tip)
        caps.append(end_cap(p, pos))
    entry = (float(pos[0]), float(pos[1]))

    down = pos[1] - (y_feed + r)
    if down < 0:
        raise ValueError(f"Left CPW-down length is negative ({down:.1f} um); increase chip_height.")
    pl, pos_a = GM.createCPWvertDown(pos, length=down, gap=fg, width=fw, layer=layer)
    polys += pl
    pl, _ = GM.createDoublearc((pos_a[0] + r, pos_a[1]), radius=r, gap=fg, width=fw,
                               angle=np.pi, layer=layer)
    polys += pl
    pos_b = (pos_a[0] + r, pos_a[1] - r)

    run = x_feed_end - pos_b[0]
    if run < 0:
        raise ValueError(f"Left horizontal CPW length is negative ({run:.1f} um).")
    pl, pos_c = GM.createCPWhor(pos_b, length=run, gap=fg, width=fw, layer=layer)
    polys += pl

    # entry arc and run down to the left mirror
    pl, pos = GM.createDoublearc((pos_c[0], pos_c[1] - r), radius=r, gap=fg, width=fw,
                                 angle=2 * np.pi, layer=layer)
    polys += pl
    pl, pos = GM.createCPWvertDown(pos, length=L["downlength"], gap=fg, width=fw, layer=layer)
    polys += pl

    # left mirror: unit cell 1 x (n-1), then unit cell 2
    for _ in range(p.left_cells - 1):
        pl, pos = GM.Braggmirror_unitcell1_left(pos, **left_kw)
        polys += pl
    pl, pos = GM.Braggmirror_unitcell2_left(pos, **left_kw)
    polys += pl

    # cavity
    cg, cw = p.cavity_xsec
    info = {"cavity": None}
    if p.cavity_meander:
        # Re-use Bragg unit cell meanders with cavity cross-section
        left_cav_kw = dict(left_kw, gap1=cg, width1=cw, gap2=cg, width2=cw)
        right_cav_kw = dict(right_kw, gap1=cg, width1=cw, gap2=cg, width2=cw)

        # 1. Left Bragg unit-cell meander
        pl, pos = GM.Braggmirror_unitcell1_left(pos, **left_cav_kw)
        polys += pl

        # 2. Flat center section
        if p.cavity_flat > 0:
            pl, pos = GM.createCPWhor(pos, length=p.cavity_flat, gap=cg, width=cw, layer=layer)
            polys += pl

        # 3. Right Bragg unit-cell meander
        pl, pos = GM.Braggmirror_unitcell1_right(pos, **right_cav_kw)
        polys += pl

        info["cavity"] = {"type": "meander_bragg_cell", "flat": p.cavity_flat}
    else:
        pl, pos = GM.createCPWhor(pos, length=p.cavity_len, gap=cg, width=cw, layer=layer)
        polys += pl

    # right mirror: unit cell 2, then unit cell 1 x (n-1)
    pl, pos = GM.Braggmirror_unitcell2_right(
        pos, **dict(right_kw, vertical_length5=L["vertical_length5"]))
    polys += pl
    for _ in range(p.right_cells - 1):
        pl, pos = GM.Braggmirror_unitcell1_right(pos, **right_kw)
        polys += pl

    # run down, arc to horizontal, feedline, arc up
    pl, pos_d = GM.createCPWvertDown(pos, length=L["downlength"], gap=fg, width=fw, layer=layer)
    polys += pl
    pl, pos_e = GM.createDoublearc((pos_d[0] + r, pos_d[1]), radius=r, gap=fg, width=fw,
                                   angle=-np.pi, layer=layer)
    polys += pl
    pos_e = (pos_e[0] + r, pos_e[1] - r)

    pos_f = pos_e
    if p.feed_right:
        pl, pos_f = GM.createCPWhor(pos_e, length=p.feed_right, gap=fg, width=fw, layer=layer)
        polys += pl
    pl, _ = GM.createDoublearc((pos_f[0], pos_f[1] + r), radius=r, gap=fg, width=fw,
                               angle=-np.pi / 2, layer=layer)
    polys += pl
    pos_g = (pos_f[0] + r, pos_f[1] + r)

    up = y_tip - pos_g[1]
    if up < 0:
        raise ValueError(f"Right CPW-up length is negative ({up:.1f} um); increase chip_height.")
    pl, pos = GM.createCPWvertUp(pos_g, length=up, gap=fg, width=fw, layer=layer)
    polys += pl

    # upper-right launcher
    x_right = float(pos[0])
    if p.launchers:
        pl, _ = launcher(x_right)
        polys += pl
    else:
        caps.append(end_cap(p, (x_right, y_tip)))
    exit_ = (x_right, float(y_tip))

    info.update(entry=entry, exit=exit_, x_left=x_left, x_right=x_right,
                probes=[(entry[0], entry[1] - 1.0), (exit_[0], exit_[1] - 1.0)])
    return polys, caps, info


# ======================================================================= #
#                           Layout and output                             #
# ======================================================================= #
def as_polyset(polys, layer):
    pts = []
    for q in polys:
        pts.extend(q.polygons if hasattr(q, "polygons") else q.get_polygons())
    return gdspy.PolygonSet(pts, layer=layer)


def bbox_of(polys):
    lo, hi = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
    for q in polys:
        bb = q.get_bounding_box()
        if bb is not None:
            lo, hi = np.minimum(lo, bb[0]), np.maximum(hi, bb[1])
    return lo, hi


def layout(p: Params, verbose=True) -> dict:
    """Build the structure and place it on the chip."""
    polys, caps, info = build(p)
    gaps = [as_polyset(polys, p.gap_layer)] + caps
    lo, hi = bbox_of(gaps)

    dx = 0.0
    if p.chip_width is None:
        chip_w = (info["x_right"] + p.launcher_height / 2.0 + p.edge_margin
                  if p.launchers else float(hi[0] + lo[0]))
    else:
        chip_w = float(p.chip_width)
        dx = 0.5 * (chip_w - (hi[0] - lo[0])) - lo[0]   # centre in x
        for g in gaps:
            g.translate(dx, 0.0)
        lo, hi = lo + np.array([dx, 0.0]), hi + np.array([dx, 0.0])

    if verbose:
        print(f"Chip width = {chip_w:.1f} um{' (auto)' if p.chip_width is None else ''}; "
              f"launchers at x = {info['x_left'] + dx:.1f} / {info['x_right'] + dx:.1f}")
        cav = info["cavity"]
        if cav is None:
            print(f"Straight cavity: {p.cavity_len:.1f} um")
        else:
            print(f"Meander cavity (Bragg unit-cell based): flat run {cav['flat']:.1f} um")

    return {"gaps": gaps, "probes": [(x + dx, y) for x, y in info["probes"]],
            "lo": lo, "hi": hi, "chip_width": chip_w, "cavity": info["cavity"]}


def extract_trace(p: Params, chip_w, all_gaps, probes):
    """Trace = the island of (chip NOT gaps) that contains the probe points."""
    chip = gdspy.Rectangle((0, 0), (chip_w, p.chip_height))
    solid = gdspy.boolean(chip, all_gaps, 'not', precision=1e-4, max_points=0)
    keep = []
    if solid is not None:
        keep = [q for q in solid.polygons
                if any(gdspy.inside([pt], [q])[0] for pt in probes)]
    if not keep:
        raise RuntimeError("Could not isolate the trace from the gap geometry.")
    return gdspy.PolygonSet(keep, layer=p.trace_layer)


def design_only(p: Params, verbose=True) -> dict:
    """Fit check without the CPW booleans or GDS write."""
    base = {"left_periods": p.left_periods, "right_periods": p.right_periods,
            "frequency": p.frequency, "qw_um": p.qw, "cavity_um": p.cavity_len}
    try:
        lay = layout(p, verbose=verbose)
    except (ValueError, RuntimeError) as e:
        return {"ok": False, "error": str(e), **base}

    lo, hi, m = lay["lo"], lay["hi"], p.edge_margin
    width_slack = float(min(lo[0], lay["chip_width"] - hi[0])) - m   # spare room beyond the margin
    height_slack = float(min(lo[1], p.chip_height - hi[1])) - m
    ok = width_slack >= 0 and height_slack >= 0

    if verbose:
        print(f"Quarter-wave length {p.qw:.1f} um; mirrors {2 * p.left_cells} / "
              f"{2 * p.right_cells} sections")
        print(f"Assembly: {hi[0] - lo[0]:.1f} x {hi[1] - lo[1]:.1f} um; slack x {width_slack:.1f}, "
              f"y {height_slack:.1f} um ({'fits' if ok else 'DOES NOT FIT'})")

    out = {"ok": ok, **base,
           "left_sections": 2 * p.left_cells, "right_sections": 2 * p.right_cells,
           "chip_width_um": lay["chip_width"],
           "width_um": float(hi[0] - lo[0]), "height_um": float(hi[1] - lo[1]),
           "width_slack_um": width_slack, "height_slack_um": height_slack}
    if not ok:
        out["error"] = (f"does not fit the chip (slack x {width_slack:.1f}, "
                        f"y {height_slack:.1f} um, edge margin {m:g} um)")
    return out


def generate_gds(p: Params, filename: str, write_gaps=False, view=False, verbose=True) -> dict:
    """Write `<filename>.gds`: ground + trace, or (write_gaps) trace + gap slots."""
    lay = layout(p, verbose=verbose)
    chip_w = lay["chip_width"]
    all_gaps = gdspy.boolean(lay["gaps"], None, 'or', layer=p.gap_layer,
                             precision=1e-4, max_points=0)
    trace = extract_trace(p, chip_w, all_gaps, lay["probes"])

    lo, hi = bbox_of([trace, all_gaps])
    off_chip = min(lo[0], lo[1], chip_w - hi[0], p.chip_height - hi[1]) < 0
    if verbose:
        print(f"\nBbox: x {lo[0]:.1f} to {hi[0]:.1f}, y {lo[1]:.1f} to {hi[1]:.1f}; slack left "
              f"{lo[0]:.1f}, right {chip_w - hi[0]:.1f}, bottom {lo[1]:.1f}, "
              f"top {p.chip_height - hi[1]:.1f} um")
        if off_chip:
            print("ERROR: geometry runs off the chip.")

    gdspy.current_library = gdspy.GdsLibrary()   # fresh library so sweeps don't collide
    cell = gdspy.Cell('merge')
    if write_gaps:
        if trace is not None:
            cell.add(trace)
        if all_gaps is not None:
            cell.add(all_gaps)
    else:
        chip = gdspy.Rectangle((0, 0), (chip_w, p.chip_height))
        cutouts = gdspy.boolean([all_gaps, trace], None, 'or', precision=1e-4, max_points=0)
        
        if cutouts is not None:
            ground = gdspy.boolean(chip, cutouts, 'not', layer=p.layer, precision=1e-4, max_points=0)
        else:
            ground = chip

        if ground is not None:
            cell.add(ground)
        if trace is not None:
            cell.add(trace)

    out_name = filename + ".gds"
    gdspy.write_gds(out_name, unit=1.0e-6, precision=1.0e-9)
    if view:
        gdspy.LayoutViewer(cells=cell)
    if verbose:
        print(f"\nWrote {out_name}  (trace layer {p.trace_layer}, "
              f"{'gaps layer ' + str(p.gap_layer) if write_gaps else 'ground layer ' + str(p.layer)})")

    return {"ok": not off_chip, "filename": out_name, "chip_width_um": chip_w,
            "width_um": float(hi[0] - lo[0]), "height_um": float(hi[1] - lo[1]),
            "bbox_lo": tuple(lo), "bbox_hi": tuple(hi), "off_chip": off_chip}


# ======================================================================= #
#                                 Sweeps                                  #
# ======================================================================= #
def sweep(base: Params, grid: dict, verbose=False) -> pd.DataFrame:
    """Fit check over the cartesian product of `grid` (field -> values)."""
    keys = list(grid)
    rows = [design_only(base.with_(**dict(zip(keys, combo))), verbose=verbose)
            for combo in product(*(grid[k] for k in keys))]
    return pd.DataFrame(rows)


def sweep_and_generate(base: Params, grid: dict, prefix="bragg", only_ok=True,
                       write_gaps=False, view=False, verbose=False) -> pd.DataFrame:
    """Like sweep(), but writes one .gds per combination (named from the swept fields)."""
    keys = list(grid)
    rows = []
    for combo in product(*(grid[k] for k in keys)):
        overrides = dict(zip(keys, combo))
        p = base.with_(**overrides)
        fit = design_only(p, verbose=verbose)
        fit["filename"] = None
        if fit["ok"] or not only_ok:
            name = "_".join([prefix] + [f"{k}-{v:g}" if isinstance(v, float) else f"{k}-{v}"
                                        for k, v in overrides.items()])
            try:
                gen = generate_gds(p, name, write_gaps=write_gaps, view=view, verbose=verbose)
                fit["filename"], fit["off_chip"] = gen["filename"], gen["off_chip"]
            except Exception as e:
                fit["ok"], fit["error"] = False, str(e)
        rows.append(fit)

    df = pd.DataFrame(rows)
    print(f"\nsweep_and_generate: {df['filename'].notna().sum()}/{len(df)} combos written.")
    return df


if __name__ == "__main__":
    params = Params()
    if design_only(params)["ok"]:
        generate_gds(params, "half_wave_resonator_top_launchers", view=True)