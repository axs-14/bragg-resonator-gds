# -*- coding: utf-8 -*-
"""
Stepped-impedance (meander Bragg) filter GDS generator.

Layers: ground plane = layer, meander trace = trace_layer, gap slots = gap_layer.

@author: Aryan Soni
"""

from dataclasses import dataclass, replace
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import gdspy
import gds_maker as GM  # launchers only


# ======================================================================= #
#                              Parameters                                 #
# ======================================================================= #
@dataclass(frozen=True)
class BraggParams:
    # Chip
    chip_height: float = 5000          # um
    chip_width: float = 10000          # um
    edge_margin: float = 250.0         # um, top/bottom keep-out

    # Mirror cross-sections
    gap1: float = 23     # ~93 ohm
    width1: float = 4
    gap2: float = 4      # ~30 ohm
    width2: float = 42

    # Mirrors start on high-Z at the outer end. Plain mirror: count rounded
    # up to odd so both ends are high-Z. Cavity: each mirror starts high-Z
    # at its feedline and alternates inward; no rounding, and whatever
    # section lands next to the defect meets it.
    high_z_ends: bool = True

    # Launchers. None gap/width -> match the connecting cross-section.
    include_launchers: bool = True
    launcher_l_gap: Optional[float] = None
    launcher_l_width: Optional[float] = None
    launcher_r_gap: Optional[float] = None
    launcher_r_width: Optional[float] = None

    # Feedline: straight CPW between mirror and launcher (or end cap).
    # Not part of the resonator length. feedline_ends: "both"|"left"|"right"
    include_feedline: bool = False
    feedline_length: float = 800.0     # um per end
    feedline_width: float = 10.0       # um
    feedline_gap: float = 6.0          # um
    feedline_ends: str = "both"

    # End cap depth when include_launchers is False (None -> section gap)
    end_cap_depth: Optional[float] = None

    # Launcher bond pad
    straightLauncher: float = 720
    taperLauncher: float = 600
    launcherHeight: float = 1040
    gapLauncher: float = 220

    # Serpentine
    bend_radius: float = 400.0
    jog_margin: float = 20.0
    lead_run: float = 500.0
    arc_step_deg: float = 1.0
    turns_min: int = 3
    turns_max: int = 14

    # Electrical
    frequency: float = 7.2e9
    epsilon: float = 11.7
    c: float = 3e8
    no_of_quarterwave_resonators: int = 6   # left mirror count when include_defect

    # Defect: lambda/2 element between two mirrors.
    #   n_right_sections: None -> same as left
    #   defect_gap/width: None -> low-Z cross-section
    include_defect: bool = False
    n_right_sections: Optional[int] = None
    defect_gap: Optional[float] = None
    defect_width: Optional[float] = None
    defect_length: Optional[float] = None   # um; None -> lambda/2 (2*qw)

    # Layers
    layer: int = 1
    trace_layer: int = 3
    gap_layer: int = 2

    # ---------------- derived ----------------
    @property
    def quarterwave_resonator_length(self) -> float:
        return 1e6 * self.c / (np.sqrt((self.epsilon + 1) / 2) * self.frequency * 4)

    @property
    def hi_lo_xsec(self):
        """(high_Z, low_Z) as (gap, width); higher Z = narrower width/gap."""
        a = (self.gap1, self.width1)
        b = (self.gap2, self.width2)
        ra = self.width1 / self.gap1 if self.gap1 else np.inf
        rb = self.width2 / self.gap2 if self.gap2 else np.inf
        return (a, b) if ra <= rb else (b, a)

    @property
    def _alternation(self):
        if self.high_z_ends:
            return self.hi_lo_xsec
        return (self.gap1, self.width1), (self.gap2, self.width2)

    def _odd_up(self, n) -> int:
        n = int(n)
        if self.high_z_ends and not self.include_defect and n % 2 == 0:
            n += 1
        return n

    @property
    def n_left_sections(self) -> int:
        return self._odd_up(self.no_of_quarterwave_resonators)

    @property
    def n_right_sections_eff(self) -> int:
        n = (self.n_right_sections if self.n_right_sections is not None
             else self.no_of_quarterwave_resonators)
        return self._odd_up(n)

    # ---------------- feedline ----------------
    @property
    def feedline_xsec(self):
        return (self.feedline_gap, self.feedline_width)

    def _feed_len(self, side: str) -> float:
        if not self.include_feedline:
            return 0.0
        if self.feedline_ends not in ("both", "left", "right"):
            raise ValueError(
                f"feedline_ends must be 'both', 'left' or 'right', "
                f"got {self.feedline_ends!r}")
        return float(self.feedline_length) if self.feedline_ends in ("both", side) else 0.0

    @property
    def feed_len_left(self) -> float:
        return self._feed_len("left")

    @property
    def feed_len_right(self) -> float:
        return self._feed_len("right")

    @property
    def lead_in(self) -> float:
        """Left lead run, stretched so the feedline stays ahead of the first bend."""
        if not self.feed_len_left:
            return self.lead_run
        return max(self.lead_run, self.feed_len_left + self.bend_radius)

    @property
    def lead_out(self) -> float:
        if not self.feed_len_right:
            return self.lead_run
        return max(self.lead_run, self.feed_len_right + self.bend_radius)

    # ---------------- section stack ----------------
    @property
    def sections(self):
        """[(length_um, gap, width, label), ...] end to end."""
        qw = self.quarterwave_resonator_length
        hi, _lo = self.hi_lo_xsec
        first, second = self._alternation

        def tag(gw):
            return "hi-Z" if tuple(gw) == tuple(hi) else "lo-Z"

        secs = []

        if self.feed_len_left:
            g, w = self.feedline_xsec
            secs.append((self.feed_len_left, g, w, "feedline L"))

        pre = "mirror-L" if self.include_defect else "section"
        for k in range(self.n_left_sections):
            gw = first if k % 2 == 0 else second
            secs.append((qw, gw[0], gw[1], f"{pre} {k + 1} [{tag(gw)}]"))

        if self.include_defect:
            dg = self.defect_gap if self.defect_gap is not None else second[0]
            dw = self.defect_width if self.defect_width is not None else second[1]
            dl = self.defect_length if self.defect_length is not None else 2.0 * qw
            label = ("DEFECT [lambda/2]" if self.defect_length is None
                     else f"DEFECT [{dl:.0f} um]")
            secs.append((dl, dg, dw, label))

            # Right mirror built outside-in from the feedline, then reversed.
            n_right = self.n_right_sections_eff
            right = [first if j % 2 == 0 else second for j in range(n_right)]
            for j, gw in enumerate(reversed(right)):
                secs.append((qw, gw[0], gw[1], f"mirror-R {j + 1} [{tag(gw)}]"))

        if self.feed_len_right:
            g, w = self.feedline_xsec
            secs.append((self.feed_len_right, g, w, "feedline R"))

        return secs

    @property
    def section_lengths(self):
        return [s[0] for s in self.sections]

    @property
    def section_xsec(self):
        return [(s[1], s[2]) for s in self.sections]

    @property
    def section_labels(self):
        return [s[3] for s in self.sections]

    @property
    def n_high_z(self) -> int:
        return sum(1 for lbl in self.section_labels if "hi-Z" in lbl)

    @property
    def n_low_z(self) -> int:
        return sum(1 for lbl in self.section_labels if "lo-Z" in lbl)

    @property
    def total_length(self) -> float:
        """Total centerline length, feedlines included."""
        return float(np.sum(self.section_lengths))

    @property
    def resonator_length_total(self) -> float:
        return self.total_length - self.feed_len_left - self.feed_len_right

    # ---------------- launcher cross-sections ----------------
    @property
    def launcher_l_xsec(self):
        g, w = self.section_xsec[0]
        return (self.launcher_l_gap if self.launcher_l_gap is not None else g,
                self.launcher_l_width if self.launcher_l_width is not None else w)

    @property
    def launcher_r_xsec(self):
        g, w = self.section_xsec[-1]
        return (self.launcher_r_gap if self.launcher_r_gap is not None else g,
                self.launcher_r_width if self.launcher_r_width is not None else w)

    def with_(self, **overrides) -> "BraggParams":
        return replace(self, **overrides)


# ======================================================================= #
#                          Centerline geometry                            #
# ======================================================================= #
def _unit(v):
    n = np.hypot(v[0], v[1])
    return v / n if n else v


def fillet_polyline(pts, radius, arc_step_deg=1.0):
    """Replace interior vertices with circular arcs; returns a dense point list."""
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
    """Cut a polyline into runs of the given arc lengths.

    Returns (runs, [(boundary_point, segment_index), ...]).
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


def cpw_center_and_gap(pts, width, gap, center_layer, gap_layer):
    """Center conductor and gap-slot PolygonSets for a centerline."""
    kw = dict(tolerance=0.01, precision=1e-4, max_points=0)
    inner = gdspy.FlexPath(pts, width, layer=center_layer, **kw).to_polygonset()
    outer = gdspy.FlexPath(pts, width + 2.0 * gap, **kw).to_polygonset()
    gap_ps = gdspy.boolean(outer, inner, 'not', layer=gap_layer,
                            precision=1e-4, max_points=0)
    return inner, gap_ps


def serpentine_vertices(x0, y_c, n_turns, leg_height, pitch, lead_in,
                        lead_out=None):
    """Boustrophedon centerline, entering and leaving horizontally on y_c."""
    if lead_out is None:
        lead_out = lead_in
    pts = [(x0, y_c)]
    x = x0 + lead_in
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
    pts.append((x + lead_out, y_c))
    return pts


def leg_height_for(total, n_turns, pitch, lead_in, radius, lead_out=None):
    """Leg height giving a filleted serpentine of exactly `total` length."""
    if lead_out is None:
        lead_out = lead_in
    delta = (2.0 - np.pi / 2.0) * radius
    return (total - lead_in - lead_out - n_turns * pitch
            + (2 * n_turns + 2) * delta) / n_turns


# ======================================================================= #
#                            Launcher helpers                             #
# ======================================================================= #
def _as_polygonset(polys, layer):
    pts = []
    for p in polys:
        pts.extend(p.polygons)
    return gdspy.PolygonSet(pts, layer=layer)


def place_launcher(template, template_final_pos, attach_pt, layer):
    """Translate a launcher copy so its narrow end lands on attach_pt."""
    ps = gdspy.PolygonSet([p.copy() for p in template.polygons], layer=layer)
    ps.translate(attach_pt[0] - template_final_pos[0],
                 attach_pt[1] - template_final_pos[1])
    return ps


def bbox_of(polys):
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    for p in polys:
        if p is None:
            continue
        bb = p.get_bounding_box()
        if bb is None:
            continue
        lo = np.minimum(lo, bb[0])
        hi = np.maximum(hi, bb[1])
    return lo, hi


def _build_launcher_template(p: BraggParams, gap, width, side, label, verbose=True):
    """Launcher gap pattern on p.gap_layer, built at (0, 0)."""
    fn = GM.createLauncherLeft if side == 'left' else GM.createLauncherRight
    raw, final_pos = fn(
        (0.0, 0.0), gap=gap, width=width,
        straight=p.straightLauncher, height=p.launcherHeight,
        taper=p.taperLauncher, gapL=p.gapLauncher, notch=True
    )
    ps = _as_polygonset(raw, p.gap_layer)
    (x_lo, _), (x_hi, _) = ps.get_bounding_box()
    length = x_hi - x_lo
    if verbose:
        print(f"{label} launcher: {length:.1f} um long, tapers to "
              f"{width:.1f}/{gap:.1f} "
              f"(final_position={tuple(round(v, 1) for v in final_pos)})")
    return ps, final_pos, length


def _launcher_templates(p: BraggParams, verbose=True):
    if not p.include_launchers:
        return None, None, 0.0, None, None, 0.0
    gap_l, width_l = p.launcher_l_xsec
    gap_r, width_r = p.launcher_r_xsec
    tmpl_l, final_l, len_l = _build_launcher_template(
        p, gap_l, width_l, 'left', "Left", verbose)
    tmpl_r, final_r, len_r = _build_launcher_template(
        p, gap_r, width_r, 'right', "Right", verbose)
    return tmpl_l, final_l, len_l, tmpl_r, final_r, len_r


# ======================================================================= #
#                              Turn search                                #
# ======================================================================= #
def _choose_turns(p: BraggParams, len_l, len_r, verbose=True):
    """Fewest turns that fit the chip.

    Returns (n_turns, leg_height, w_total, span, verts, dense).
    """
    pitch = 2.0 * p.bend_radius + p.jog_margin
    cpw_halfspan = max(w / 2.0 + g for g, w in p.section_xsec)
    y_c = p.chip_height / 2.0
    lead_in, lead_out = p.lead_in, p.lead_out

    if verbose:
        print("\nturns   leg height   mirror width   total width   vertical span")
    chosen = None
    for n_turns in range(p.turns_min, p.turns_max + 1):
        h = leg_height_for(p.total_length, n_turns, pitch, lead_in,
                           p.bend_radius, lead_out)
        span = h + 2.0 * cpw_halfspan
        w_mirror = lead_in + lead_out + n_turns * pitch
        w_total = w_mirror + len_l + len_r
        ok = (h >= 4.0 * p.bend_radius
              and span <= p.chip_height - 2.0 * p.edge_margin
              and w_total <= p.chip_width)
        if verbose:
            print(f"{n_turns:5d}   {h:10.1f}   {w_mirror:12.1f}   {w_total:11.1f}"
                  f"   {span:13.1f}   {'fits' if ok else ''}")
        if ok and chosen is None:
            x0 = 0.5 * (p.chip_width - w_total) + len_l
            verts = serpentine_vertices(x0, y_c, n_turns, h, pitch,
                                        lead_in, lead_out)
            try:
                dense = fillet_polyline(verts, p.bend_radius, p.arc_step_deg)
                chosen = (n_turns, h, w_total, span, verts, dense)
            except ValueError:
                chosen = None

    if chosen is None:
        raise RuntimeError(
            "No turn count fits. Try a smaller bend_radius, a smaller "
            "edge_margin, a shorter feedline_length, or split the mirror "
            "into two stacked banks."
        )
    return chosen


# ======================================================================= #
#                        Fast path: fit check only                        #
# ======================================================================= #
def design_only(p: BraggParams, verbose=True) -> dict:
    """Everything up to the CPW booleans and GDS write."""
    tmpl_l, final_l, len_l, tmpl_r, final_r, len_r = _launcher_templates(p, verbose=verbose)

    if verbose:
        print(f"\nStack: {len(p.sections)} sections -- "
              f"{p.n_high_z} high-Z, {p.n_low_z} low-Z"
              + (", + half-wave defect" if p.include_defect else "")
              + (f", + feedline ({p.feedline_ends})" if p.include_feedline else ""))
        print("  " + " | ".join(p.section_labels))
        if p.n_left_sections != p.no_of_quarterwave_resonators:
            print(f"  note: section count rounded {p.no_of_quarterwave_resonators}"
                  f" -> {p.n_left_sections} so both ends are high-Z.")
        if p.include_defect and p.n_left_sections != p.n_right_sections_eff:
            print(f"  note: asymmetric mirror section counts -- left "
                  f"{p.n_left_sections}, right {p.n_right_sections_eff}.")

    try:
        n_turns, leg_height, w_total, span, verts, dense = _choose_turns(
            p, len_l, len_r, verbose=verbose)
    except RuntimeError as e:
        return {
            "ok": False, "error": str(e),
            "no_of_quarterwave_resonators": p.no_of_quarterwave_resonators,
            "n_sections": len(p.sections),
            "n_high_z": p.n_high_z, "n_low_z": p.n_low_z,
            "high_z_ends": p.high_z_ends,
            "include_feedline": p.include_feedline,
            "feedline_length": p.feedline_length,
            "gap1": p.gap1, "width1": p.width1, "gap2": p.gap2, "width2": p.width2,
            "frequency": p.frequency,
        }

    runs, bounds = split_by_lengths(dense, p.section_lengths)
    actual_len = arclen(dense)

    bend_warnings = 0
    for k, (q, i) in enumerate(bounds):
        a = np.asarray(dense[max(i - 1, 0)])
        b = np.asarray(dense[i])
        cpt = np.asarray(dense[min(i + 2, len(dense) - 1)])
        d1, d2 = _unit(b - a), _unit(cpt - b)
        if abs(np.degrees(np.arccos(np.clip(np.dot(d1, d2), -1, 1)))) > 0.5:
            bend_warnings += 1
            if verbose:
                print(f"  WARNING: step {k + 1}->{k + 2} at "
                      f"({q[0]:.1f}, {q[1]:.1f}) falls inside a bend.")

    pitch = 2.0 * p.bend_radius + p.jog_margin
    cpw_halfspan = max(w / 2.0 + g for g, w in p.section_xsec)
    ground_gap = pitch - 2.0 * cpw_halfspan
    coupling_warning = ground_gap < 100.0

    if verbose:
        print(f"\nSerpentine: {n_turns} turns, leg height {leg_height:.1f} um, "
              f"pitch {pitch:.1f} um, radius {p.bend_radius:.1f} um")
        if p.include_feedline:
            print(f"Leads: in {p.lead_in:.1f} um, out {p.lead_out:.1f} um "
                  f"(feedline {p.feed_len_left:.1f} / {p.feed_len_right:.1f} um "
                  f"at {p.feedline_width:.1f}/{p.feedline_gap:.1f})")
        print(f"Centerline length {actual_len:.1f} um "
              f"(target {p.total_length:.1f}, error {actual_len - p.total_length:+.3f})")
        if coupling_warning:
            print(f"WARNING: only {ground_gap:.1f} um of ground between "
                  f"adjacent legs.")

    return {
        "ok": True,
        "no_of_quarterwave_resonators": p.no_of_quarterwave_resonators,
        "high_z_ends": p.high_z_ends,
        "n_sections": len(p.sections),
        "n_left_sections": p.n_left_sections,
        "n_right_sections_eff": p.n_right_sections_eff if p.include_defect else None,
        "n_high_z": p.n_high_z,
        "n_low_z": p.n_low_z,
        "include_defect": p.include_defect,
        "n_right_sections": p.n_right_sections,
        "include_feedline": p.include_feedline,
        "feedline_ends": p.feedline_ends,
        "feedline_length": p.feedline_length,
        "feedline_width": p.feedline_width,
        "feedline_gap": p.feedline_gap,
        "feed_len_total_um": p.feed_len_left + p.feed_len_right,
        "gap1": p.gap1, "width1": p.width1, "gap2": p.gap2, "width2": p.width2,
        "frequency": p.frequency, "epsilon": p.epsilon,
        "quarterwave_length_um": p.quarterwave_resonator_length,
        "resonator_length_um": p.resonator_length_total,
        "n_turns": n_turns,
        "leg_height_um": leg_height,
        "w_total_um": w_total,
        "vertical_span_um": span,
        "width_slack_um": p.chip_width - w_total,
        "height_slack_um": (p.chip_height - 2.0 * p.edge_margin) - span,
        "actual_length_um": actual_len,
        "length_error_um": actual_len - p.total_length,
        "bend_step_warnings": bend_warnings,
        "ground_gap_um": ground_gap,
        "coupling_warning": coupling_warning,
        "left_launcher_len_um": len_l,
        "right_launcher_len_um": len_r,
    }


# ======================================================================= #
#                        Slow path: geometry + GDS                        #
# ======================================================================= #
def generate_gds(p: BraggParams, filename: str, write_gaps: bool = False,
                  view: bool = False, verbose: bool = True) -> dict:
    """Build the CPW geometry and write `<filename>.gds`.

    write_gaps=False: ground on p.layer, trace on p.trace_layer.
    write_gaps=True:  trace on p.trace_layer, gap slots on p.gap_layer, no ground.
    """
    tmpl_l, final_l, len_l, tmpl_r, final_r, len_r = _launcher_templates(p, verbose=verbose)
    n_turns, leg_height, w_total, span, verts, dense = _choose_turns(
        p, len_l, len_r, verbose=verbose)

    sections = p.sections
    runs, bounds = split_by_lengths(dense, p.section_lengths)
    # Keep the first mirror section horizontal where it meets the feedline
    if len(runs) > 1 and len(runs[1]) > 1:
        runs[1][1][1] = runs[1][0][1]
    assert len(runs) == len(sections)

    center_polys, gap_polys = [], []
    for run, (_len_k, g_k, w_k, label) in zip(runs, sections):
        inner, gap_ps = cpw_center_and_gap(run, w_k, g_k, p.trace_layer, p.gap_layer)
        center_polys.append(inner)
        gap_polys.append(gap_ps)
        if verbose:
            print(f"  {label:<22}: len {arclen(run):8.1f} um  "
                  f"width {w_k:5.1f}  gap {g_k:5.1f}  -> trace layer {p.trace_layer}")

    entry_pt = (float(dense[0][0]), float(dense[0][1]))
    exit_pt = (float(dense[-1][0]), float(dense[-1][1]))

    if p.include_launchers:
        launcher_l = place_launcher(tmpl_l, final_l, entry_pt, p.gap_layer)
        launcher_r = place_launcher(tmpl_r, final_r, exit_pt, p.gap_layer)
        gap_polys += [launcher_l, launcher_r]
        if verbose:
            for name, pt, (g_k, w_k), (g_L, w_L) in (
                    ("left", entry_pt, p.section_xsec[0], p.launcher_l_xsec),
                    ("right", exit_pt, p.section_xsec[-1], p.launcher_r_xsec)):
                step_outer = abs((w_k / 2.0 + g_k) - (w_L / 2.0 + g_L))
                step_inner = abs(w_k - w_L)
                print(f"{name:>5} launcher: taper ends {w_L:.1f}/{g_L:.1f}, butts "
                      f"onto {w_k:.1f}/{g_k:.1f} at x={pt[0]:.1f}  ->  "
                      f"conductor step {step_inner:.2f} um, "
                      f"ground step {step_outer:.2f} um")
    else:
        # Open end cap: joins the side gaps across the trace tip
        g_l, w_l = p.section_xsec[0]
        g_r, w_r = p.section_xsec[-1]
        cap_l_depth = p.end_cap_depth if p.end_cap_depth is not None else g_l
        cap_r_depth = p.end_cap_depth if p.end_cap_depth is not None else g_r
        halfspan_l = w_l / 2.0 + g_l
        halfspan_r = w_r / 2.0 + g_r

        cap_l = gdspy.Rectangle(
            (entry_pt[0] - cap_l_depth, entry_pt[1] - halfspan_l),
            (entry_pt[0], entry_pt[1] + halfspan_l),
            layer=p.gap_layer)
        cap_r = gdspy.Rectangle(
            (exit_pt[0], exit_pt[1] - halfspan_r),
            (exit_pt[0] + cap_r_depth, exit_pt[1] + halfspan_r),
            layer=p.gap_layer)
        gap_polys += [cap_l, cap_r]

        if verbose:
            print(f"  left end cap:  {cap_l_depth:.1f} um gap at x={entry_pt[0]:.1f}")
            print(f"  right end cap: {cap_r_depth:.1f} um gap at x={exit_pt[0]:.1f}")

    all_center = gdspy.boolean(center_polys, None, 'or', layer=p.trace_layer,
                                precision=1e-4, max_points=0)
    all_gaps = gdspy.boolean(gap_polys, None, 'or', layer=p.gap_layer,
                              precision=1e-4, max_points=0)

    lo, hi = bbox_of([all_center, all_gaps])
    off_chip = min(lo[0], lo[1], p.chip_width - hi[0], p.chip_height - hi[1]) < 0

    if verbose:
        print(f"\nAssembly bbox: x {lo[0]:.1f} to {hi[0]:.1f} "
              f"({hi[0] - lo[0]:.1f} wide)")
        print(f"               y {lo[1]:.1f} to {hi[1]:.1f} "
              f"({hi[1] - lo[1]:.1f} tall)")
        print(f"Slack: left {lo[0]:.1f}, right {p.chip_width - hi[0]:.1f}, "
              f"bottom {lo[1]:.1f}, top {p.chip_height - hi[1]:.1f} um")
        if off_chip:
            print("ERROR: geometry runs off the chip.")

    # Fresh library per call so sweeps don't collide on cell names
    gdspy.current_library = gdspy.GdsLibrary()
    merged_cell = gdspy.Cell('merge')

    if write_gaps:
        merged_cell.add(all_center)
        merged_cell.add(all_gaps)
    else:
        chip_rect = gdspy.Rectangle((0, 0), (p.chip_width, p.chip_height))
        # Ground excludes both gaps and trace so the layers never overlap
        cutouts = gdspy.boolean([all_gaps, all_center], None, 'or',
                                 precision=1e-4, max_points=0)
        ground_plane = gdspy.boolean(chip_rect, cutouts, 'not', layer=p.layer,
                                      precision=1e-4, max_points=0)
        merged_cell.add(ground_plane)
        merged_cell.add(all_center)

    out_name = filename + ".gds"
    gdspy.write_gds(out_name, unit=1.0e-6, precision=1.0e-9)

    if view:
        gdspy.LayoutViewer(cells=merged_cell)

    if verbose:
        layer_note = f"trace -> layer {p.trace_layer}, ground -> layer {p.layer}"
        if write_gaps:
            layer_note += f", gap slots -> layer {p.gap_layer}"
        print(f"\nWrote {out_name}  ({layer_note})")

    return {
        "ok": not off_chip,
        "filename": out_name,
        "n_turns": n_turns,
        "n_sections": len(sections),
        "n_high_z": p.n_high_z,
        "n_low_z": p.n_low_z,
        "w_total_um": w_total,
        "vertical_span_um": span,
        "bbox_lo": tuple(lo),
        "bbox_hi": tuple(hi),
        "off_chip": off_chip,
    }


# ======================================================================= #
#                                 Sweeps                                  #
# ======================================================================= #
def sweep(base: BraggParams, grid: dict, verbose: bool = False) -> pd.DataFrame:
    """Cartesian sweep of `grid` (field -> values) using design_only."""
    keys = list(grid.keys())
    rows = []
    for combo in product(*(grid[k] for k in keys)):
        p = base.with_(**dict(zip(keys, combo)))
        rows.append(design_only(p, verbose=verbose))
    return pd.DataFrame(rows)


def _auto_name(prefix: str, overrides: dict) -> str:
    parts = [prefix]
    for k, v in overrides.items():
        v_str = f"{v:g}" if isinstance(v, float) else str(v)
        parts.append(f"{k}-{v_str}")
    return "_".join(parts)


def sweep_and_generate(base: BraggParams, grid: dict, prefix: str = "AS_bragg",
                        only_ok: bool = True, write_gaps: bool = False,
                        view: bool = False, verbose: bool = False) -> pd.DataFrame:
    """Like sweep(), but writes one .gds per combination.

    Filenames come from the swept fields only. With only_ok=True, combos
    that fail design_only are skipped (filename=None).
    """
    keys = list(grid.keys())
    rows = []
    for combo in product(*(grid[k] for k in keys)):
        overrides = dict(zip(keys, combo))
        p = base.with_(**overrides)

        fit = design_only(p, verbose=verbose)
        if not fit["ok"] and only_ok:
            fit["filename"] = None
            rows.append(fit)
            continue

        name = _auto_name(prefix, overrides)
        try:
            gen = generate_gds(p, filename=name, write_gaps=write_gaps,
                                view=view, verbose=verbose)
            fit["filename"] = gen["filename"]
            fit["off_chip"] = gen["off_chip"]
        except Exception as e:
            fit["ok"] = False
            fit["error"] = str(e)
            fit["filename"] = None
        rows.append(fit)

    df = pd.DataFrame(rows)
    n_written = df["filename"].notna().sum() if "filename" in df else 0
    print(f"\nsweep_and_generate: {n_written}/{len(df)} combos written to disk.")
    return df