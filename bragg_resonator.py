# -*- coding: utf-8 -*-
"""Half-wave CPW resonator between two Bragg mirrors, with top launchers.

Launcher -> arc -> left mirror -> cavity -> right mirror -> arc -> launcher.
The cavity is either a straight line or a serpentine meander whose centreline
length is held fixed (half wavelength by default) while the number of legs sets
the trade-off between horizontal width and vertical height.

@author: Aryan Soni
"""

from dataclasses import dataclass, replace
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import gdspy
import gds_maker as GM

C = 299_792_458    # speed of light, m/s
ARC_STEP_DEG = 1.0  # meander arc discretisation


# ======================================================================= #
#                               Parameters                                #
# ======================================================================= #
@dataclass(frozen=True)
class Params:
    # chip (chip_width=None -> set by launcher positions)
    chip_height: float = 5000
    chip_width: Optional[float] = 10000
    edge_margin: float = 500    # clear ground on all four sides, launchers included

    # mirror cross-sections (um)
    gap1: float = 6
    width1: float = 30
    gap2: float = 16.5
    width2: float = 9
    high_z_first: bool = False   # True: high-Z next to each feedline, False: gap1/width1

    # mirror drawing style:
    #   "cells"      -> gds_maker Bragg unit cells (4 quarter-wave sections per cell)
    #   "serpentine" -> one continuous serpentine per mirror on the feed axis, same section
    #                   sequence, height capped by mirror_height_max
    mirror_style: str = "cells"
    mirror_height_max: Optional[float] = None   # outer height; None -> chip_height - 2*edge_margin - 2*mirror_clearance
    mirror_clearance: float = 50                # extra room kept inside the edge margin (top and bottom)
    mirror_radius: Optional[float] = None       # serpentine bend radius; None -> radius
    mirror_pitch: Optional[float] = None        # leg centre-to-centre spacing (>= 2r); None -> 2 * radius.
                                                # pitch > 2r puts a straight in each 180 deg turn
    mirror_lead: float = 50                     # horizontal run at each mirror end
    mirror_step_clear: float = 25               # min distance from an impedance step to a bend
    min_ground: Optional[float] = None          # min ground strip between adjacent legs (fab rule); None -> one CPW width

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

    # serpentine cavity, centreline length = cavity_len exactly (flat included):
    #   lead -> [meander half] -> flat (on axis, centred) -> [mirrored meander half] -> lead
    #   each half: 90 deg bend -> half leg -> (legs joined by 180 deg bends) -> half leg -> 90 deg bend
    cavity_meander: bool = True
    cavity_flat: float = 500                  # straight run at the electrical centre of the cavity
    cavity_legs: Optional[int] = None         # full-height legs PER HALF; None -> fewest with amplitude <= cavity_amp_max
    cavity_amp_max: float = 1200              # max centreline excursion from the cavity axis (used when legs=None)
    cavity_radius: Optional[float] = None     # bend radius; None -> radius
    cavity_pitch: Optional[float] = None      # leg centre-to-centre spacing (>= 2r); None -> 2 * cavity radius
    cavity_lead: float = 100                  # straight run at each end, clearance from the mirrors
    cavity_first_up: bool = True              # first excursion goes up (+y) or down (-y);
                                              # with 0 legs/half this is the only side used

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

    # ---- serpentine cavity -------------------------------------------- #
    @property
    def cav_r(self) -> float:
        return self.radius if self.cavity_radius is None else self.cavity_radius

    @property
    def cav_pitch(self) -> float:
        return 2 * self.cav_r if self.cavity_pitch is None else self.cavity_pitch

    def meander_amplitude(self, legs: int) -> float:
        """Centreline excursion A that makes the meander exactly cavity_len long."""
        return meander_amplitude(self.cavity_len, legs, self.cav_r, self.cavity_lead,
                                 self.cavity_flat, pitch=self.cav_pitch)

    @property
    def meander_legs(self) -> int:
        if self.cavity_legs is not None:
            return int(self.cavity_legs)
        for n in range(0, 200):
            if self.meander_amplitude(n) <= self.cavity_amp_max:
                return n
        raise ValueError("No leg count gives amplitude <= cavity_amp_max.")

    @property
    def meander_geometry(self) -> dict:
        n, r, d = self.meander_legs, self.cav_r, self.cav_pitch
        a = self.meander_amplitude(n)
        cg, cw = self.cavity_xsec
        return {"legs": n, "amplitude": a, "radius": r, "pitch": d,
                "ground_between_legs": d - (cw + 2 * cg),
                "flat": self.cavity_flat,
                # centreline span in x: per half 2r (two 90 deg bends) + (n+1) turns of pitch d
                "width": 2 * self.cavity_lead + self.cavity_flat + 2 * (2 * r + d * (n + 1)),
                # outer CPW extent in y; 0 legs/half only rises on one side of the axis
                "height": (a if n == 0 else 2 * a) + cw + 2 * cg,
                "length": self.cavity_len}

    # ---- serpentine mirrors -------------------------------------------- #
    @property
    def mir_r(self) -> float:
        return self.radius if self.mirror_radius is None else self.mirror_radius

    @property
    def mir_pitch(self) -> float:
        return 2 * self.mir_r if self.mirror_pitch is None else self.mirror_pitch

    @property
    def mir_h_max(self) -> float:
        if self.mirror_height_max is not None:
            return self.mirror_height_max
        return self.chip_height - 2 * self.edge_margin - 2 * self.mirror_clearance

    @property
    def mirror_sequences(self):
        """Cross-sections (gap, width) feed -> cavity on the left, cavity -> feed on the right.

        Same order the gds_maker cells draw: left [first, second] x 2 per cell,
        right [second, first] x 2 per cell.
        """
        (lf, ls), (rf, rs) = self.mirror_xsecs
        return [lf, ls] * (2 * self.left_cells), [rs, rf] * (2 * self.right_cells)

    def check(self):
        if self.left_cells < 1 or self.right_cells < 1:
            raise ValueError("Each mirror needs at least 2 periods (one unit-cell structure).")
        if self.mirror_style not in ("cells", "serpentine"):
            raise ValueError("mirror_style must be 'cells' or 'serpentine'.")
        if self.mirror_style == "serpentine":
            seqs = self.mirror_sequences
            xw = max(w + 2 * g for g, w in seqs[0] + seqs[1])
            if self.mir_pitch < 2 * self.mir_r - 1e-9:
                raise ValueError(f"mirror_pitch {self.mir_pitch:g} < 2 x mirror radius "
                                 f"{2 * self.mir_r:g} um.")
            need = xw if self.min_ground is None else self.min_ground
            if self.mir_pitch - xw < need - 1e-9:
                raise ValueError(f"Ground between mirror legs {self.mir_pitch - xw:.1f} um < "
                                 f"{need:g} um; increase mirror_pitch.")
        else:
            for name, v in self.cell_lengths.items():
                if v < 0:
                    raise ValueError(f"{name} is negative ({v:.2f} um); check radius, "
                                     f"length1, addlength, frequency, epsilon.")
        if self.cavity_meander:
            if self.cavity_flat < 0 or self.cavity_lead < 0:
                raise ValueError("cavity_flat and cavity_lead must be >= 0.")
            cg, cw = self.cavity_xsec
            g = self.meander_geometry
            if g["amplitude"] < 2 * g["radius"]:
                raise ValueError(f"Meander amplitude {g['amplitude']:.1f} um < 2 x radius; "
                                 f"use fewer legs or a smaller cavity_radius.")
            if g["pitch"] < 2 * g["radius"] - 1e-9:
                raise ValueError(f"cavity_pitch {g['pitch']:g} < 2 x cavity radius "
                                 f"{2 * g['radius']:g} um.")
            need = cw + 2 * cg if self.min_ground is None else self.min_ground
            if g["ground_between_legs"] < need - 1e-9:
                raise ValueError(f"Ground between cavity legs {g['ground_between_legs']:.1f} um < "
                                 f"{need:g} um; increase cavity_pitch.")

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


# ======================================================================= #
#                               Geometry                                  #
# ======================================================================= #
def turn_len(radius, pitch):
    """Centreline length of one 180 deg turn: two quarter arcs + a straight of pitch - 2r."""
    return np.pi * radius + (pitch - 2 * radius)


def serpentine_amplitude(length, legs, radius, lead_a, lead_b, pitch=None):
    """Amplitude A that makes serpentine_ops(...) exactly `length` long.

    L = lead_a + lead_b + pi*r (two 90 deg bends) + (n+1)*turn + 2*(A - 2r) + 2*n*(A - r)
    """
    r, n = radius, legs
    d = 2 * r if pitch is None else pitch
    return ((length - lead_a - lead_b - np.pi * r - (n + 1) * turn_len(r, d)
             + 4 * r + 2 * r * n) / (2 + 2 * n))


def meander_amplitude(length, legs, radius, lead, flat=0.0, pitch=None):
    """Amplitude for two mirrored serpentine halves (`legs` full legs each) around a centred flat."""
    half = (length - 2 * lead - flat) / 2
    return serpentine_amplitude(half, legs, radius, 0.0, 0.0, pitch)


def serpentine_ops(lead_a, legs, amplitude, radius, lead_b, first_up=True, pitch=None):
    """Centreline route (heading +x in and out, same axis) as ('S', length) / ('A', radius, signed angle).

    Each 180 deg turn is quarter arc -> straight (pitch - 2r) -> quarter arc, so legs sit `pitch` apart.
    """
    r, a, n = radius, amplitude, legs
    d = 2 * r if pitch is None else pitch
    half_leg, full_leg = a - 2 * r, 2 * (a - r)
    s = 1 if first_up else -1

    def u_turn(sg):   # sg: current leg direction (+1 up), turning over the top (+1) or bottom (-1)
        return [("A", r, -sg * np.pi / 2), ("S", d - 2 * r), ("A", r, -sg * np.pi / 2)]

    ops = [("S", lead_a), ("A", r, s * np.pi / 2), ("S", half_leg)]
    for _ in range(n):
        ops += u_turn(s) + [("S", full_leg)]
        s = -s
    ops += u_turn(s) + [("S", half_leg), ("A", r, s * np.pi / 2), ("S", lead_b)]
    return [op for op in ops if not (op[0] == "S" and op[1] <= 1e-9)]


def route_length(ops):
    return sum(op[1] if op[0] == "S" else abs(op[2]) * op[1] for op in ops)


def cpw_meander(start, length, legs, radius, lead, gap, width, layer, first_up=True,
                flat=0.0, pitch=None, tolerance=0.01):
    """Serpentine CPW (gap slots only) heading +x from `start`, centreline `length`.

    lead -> half A -> flat on the axis -> half B (mirror image of A) -> lead.
    The flat sits at the electrical midpoint. Ends on the same axis heading +x.
    Returns (polygons, end point, amplitude).
    """
    r = radius
    a = meander_amplitude(length, legs, radius, lead, flat, pitch)
    if a - 2 * r < 0:
        raise ValueError(f"Meander amplitude {a:.1f} um too small for radius {r} um.")
    # half A ends on the side it started if legs is even, so half B starts there to mirror it
    second_up = first_up if legs % 2 == 0 else not first_up
    ops = (serpentine_ops(lead, legs, a, r, 0.0, first_up, pitch) + [("S", flat)]
           + serpentine_ops(0.0, legs, a, r, lead, second_up, pitch))
    ops = [op for op in ops if not (op[0] == "S" and op[1] <= 1e-9)]
    polys, end, _, _ = draw_route(start, ops, [(route_length(ops), gap, width)], layer, tolerance)
    if abs(end[1] - start[1]) > 1e-6:
        raise RuntimeError("Meander did not close on its axis.")
    return polys, end, a


def draw_route(start, ops, pieces, layer, tolerance=0.01):
    """Draw CPW gap slots along `ops`, switching cross-section per `pieces` [(length, gap, width)].

    Piece boundaries can fall anywhere, including inside arcs. Returns
    (polygons, end point, bends [(s0, s1)], steps [s]) with s = distance along the centreline.
    """
    total_ops = sum(op[1] if op[0] == "S" else abs(op[2]) * op[1] for op in ops)
    total_pieces = sum(pc[0] for pc in pieces)
    if abs(total_ops - total_pieces) > 1e-6:
        raise RuntimeError(f"Route length {total_ops:.4f} != section total {total_pieces:.4f} um.")

    x, y, h = float(start[0]), float(start[1]), 0.0
    polys, bends, steps = [], [], []
    s_pos, k, k_rem = 0.0, 0, pieces[0][0]

    for op in ops:
        op_len = op[1] if op[0] == "S" else abs(op[2]) * op[1]
        if op[0] == "A":
            bends.append((s_pos, s_pos + op_len))
        done = 0.0
        while done < op_len - 1e-9:
            chunk = min(op_len - done, k_rem)
            g, w = pieces[k][1], pieces[k][2]
            if op[0] == "S":
                ux, uy = np.cos(h), np.sin(h)
                nx, ny = -uy, ux
                x1, y1 = x + chunk * ux, y + chunk * uy
                for sgn in (1, -1):
                    o_in, o_out = sgn * w / 2, sgn * (w / 2 + g)
                    polys.append(gdspy.Polygon([(x + o_in * nx, y + o_in * ny), (x1 + o_in * nx, y1 + o_in * ny),
                                                (x1 + o_out * nx, y1 + o_out * ny), (x + o_out * nx, y + o_out * ny)],
                                               layer=layer))
                x, y = x1, y1
            else:
                r, sgn = op[1], np.sign(op[2])
                cx, cy = x - sgn * r * np.sin(h), y + sgn * r * np.cos(h)
                phi0 = np.arctan2(y - cy, x - cx)
                dphi = sgn * chunk / r
                a0, a1 = sorted((phi0, phi0 + dphi))
                for ro, ri in ((r + w / 2 + g, r + w / 2), (r - w / 2, r - w / 2 - g)):
                    polys.append(gdspy.Round((cx, cy), ro, inner_radius=ri, initial_angle=a0,
                                             final_angle=a1, tolerance=tolerance, layer=layer))
                x, y = cx + r * np.cos(phi0 + dphi), cy + r * np.sin(phi0 + dphi)
                h += dphi
            done += chunk
            s_pos += chunk
            k_rem -= chunk
            if k_rem <= 1e-9 and k < len(pieces) - 1:
                k += 1
                k_rem = pieces[k][0]
                steps.append(s_pos)
    return polys, (x, y), bends, steps


def serpentine_mirror(p: Params, start, xsecs, feed_side):
    """One Bragg mirror as a serpentine heading +x along the axis through `start`.

    xsecs: section cross-sections in drawing order; each section is a quarter wave.
    feed_side: 'start' (left mirror) or 'end' (right mirror): where the 50-ohm padding goes.
    Pads the feed end with up to one leg of 50-ohm line so no impedance step sits on a bend.
    """
    r, d, qw, lead = p.mir_r, p.mir_pitch, p.qw, p.mirror_lead
    feed = (p.feed_gap, p.feed_width)
    xw = max(w + 2 * g for g, w in xsecs)
    a_max = (p.mir_h_max - xw) / 2
    n_qw = len(xsecs) * qw                      # every section is exactly one quarter wave
    left = feed_side == "start"

    def route(pad, extra):
        """Route + pieces. The 50-ohm end carries `lead + pad` of feed line; the cavity end has a
        horizontal lead of `lead + extra` that is part of the adjacent section's quarter wave."""
        la, lb = (lead, lead + extra) if left else (lead + extra, lead)
        a = serpentine_amplitude(n_qw + lead + pad, legs, r, la, lb, d)
        sec = [(qw, g, w) for g, w in xsecs]
        pieces = [(lead + pad, *feed)] + sec if left else sec + [(lead + pad, *feed)]
        return a, la, lb, pieces

    def clearance(a, la, lb, pieces):
        ops = serpentine_ops(la, legs, a, r, lb, pitch=d)
        lens = np.array([op[1] if op[0] == "S" else abs(op[2]) * op[1] for op in ops])
        s0 = np.concatenate(([0.0], np.cumsum(lens)[:-1]))
        is_arc = np.array([op[0] == "A" for op in ops])
        b0, b1 = s0[is_arc], (s0 + lens)[is_arc]
        st = np.cumsum([pc[0] for pc in pieces])[:-1][:, None]
        if ((st > b0) & (st < b1)).any():
            return -1.0, ops
        return float(np.minimum(np.abs(st - b0), np.abs(st - b1)).min()), ops

    # fewest legs whose amplitude fits under the height cap
    legs = next((n for n in range(0, 400)
                 if serpentine_amplitude(n_qw + lead, n, r, lead, lead, d) <= a_max), None)
    if legs is None:
        raise ValueError("Mirror cannot meet mirror_height_max.")

    # Search two knobs for the least extra line that keeps every impedance step at least
    # mirror_step_clear from a bend: 50-ohm padding (absorbed into the amplitude) and the
    # cavity-side lead (costs width, at most one pitch). Cost = padding + extra lead.
    leg_len = 2 * (a_max - r) + turn_len(r, d)
    best, fallback = None, None
    for extra in np.arange(0.0, d + 1e-9, 5.0):
        if best is not None and extra >= best[1] + best[2]:
            break                                   # can no longer beat the best cost
        for pad in np.arange(0.0, leg_len, 4.0):
            if best is not None and pad + extra >= best[1] + best[2]:
                break
            a, la, lb, pieces = route(pad, extra)
            if a > a_max:
                break
            if a < 2 * r:
                continue
            clear, ops = clearance(a, la, lb, pieces)
            if fallback is None or clear > fallback[0]:
                fallback = (clear, pad, extra, a, pieces, ops)
            if clear >= p.mirror_step_clear:
                best = (clear, pad, extra, a, pieces, ops)
                break
    best = best or fallback
    if best is None:
        raise ValueError("Could not place the mirror serpentine.")
    clear, pad, extra, a, pieces, ops = best

    polys, end, bends, steps = draw_route(start, ops, pieces, p.gap_layer)
    if abs(end[1] - start[1]) > 1e-6:
        raise RuntimeError("Mirror serpentine did not close on its axis.")
    in_bend = sum(any(b0 < st < b1 for b0, b1 in bends) for st in steps)
    if clear < p.mirror_step_clear:
        print(f"WARNING: {'left' if left else 'right'} mirror: nearest impedance step is "
              f"{clear:.1f} um from a bend (target {p.mirror_step_clear:g}); nudge mirror_pitch, "
              f"mirror_radius or mirror_height_max, or lower mirror_step_clear.")
    info = {"legs": legs, "amplitude": a, "height": 2 * a + xw, "radius": r, "pitch": d,
            "step_clear_met": bool(clear >= p.mirror_step_clear),
            "ground_between_legs": d - xw,
            "width": end[0] - start[0], "sections": len(xsecs), "pad": pad, "extra_lead": extra,
            "min_step_to_bend": clear, "steps_in_bends": in_bend}
    return polys, end, info


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

    serp = p.mirror_style == "serpentine"
    seq_left, seq_right = p.mirror_sequences
    mirrors = {}
    if serp:
        # mirror starts on the feed axis, heading +x
        marks = [len(polys)]
        pl, pos, mirrors["left"] = serpentine_mirror(p, (float(pos_c[0]), float(pos_c[1])),
                                                     seq_left, feed_side="start")
        polys += pl
    else:
        # entry arc and run down to the left mirror
        pl, pos = GM.createDoublearc((pos_c[0], pos_c[1] - r), radius=r, gap=fg, width=fw,
                                     angle=2 * np.pi, layer=layer)
        polys += pl
        pl, pos = GM.createCPWvertDown(pos, length=L["downlength"], gap=fg, width=fw, layer=layer)
        polys += pl

        # left mirror: unit cell 1 x (n-1), then unit cell 2
        marks = [len(polys)]
        for _ in range(p.left_cells - 1):
            pl, pos = GM.Braggmirror_unitcell1_left(pos, **left_kw)
            polys += pl
        pl, pos = GM.Braggmirror_unitcell2_left(pos, **left_kw)
        polys += pl

    # cavity
    marks.append(len(polys))
    cg, cw = p.cavity_xsec
    info = {"cavity": None}
    if p.cavity_meander:
        g = p.meander_geometry
        cav_start = (float(pos[0]), float(pos[1]))
        pl, pos, _ = cpw_meander(cav_start, p.cavity_len, g["legs"], g["radius"],
                                 p.cavity_lead, cg, cw, layer, first_up=p.cavity_first_up,
                                 flat=p.cavity_flat, pitch=g["pitch"])
        polys += pl
        info["cavity"] = {"type": "serpentine", "start": cav_start, **g}
    else:
        pl, pos = GM.createCPWhor(pos, length=p.cavity_len, gap=cg, width=cw, layer=layer)
        polys += pl

    marks.append(len(polys))
    if serp:
        pl, pos, mirrors["right"] = serpentine_mirror(p, (float(pos[0]), float(pos[1])),
                                                      seq_right, feed_side="end")
        polys += pl
        marks.append(len(polys))
        pos_e = (float(pos[0]), float(pos[1]))       # already horizontal on the feed axis
    else:
        # right mirror: unit cell 2, then unit cell 1 x (n-1)
        pl, pos = GM.Braggmirror_unitcell2_right(
            pos, **dict(right_kw, vertical_length5=L["vertical_length5"]))
        polys += pl
        for _ in range(p.right_cells - 1):
            pl, pos = GM.Braggmirror_unitcell1_right(pos, **right_kw)
            polys += pl

        # run down, arc to horizontal
        marks.append(len(polys))
        pl, pos_d = GM.createCPWvertDown(pos, length=L["downlength"], gap=fg, width=fw, layer=layer)
        polys += pl
        pl, pos_e = GM.createDoublearc((pos_d[0] + r, pos_d[1]), radius=r, gap=fg, width=fw,
                                       angle=-np.pi, layer=layer)
        polys += pl
        pos_e = (pos_e[0] + r, pos_e[1] - r)

    # feedline, arc up

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

    marks.append(len(polys))
    names = ["left launcher + feed", "left mirror", "cavity", "right mirror",
             "right feed + launcher"]
    bounds = [0] + marks
    info.update(entry=entry, exit=exit_, x_left=x_left, x_right=x_right, mirrors=mirrors,
                probes=[(entry[0], entry[1] - 1.0), (exit_[0], exit_[1] - 1.0)],
                sections={n: (bounds[i], bounds[i + 1]) for i, n in enumerate(names)})
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
            print(f"Serpentine cavity: length {cav['length']:.1f} um, {cav['legs']} legs/half, "
                  f"r {cav['radius']:g}, pitch {cav['pitch']:g} (ground {cav['ground_between_legs']:g}), "
                  f"amplitude {cav['amplitude']:.1f} um, footprint {cav['width']:.1f} x "
                  f"{cav['height']:.1f} um (vs {p.cavity_len:.1f} um straight)")

    if verbose:
        for side, m_ in info["mirrors"].items():
            print(f"Serpentine {side} mirror: {m_['sections']} quarter-wave sections, {m_['legs']} legs, "
                  f"r {m_['radius']:g}, pitch {m_['pitch']:g} (ground {m_['ground_between_legs']:g}), amplitude {m_['amplitude']:.1f} um, footprint {m_['width']:.1f} x "
                  f"{m_['height']:.1f} um; 50-ohm pad {m_['pad']:.0f} um, extra cavity-side lead {m_['extra_lead']:.0f} um; nearest step-to-bend "
                  f"{m_['min_step_to_bend']:.1f} um ({m_['steps_in_bends']} steps inside bends)")

    # per-section bounding boxes on the final chip, and how far each one sits from each edge
    m, H = p.edge_margin, p.chip_height
    sections = {}
    for name, (i0, i1) in info["sections"].items():
        if i1 <= i0:
            continue
        slo, shi = bbox_of(polys[i0:i1])
        slo, shi = slo + np.array([dx, 0.0]), shi + np.array([dx, 0.0])
        sections[name] = {"lo": slo, "hi": shi,
                          "left": slo[0] - m, "right": chip_w - shi[0] - m,
                          "bottom": slo[1] - m, "top": H - shi[1] - m}

    return {"gaps": gaps, "probes": [(x + dx, y) for x, y in info["probes"]],
            "lo": lo, "hi": hi, "chip_width": chip_w, "cavity": info["cavity"],
            "sections": sections, "mirrors": info["mirrors"]}


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
    if p.cavity_meander:
        try:
            g = p.meander_geometry
            base.update(cavity_legs=g["legs"], cavity_amp_um=g["amplitude"],
                        cavity_w_um=g["width"], cavity_h_um=g["height"])
        except ValueError as e:
            return {"ok": False, "error": str(e), **base}
    try:
        lay = layout(p, verbose=verbose)
    except (ValueError, RuntimeError) as e:
        return {"ok": False, "error": str(e), **base}

    lo, hi, m = lay["lo"], lay["hi"], p.edge_margin
    width_slack = float(min(lo[0], lay["chip_width"] - hi[0])) - m   # spare room beyond the margin
    height_slack = float(min(lo[1], p.chip_height - hi[1])) - m
    ok = width_slack >= 0 and height_slack >= 0

    if verbose:
        print(f"Quarter-wave length {p.qw:.1f} um; mirrors {4 * p.left_cells} / "
              f"{4 * p.right_cells} quarter-wave sections ({2 * p.left_cells} / "
              f"{2 * p.right_cells} periods)")
        print(f"Assembly: {hi[0] - lo[0]:.1f} x {hi[1] - lo[1]:.1f} um; slack x {width_slack:.1f}, "
              f"y {height_slack:.1f} um ({'fits' if ok else 'DOES NOT FIT'})")
        print("Per-section clearance beyond the edge margin (negative = intrudes), um:")
        print(f"  {'section':<24}{'left':>9}{'right':>9}{'bottom':>9}{'top':>9}   y-span")
        for name, s in lay["sections"].items():
            flag = "  <-- " if min(s["left"], s["right"], s["bottom"], s["top"]) < 0 else ""
            print(f"  {name:<24}{s['left']:9.1f}{s['right']:9.1f}{s['bottom']:9.1f}"
                  f"{s['top']:9.1f}   {s['lo'][1]:.0f} to {s['hi'][1]:.0f}{flag}")

    out = {"ok": ok, **base,
           "left_sections": 2 * p.left_cells, "right_sections": 2 * p.right_cells,
           "chip_width_um": lay["chip_width"],
           "width_um": float(hi[0] - lo[0]), "height_um": float(hi[1] - lo[1]),
           "width_slack_um": width_slack, "height_slack_um": height_slack}
    for side, m_ in lay["mirrors"].items():
        out[f"{side}_mirror_legs"] = m_["legs"]
        out[f"{side}_mirror_h_um"] = m_["height"]
        out[f"{side}_mirror_w_um"] = m_["width"]
        out[f"{side}_steps_in_bends"] = m_["steps_in_bends"]
        out[f"{side}_step_clear_um"] = m_["min_step_to_bend"]
    for name, s in lay["sections"].items():
        key = name.replace(" + ", "_").replace(" ", "_")       # e.g. left_mirror, cavity
        out[f"{key}_min_clear_um"] = float(min(s["left"], s["right"], s["bottom"], s["top"]))
    if not ok:
        culprits = [n for n, s in lay["sections"].items()
                    if min(s["left"], s["right"], s["bottom"], s["top"]) < 0]
        out["culprits"] = ", ".join(culprits)
        out["error"] = (f"does not fit the chip [{', '.join(culprits) or 'unknown'}] "
                        f"(slack x {width_slack:.1f}, "
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
    def gds_safe(ps):   # GDSII allows at most 8190 vertices per polygon
        return ps.fracture(max_points=8190, precision=1e-4)

    if write_gaps:
        if trace is not None:
            cell.add(gds_safe(trace))
        if all_gaps is not None:
            cell.add(gds_safe(all_gaps))
    else:
        chip = gdspy.Rectangle((0, 0), (chip_w, p.chip_height))
        cutouts = gdspy.boolean([all_gaps, trace], None, 'or', precision=1e-4, max_points=0)
        
        if cutouts is not None:
            ground = gdspy.boolean(chip, cutouts, 'not', layer=p.layer, precision=1e-4, max_points=0)
        else:
            ground = chip

        if ground is not None:
            cell.add(gds_safe(ground) if hasattr(ground, "fracture") else ground)
        if trace is not None:
            cell.add(gds_safe(trace))

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


def fit(base: Params, verbose=True, **grid):
    """Sweep any Params fields (field=[values]); return (best Params or None, table).

    'Best' = fits the chip and maximises the smaller of the x / y slack.
    Example: fit(p, mirror_radius=[50, 60, 70], cavity_legs=[0, 1])
    """
    grid = {k: list(v) for k, v in grid.items()}
    combos = list(product(*grid.values()))
    df = sweep(base, grid)
    for j, k in enumerate(grid):                 # make sure every swept field is a column
        df[k] = [c[j] for c in combos]
    ok = df[df["ok"]]
    if verbose:
        show = list(grid) + [c for c in ("cavity_w_um", "cavity_h_um", "left_mirror_w_um",
                                         "right_mirror_w_um", "left_mirror_h_um",
                                         "width_slack_um", "height_slack_um", "ok", "culprits")
                             if c in df and c not in grid]
        print(df[show].to_string(index=False, float_format=lambda v: f"{v:.1f}", na_rep="-"))
    if ok.empty:
        if verbose:
            print("\nNothing fits. Distinct errors:")
            for e in df["error"].dropna().unique():
                print("  ", e)
        return None, df
    i = ok[["width_slack_um", "height_slack_um"]].min(axis=1).idxmax()
    best = base.with_(**dict(zip(grid, combos[i])))
    if verbose:
        print(f"\nBest: {dict(zip(grid, combos[i]))} (slack x {ok.loc[i, 'width_slack_um']:.1f}, "
              f"y {ok.loc[i, 'height_slack_um']:.1f} um)")
    return best, df


def fit_cavity(base: Params, legs=range(0, 11), radius=(None,), lead=(None,), verbose=True):
    """Sweep cavity legs per half (and optionally bend radius / lead)."""
    return fit(base.with_(cavity_meander=True), verbose=verbose, cavity_legs=list(legs),
               cavity_radius=[base.cavity_radius if v is None else v for v in radius],
               cavity_lead=[base.cavity_lead if v is None else v for v in lead])


if __name__ == "__main__":
    # serpentine mirrors + one-sided cavity meander with the 500 um flat; fits 10000 x 5000 with 500 margin
    params = Params(frequency=7e9, cavity_length=6687.5, left_periods=13, right_periods=7,
                    mirror_style="serpentine", mirror_radius=60, cavity_legs=0)
    # to explore alternatives: best, table = fit(params, mirror_radius=[50, 60, 70], cavity_legs=[0, 1])
    if design_only(params)["ok"]:
        generate_gds(params, "bragg_resonator_serpentine", view=True)