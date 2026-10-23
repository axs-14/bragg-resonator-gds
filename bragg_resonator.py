# -*- coding: utf-8 -*-
"""Bragg-mirror CPW resonator chip layouts.

Chip: launcher -> feed -> left mirror -> cavity -> right mirror -> feed -> launcher

Mirror styles: cells (gds_maker unit cells), cell_meander (the cell pattern as one
continuous meander; symmetric=True gives a mirror-image chip), loops, serpentine.
Cavity styles: continuous (the meander carries on into the cavity up to a centred flat),
loops, spine, meander, straight.

@author: Aryan Soni
"""

from dataclasses import dataclass, replace
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import gdspy
import gds_maker as GM

C = 299_792_458
MIRROR_STYLES = ("cells", "cell_meander", "loops", "serpentine")
CAVITY_STYLES = ("straight", "meander", "spine", "loops", "continuous")
SECTION_NAMES = ("left launcher + feed", "left mirror", "cavity", "right mirror",
                 "right feed + launcher")


# --------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class Params:
    # chip
    chip_width: Optional[float] = 10000     # None: sized to the launchers
    chip_height: float = 5000
    edge_margin: float = 500

    # electrical
    frequency: float = 7e9
    epsilon: float = 11.9

    # mirrors
    mirror_style: str = "cell_meander"
    left_periods: int = 8
    right_periods: Optional[int] = 8        # None: same as left
    gap1: float = 23
    width1: float = 4
    gap2: float = 4
    width2: float = 42
    high_z_first: bool = True
    radius: float = 70
    length1: float = 337
    addlength: float = 150
    sections_per_column: float = 3          # cell_meander: quarter waves per 4-leg column
    cell_pitch: Optional[float] = None      # cell_meander leg spacing; None: 2 * radius
    symmetric: bool = False                 # cell_meander: mirror-image chip
    mirror_lead: float = 50
    mirror_step_clear: float = 25           # min distance from an impedance step to a bend
    mirror_clearance: float = 50            # extra room inside the edge margin
    min_ground: Optional[float] = None      # None: one CPW width
    loop_jog_extra: float = 0               # loops style
    mirror_radius: Optional[float] = None   # serpentine style
    mirror_pitch: Optional[float] = None    # serpentine style
    mirror_height_max: Optional[float] = None   # serpentine style

    # feedline and launchers
    feed_gap: float = 14
    feed_width: float = 22
    feed_left: float = 150
    feed_right: float = 150                 # ignored when symmetric
    launchers: bool = True
    launcher_straight: float = 425
    launcher_taper: float = 425
    launcher_height: float = 622
    launcher_gap: float = 161

    # cavity
    cavity_style: str = "spine"
    cavity_gap: Optional[float] = 24        # None: feed cross-section
    cavity_width: Optional[float] = 2
    cavity_length: Optional[float] = 6687.5  # None: half wave
    cavity_radius: Optional[float] = 70
    cavity_pitch: Optional[float] = 140
    cavity_lead: float = 50
    cavity_first_up: bool = True
    cavity_flat: float = 500                # continuous: the minimum flat
    flat_clearance: float = 100
    cavity_loops: int = 1                   # loops style
    cavity_legs: Optional[int] = 0          # meander style
    cavity_amp_max: float = 1200
    spine_length: Optional[float] = None    # spine style
    spine_legs: Optional[int] = None
    spine_pitch: Optional[float] = None
    spine_centered: bool = True             # centre the cavity on the chip

    # sample
    sample_size: float = 500                # 0: none
    sample_clearance: float = 50
    draw_sample: bool = True
    sample_layer: int = 10

    # output
    arc_tolerance: float = 0.01             # ~0.5 for EM simulation
    layer: int = 1
    gap_layer: int = 2
    trace_layer: int = 3

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    @property
    def qw(self) -> float:
        return 1e6 * C / (np.sqrt((self.epsilon + 1) / 2) * self.frequency * 4)

    @property
    def cavity_len(self) -> float:
        return 2 * self.qw if self.cavity_length is None else self.cavity_length

    @property
    def feed_xsec(self):
        return self.feed_gap, self.feed_width

    @property
    def cavity_xsec(self):
        return (self.feed_gap if self.cavity_gap is None else self.cavity_gap,
                self.feed_width if self.cavity_width is None else self.cavity_width)

    @property
    def cav_xw(self) -> float:
        g, w = self.cavity_xsec
        return w + 2 * g

    @property
    def mirror_xsecs(self):
        """((left first, left second), (right first, right second)) as (gap, width)."""
        a, b = (self.gap1, self.width1), (self.gap2, self.width2)
        if self.high_z_first:
            hi, lo = (a, b) if self.width1 / self.gap1 <= self.width2 / self.gap2 else (b, a)
            return (hi, lo), (lo, hi)
        return (a, b), (a, b)

    @property
    def periods(self):
        """(left, right) as drawn; cell-based styles round down to whole cells."""
        right = self.left_periods if self.right_periods is None else self.right_periods
        if self.mirror_style == "cell_meander":
            return self.left_periods, right
        return 2 * (self.left_periods // 2), 2 * (right // 2)

    @property
    def is_symmetric(self) -> bool:
        return self.symmetric and self.mirror_style == "cell_meander"

    @property
    def mirror_sequences(self):
        """Cross-sections: left mirror feed -> cavity, right mirror cavity -> feed."""
        (lf, ls), (rf, rs) = self.mirror_xsecs
        nl, nr = self.periods
        if self.is_symmetric:
            return [lf, ls] * nl, [ls, lf] * nr
        return [lf, ls] * nl, [rs, rf] * nr

    @property
    def full_sequence(self):
        """Left-oriented sequence for the longer mirror (the path both symmetric mirrors follow)."""
        left, right = self.mirror_sequences
        if len(left) >= len(right):
            return left
        return self.with_(left_periods=max(self.periods)).mirror_sequences[0]

    @property
    def cm_pitch(self) -> float:
        return 2 * self.radius if self.cell_pitch is None else self.cell_pitch

    @property
    def mirror_xw(self) -> float:
        left, right = self.mirror_sequences
        return max(w + 2 * g for g, w in left + right)

    @property
    def ground_rule(self) -> float:
        return self.mirror_xw if self.min_ground is None else self.min_ground

    @property
    def cell_lengths(self) -> dict:
        r, qw, l1, add = self.radius, self.qw, self.length1, self.addlength
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

    @property
    def loop_len(self) -> float:
        r = self.radius
        return 2 * self.addlength + 2 * self.length1 + 4 * r + 3 * np.pi * r

    @property
    def loop_height(self) -> float:
        return self.addlength + self.length1 + 3 * self.radius

    def loop_leg(self, first=False, last=False) -> float:
        """loops style: leg either side of a loop that makes the section one quarter wave."""
        jog = np.pi * self.radius + self.loop_jog_extra
        conn = (np.pi * self.radius / 2 if first else 0.0) + (np.pi * self.radius / 2 if last else jog)
        return (self.qw - self.loop_len - conn) / 2

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
    def cav_r(self) -> float:
        return self.radius if self.cavity_radius is None else self.cavity_radius

    @property
    def cav_pitch(self) -> float:
        return 2 * self.cav_r if self.cavity_pitch is None else self.cavity_pitch

    @property
    def spine_d(self) -> float:
        """Spine-to-riser spacing, widened so the risers clear the sample."""
        d = self.cav_pitch if self.spine_pitch is None else self.spine_pitch
        if self.sample_size > 0:
            d = max(d, self.sample_size / 2 + self.sample_clearance + self.cav_xw / 2)
        return d

    def meander_amplitude(self, legs: int) -> float:
        return meander_amplitude(self.cavity_len, legs, self.cav_r, self.cavity_lead,
                                 self.cavity_flat, pitch=self.cav_pitch)

    @property
    def meander_legs(self) -> int:
        if self.cavity_legs is not None:
            return int(self.cavity_legs)
        for n in range(200):
            if self.meander_amplitude(n) <= self.cavity_amp_max:
                return n
        raise ValueError("No leg count gives amplitude <= cavity_amp_max.")

    @property
    def meander_geometry(self) -> dict:
        n, r, d = self.meander_legs, self.cav_r, self.cav_pitch
        a = self.meander_amplitude(n)
        return {"type": "meander", "legs": n, "amplitude": a, "radius": r, "pitch": d,
                "ground_between_legs": d - self.cav_xw, "flat": self.cavity_flat,
                "width": 2 * self.cavity_lead + self.cavity_flat + 2 * (2 * r + d * (n + 1)),
                "height": (a if n == 0 else 2 * a) + self.cav_xw,
                "length": self.cavity_len}

    def check(self):
        if self.mirror_style not in MIRROR_STYLES:
            raise ValueError(f"mirror_style must be one of {MIRROR_STYLES}.")
        if self.cavity_style not in CAVITY_STYLES:
            raise ValueError(f"cavity_style must be one of {CAVITY_STYLES}.")
        if min(self.periods) < 1:
            raise ValueError("Each mirror needs at least one period (two for cell-based styles).")
        {"cells": self._check_cells, "cell_meander": self._check_cell_meander,
         "loops": self._check_loops, "serpentine": self._check_serpentine}[self.mirror_style]()
        if self.cavity_style == "meander":
            self._check_meander_cavity()
        if self.cavity_style == "spine":
            self._check_spine()
        if self.cavity_style == "loops" and self.cavity_loops < 1:
            raise ValueError("cavity_loops must be >= 1.")
        if self.cavity_style == "continuous" and not self.is_symmetric:
            raise ValueError("cavity_style='continuous' needs mirror_style='cell_meander' "
                             "and symmetric=True.")

    def _check_cells(self):
        for name, v in self.cell_lengths.items():
            if v < 0:
                raise ValueError(f"{name} is negative ({v:.2f} um); check radius, length1, "
                                 f"addlength, frequency, epsilon.")

    def _check_cell_meander(self):
        if self.cm_pitch < 2 * self.radius - 1e-9:
            raise ValueError(f"cell_pitch {self.cm_pitch:g} < 2 x radius {2 * self.radius:g} um.")
        ground = self.cm_pitch - self.mirror_xw
        if ground < self.ground_rule - 1e-9:
            raise ValueError(f"Ground between legs {ground:.1f} um < {self.ground_rule:g} um; "
                             f"increase radius.")

    def _check_loops(self):
        ground = 2 * self.radius + 2 * self.loop_jog_extra - self.mirror_xw
        if ground < self.ground_rule - 1e-9:
            raise ValueError(f"Ground between neighbouring loops {ground:.1f} um < "
                             f"{self.ground_rule:g} um; increase loop_jog_extra.")
        if self.loop_leg(first=True) < 0:
            raise ValueError("Loop longer than a quarter wave; reduce length1 / addlength.")
        if self.cavity_style == "meander" and self.cavity_first_up:
            raise ValueError("loops mirrors need cavity_first_up=False with a meander cavity: "
                             "the sections beside the cavity loop upward.")

    def _check_serpentine(self):
        if self.mir_pitch < 2 * self.mir_r - 1e-9:
            raise ValueError(f"mirror_pitch {self.mir_pitch:g} < 2 x mirror radius {2 * self.mir_r:g} um.")
        if self.mir_pitch - self.mirror_xw < self.ground_rule - 1e-9:
            raise ValueError(f"Ground between mirror legs {self.mir_pitch - self.mirror_xw:.1f} um < "
                             f"{self.ground_rule:g} um; increase mirror_pitch.")

    def _check_meander_cavity(self):
        if self.cavity_flat < 0 or self.cavity_lead < 0:
            raise ValueError("cavity_flat and cavity_lead must be >= 0.")
        g = self.meander_geometry
        need = self.cav_xw if self.min_ground is None else self.min_ground
        if g["amplitude"] < 2 * g["radius"]:
            raise ValueError(f"Meander amplitude {g['amplitude']:.1f} um < 2 x radius; use fewer "
                             f"legs or a smaller cavity_radius.")
        if g["pitch"] < 2 * g["radius"] - 1e-9:
            raise ValueError(f"cavity_pitch {g['pitch']:g} < 2 x cavity radius {2 * g['radius']:g} um.")
        if g["ground_between_legs"] < need - 1e-9:
            raise ValueError(f"Ground between cavity legs {g['ground_between_legs']:.1f} um < "
                             f"{need:g} um; increase cavity_pitch.")

    def _check_spine(self):
        need = self.cav_xw if self.min_ground is None else self.min_ground
        if self.spine_d < 2 * self.cav_r - 1e-9:
            raise ValueError(f"spine_pitch {self.spine_d:g} < 2 x cavity radius {2 * self.cav_r:g} um.")
        for name, d in (("cavity_pitch", self.cav_pitch), ("spine_pitch", self.spine_d)):
            if d - self.cav_xw < need - 1e-9:
                raise ValueError(f"Ground at {name} = {d:g} um is {d - self.cav_xw:.1f} um < "
                                 f"{need:g} um; increase {name}.")


# --------------------------------------------------------------------------- routes
def op_length(op) -> float:
    return op[1] if op[0] == "S" else abs(op[2]) * op[1]


def route_length(ops) -> float:
    return sum(op_length(op) for op in ops)


def drop_empty(ops):
    return [op for op in ops if not (op[0] == "S" and op[1] <= 1e-9)]


def flip_ops(ops):
    """Mirror a route about its axis."""
    return [(op[0], op[1], -op[2]) if op[0] == "A" else op for op in ops]


def heading_after(ops, heading=0.0) -> float:
    return heading + sum(op[2] for op in ops if op[0] == "A")


def split_ops(ops, s):
    """Split a route at distance s, which must fall on a straight. Returns (head, tail)."""
    acc = 0.0
    for i, op in enumerate(ops):
        length = op_length(op)
        if acc + length >= s - 1e-9:
            if op[0] != "S":
                raise ValueError(f"Split point {s:.1f} um falls inside an arc.")
            a = s - acc
            head = ops[:i] + ([("S", a)] if a > 1e-9 else [])
            tail = ([("S", length - a)] if length - a > 1e-9 else []) + ops[i + 1:]
            return head, tail
        acc += length
    return list(ops), []


def turn_len(radius, pitch) -> float:
    """Centreline length of a 180-degree turn joining legs `pitch` apart."""
    return np.pi * radius + (pitch - 2 * radius)


def u_turn(r, direction, pitch):
    """180-degree turn from a leg heading `direction` (+1 up) to one `pitch` further along x."""
    return [("A", r, -direction * np.pi / 2), ("S", pitch - 2 * r), ("A", r, -direction * np.pi / 2)]


def loop_ops(r, add, l1, top, extra=0.0):
    """gds_maker loop_up (top=True, heading up) or loop_down; ends 2r + extra further along x."""
    s = 1 if top else -1
    return [("S", add), ("A", r, s * np.pi / 2), ("A", r, -s * np.pi / 2), ("S", l1),
            ("A", r, -s * np.pi / 2), ("S", 4 * r + extra), ("A", r, -s * np.pi / 2), ("S", l1),
            ("A", r, -s * np.pi / 2), ("A", r, s * np.pi / 2), ("S", add)]


def draw_route(start, ops, pieces, layer, tolerance=0.01, heading=0.0):
    """Draw CPW gap slots along `ops`; `pieces` = [(length, gap, width), ...] in order.

    Cross-section changes may fall anywhere, including inside arcs. Returns
    (polygons, end, bends [(s0, s1)], steps [s]) with s the distance along the route.
    """
    if abs(route_length(ops) - sum(pc[0] for pc in pieces)) > 1e-6:
        raise RuntimeError(f"Route length {route_length(ops):.4f} != section total "
                           f"{sum(pc[0] for pc in pieces):.4f} um.")
    x, y, h = float(start[0]), float(start[1]), float(heading)
    polys, bends, steps = [], [], []
    s_pos, k, k_rem = 0.0, 0, pieces[0][0]

    for op in ops:
        length = op_length(op)
        if op[0] == "A":
            bends.append((s_pos, s_pos + length))
        done = 0.0
        while done < length - 1e-9:
            chunk = min(length - done, k_rem)
            g, w = pieces[k][1], pieces[k][2]
            if op[0] == "S":
                ux, uy = np.cos(h), np.sin(h)
                nx, ny = -uy, ux
                x1, y1 = x + chunk * ux, y + chunk * uy
                for sgn in (1, -1):
                    o_in, o_out = sgn * w / 2, sgn * (w / 2 + g)
                    polys.append(gdspy.Polygon(
                        [(x + o_in * nx, y + o_in * ny), (x1 + o_in * nx, y1 + o_in * ny),
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


def vertices(polys) -> np.ndarray:
    return np.vstack([pp for q in polys
                      for pp in (q.polygons if hasattr(q, "polygons") else q.get_polygons())])


def side_extent(polys, y0, side, x_fn):
    """x_fn (max or min) over vertices more than 30 um to `side` (+1 above, -1 below) of y0."""
    pts = vertices(polys)
    sel = pts[side * (pts[:, 1] - y0) > 30.0]
    return None if len(sel) == 0 else float(x_fn(sel[:, 0]))


def mirror_pieces(p: Params, xsecs, feed_at_start: bool, pad: float):
    """Quarter-wave sections plus the 50-ohm lead (mirror_lead + pad) at the feed end."""
    sec = [(p.qw, g, w) for g, w in xsecs]
    feed = (p.mirror_lead + pad, *p.feed_xsec)
    return [feed] + sec if feed_at_start else sec + [feed]


def step_positions(pieces) -> np.ndarray:
    return np.cumsum([pc[0] for pc in pieces])[:-1]


def clear_of_arcs(ops, pieces) -> float:
    """Smallest distance from a cross-section step to an arc end; -1 if a step is inside an arc."""
    lens = np.array([op_length(op) for op in ops])
    s0 = np.concatenate(([0.0], np.cumsum(lens)[:-1]))
    is_arc = np.array([op[0] == "A" for op in ops])
    b0, b1 = s0[is_arc], (s0 + lens)[is_arc]
    st = step_positions(pieces)[:, None]
    if ((st > b0) & (st < b1)).any():
        return -1.0
    return float(np.minimum(np.abs(st - b0), np.abs(st - b1)).min())


def clear_on_legs(ops, is_leg, pieces) -> float:
    """Smallest distance from a step to the end of the leg it sits on; -1 if one is off the legs."""
    lens = np.array([op_length(op) for op in ops])
    s0 = np.concatenate(([0.0], np.cumsum(lens)[:-1]))
    leg = np.array(is_leg)
    a, b = s0[leg], (s0 + lens)[leg]
    worst = np.inf
    for st in step_positions(pieces):
        inside = (st >= a) & (st <= b)
        if not inside.any():
            return -1.0
        worst = min(worst, float(np.max(np.minimum(st - a[inside], b[inside] - st))))
    return worst


def warn_step_clear(p: Params, side: str, clear: float):
    if clear < p.mirror_step_clear:
        print(f"WARNING: {side} mirror: nearest impedance step is {clear:.1f} um from a bend "
              f"(target {p.mirror_step_clear:g}).")


# --------------------------------------------------------------------------- mirrors
# builder(p, start, xsecs, side) -> (polygons, end, info); starts and ends heading +x

def serpentine_ops(lead_a, legs, amplitude, radius, lead_b, first_up=True, pitch=None):
    r, a = radius, amplitude
    d = 2 * r if pitch is None else pitch
    s = 1 if first_up else -1
    ops = [("S", lead_a), ("A", r, s * np.pi / 2), ("S", a - 2 * r)]
    for _ in range(legs):
        ops += u_turn(r, s, d) + [("S", 2 * (a - r))]
        s = -s
    ops += u_turn(r, s, d) + [("S", a - 2 * r), ("A", r, s * np.pi / 2), ("S", lead_b)]
    return drop_empty(ops)


def serpentine_amplitude(length, legs, radius, lead_a, lead_b, pitch=None):
    """Amplitude that makes serpentine_ops(...) exactly `length` long."""
    r, n = radius, legs
    d = 2 * r if pitch is None else pitch
    return ((length - lead_a - lead_b - np.pi * r - (n + 1) * turn_len(r, d)
             + 4 * r + 2 * r * n) / (2 + 2 * n))


def serpentine_mirror(p: Params, start, xsecs, side):
    """Plain serpentine at the full chip height. 50-ohm padding at the feed end and extra
    lead at the cavity end are searched so no impedance step sits on a bend."""
    r, d, lead = p.mir_r, p.mir_pitch, p.mirror_lead
    xw = max(w + 2 * g for g, w in xsecs)
    a_max = (p.mir_h_max - xw) / 2
    n_qw = len(xsecs) * p.qw
    left = side == "left"

    legs = next((n for n in range(400)
                 if serpentine_amplitude(n_qw + lead, n, r, lead, lead, d) <= a_max), None)
    if legs is None:
        raise ValueError("Mirror cannot meet mirror_height_max.")

    def candidate(pad, extra):
        la, lb = (lead, lead + extra) if left else (lead + extra, lead)
        a = serpentine_amplitude(n_qw + lead + pad, legs, r, la, lb, d)
        return a, serpentine_ops(la, legs, a, r, lb, pitch=d), mirror_pieces(p, xsecs, left, pad)

    leg_len = 2 * (a_max - r) + turn_len(r, d)
    best, fallback = None, None
    for extra in np.arange(0.0, d + 1e-9, 5.0):
        if best is not None and extra >= best[1] + best[2]:
            break
        for pad in np.arange(0.0, leg_len, 4.0):
            if best is not None and pad + extra >= best[1] + best[2]:
                break
            a, ops, pieces = candidate(pad, extra)
            if a > a_max:
                break
            if a < 2 * r:
                continue
            clear = clear_of_arcs(ops, pieces)
            if fallback is None or clear > fallback[0]:
                fallback = (clear, pad, extra, a, pieces, ops)
            if clear >= p.mirror_step_clear:
                best = (clear, pad, extra, a, pieces, ops)
                break
    best = best or fallback
    if best is None:
        raise ValueError("Could not place the serpentine mirror.")
    clear, pad, extra, a, pieces, ops = best
    warn_step_clear(p, side, clear)

    polys, end, bends, steps = draw_route(start, ops, pieces, p.gap_layer, p.arc_tolerance)
    in_bend = sum(any(b0 < st < b1 for b0, b1 in bends) for st in steps)
    info = {"sections": len(xsecs), "legs": legs, "amplitude": a, "height": 2 * a + xw,
            "width": end[0] - start[0], "radius": r, "pitch": d, "ground_between_legs": d - xw,
            "pad": pad, "extra_lead": extra, "min_step_to_bend": clear, "steps_in_bends": in_bend}
    return polys, end, info


def loop_mirror(p: Params, start, xsecs, side):
    """One gds_maker loop per section; loops alternate below / above the axis, S-jogs between.

    The outer sections loop down, away from the launcher feeds.
    """
    r, n = p.radius, len(xsecs)
    down = True if side == "left" else (n - 1) % 2 == 0
    x, y = float(start[0]), float(start[1])
    polys, lengths = [], []
    for i, (g, w) in enumerate(xsecs):
        first, last = i == 0, i == n - 1
        leg = p.loop_leg(first=first, last=last)
        sgn = -1 if down else 1
        ops_a = ([("A", r, sgn * np.pi / 2)] if first else []) + [("S", leg)]
        pl, (x, y), _, _ = draw_route((x, y), ops_a, [(route_length(ops_a), g, w)],
                                      p.gap_layer, p.arc_tolerance,
                                      heading=0.0 if first else sgn * np.pi / 2)
        polys += pl
        loop = GM.loop_down if down else GM.loop_up
        pl, end = loop([x, y], addlength=p.addlength, length1=p.length1, radius=r,
                       gap=g, width=w, layer=p.gap_layer)
        polys += pl
        x, y = float(end[0]), float(end[1])
        if last:
            ops_b = [("S", leg), ("A", r, sgn * np.pi / 2)]
        else:
            ops_b = drop_empty([("S", leg), ("A", r, sgn * np.pi / 2), ("S", p.loop_jog_extra),
                                ("A", r, -sgn * np.pi / 2)])
        pl, (x, y), _, _ = draw_route((x, y), ops_b, [(route_length(ops_b), g, w)],
                                      p.gap_layer, p.arc_tolerance,
                                      heading=-sgn * np.pi / 2)
        polys += pl
        lengths.append(route_length(ops_a) + p.loop_len + route_length(ops_b))
        down = not down

    if abs(y - start[1]) > 1e-6:
        raise RuntimeError(f"Loop mirror did not return to its axis (dy = {y - start[1]:.3f} um).")
    pts = vertices(polys)
    info = {"sections": n, "legs": n, "width": x - float(start[0]),
            "height": float(pts[:, 1].max() - pts[:, 1].min()),
            "leg_first": p.loop_leg(first=True), "leg_mid": p.loop_leg(), "leg_last": p.loop_leg(last=True),
            "section_len_min": min(lengths), "section_len_max": max(lengths),
            "ground_between_legs": 2 * r + 2 * p.loop_jog_extra - max(w + 2 * g for g, w in xsecs),
            "min_step_to_bend": 0.0, "steps_in_bends": 0}
    return polys, (x, y), info


def cell_meander_ops(p: Params, legs: int, F: float, lead_a: float, lead_b: float,
                     drop: float = 0.0, flip: bool = False):
    """The gds_maker cell pattern as one meander with leg height F. Returns (ops, is_leg).

    Legs 2r apart; top turns go loop, U, loop, U..., bottom turns U, loop, U..., so same-side
    loops are 8r apart as in the cells. Half legs at both ends; the last turn is a plain U-turn
    lowered by `drop`, which closes the total length.
    """
    r, add, l1, d = p.radius, p.addlength, p.length1, p.cm_pitch
    extra = d - 2 * r

    def plain(direction):
        return [("A", r, -direction * np.pi)] if extra <= 1e-9 else u_turn(r, direction, d)

    ops, leg = [("S", lead_a), ("A", r, np.pi / 2), ("S", F / 2 - r)], [False, False, True]
    for i in range(legs - 1):
        last_turn = i == legs - 2
        if i % 2 == 0:
            turn = loop_ops(r, add, l1, True, extra) if i % 4 == 0 and not last_turn else plain(1)
        else:
            turn = loop_ops(r, add, l1, False, extra) if i % 4 == 3 and not last_turn else plain(-1)
        if last_turn:
            ops[-1] = ("S", ops[-1][1] - drop)
        ops += turn
        leg += [False] * len(turn)
        ops.append(("S", F / 2 - r - drop if last_turn else F))
        leg.append(True)
    going_up = (legs - 1) % 2 == 0
    ops += [("A", r, -np.pi / 2 if going_up else np.pi / 2), ("S", lead_b)]
    leg += [False, False]
    if flip:
        ops = flip_ops(ops)
    keep = [i for i, op in enumerate(ops) if not (op[0] == "S" and op[1] <= 1e-9)]
    return [ops[i] for i in keep], [leg[i] for i in keep]


def cell_flip(legs: int, feed_at_start: bool) -> bool:
    """Flip so the loop nearest the launcher hangs below the axis."""
    loops = [i for i in range(legs - 2) if i % 4 in (0, 3)]
    if not loops:
        return False
    return (loops[0] if feed_at_start else loops[-1]) % 2 == 0


def cell_leg_height(p: Params, xw: float) -> float:
    """cell_column_F, checked against the chip height."""
    f_max = p.chip_height - 2 * p.edge_margin - 2 * p.mirror_clearance - 2 * p.loop_height - xw
    F = cell_column_F(p)
    if F > f_max:
        raise ValueError(f"sections_per_column={p.sections_per_column:g} needs {F:.0f} um legs; "
                         f"only {f_max:.0f} um fit. Lower it.")
    return F


def cell_column_F(p: Params) -> float:
    """Leg height giving exactly sections_per_column quarter waves per 4-leg column."""
    extra = p.cm_pitch - 2 * p.radius
    turns = 2 * (p.loop_len + extra) + 2 * (np.pi * p.radius + extra)
    return (p.sections_per_column * p.qw - turns) / 4


def cell_meander_mirror(p: Params, start, xsecs, side):
    """Cell pattern, periodic so the impedance steps repeat at the same spots in every column.

    When symmetric, both mirrors follow one path laid out for the longer mirror; the right
    mirror is its mirror image and the shorter one carries 50-ohm line on its outer part.
    """
    if not p.is_symmetric:
        plan = cell_meander_plan(p, xsecs, side)
        return draw_cell_meander(p, start, xsecs, side, plan, plan["ops"], plan["pieces"])
    full = p.full_sequence
    plan = cell_meander_plan(p, full, "left")
    pad = plan["pad"] + (len(full) - len(xsecs)) * p.qw
    ops = plan["ops"] if side == "left" else list(reversed(plan["ops"]))
    return draw_cell_meander(p, start, xsecs, side, plan, ops, mirror_pieces(p, xsecs, side == "left", pad))


def cell_meander_plan(p: Params, xsecs, side) -> dict:
    """Search legs, 50-ohm padding and the last U-turn drop so every step sits on a leg."""
    r, qw, lead = p.radius, p.qw, p.mirror_lead
    xw = max(w + 2 * g for g, w in xsecs)
    F = cell_leg_height(p, xw)
    left = side == "left"
    n_qw = len(xsecs) * qw
    drop_max = F / 2 - r - 2 * p.mirror_step_clear

    best = None
    for n in range(2, 400):
        flip = cell_flip(n, left)
        r0 = route_length(cell_meander_ops(p, n, F, lead, lead, flip=flip)[0])
        pad_lo = max(0.0, r0 - 2 * drop_max - lead - n_qw)
        pad_hi = min(qw + 2000.0, r0 - lead - n_qw)
        if pad_hi < 0:
            continue
        for pad in np.arange(np.ceil(pad_lo / 2) * 2, pad_hi + 1e-9, 2.0):
            drop = (r0 - lead - n_qw - pad) / 2
            ops, is_leg = cell_meander_ops(p, n, F, lead, lead, drop, flip)
            pieces = mirror_pieces(p, xsecs, left, pad)
            clear = clear_on_legs(ops, is_leg, pieces)
            if best is None or clear > best[0]:
                best = (clear, n, pad, drop, ops, pieces)
            if clear >= p.mirror_step_clear:
                break
        if best is not None and best[0] >= p.mirror_step_clear:
            break
        if pad_lo > qw + 2000.0:
            break
    if best is None:
        raise ValueError("Could not lay out the cell_meander mirror.")
    clear, n, pad, drop, ops, pieces = best
    warn_step_clear(p, side, clear)
    return {"clear": clear, "legs": n, "pad": pad, "drop": drop, "ops": ops, "pieces": pieces,
            "F": F, "xw": xw}


def draw_cell_meander(p: Params, start, xsecs, side, plan, ops, pieces, heading=0.0):
    polys, end, _, steps = draw_route(start, ops, pieces, p.gap_layer, p.arc_tolerance, heading)
    seg = np.diff([0.0] + steps + [route_length(ops)])
    n = len(xsecs)
    sec_lens = seg[-n:] if side == "left" else seg[:n]
    pad = pieces[0][0] - p.mirror_lead if side == "left" else pieces[-1][0] - p.mirror_lead
    pts = vertices(polys)
    info = {"sections": n, "legs": plan["legs"], "leg_height": plan["F"], "pad": pad,
            "tail_drop": plan["drop"], "width": end[0] - float(start[0]),
            "height": float(pts[:, 1].max() - pts[:, 1].min()),
            "section_len_min": float(sec_lens.min()), "section_len_max": float(sec_lens.max()),
            "ground_between_legs": p.cm_pitch - plan["xw"], "min_step_to_bend": plan["clear"],
            "steps_in_bends": 0}
    return polys, end, info


def continuous_plan(p: Params) -> dict:
    """One half-chip meander: 50-ohm lead, mirror sections, then cavity line up to the flat.

    Searched like cell_meander_plan, with the mirror/cavity boundary also on a leg. The
    cavity part c of each half sets the flat: flat = cavity_len - 2c >= cavity_flat.
    """
    r, qw, lead = p.radius, p.qw, p.mirror_lead
    full = p.full_sequence
    xw = max([w + 2 * g for g, w in full] + [p.cav_xw])
    F = cell_leg_height(p, xw)
    n_qw = len(full) * qw
    c_max = (p.cavity_len - p.cavity_flat) / 2
    if c_max <= 0:
        raise ValueError("cavity_flat is longer than the cavity.")
    sec = [(qw, g, w) for g, w in full]

    # padding places the steps; the last U-turn drop closes the length; cost = half-chip width
    drop_max = F / 2 - r - 2 * p.mirror_step_clear
    best, fallback = None, None
    for n in range(2, 400):
        flip = cell_flip(n, True)
        spare = route_length(cell_meander_ops(p, n, F, lead, 0.0, 0.0, flip)[0]) - lead - n_qw
        if spare < 0:
            continue
        if spare - 2 * drop_max - c_max > qw + 2000.0:
            break
        found = None
        for pad in np.arange(0.0, min(qw + 2000.0, spare) + 1e-9, 2.0):
            drop = max(0.0, (spare - pad - c_max) / 2)
            if drop > drop_max:
                continue
            c = spare - pad - 2 * drop
            ops, is_leg = cell_meander_ops(p, n, F, lead, 0.0, drop, flip)
            pieces = [(lead + pad, *p.feed_xsec)] + sec + [(c, *p.cavity_xsec)]
            clear = clear_on_legs(ops, is_leg, pieces)
            if fallback is None or clear > fallback[0]:
                fallback = (clear, n, pad, c, ops, drop)
            if clear >= p.mirror_step_clear:
                cost = n * p.cm_pitch + (p.cavity_len - 2 * c) / 2
                if found is None or cost < found[0]:
                    found = (cost, (clear, n, pad, c, ops, drop))
                if drop > 0:
                    break
        if found is not None and (best is None or found[0] < best[0]):
            best = found
    if best is None and fallback is None:
        raise ValueError("Could not lay out the continuous meander.")
    best = best[1] if best is not None else fallback
    clear, n, pad, c, ops, drop = best
    warn_step_clear(p, "half-chip", clear)
    return {"clear": clear, "legs": n, "pad": pad, "c": c, "flat": p.cavity_len - 2 * c,
            "ops": ops, "F": F, "xw": xw, "drop": drop, "full": full}


def cells_mirror(p: Params, start, xsecs, side):
    """gds_maker Bragg unit cells. Starts heading down (left) / ends heading down (right)."""
    L = p.cell_lengths
    (lf, ls), (rf, rs) = p.mirror_xsecs
    a, b = (lf, ls) if side == "left" else (rf, rs)
    kw = dict(addlength=p.addlength, length1=p.length1, radius=p.radius, layer=p.gap_layer,
              gap1=a[0], width1=a[1], gap2=b[0], width2=b[1],
              **{k: L[k] for k in ("vertical_length1", "vertical_length2",
                                   "vertical_length3", "vertical_length4")})
    cells = len(xsecs) // 4
    polys, pos = [], start
    if side == "left":
        for _ in range(cells - 1):
            pl, pos = GM.Braggmirror_unitcell1_left(pos, **kw)
            polys += pl
        pl, pos = GM.Braggmirror_unitcell2_left(pos, **kw)
        polys += pl
    else:
        pl, pos = GM.Braggmirror_unitcell2_right(pos, **dict(kw, vertical_length5=L["vertical_length5"]))
        polys += pl
        for _ in range(cells - 1):
            pl, pos = GM.Braggmirror_unitcell1_right(pos, **kw)
            polys += pl
    return polys, pos, None


MIRROR_BUILDERS = {"cells": cells_mirror, "cell_meander": cell_meander_mirror,
                   "loops": loop_mirror, "serpentine": serpentine_mirror}


# --------------------------------------------------------------------------- cavities
def meander_amplitude(length, legs, radius, lead, flat=0.0, pitch=None):
    """Amplitude for two mirrored serpentine halves around a centred flat."""
    return serpentine_amplitude((length - 2 * lead - flat) / 2, legs, radius, 0.0, 0.0, pitch)


def meander_cavity(p: Params, start):
    """Serpentine cavity with the flat at the electrical centre."""
    g = p.meander_geometry
    r, n, a = g["radius"], g["legs"], g["amplitude"]
    if a - 2 * r < 0:
        raise ValueError(f"Meander amplitude {a:.1f} um too small for radius {r} um.")
    second_up = p.cavity_first_up if n % 2 == 0 else not p.cavity_first_up
    ops = drop_empty(serpentine_ops(p.cavity_lead, n, a, r, 0.0, p.cavity_first_up, g["pitch"])
                     + [("S", p.cavity_flat)]
                     + serpentine_ops(0.0, n, a, r, p.cavity_lead, second_up, g["pitch"]))
    polys, end, _, _ = draw_route(start, ops, [(route_length(ops), *p.cavity_xsec)], p.gap_layer,
                                  p.arc_tolerance)
    if abs(end[1] - start[1]) > 1e-6:
        raise RuntimeError("Meander cavity did not close on its axis.")
    return polys, end, dict(g, start=start)


def spine_plan(p: Params, lead: float) -> dict:
    """Spine length, serpentine legs per half and amplitude for the spine cavity.

    Half route: lead -> bend -> [half leg, (k-1) x (U-turn + leg), U-turn] -> riser ->
    U-turn (spine pitch) -> spine. With U(d) = pi*r + d - 2r:
      k = 0 : half = lead + pi*r/2 + Vs/2 - r + U(D)
      k >= 1: half = lead + pi*r/2 + k*U(d) + U(D) + Vs/2 - (2k+1)*r + 2k*A
      total = 2*half + Vs
    """
    L, r, d, D, xw = p.cavity_len, p.cav_r, p.cav_pitch, p.spine_d, p.cav_xw
    usable = p.chip_height - 2 * p.edge_margin
    vs_cap = usable - 2 * r - xw
    a_cap = min(p.cavity_amp_max, (usable - xw) / 2)
    vs_zero_legs = (L - 2 * lead - np.pi * r + 2 * r - 2 * turn_len(r, D)) / 2

    def amplitude(k, vs):
        half = (L - vs) / 2
        return (half - lead - np.pi * r / 2 - k * turn_len(r, d) - turn_len(r, D) - vs / 2
                + (2 * k + 1) * r) / (2 * k)

    vs = min(vs_zero_legs, vs_cap) if p.spine_length is None else float(p.spine_length)
    if vs <= 0:
        raise ValueError("Cavity too short for a spine with these radius / pitch / lead values.")
    if vs > vs_cap + 1e-6:
        raise ValueError(f"spine_length {vs:.0f} um exceeds the {vs_cap:.0f} um the chip height allows.")
    if vs > vs_zero_legs + 1e-6:
        raise ValueError(f"spine_length {vs:.0f} um needs more line than the cavity has; max is "
                         f"{vs_zero_legs:.0f} um at this cavity_length.")

    if abs(vs - vs_zero_legs) < 1e-6 and not p.spine_legs:
        k, a = 0, None
    elif p.spine_legs is not None:
        k = int(p.spine_legs)
        if k < 1:
            raise ValueError("spine_legs = 0 only works with spine_length=None (the longest spine).")
        a = amplitude(k, vs)
    else:
        k = next((n for n in range(1, 200) if amplitude(n, vs) <= a_cap), None)
        if k is None:
            raise ValueError("No leg count keeps the cavity serpentine inside the chip.")
        a = amplitude(k, vs)
    if a is not None and a < 2 * r:
        raise ValueError(f"Serpentine amplitude {a:.1f} um < 2 x radius: too little spare line for "
                         f"{k} legs; use fewer legs or a longer spine.")
    if a is not None and a > a_cap + 1e-6:
        raise ValueError(f"Serpentine amplitude {a:.1f} um exceeds {a_cap:.0f} um; use more legs.")

    half_w = lead + r + k * d + D
    return {"type": "spine", "length": L, "spine_length": vs, "legs": k, "amplitude": a,
            "lead": lead, "radius": r, "pitch": d, "spine_pitch": D,
            "ground_between_legs": d - xw, "spine_ground": D - xw,
            "width": 2 * half_w, "half_width": half_w,
            "height": vs + 2 * r + xw if a is None else max(vs + 2 * r, 2 * a) + xw}


def spine_ops(plan: dict, up: bool):
    """Full spine cavity; the right half is the left half rotated 180 degrees."""
    r, d, D, k, a, vs, s = (plan["radius"], plan["pitch"], plan["spine_pitch"], plan["legs"],
                            plan["amplitude"], plan["spine_length"], plan["lead"])
    sg = 1 if up else -1
    half = [("S", s)]
    if k == 0:
        half += [("A", r, sg * np.pi / 2), ("S", vs / 2 - r)]
    else:
        direction = sg if k % 2 == 0 else -sg
        half += [("A", r, direction * np.pi / 2), ("S", a - 2 * r)]
        for _ in range(k - 1):
            half += u_turn(r, direction, d) + [("S", 2 * (a - r))]
            direction = -direction
        half += u_turn(r, direction, d) + [("S", vs / 2 + a - r)]
    half += u_turn(r, sg, D)
    return drop_empty(half + [("S", vs)] + flip_ops(list(reversed(half))))


def spine_cavity(p: Params, start, left_mirror_polys, right_mirror_template):
    """Spine cavity with the lead raised until the first riser clears both mirrors."""
    ground = p.min_ground if p.min_ground is not None else p.cav_pitch - p.cav_xw
    lead = p.cavity_lead
    for _ in range(4):                     # the lead can change the leg parity
        plan = spine_plan(p, lead)
        up = 1 if p.cavity_first_up else -1
        depart = up if plan["legs"] % 2 == 0 else -up
        need = lead
        mx = side_extent(left_mirror_polys, start[1], depart, np.max)
        if mx is not None:
            need = max(need, mx - start[0] + ground + p.cav_xw / 2 - plan["radius"])
        mn = side_extent(right_mirror_template, 0.0, -depart, np.min)
        if mn is not None:
            need = max(need, -mn + ground + p.cav_xw / 2 - plan["radius"])
        if need <= lead + 1e-6:
            break
        lead = float(np.ceil(need))

    ops = spine_ops(plan, p.cavity_first_up)
    polys, end, _, _ = draw_route(start, ops, [(route_length(ops), *p.cavity_xsec)], p.gap_layer,
                                  p.arc_tolerance)
    if abs(route_length(ops) - p.cavity_len) > 1e-6 or abs(end[1] - start[1]) > 1e-6:
        raise RuntimeError("Spine cavity length or closure is wrong.")
    info = dict(plan, start=start, spine_x=start[0] + plan["half_width"],
                centre_x=start[0] + plan["half_width"],
                spine_y=(start[1] - plan["spine_length"] / 2, start[1] + plan["spine_length"] / 2))
    return polys, end, info


def loops_cavity_half(p: Params, lead_out: float, lead_in: float, h: float):
    """Left half: lead, loop excursions (leg, loop, leg), lead to the flat."""
    r, d = p.radius, p.cm_pitch
    up = 1 if p.cavity_first_up else -1
    ops = [("S", lead_out)]
    for j in range(p.cavity_loops):
        s = up if j % 2 == 0 else -up
        ops += ([("A", r, s * np.pi / 2), ("S", h)]
                + loop_ops(r, p.addlength, p.length1, s > 0, d - 2 * r)
                + [("S", h), ("A", r, s * np.pi / 2)])
    return drop_empty(ops + [("S", lead_in)])


def loops_cavity_leg(p: Params, lead_out: float, lead_in: float) -> float:
    """Leg height that makes the whole cavity exactly cavity_len long."""
    half = (p.cavity_len - p.cavity_flat) / 2
    one = route_length(loops_cavity_half(p, 0.0, 0.0, 0.0)) / p.cavity_loops
    return (half - lead_out - lead_in - p.cavity_loops * one) / (2 * p.cavity_loops)


def loops_cavity(p: Params, start, left_mirror_polys, right_mirror_template):
    """Loop-meander cavity with a centred flat; the right half mirrors the left.

    lead_out (mirror side) is raised until the first loop clears both mirrors; lead_in
    keeps the loops at least flat_clearance clear of the flat's ends.
    """
    xw = p.cav_xw
    ground = p.min_ground if p.min_ground is not None else p.cm_pitch - xw
    depart = 1 if p.cavity_first_up else -1

    probe, _, _, _ = draw_route((0.0, 0.0), loops_cavity_half(p, 0.0, 0.0, 10.0),
                                [(route_length(loops_cavity_half(p, 0.0, 0.0, 10.0)), *p.cavity_xsec)],
                                p.gap_layer, p.arc_tolerance)
    pts = vertices(probe)
    off_pts = pts[np.abs(pts[:, 1]) > 30.0]
    reach_back = float(-off_pts[:, 0].min())
    reach_fwd = float(off_pts[:, 0].max() - pts[:, 0].max())

    lead_in = max(p.cavity_lead, reach_fwd + p.flat_clearance)
    lead_out = p.cavity_lead
    mx = side_extent(left_mirror_polys, start[1], depart, np.max)
    if mx is not None:
        lead_out = max(lead_out, mx - start[0] + ground + reach_back)
    mn = side_extent(right_mirror_template, 0.0, depart, np.min)
    if mn is not None:
        lead_out = max(lead_out, -mn + ground + reach_back)
    lead_out = float(np.ceil(lead_out))
    lead_in = float(np.ceil(lead_in))

    h = loops_cavity_leg(p, lead_out, lead_in)
    if h < 1.0:
        raise ValueError(f"Cavity too short for {p.cavity_loops} loop(s) per half with a "
                         f"{p.cavity_flat:g} um flat (leg {h:.0f} um); use fewer loops or a shorter flat.")
    half = loops_cavity_half(p, lead_out, lead_in, h)
    ops = drop_empty(half + [("S", p.cavity_flat)] + list(reversed(half)))
    polys, end, _, _ = draw_route(start, ops, [(route_length(ops), *p.cavity_xsec)], p.gap_layer,
                                  p.arc_tolerance)
    if abs(route_length(ops) - p.cavity_len) > 1e-6 or abs(end[1] - start[1]) > 1e-6:
        raise RuntimeError("Loops cavity length or closure is wrong.")
    half_len = route_length(half)
    centre_x = start[0] + (end[0] - start[0]) / 2
    cav_pts = vertices(polys)
    return polys, end, {"type": "loops", "length": p.cavity_len, "flat": p.cavity_flat,
                        "loops": p.cavity_loops, "leg": h, "lead_out": lead_out, "lead_in": lead_in,
                        "half_length": half_len, "start": start, "centre_x": centre_x,
                        "width": end[0] - start[0],
                        "height": float(cav_pts[:, 1].max() - cav_pts[:, 1].min())}


# --------------------------------------------------------------------------- assembly
def continuous_chain(p: Params, start):
    """Left mirror, cavity and right mirror as one mirror-symmetric meander."""
    plan = continuous_plan(p)
    ops, c, flat = plan["ops"], plan["c"], plan["flat"]
    seq_left, seq_right = p.mirror_sequences
    mir_ops, cav_ops = split_ops(ops, route_length(ops) - c)
    h_mirror_end = heading_after(mir_ops)
    extra_l = (len(plan["full"]) - len(seq_left)) * p.qw
    extra_r = (len(plan["full"]) - len(seq_right)) * p.qw

    left_polys, pos, left_info = draw_cell_meander(
        p, start, seq_left, "left", plan, mir_ops, mirror_pieces(p, seq_left, True, plan["pad"] + extra_l))

    cav = p.cavity_xsec
    pl_a, flat_start, _, _ = draw_route(pos, cav_ops, [(c, *cav)], p.gap_layer, p.arc_tolerance, h_mirror_end)
    pl_f, flat_end, _, _ = draw_route(flat_start, [("S", flat)], [(flat, *cav)], p.gap_layer, p.arc_tolerance)
    back = list(reversed(cav_ops))
    pl_b, pos, _, _ = draw_route(flat_end, back, [(c, *cav)], p.gap_layer, p.arc_tolerance)
    cavity_polys = pl_a + pl_f + pl_b

    right_ops = list(reversed(mir_ops))
    right_polys, end, right_info = draw_cell_meander(
        p, pos, seq_right, "right", plan, right_ops,
        mirror_pieces(p, seq_right, False, plan["pad"] + extra_r), heading_after(back))
    if abs(end[1] - start[1]) > 1e-6:
        raise RuntimeError("Continuous meander did not close on its axis.")

    pts = vertices(cavity_polys)
    cavity = {"type": "continuous", "length": p.cavity_len, "flat": flat, "meander_part": c,
              "start": (float(flat_start[0]), float(flat_start[1])),
              "centre_x": float(flat_start[0]) + flat / 2,
              "width": float(pts[:, 0].max() - pts[:, 0].min()),
              "height": float(pts[:, 1].max() - pts[:, 1].min())}
    return [left_polys, cavity_polys, right_polys], end, {"left": left_info, "right": right_info}, cavity


def separate_chain(p: Params, start):
    """Left mirror, cavity and right mirror built one after another."""
    build_mirror = MIRROR_BUILDERS[p.mirror_style]
    seq_left, seq_right = p.mirror_sequences
    mirrors = {}
    left, pos, mirrors["left"] = build_mirror(p, start, seq_left, "left")
    cav_start = (float(pos[0]), float(pos[1]))
    if p.cavity_style in ("spine", "loops"):
        template, _, _ = build_mirror(p, (0.0, 0.0), seq_right, "right")
        make = spine_cavity if p.cavity_style == "spine" else loops_cavity
        cav, pos, cavity = make(p, cav_start, left, template)
    elif p.cavity_style == "meander":
        cav, pos, cavity = meander_cavity(p, cav_start)
    else:
        cav, pos = GM.createCPWhor(pos, length=p.cavity_len, gap=p.cavity_xsec[0],
                                   width=p.cavity_xsec[1], layer=p.gap_layer)
        cavity = None
    right, pos, mirrors["right"] = build_mirror(p, (float(pos[0]), float(pos[1])), seq_right, "right")
    return [left, cav, right], pos, mirrors, cavity


def end_cap(p: Params, tip):
    """Open end above a downward line tip (no launcher)."""
    half = p.feed_width / 2.0 + p.feed_gap
    return gdspy.Rectangle((tip[0] - half, tip[1]), (tip[0] + half, tip[1] + p.feed_gap),
                           layer=p.gap_layer)


def build(p: Params):
    """Chain launchers, feeds, mirrors and cavity. Returns (gap polygons, end caps, info)."""
    p.check()
    r, layer = p.radius, p.gap_layer
    fg, fw = p.feed_xsec
    L = p.cell_lengths
    cells = p.mirror_style == "cells"

    y_axis = p.chip_height / 2.0
    y_pad = p.chip_height - p.edge_margin - p.launcher_gap
    y_tip = y_pad - (p.launcher_straight + p.launcher_taper)
    x_left = p.launcher_height / 2.0 + p.edge_margin
    x_feed_end = p.launcher_gap + p.launcher_straight + p.launcher_taper + p.feed_left

    def launcher(x):
        return GM.Createlauncherdown([x, y_pad], straight=p.launcher_straight, height=p.launcher_height,
                                     taper=p.launcher_taper, gapL=p.launcher_gap, gap=fg, width=fw,
                                     layer=layer, notch=True)

    polys, caps, marks = [], [], []

    if p.launchers:
        pl, pos = launcher(x_left)
        polys += pl
    else:
        pos = (x_left, y_tip)
        caps.append(end_cap(p, pos))
    entry = (float(pos[0]), float(pos[1]))
    down = pos[1] - (y_axis + r)
    if down < 0:
        raise ValueError(f"Left feed down length is negative ({down:.1f} um); increase chip_height.")
    pl, pos = GM.createCPWvertDown(pos, length=down, gap=fg, width=fw, layer=layer)
    polys += pl
    pl, _ = GM.createDoublearc((pos[0] + r, pos[1]), radius=r, gap=fg, width=fw, angle=np.pi, layer=layer)
    polys += pl
    pos = (pos[0] + r, pos[1] - r)
    run = x_feed_end - pos[0]
    if run < 0:
        raise ValueError(f"Left feed horizontal length is negative ({run:.1f} um).")
    pl, pos = GM.createCPWhor(pos, length=run, gap=fg, width=fw, layer=layer)
    polys += pl
    if cells:
        pl, pos = GM.createDoublearc((pos[0], pos[1] - r), radius=r, gap=fg, width=fw,
                                     angle=2 * np.pi, layer=layer)
        polys += pl
        pl, pos = GM.createCPWvertDown(pos, length=L["downlength"], gap=fg, width=fw, layer=layer)
        polys += pl

    chain = continuous_chain if p.cavity_style == "continuous" else separate_chain
    parts, pos, mirrors, cavity = chain(p, (float(pos[0]), float(pos[1])))
    for pl in parts:
        marks.append(len(polys))
        polys += pl
    marks.append(len(polys))
    if cells:
        pl, pos = GM.createCPWvertDown(pos, length=L["downlength"], gap=fg, width=fw, layer=layer)
        polys += pl
        pl, pos = GM.createDoublearc((pos[0] + r, pos[1]), radius=r, gap=fg, width=fw,
                                     angle=-np.pi, layer=layer)
        polys += pl
        pos = (pos[0] + r, pos[1] - r)

    right_run = run if p.is_symmetric else p.feed_right
    if right_run:
        pl, pos = GM.createCPWhor(pos, length=right_run, gap=fg, width=fw, layer=layer)
        polys += pl
    pl, _ = GM.createDoublearc((pos[0], pos[1] + r), radius=r, gap=fg, width=fw,
                               angle=-np.pi / 2, layer=layer)
    polys += pl
    pos = (pos[0] + r, pos[1] + r)
    up = y_tip - pos[1]
    if up < 0:
        raise ValueError(f"Right feed up length is negative ({up:.1f} um); increase chip_height.")
    pl, pos = GM.createCPWvertUp(pos, length=up, gap=fg, width=fw, layer=layer)
    polys += pl
    x_right = float(pos[0])
    if p.launchers:
        pl, _ = launcher(x_right)
        polys += pl
    else:
        caps.append(end_cap(p, (x_right, y_tip)))
    exit_ = (x_right, float(y_tip))
    marks.append(len(polys))

    bounds = [0] + marks
    info = {"cavity": cavity, "mirrors": {k: v for k, v in mirrors.items() if v is not None},
            "entry": entry, "exit": exit_, "x_left": x_left, "x_right": x_right,
            "probes": [(entry[0], entry[1] - 1.0), (exit_[0], exit_[1] - 1.0)],
            "sections": {n: (bounds[i], bounds[i + 1]) for i, n in enumerate(SECTION_NAMES)}}
    return polys, caps, info


# --------------------------------------------------------------------------- layout and checks
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
    """Build and place on the chip, centring the cavity or the whole assembly."""
    polys, caps, info = build(p)
    gaps = [as_polyset(polys, p.gap_layer)] + caps
    lo, hi = bbox_of(gaps)
    cav = info["cavity"]
    centred = cav is not None and cav["type"] in ("spine", "loops", "continuous")

    dx = 0.0
    if p.chip_width is None:
        chip_w = (info["x_right"] + p.launcher_height / 2.0 + p.edge_margin
                  if p.launchers else float(hi[0] + lo[0]))
    else:
        chip_w = float(p.chip_width)
        dx = (chip_w / 2 - cav["centre_x"] if centred and p.spine_centered
              else 0.5 * (chip_w - (hi[0] - lo[0])) - lo[0])
        for g in gaps:
            g.translate(dx, 0.0)
        lo, hi = lo + np.array([dx, 0.0]), hi + np.array([dx, 0.0])

    m = p.edge_margin
    sections = {}
    for name, (i0, i1) in info["sections"].items():
        if i1 > i0:
            slo, shi = bbox_of(polys[i0:i1])
            slo, shi = slo + np.array([dx, 0.0]), shi + np.array([dx, 0.0])
            sections[name] = {"lo": slo, "hi": shi, "left": slo[0] - m, "right": chip_w - shi[0] - m,
                              "bottom": slo[1] - m, "top": p.chip_height - shi[1] - m}

    sample = None
    if centred and p.sample_size > 0:
        sx, sy, h = cav["centre_x"] + dx, cav["start"][1], p.sample_size / 2
        sample = (sx - h, sy - h, sx + h, sy + h)

    lay = {"gaps": gaps, "probes": [(x + dx, y) for x, y in info["probes"]], "sample": sample,
           "lo": lo, "hi": hi, "chip_width": chip_w, "dx": dx, "cavity": cav,
           "sections": sections, "mirrors": info["mirrors"],
           "launchers_x": (info["x_left"] + dx, info["x_right"] + dx)}
    if verbose:
        print_layout(p, lay)
    return lay


def print_layout(p: Params, lay: dict):
    chip_w, dx, cav = lay["chip_width"], lay["dx"], lay["cavity"]
    print(f"Chip {chip_w:.0f} x {p.chip_height:.0f} um; launchers at x = "
          f"{lay['launchers_x'][0]:.1f} / {lay['launchers_x'][1]:.1f}")
    if cav is None:
        print(f"Straight cavity: {p.cavity_len:.1f} um")
    elif cav["type"] == "continuous":
        print(f"Continuous cavity: {cav['length']:.1f} um = 2 x {cav['meander_part']:.1f} um in the "
              f"meander + {cav['flat']:.1f} um flat centred at x = {cav['centre_x'] + dx:.1f}")
    elif cav["type"] == "loops":
        print(f"Loops cavity: {cav['length']:.1f} um; {cav['flat']:g} um flat centred at x = "
              f"{cav['centre_x'] + dx:.1f}; {cav['loops']} loop(s)/half with {cav['leg']:.0f} um legs; "
              f"leads {cav['lead_out']:.0f} (mirror side) / {cav['lead_in']:.0f} (flat side) um; "
              f"footprint {cav['width']:.0f} x {cav['height']:.0f} um")
    elif cav["type"] == "spine":
        amp = "" if cav["amplitude"] is None else f", serpentine amplitude {cav['amplitude']:.1f} um"
        print(f"Spine cavity: {cav['length']:.1f} um; spine {cav['spine_length']:.1f} um at x = "
              f"{cav['spine_x'] + dx:.1f}, y {cav['spine_y'][0]:.0f} to {cav['spine_y'][1]:.0f}; "
              f"{cav['legs']} legs/half{amp}; lead {cav['lead']:.0f} um; footprint {cav['width']:.1f} x "
              f"{cav['height']:.1f} um; spine-to-riser {cav['spine_pitch']:.0f} um")
    else:
        print(f"Meander cavity: {cav['length']:.1f} um, {cav['legs']} legs/half, r {cav['radius']:g}, "
              f"pitch {cav['pitch']:g}, amplitude {cav['amplitude']:.1f} um, footprint "
              f"{cav['width']:.1f} x {cav['height']:.1f} um")
    if lay["sample"] is not None:
        x0, y0, x1, y1 = lay["sample"]
        print(f"Sample {p.sample_size:g} x {p.sample_size:g} um at ({(x0 + x1) / 2:.1f}, "
              f"{(y0 + y1) / 2:.1f}), {p.sample_clearance:g} um clearance"
              f"{'' if p.draw_sample else ' (not drawn)'}")
    for side, m in lay["mirrors"].items():
        head = (f"{p.mirror_style} {side} mirror: {m['sections']} sections, "
                f"footprint {m['width']:.0f} x {m['height']:.0f} um")
        if p.mirror_style == "cell_meander":
            print(f"{head}, {m['legs']} legs of {m['leg_height']:.0f} um, 50-ohm pad {m['pad']:.0f} um, "
                  f"nearest step to a bend {m['min_step_to_bend']:.1f} um")
        elif p.mirror_style == "loops":
            print(f"{head}, legs {m['leg_first']:.0f}/{m['leg_mid']:.0f}/{m['leg_last']:.0f} um, "
                  f"ground between loops {m['ground_between_legs']:.0f} um")
        else:
            print(f"{head}, {m['legs']} legs, r {m['radius']:g}, pitch {m['pitch']:g}, 50-ohm pad "
                  f"{m['pad']:.0f} um, extra lead {m['extra_lead']:.0f} um, nearest step to a bend "
                  f"{m['min_step_to_bend']:.1f} um")


def extract_trace(p: Params, chip_w, all_gaps, probes):
    """The island of (chip NOT gaps) containing the launcher probe points."""
    chip = gdspy.Rectangle((0, 0), (chip_w, p.chip_height))
    solid = gdspy.boolean(chip, all_gaps, "not", precision=1e-4, max_points=0)
    keep = [] if solid is None else [q for q in solid.polygons
                                     if any(gdspy.inside([pt], [q])[0] for pt in probes)]
    if not keep:
        raise RuntimeError("Could not isolate the trace from the gap geometry.")
    return gdspy.PolygonSet(keep, layer=p.trace_layer)


def keepout_zones(p: Params, lay: dict):
    """[(name, zone rectangle grown by clearance, allowed strip)] for the sample and the flat."""
    cav, zones = lay["cavity"], []
    if cav is None:
        return zones
    y = cav["start"][1]
    if lay["sample"] is not None:
        x0, y0, x1, y1 = lay["sample"]
        sx, half = (x0 + x1) / 2, p.cav_xw / 2 + 0.5
        if cav["type"] == "spine":
            allowed = gdspy.Rectangle((sx - half, y0 - p.sample_size), (sx + half, y1 + p.sample_size))
        else:
            allowed = gdspy.Rectangle((x0 - p.sample_size, y - half), (x1 + p.sample_size, y + half))
        zones.append(("sample", gdspy.offset(gdspy.Rectangle((x0, y0), (x1, y1)), p.sample_clearance - 1e-3,
                                             join="round", tolerance=0.5), allowed))
    if cav["type"] == "loops":
        cx, f, half = cav["centre_x"] + lay["dx"], p.cavity_flat / 2, p.cav_xw / 2 + 0.5
        flat = gdspy.Rectangle((cx - f, y - half), (cx + f, y + half))
        zone = gdspy.offset(flat, p.flat_clearance - 1e-3, join="round", tolerance=0.5)
        allowed = gdspy.Rectangle((cx - f - p.flat_clearance - 1, y - half),
                                  (cx + f + p.flat_clearance + 1, y + half))
        zones.append(("flat", zone, allowed))
    if cav["type"] == "continuous":
        cx, f, half = cav["centre_x"] + lay["dx"], cav["flat"] / 2, p.cav_xw / 2
        zone = gdspy.Rectangle((cx - f, y - half - p.flat_clearance), (cx + f, y + half + p.flat_clearance))
        allowed = gdspy.Rectangle((cx - f - 1, y - half - 0.5), (cx + f + 1, y + half + 0.5))
        zones.append(("flat", zone, allowed))
    return zones


def keepouts_clear(p: Params, lay: dict, gaps) -> dict:
    """{zone name: True if no slot outside its allowed strip enters it}."""
    out = {}
    for name, zone, allowed in keepout_zones(p, lay):
        others = gdspy.boolean(gaps, allowed, "not", max_points=0)
        hit = gdspy.boolean(others, zone, "and", max_points=0)
        out[name] = hit is None or hit.area() < 1e-6
    return out


def check_integrity(p: Params, verbose=True) -> dict:
    """Continuous trace, no floating ground, no overlapping sections, keep-outs clear."""
    lay = layout(p, verbose=False)
    gaps = gdspy.boolean(lay["gaps"], None, "or", max_points=0)
    solid = gdspy.boolean(gdspy.Rectangle((0, 0), (lay["chip_width"], p.chip_height)), gaps,
                          "not", max_points=0)
    islands = solid.polygons if solid is not None else []
    areas = [gdspy.PolygonSet([q]).area() for q in islands]
    hits = [{i for i, q in enumerate(islands) if gdspy.inside([pt], [q])[0]} for pt in lay["probes"]]
    ground_id = int(np.argmax(areas)) if areas else None
    floating = [(np.round(islands[i].min(0)).tolist(), np.round(islands[i].max(0)).tolist())
                for i in range(len(islands)) if i not in hits[0] | hits[1] and i != ground_id]

    built, _, info = build(p)
    merged = {n: gdspy.boolean(built[a:b], None, "or", max_points=0)
              for n, (a, b) in info["sections"].items() if b > a}
    names = list(merged)
    overlaps = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            o = gdspy.boolean(merged[names[i]], merged[names[j]], "and", max_points=0)
            if o is not None and o.area() > 1.0:
                overlaps.append((names[i], names[j], round(o.area(), 1)))

    keepouts = keepouts_clear(p, lay, gaps)
    out = {"trace_continuous": len(hits[0] & hits[1]) == 1, "floating_ground": floating,
           "section_overlaps": overlaps, "keepouts": keepouts,
           "sample_clearance": keepouts.get("sample")}
    out["ok"] = out["trace_continuous"] and not floating and not overlaps and all(keepouts.values())
    if verbose:
        keepout = "".join(f"; {k} keep-out {'clear' if v else 'VIOLATED'}" for k, v in keepouts.items())
        print(f"Integrity: trace continuous {out['trace_continuous']}; floating ground "
              f"{len(floating)}{' ' + str(floating) if floating else ''}; section overlaps "
              f"{overlaps or 'none'}{keepout} -> {'OK' if out['ok'] else 'PROBLEM'}")
    return out


def design_only(p: Params, verbose=True) -> dict:
    """Fit check (no booleans, no file)."""
    out = {"left_periods": p.left_periods, "right_periods": p.right_periods,
           "frequency": p.frequency, "qw_um": p.qw, "cavity_um": p.cavity_len}
    try:
        if p.cavity_style == "meander":
            g = p.meander_geometry
            out.update(cavity_legs=g["legs"], cavity_amp_um=g["amplitude"],
                       cavity_w_um=g["width"], cavity_h_um=g["height"])
        lay = layout(p, verbose=verbose)
    except (ValueError, RuntimeError) as e:
        return {"ok": False, "error": str(e), **out}

    lo, hi, m = lay["lo"], lay["hi"], p.edge_margin
    width_slack = float(min(lo[0], lay["chip_width"] - hi[0])) - m
    height_slack = float(min(lo[1], p.chip_height - hi[1])) - m
    ok = width_slack >= 0 and height_slack >= 0
    periods_l, periods_r = p.periods
    clear = {n: float(min(s["left"], s["right"], s["bottom"], s["top"])) for n, s in lay["sections"].items()}

    out.update(ok=ok, left_sections=periods_l, right_sections=periods_r,
               chip_width_um=lay["chip_width"], width_um=float(hi[0] - lo[0]),
               height_um=float(hi[1] - lo[1]), width_slack_um=width_slack, height_slack_um=height_slack)
    for side, mi in lay["mirrors"].items():
        out.update({f"{side}_mirror_legs": mi["legs"], f"{side}_mirror_h_um": mi["height"],
                    f"{side}_mirror_w_um": mi["width"], f"{side}_steps_in_bends": mi["steps_in_bends"],
                    f"{side}_step_clear_um": mi["min_step_to_bend"]})
    for name, c in clear.items():
        out[f"{name.replace(' + ', '_').replace(' ', '_')}_min_clear_um"] = c
    if not ok:
        culprits = [n for n, c in clear.items() if c < 0]
        out["culprits"] = ", ".join(culprits)
        out["error"] = (f"does not fit the chip [{', '.join(culprits) or 'unknown'}] (slack x "
                        f"{width_slack:.1f}, y {height_slack:.1f} um, edge margin {m:g} um)")

    if verbose:
        rp = p.left_periods if p.right_periods is None else p.right_periods
        if (periods_l, periods_r) != (p.left_periods, rp):
            print(f"NOTE: {p.mirror_style} draws whole 2-period cells: {p.left_periods}/{rp} periods "
                  f"drawn as {periods_l}/{periods_r}.")
        print(f"Quarter wave {p.qw:.1f} um; mirrors {periods_l} / {periods_r} periods")
        print(f"Assembly {hi[0] - lo[0]:.1f} x {hi[1] - lo[1]:.1f} um; slack x {width_slack:.1f}, "
              f"y {height_slack:.1f} um ({'fits' if ok else 'DOES NOT FIT'})")
        print(f"  {'clearance beyond margin (um)':<30}{'left':>8}{'right':>8}{'bottom':>8}{'top':>8}")
        for name, s in lay["sections"].items():
            flag = "  <--" if clear[name] < 0 else ""
            print(f"  {name:<30}{s['left']:8.1f}{s['right']:8.1f}{s['bottom']:8.1f}{s['top']:8.1f}{flag}")
    return out


def generate_gds(p: Params, filename: str, write_gaps=False, view=False, verbose=True) -> dict:
    """Write `<filename>.gds`: ground + trace, or (write_gaps) trace + gap slots."""
    lay = layout(p, verbose=False)
    chip_w = lay["chip_width"]
    all_gaps = gdspy.boolean(lay["gaps"], None, "or", layer=p.gap_layer, precision=1e-4, max_points=0)
    trace = extract_trace(p, chip_w, all_gaps, lay["probes"])
    lo, hi = bbox_of([trace, all_gaps])
    off_chip = min(lo[0], lo[1], chip_w - hi[0], p.chip_height - hi[1]) < 0

    def gds_safe(ps):                       # GDSII vertex limit
        return ps.fracture(max_points=8190, precision=1e-4)

    gdspy.current_library = gdspy.GdsLibrary()
    cell = gdspy.Cell("merge")
    if write_gaps:
        cell.add(gds_safe(trace))
        cell.add(gds_safe(all_gaps))
    else:
        cutouts = gdspy.boolean([all_gaps, trace], None, "or", precision=1e-4, max_points=0)
        ground = gdspy.boolean(gdspy.Rectangle((0, 0), (chip_w, p.chip_height)), cutouts, "not",
                               layer=p.layer, precision=1e-4, max_points=0)
        cell.add(gds_safe(ground))
        cell.add(gds_safe(trace))
    if lay["sample"] is not None and p.draw_sample:
        x0, y0, x1, y1 = lay["sample"]
        cell.add(gdspy.Rectangle((x0, y0), (x1, y1), layer=p.sample_layer))

    out_name = filename + ".gds"
    gdspy.write_gds(out_name, unit=1.0e-6, precision=1.0e-9)
    if verbose:
        if off_chip:
            print("ERROR: geometry runs off the chip.")
        layers = f"gaps {p.gap_layer}" if write_gaps else f"ground {p.layer}"
        sample = f", sample {p.sample_layer}" if lay["sample"] is not None and p.draw_sample else ""
        print(f"Wrote {out_name} (layers: trace {p.trace_layer}, {layers}{sample})")
    if view:
        gdspy.LayoutViewer(cells=cell)
    return {"ok": not off_chip, "filename": out_name, "chip_width_um": chip_w,
            "width_um": float(hi[0] - lo[0]), "height_um": float(hi[1] - lo[1]),
            "bbox_lo": tuple(lo), "bbox_hi": tuple(hi), "off_chip": off_chip}


# --------------------------------------------------------------------------- sweeps
def sweep(base: Params, grid: dict, verbose=False) -> pd.DataFrame:
    """Fit check over every combination of `grid` (field -> values)."""
    keys = list(grid)
    rows = []
    for combo in product(*(grid[k] for k in keys)):
        row = design_only(base.with_(**dict(zip(keys, combo))), verbose=verbose)
        rows.append({**dict(zip(keys, combo)), **row})
    return pd.DataFrame(rows)


def sweep_and_generate(base: Params, grid: dict, prefix="bragg", only_ok=True,
                       write_gaps=False, verbose=False) -> pd.DataFrame:
    """sweep(), writing one GDS per combination that fits (or every one if only_ok=False)."""
    keys = list(grid)
    rows = []
    for combo in product(*(grid[k] for k in keys)):
        overrides = dict(zip(keys, combo))
        p = base.with_(**overrides)
        row = {**overrides, **design_only(p, verbose=verbose), "filename": None}
        if row["ok"] or not only_ok:
            name = "_".join([prefix] + [f"{k}-{v:g}" if isinstance(v, float) else f"{k}-{v}"
                                        for k, v in overrides.items()])
            try:
                gen = generate_gds(p, name, write_gaps=write_gaps, verbose=verbose)
                row.update(filename=gen["filename"], off_chip=gen["off_chip"])
            except Exception as e:
                row.update(ok=False, error=str(e))
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"sweep_and_generate: {df['filename'].notna().sum()}/{len(df)} written.")
    return df


def fit(base: Params, verbose=True, **grid):
    """Sweep fields (field=[values]); return (best Params or None, table).

    Best = fits and has the largest smaller-of-x/y slack. Example: fit(p, left_periods=[6, 7, 8])
    """
    grid = {k: list(v) for k, v in grid.items()}
    combos = list(product(*grid.values()))
    df = sweep(base, grid)
    ok = df[df["ok"]]
    if verbose:
        show = list(grid) + [c for c in ("width_slack_um", "height_slack_um", "ok", "culprits") if c in df]
        print(df[show].to_string(index=False, float_format=lambda v: f"{v:.1f}", na_rep="-"))
    if ok.empty:
        if verbose:
            print("Nothing fits:")
            for e in df["error"].dropna().unique():
                print("  ", e)
        return None, df
    i = ok[["width_slack_um", "height_slack_um"]].min(axis=1).idxmax()
    choice = dict(zip(grid, combos[i]))
    if verbose:
        print(f"Best: {choice} (slack x {df.loc[i, 'width_slack_um']:.1f}, "
              f"y {df.loc[i, 'height_slack_um']:.1f} um)")
    return base.with_(**choice), df
