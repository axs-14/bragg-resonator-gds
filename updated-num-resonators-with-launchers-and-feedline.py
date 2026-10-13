# -*- coding: utf-8 -*-
"""run_bragg.py -- build one Bragg resonator chip with bragg_resonator.py."""

from bragg_resonator import Params, check_integrity, design_only, fit, generate_gds


# ---------------------------------------------------------------- run options
OUTPUT_NAME = "bragg_mirror_7GHz_loops_8-4_2um_cavity"
VIEW = True
WRITE_GAPS = False      # True: write trace + gap slots instead of ground + trace
DRY_RUN = False         # True: stop after the fit check
SWEEP = None            # e.g. {"cavity_legs": [0, 1]}


# ---------------------------------------------------------------- design
CHIP = dict(
    chip_width=10000,
    chip_height=5000,
    edge_margin=500,
)

ELECTRICAL = dict(
    frequency=7e9,
    epsilon=11.9,
)

MIRRORS = dict(
    mirror_style="loops",       # "loops" | "cells" | "serpentine"
    left_periods=8,
    right_periods=4,
    gap1=23, width1=4,
    gap2=4, width2=42,
    high_z_first=True,
    radius=70,
    length1=337,
    addlength=150,
)

FEEDLINE = dict(
    feed_gap=14,
    feed_width=22,
    feed_left=20,
    feed_right=150,
)

CAVITY = dict(
    cavity_gap=24,
    cavity_width=2,
    cavity_length=6687.5,
    cavity_meander=True,
    cavity_flat=500,
    cavity_legs=0,
    cavity_lead=50,
    cavity_first_up=False,      # must be False with mirror_style="loops"
    cavity_radius=70,
    cavity_pitch=140,
)

PARAMS = Params(**CHIP, **ELECTRICAL, **MIRRORS, **FEEDLINE, **CAVITY)


# ---------------------------------------------------------------- steps
def choose_params():
    if not SWEEP:
        return PARAMS
    print("== sweep ==")
    best, _ = fit(PARAMS, **SWEEP)
    return best


def main():
    p = choose_params()
    if p is None:
        return None

    print("== fit check ==")
    fit_result = design_only(p)
    if not fit_result["ok"]:
        print(f"Does not fit: {fit_result['error']}")
        return fit_result
    if DRY_RUN:
        return fit_result

    print("\n== integrity check ==")
    integrity = check_integrity(p)
    if not integrity["ok"]:
        print("Not writing the GDS.")
        return integrity

    print("\n== writing GDS ==")
    result = generate_gds(p, OUTPUT_NAME, write_gaps=WRITE_GAPS, view=VIEW)
    if result["off_chip"]:
        print("WARNING: geometry runs off the chip.")
    return result


if __name__ == "__main__":
    main()