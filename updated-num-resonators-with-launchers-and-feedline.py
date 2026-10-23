# -*- coding: utf-8 -*-
"""Build one Bragg resonator chip with bragg_resonator.py."""

from bragg_resonator import Params, check_integrity, design_only, fit, generate_gds


# --------------------------------------------------------------------------- run options
OUTPUT_NAME = "bragg_7GHz_continuous_5-5_pitch200"
VIEW = True
WRITE_GAPS = False      # True: trace + gap slots instead of ground + trace
DRY_RUN = False         # True: stop after the fit check
SWEEP = None            # e.g. {"cell_pitch": [180, 190, 200]}


# --------------------------------------------------------------------------- design
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
    mirror_style="cell_meander",
    symmetric=True,
    left_periods=5,
    right_periods=5,
    gap1=23, width1=4,
    gap2=4, width2=42,
    high_z_first=True,
    radius=70,
    length1=337,
    addlength=150,
    cell_pitch=200,     # ground between legs = pitch - 50
    sections_per_column=3,
    mirror_lead=50,
)

FEEDLINE = dict(
    feed_gap=14,
    feed_width=22,
    feed_left=20,
)

CAVITY = dict(
    cavity_style="continuous",
    cavity_gap=24,
    cavity_width=2,
    cavity_length=None,     # half wave
    cavity_flat=1000,       # minimum flat
    flat_clearance=100,
)

SAMPLE = dict(
    sample_size=500,
    sample_clearance=50,
    draw_sample=True,       # layer 10
)

PARAMS = Params(**CHIP, **ELECTRICAL, **MIRRORS, **FEEDLINE, **CAVITY, **SAMPLE)


# --------------------------------------------------------------------------- steps
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
    result = design_only(p)
    if not result["ok"]:
        print(f"Does not fit: {result['error']}")
        return result
    if DRY_RUN:
        return result

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
