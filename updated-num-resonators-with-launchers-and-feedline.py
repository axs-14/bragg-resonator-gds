# -*- coding: utf-8 -*-
"""run_bragg.py -- single-run driver for bragg_resonator.py."""

from bragg_resonator import Params, design_only, generate_gds

OUTPUT_NAME = "bragg_mirror_7GHz_straight_2um_cavity"
VIEW = True          # open the gdspy viewer when done
WRITE_GAPS = False   # False: ground + trace; True: trace + gap slots only
DRY_RUN = False      # True: fit check only, no file written

p = Params(
    left_periods=12,
    right_periods=6,
    frequency=7e9,
    epsilon=11.9,

    gap1=23, width1=4,
    gap2=4, width2=42,
    high_z_first=True,      # False: gap1/width1 first in both mirrors

    feed_gap=14,            # 50-ohm line
    feed_width=22,
    feed_left=150,
    feed_right=400,

    cavity_gap=24,
    cavity_width=2,
    cavity_length=6687.5,   # total centerline length

    cavity_meander=True,    # False: straight cavity
    cavity_flat=500,       # flat run in the centre

    chip_width=12000,
    chip_height=6000,
    edge_margin=200,
)


def main():
    print("STEP 1 -- fit check (nothing written)")
    fit = design_only(p)
    if not fit["ok"]:
        print(f"\nDesign does not fit: {fit.get('error', 'unknown reason')}")
        return fit

    print(f"\nSections     : {fit['left_sections']} left / {fit['right_sections']} right")
    print(f"Quarter-wave : {fit['qw_um']:.1f} um")
    print(f"Footprint    : {fit['width_um']:.1f} x {fit['height_um']:.1f} um "
          f"(chip {fit['chip_width_um']:.1f} x {p.chip_height:.1f} um)")
    print(f"Slack        : {fit['width_slack_um']:.1f} um wide, {fit['height_slack_um']:.1f} um tall")

    if DRY_RUN:
        print("\nDRY_RUN is True -- stopping before the GDS write.")
        return fit

    print("\nSTEP 2 -- writing GDS")
    gen = generate_gds(p, OUTPUT_NAME, write_gaps=WRITE_GAPS, view=VIEW)
    if gen["off_chip"]:
        print("\nWARNING: geometry runs off the chip -- check the bbox above.")
    return gen


if __name__ == "__main__":
    main()