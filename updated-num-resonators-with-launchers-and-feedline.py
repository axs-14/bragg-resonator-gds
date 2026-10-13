# -*- coding: utf-8 -*-
"""run_bragg.py -- single-run driver for bragg_resonator.py."""

from bragg_resonator import Params, design_only, generate_gds, fit

OUTPUT_NAME = "bragg_mirror_7GHz_meander_2um_cavity"
VIEW = True          # open the gdspy viewer when done
WRITE_GAPS = False   # False: ground + trace; True: trace + gap slots only
DRY_RUN = False      # True: fit check only, no file written
SWEEP = None         # e.g. dict(mirror_radius=[50, 60, 70], cavity_legs=[0, 1]) -> pick the best fit first

p = Params(
    left_periods=9,        # 12 periods -> 6 unit cells -> 24 quarter-wave sections
    right_periods=5,        #  6 periods -> 3 unit cells -> 12 quarter-wave sections
    frequency=7e9,
    epsilon=11.9,

    gap1=23, width1=4,
    gap2=4, width2=42,
    high_z_first=True,      # False: gap1/width1 first in both mirrors

    # mirrors: "serpentine" fits the chip; "cells" = original gds_maker unit cells
    mirror_style="serpentine",
    mirror_radius=100,       # bend radius
    mirror_pitch=200,       # leg centre-to-centre (>= 2r); ground between legs = pitch - (width + 2*gap)
                            # ~170 um is the most that fits this chip with these settings
    min_ground=None,        # fab rule for the ground strip between legs; None -> one CPW width
    mirror_height_max=None, # None -> chip_height - 2*edge_margin - 2*mirror_clearance
    mirror_clearance=50,

    feed_gap=14,            # 50-ohm line
    feed_width=22,
    feed_left=150,
    feed_right=400,

    cavity_gap=24,
    cavity_width=2,
    cavity_length=6687.5,   # total centreline length, flat included

    cavity_meander=True,    # False: straight cavity
    cavity_flat=600,        # flat run at the electrical centre
    cavity_legs=0,          # full legs per meander half; None -> auto from cavity_amp_max
    cavity_lead=100,
    cavity_radius=75,
    cavity_pitch=166,

    chip_width=10000,
    chip_height=5000,
    edge_margin=500,
)


def main():
    params = p
    if SWEEP:
        print("STEP 0 -- parameter sweep")
        params, _ = fit(p, **SWEEP)
        if params is None:
            return None
        print()

    print("STEP 1 -- fit check (nothing written)")
    res = design_only(params)
    if not res["ok"]:
        print(f"\nDesign does not fit: {res.get('error', 'unknown reason')}")
        return res

    print(f"\nMirrors      : {4 * params.left_cells} left / {4 * params.right_cells} right "
          f"quarter-wave sections ({res['left_sections']} / {res['right_sections']} periods)")
    print(f"Quarter-wave : {res['qw_um']:.1f} um; cavity {res['cavity_um']:.1f} um")
    if params.mirror_style == "serpentine":
        for side in ("left", "right"):
            print(f"{side.capitalize():<13}: {res[f'{side}_mirror_w_um']:.1f} x "
                  f"{res[f'{side}_mirror_h_um']:.1f} um, {res[f'{side}_mirror_legs']} legs, "
                  f"{res[f'{side}_steps_in_bends']} steps in bends")
    print(f"Footprint    : {res['width_um']:.1f} x {res['height_um']:.1f} um "
          f"(chip {res['chip_width_um']:.1f} x {params.chip_height:.1f} um)")
    print(f"Slack        : {res['width_slack_um']:.1f} um wide, {res['height_slack_um']:.1f} um tall "
          f"(beyond the {params.edge_margin:g} um margin)")

    if DRY_RUN:
        print("\nDRY_RUN is True -- stopping before the GDS write.")
        return res

    print("\nSTEP 2 -- writing GDS")
    gen = generate_gds(params, OUTPUT_NAME, write_gaps=WRITE_GAPS, view=VIEW)
    if gen["off_chip"]:
        print("\nWARNING: geometry runs off the chip -- check the bbox above.")
    return gen


if __name__ == "__main__":
    main()