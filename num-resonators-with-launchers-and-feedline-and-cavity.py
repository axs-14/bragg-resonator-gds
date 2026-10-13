# -*- coding: utf-8 -*-
"""
run_bragg.py -- single-run driver for bragg_filter.py.
"""

from bragg_filter import BraggParams, design_only, generate_gds


# ======================================================================= #
#                                 CONFIG                                  #
# ======================================================================= #
OUTPUT_NAME = "AS_2um_6687.5_bragg_feedline_launcher_cavity_12_meme"   # .gds is appended automatically
VIEW = True                    # pop the gdspy LayoutViewer when done
WRITE_GAPS = False             # False -> ground + trace (negative mask)
                               # True  -> trace + gap slots, no ground fill
DRY_RUN = False                # True -> fit check only, never write a file

p = BraggParams(
    # --- what the filter is ---
    no_of_quarterwave_resonators= 6,   
    high_z_ends=True,                 # high-Z sections terminate both ends
    frequency=7.0e9,
    epsilon=11.9,

    # --- Bragg cross-sections ---
    gap1=23, width1=4,                # high-Z section
    gap2=4,  width2=42,               # low-Z section

    # --- feedline ---
    include_feedline=True,
    feedline_length=50.0,
    feedline_width=22.0,
    feedline_gap=14.0,
    feedline_ends="both",             # "both" | "left" | "right"

    # --- ends ---
    include_launchers=True,           # False -> bare ends + end caps
    # end_cap_depth=None,             # only used when launchers are off

    # --- cavity (off = one plain Bragg mirror) ---
    include_defect=True,
    #n_right_sections= 4,          # defect builds only
    defect_gap= 24, defect_width= 2, 
    defect_length=6687.5,

    # --- packing / chip ---
    chip_width=10000, chip_height=5000,
    bend_radius=200.0,
    jog_margin=20.0,
    lead_run=500.0,
    edge_margin=300.0,
)


# ======================================================================= #
#                                  RUN                                    #
# ======================================================================= #
def main():
    print("=" * 72)
    print("STEP 1 -- fit check (nothing written)")
    print("=" * 72)
    fit = design_only(p, verbose=True)

    if not fit["ok"]:
        print("\nDesign does not fit:")
        print("  " + fit.get("error", "unknown reason"))
        print("\nNothing written. Try fewer sections, a smaller bend_radius, "
              "a shorter feedline, or a bigger chip.")
        return fit

    print("\nSummary")
    print(f"  sections        : {fit['n_sections']} "
          f"({fit['n_high_z']} high-Z, {fit['n_low_z']} low-Z)")
    print(f"  quarter-wave    : {fit['quarterwave_length_um']:.1f} um")
    print(f"  resonator length: {fit['resonator_length_um']:.1f} um "
          f"(+ {fit['feed_len_total_um']:.1f} um feedline)")
    print(f"  serpentine      : {fit['n_turns']} turns, "
          f"leg height {fit['leg_height_um']:.1f} um")
    print(f"  footprint       : {fit['w_total_um']:.1f} x "
          f"{fit['vertical_span_um']:.1f} um")
    print(f"  slack           : {fit['width_slack_um']:.1f} um wide, "
          f"{fit['height_slack_um']:.1f} um tall")
    print(f"  ground between legs: {fit['ground_gap_um']:.1f} um"
          + ("   <-- check coupling" if fit["coupling_warning"] else ""))
    if fit["bend_step_warnings"]:
        print(f"  {fit['bend_step_warnings']} impedance step(s) land inside a "
              f"bend -- nudge feedline_length / jog_margin if that matters")

    if DRY_RUN:
        print("\nDRY_RUN is True -- stopping before the GDS write.")
        return fit

    print()
    print("=" * 72)
    print("STEP 2 -- building geometry and writing GDS")
    print("=" * 72)
    gen = generate_gds(p, filename=OUTPUT_NAME, write_gaps=WRITE_GAPS,
                       view=VIEW, verbose=True)

    if gen["off_chip"]:
        print("\nWARNING: geometry runs off the chip -- file written anyway, "
              "but check the bbox above.")
    return gen


if __name__ == "__main__":
    main()