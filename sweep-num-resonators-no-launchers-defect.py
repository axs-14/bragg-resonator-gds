# -*- coding: utf-8 -*-
"""
Sweep: symmetric mirror-defect-mirror cavity, quarter-wave sections per
mirror swept at 3, 5, 7, 9, 11, 13 (each mirror is H-L-H-...-L-H, ending
and starting high, flanking a half-wave defect in the middle -- e.g. n=5
gives H L H L H -- DEFECT -- H L H L H). No launchers, no lumped RLC:
this is meant as a fully lossless cavity for an eigenmode check.

FITTING THE LONGER STRUCTURE
------------------------------
A defect cavity is roughly TWO mirrors end to end (plus one defect
section), so for the same n it's about double the total path length of
the old single-mirror sweep. At the original bend_radius=400um /
turns_max=14, only n=3 fits -- everything from n=5 up runs out of turns
before the serpentine gets narrow enough to fit the chip height.

Fix: shrink bend_radius (tighter turns pack more length into the same
mirror width) and raise turns_max so there's room to use them. At
bend_radius=150um, turns_max=40, all six values fit -- n=11 is the
tightest (~62um of height slack), everything else has healthy margin.
If you push further out (more sections per side, or a taller/narrower
mirror), re-run the fit-check below before trusting a value; shrink
bend_radius further and/or raise turns_max again if something stops
fitting.

Requires bragg_filter.py in the same directory (or on PYTHONPATH).
"""

from bragg_filter import BraggParams, design_only, generate_gds
import pandas as pd

base = BraggParams(
    frequency=7e9,
    include_launchers=False,
    include_defect=True,
    bend_radius=150.0,
    turns_max=40,
)

# no_of_quarterwave_resonators = sections per mirror (n_left); n_right
# defaults to the same value, so this sweep is symmetric on both sides
# of the defect. 5 (H L H L H on each side) is a reasonable starting
# point -- good stopband depth without an oversized chip -- but the
# sweep covers a range around it so you can see the width/Q tradeoff.
grid = {
    "no_of_quarterwave_resonators": [3, 5, 7, 9, 11, 13],
}

configs = {n: base.with_(no_of_quarterwave_resonators=n)
           for n in grid["no_of_quarterwave_resonators"]}

# 1) quick fit check first
preview = pd.DataFrame([design_only(p, verbose=False) for p in configs.values()])
print(preview[["no_of_quarterwave_resonators", "ok", "n_turns",
                "w_total_um", "width_slack_um", "height_slack_um"]]
      .to_string(index=False))

# 2) write a .gds for each value that fits
rows = []
for n, p in configs.items():
    fit = design_only(p, verbose=False)
    if not fit["ok"]:
        fit["filename"] = None
        rows.append(fit)
        continue
    name = f"AS_bragg_cavity_n-{n}"
    gen = generate_gds(p, filename=name, verbose=False)
    fit["filename"] = gen["filename"]
    fit["off_chip"] = gen["off_chip"]
    rows.append(fit)

results = pd.DataFrame(rows)
print("\n" + results[["no_of_quarterwave_resonators", "ok", "filename"]]
      .to_string(index=False))