# -*- coding: utf-8 -*-
"""
Sweep 1: number of quarter-wave resonators.

Requires bragg_filter.py in the same directory (or on PYTHONPATH).
"""

from bragg_filter import BraggParams, design_only, generate_gds
import pandas as pd

base = BraggParams(frequency=7e9, include_launchers=True)

grid = {
    "no_of_quarterwave_resonators": [3,5,7,9,11,13],
}

def _with_matched_right_launcher(p: BraggParams) -> BraggParams:
    last_is_high = (p.no_of_quarterwave_resonators - 1) % 2 == 0
    if last_is_high:
        return p.with_(launcher_r_gap=p.gap1, launcher_r_width=p.width1)
    return p.with_(launcher_r_gap=p.gap2, launcher_r_width=p.width2)


configs = {n: _with_matched_right_launcher(base.with_(no_of_quarterwave_resonators=n))
           for n in grid["no_of_quarterwave_resonators"]}

for n, p in configs.items():
    print(f"n={n}: right launcher set to {p.launcher_r_width}/{p.launcher_r_gap} "
          f"({'high' if (n - 1) % 2 == 0 else 'low'}-Z, to match the last section)")

# 1) quick fit check first
preview = pd.DataFrame([design_only(p, verbose=False) for p in configs.values()])
print("\n" + preview[["no_of_quarterwave_resonators", "ok", "n_turns",
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
    name = f"AS_bragg_no_of_res-{n}"
    gen = generate_gds(p, filename=name, verbose=False)
    fit["filename"] = gen["filename"]
    fit["off_chip"] = gen["off_chip"]
    rows.append(fit)

results = pd.DataFrame(rows)
print("\n" + results[["no_of_quarterwave_resonators", "ok", "filename"]]
      .to_string(index=False))