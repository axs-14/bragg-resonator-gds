# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:45:40 2024
Modified: 5000 x 10000 um chip, left launcher removed, space reserved on the
left for a future IDC, all packing geometry auto-scaled to fit.

@author: Anusree V P
"""

import numpy as np
import gdspy
import gds_maker as GM  # Custom module

#---------------------------------------------#
#              Design Parameters              #
#---------------------------------------------#
filename = '10_meander_bragg_resonator_IDC_ready'
write_gaps = False  # True to write just shapes, False to subtract from background

# --- Target chip envelope ---
chip_height = 5000     # microns (unchanged)
chip_width  = 10000    # microns (new target footprint)

# --- Space reserved on the left for a future IDC ---
# The feedline starts here; everything to the left stays solid ground plane
# (or open, if write_gaps=True) for you to pattern the IDC into later.
idc_space = 1500       # microns -- replace with your actual IDC footprint

# --- Base (scale=1) packing geometry: bend radius, CPW cross-sections, ---
# --- launcher dimensions, feedline length. These are what get shrunk.  ---
radius_0            = 400
gap_0               = 30
width_0             = 50
gap1_0, width1_0     = 5, 100     # high-impedance Bragg section
gap2_0, width2_0     = 45, 20     # low-impedance Bragg section
feedline_length_0    = 500
straightLauncher_0   = 1000
taperLauncher_0      = 1000
launcherHeight_0     = 1000
gapLauncher_0        = 275

# --- Physics that is NOT allowed to scale: it sets the operating frequency ---
frequency = 6e9              # Hz
epsilon = 11.5               # substrate relative permittivity
c = 3e8                      # m/s
no_of_quarterwave_resonators = 6
layer = 1

quarterwave_resonator_length = 1e6 * c / (np.sqrt((epsilon + 1) / 2) * frequency * 4)
print(f"Quarterwave resonator length (frequency-locked) = {quarterwave_resonator_length:.1f} um")


#---------------------------------------------#
#   Build the layout at a given packing scale  #
#---------------------------------------------#
def build_layout(packing_scale, start_x):
    """
    Builds the meander-Bragg resonator (no left launcher) starting at
    x = start_x, with all packing dimensions (radius, CPW cross-sections,
    launcher geometry) multiplied by packing_scale. The electrical length
    of the resonator (quarterwave_resonator_length) is NEVER scaled, so the
    resonant frequency stays fixed regardless of packing_scale.

    Returns (list_of_polygons, chip_width_used_by_this_layout).
    """
    radius = radius_0 * packing_scale
    gap = gap_0 * packing_scale
    width = width_0 * packing_scale
    gap1, width1 = gap1_0 * packing_scale, width1_0 * packing_scale
    gap2, width2 = gap2_0 * packing_scale, width2_0 * packing_scale
    feedline_length = feedline_length_0 * packing_scale
    straightLauncher = straightLauncher_0 * packing_scale
    taperLauncher = taperLauncher_0 * packing_scale
    launcherHeight = launcherHeight_0 * packing_scale
    gapLauncher = gapLauncher_0 * packing_scale

    length_circle = np.pi * radius * 2
    length_straight = (quarterwave_resonator_length - length_circle) / 4
    if length_straight <= 0:
        raise ValueError(
            f"packing_scale={packing_scale:.4f}: bend radius alone exceeds "
            f"the quarter-wavelength -- shrink radius further."
        )

    if 7 * (2 * gap + width) > 2 * radius:
        print(f"Warning (packing_scale={packing_scale:.4f}): CTL spacing may "
              f"be too tight relative to bend radius.")

    structure = []
    start = [start_x, chip_height / 2]

    # No left launcher -- feed straight in from the IDC side.
    [poly2, pos2] = GM.createCPWhor(start, length=feedline_length, gap=gap, width=width)
    structure += poly2

    [poly3, pos3] = GM.createDoublearc(
        center=(pos2[0], pos2[1] + radius), radius=radius,
        gap=gap, width=width, angle=np.pi * 3 / 2, layer=layer
    )
    pos3 = pos3[0] + radius, pos3[1] + radius
    structure += poly3

    [poly4, pos4] = GM.CreateAlternateMeanderUp(
        pos3, length=quarterwave_resonator_length,
        no_meander=no_of_quarterwave_resonators,
        radius=radius, gap1=gap1, width1=width1,
        gap2=gap2, width2=width2, layer=layer
    )
    structure += poly4

    [poly5, pos5] = GM.createCPWvertUp(pos4, length=length_straight, gap=gap, width=width, layer=layer)

    [poly6, pos6] = GM.createSemiarc(
        (pos5[0] + radius, pos5[1]), radius=radius,
        gap=gap, width=width, angle=np.pi * 2, layer=layer
    )
    structure += poly5 + poly6

    [poly7, pos7] = GM.createCPWvertDown(pos6, length=length_straight, gap=gap, width=width, layer=layer)
    structure += poly7   # NOTE: original script appended poly7 twice -- fixed here

    [poly8, pos_8] = GM.createDoublearc(
        (pos7[0] + radius, pos7[1]), radius=radius,
        gap=gap, width=width, angle=-np.pi, layer=layer
    )
    pos8 = (pos_8[0] + 2 * radius, pos_8[1] - radius)
    structure += poly8

    # Single right-side launcher (mirrored placement, same trick as the
    # original: a dummy call first to get its own width for symmetry).
    [_, pos9] = GM.createLauncherRight(
        pos8, gap=gap, width=width, straight=straightLauncher,
        height=launcherHeight, taper=taperLauncher, gapL=gapLauncher, notch=True
    )
    pos9 = [pos8[0] - pos9[0] + pos8[0] - radius, pos8[1]]

    [poly10, pos10] = GM.createLauncherRight(
        pos9, gap=gap, width=width, straight=straightLauncher,
        height=launcherHeight, taper=taperLauncher, gapL=gapLauncher, notch=True
    )
    structure += poly10

    used_width = pos8[0] + 2 * gapLauncher + taperLauncher + straightLauncher - radius
    return structure, used_width


#---------------------------------------------#
#  Auto-fit: search for the largest packing    #
#  scale that still fits in the available      #
#  width (chip_width minus the IDC margin)     #
#---------------------------------------------#
available_width = chip_width - idc_space
start_x = idc_space

scale_lo, scale_hi = 0.05, 1.0
best_scale = None
for _ in range(40):
    mid = (scale_lo + scale_hi) / 2
    try:
        _, used_width = build_layout(mid, start_x)
    except ValueError:
        scale_hi = mid
        continue
    if used_width <= available_width:
        best_scale = mid
        scale_lo = mid
    else:
        scale_hi = mid

if best_scale is None:
    raise RuntimeError(
        "No packing scale in [0.05, 1.0] fits the target chip_width. "
        "Increase chip_width, reduce idc_space, or reduce "
        "no_of_quarterwave_resonators."
    )

packing_scale = best_scale
total_structure, used_width = build_layout(packing_scale, start_x)

print(f"Chosen packing scale: {packing_scale:.4f} "
      f"(radius -> {radius_0 * packing_scale:.1f} um, "
      f"width -> {width_0 * packing_scale:.1f} um, gap -> {gap_0 * packing_scale:.1f} um)")
print(f"Design footprint used: {used_width:.1f} um of {available_width:.1f} um available, "
      f"plus {idc_space} um reserved for the IDC on the left.")
print("NOTE: this search assumes footprint width decreases monotonically as "
      "packing_scale decreases. Check the viewer output below to confirm.")

#---------------------------------------------#
#         Boolean Merge and Background        #
#---------------------------------------------#
gdspy.current_library.cells.clear()

merged_cell = gdspy.Cell('merge')
merged_path = gdspy.fast_boolean(total_structure, None, 'or', max_points=0)

if write_gaps:
    final_merged = merged_path
else:
    chip_rect = gdspy.Rectangle((0, 0), (chip_width, chip_height))
    subtracted_path = gdspy.fast_boolean(chip_rect, merged_path, 'not', layer=2)
    final_merged = gdspy.fast_boolean(subtracted_path, None, 'or', max_points=0)

merged_cell.add(final_merged)

#---------------------------------------------#
#         Export GDSII and View Layout        #
#---------------------------------------------#
gdspy.write_gds(filename + ".gds", unit=1.0e-6, precision=1.0e-9)
gdspy.LayoutViewer(cells=merged_cell)

print('Finished!')
