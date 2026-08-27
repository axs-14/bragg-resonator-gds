# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:45:40 2024
Modified: 5000 x 10000 um chip. Left launcher removed and replaced with an
IDC quarter-wave resonator (per Fig. 6 of the reference), galvanically
connected into the stepped-impedance (meander Bragg) filter. Whole
assembly centered on the chip.

@author: Aryan Soni
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
chip_width  = 10000    # microns (target footprint)

# --- IDC quarter-wave resonator (Fig. 6) ---
# l0 is the TOTAL length of this element (not a repeat period) -- per
# "the resonator has a length of l0 = 240 um" in the source text.
idc_l0 = 240.0   # total resonator length, um
idc_w  = 2.0     # inner trunk track width, um
idc_s  = 2.0     # finger width, um (ASSUMED also = lateral finger-to-finger gap)
idc_h  = 46.0    # finger length, um
idc_g  = 2.0     # finger-tip-to-opposite-conductor clearance, um
idc_Z0_reference = 118.0  # ohms, quoted resonator impedance (Sonnet-simulated, not recomputed here)

# --- Base (packing_scale=1) packing geometry for the meander/Bragg section ---
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
print(f"Meander quarter-wave length (frequency-locked) = {quarterwave_resonator_length:.1f} um")
print(f"IDC resonator length l0 = {idc_l0:.1f} um (Z0 ~= {idc_Z0_reference:.0f} ohm, per reference)")


#---------------------------------------------#
#   IDC quarter-wave resonator (Fig. 6 style)  #
#---------------------------------------------#
def _rect(x_lo, y_lo, x_hi, y_hi, layer):
    return gdspy.Rectangle((x_lo, y_lo), (x_hi, y_hi), layer=layer)


def build_IDC(x0, y_center, l0, w, s, h, g, layer):
    """
    Interdigitated-capacitor quarter-wave section.

    Trunk (width w) runs continuously along the centerline -- galvanic
    connection preserved throughout, as in the reference. Above and below
    the trunk, alternating fingers reach in from the trunk and from the
    outer ground plane, each of width s, each stopping short of the
    opposite conductor by a gap g. Left end (x0) is left unslotted, so the
    trunk grows directly out of the surrounding solid ground plane there
    (a galvanic short to ground, mirroring the reference sketch's
    grounded end). Right end hands off to the stepped-impedance filter
    with no taper (intentional impedance step).

    Returns (list_of_gap_polygons, (end_x, end_y)).
    """
    polys = []
    seg = s
    n_segments = int(l0 // seg)

    for j in range(n_segments):
        x_lo = x0 + j * seg
        x_hi = x_lo + seg
        if j % 2 == 1:
            # lateral gap between adjacent fingers: full comb-band height
            polys.append(_rect(x_lo, y_center + w / 2, x_hi, y_center + w / 2 + h + g, layer))
            polys.append(_rect(x_lo, y_center - w / 2 - h - g, x_hi, y_center - w / 2, layer))
        else:
            finger_pair_index = j // 2
            if finger_pair_index % 2 == 0:
                # trunk finger: reaches out from the trunk, tip gap g from ground
                polys.append(_rect(x_lo, y_center + w / 2 + h, x_hi, y_center + w / 2 + h + g, layer))
                polys.append(_rect(x_lo, y_center - w / 2 - h - g, x_hi, y_center - w / 2 - h, layer))
            else:
                # ground finger: reaches in from ground, tip gap g from trunk
                polys.append(_rect(x_lo, y_center + w / 2, x_hi, y_center + w / 2 + g, layer))
                polys.append(_rect(x_lo, y_center - w / 2 - g, x_hi, y_center - w / 2, layer))

    n_pairs = n_segments // 4
    print(f"IDC: {n_pairs} interdigitated finger pairs packed into {n_segments * seg:.1f} um "
          f"(target l0={l0:.1f} um)")
    return polys, (x0 + n_segments * seg, y_center)


#---------------------------------------------#
#  Meander/Bragg resonator (no left launcher)  #
#---------------------------------------------#
def build_layout(packing_scale, start_x):
    """
    Builds the meander-Bragg resonator starting at x = start_x, with all
    packing dimensions (radius, CPW cross-sections, launcher geometry)
    multiplied by packing_scale. quarterwave_resonator_length is never
    scaled, so resonant frequency stays fixed regardless of packing_scale.

    Returns (list_of_polygons, chip_width_used) where chip_width_used is
    the absolute rightmost x-coordinate reached (i.e. measured from x=0,
    not from start_x).
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
#  Auto-fit packing scale, then center the     #
#  whole IDC + resonator assembly on the chip  #
#---------------------------------------------#
available_width = chip_width - idc_l0   # room left for the meander/launcher section

scale_lo, scale_hi = 0.05, 1.0
best_scale = None
for _ in range(40):
    mid = (scale_lo + scale_hi) / 2
    try:
        _, w_used = build_layout(mid, start_x=0.0)  # reference: intrinsic width from x=0
    except ValueError:
        scale_hi = mid
        continue
    if w_used <= available_width:
        best_scale = mid
        scale_lo = mid
    else:
        scale_hi = mid

if best_scale is None:
    raise RuntimeError(
        "No packing scale in [0.05, 1.0] fits the target chip_width. "
        "Increase chip_width or reduce no_of_quarterwave_resonators."
    )

packing_scale = best_scale
resonator_width, _ = None, None
_, resonator_width = build_layout(packing_scale, start_x=0.0)

total_assembly_width = idc_l0 + resonator_width
x_idc_start = (chip_width - total_assembly_width) / 2.0
feedline_start_x = x_idc_start + idc_l0

idc_polys, idc_end = build_IDC(x_idc_start, chip_height / 2, idc_l0, idc_w, idc_s, idc_h, idc_g, layer)
resonator_polys, used_width = build_layout(packing_scale, feedline_start_x)

total_structure = idc_polys + resonator_polys

print(f"Chosen packing scale: {packing_scale:.4f} "
      f"(radius -> {radius_0 * packing_scale:.1f} um, "
      f"width -> {width_0 * packing_scale:.1f} um, gap -> {gap_0 * packing_scale:.1f} um)")
print(f"Assembly spans x = {x_idc_start:.1f} to {used_width:.1f} um "
      f"(total {total_assembly_width:.1f} um), centered on chip_width={chip_width} um "
      f"(margins {x_idc_start:.1f} um each side).")

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
