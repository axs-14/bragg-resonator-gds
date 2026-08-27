# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:45:40 2024

@author: Anusree V P
"""

import numpy as np
import gdspy
import gds_maker as GM  # Custom module

#---------------------------------------------#
#              Design Parameters              #
#---------------------------------------------#
filename = '10_meander_bragg resonator_with  symmetry'
write_gaps = False  # True to write just shapes, False to subtract from background
chip_height = 5000  # in microns

# CPW and resonator properties
feedline_length = 500        # CPW feedline length
radius = 400                 # Bend radius for arcs
gap = 30                     # Gap between center conductor and ground
width = 50                  # Width of the center conductor
frequency = 6e9              # Operating frequency (Hz)
no_of_quarterwave_resonators = 6  # Number of CTLs in the meandered structure

# Alternating CPW parameters for Bragg
gap1 = 5
width1 = 100
gap2 = 45
width2 = 20

# Launcher parameters
gapLauncher = 275
straightLauncher = 1000
taperLauncher = 1000
height = 1000
layer = 1

#---------------------------------------------#
#    Calculate Quarter-Wave Resonator Length  #
#---------------------------------------------#
c = 3e8            # Speed of light in m/s
epsilon = 11.5     # Relative permittivity of substrate
# Effective quarter-wavelength (in microns)
quarterwave_resonator_length = 1e6 * c / (np.sqrt((epsilon + 1) / 2) * frequency * 4)
print("Quarterwave resonator length =", quarterwave_resonator_length)

# Length of the double arc bend (semi-circular back and forth)
length_circle = np.pi * radius * 2
# Remaining length to be split across 4 vertical CPWs
length_straight = (quarterwave_resonator_length - length_circle) / 4

# Starting point on left-middle of chip
start = [gapLauncher, chip_height / 2]

# Warning for minimum spacing between CTLs
if 7 * (2 * gap + width) > 2 * radius:
    print("Warning: Distance between adjacent CTLs is too small. Increase radius to reduce coupling.")

# List to hold all polygons for final layout
total_structure = []

#---------------------------------------------#
#        Create Launchers and Feedline        #
#---------------------------------------------#

# Left launcher
[poly1, pos1] = GM.createLauncherLeft(
    start, gap=gap, width=width, straight=straightLauncher,
    height=height, taper=taperLauncher, gapL=gapLauncher, notch=True
)
total_structure += poly1

# Horizontal feedline
[poly2, pos2] = GM.createCPWhor(pos1, length=feedline_length, gap=gap, width=width)
total_structure += poly2

# Double arc bend to bring the line upward
[poly3, pos3] = GM.createDoublearc(
    center=(pos2[0], pos2[1] + radius), radius=radius,
    gap=gap, width=width, angle=np.pi * 3/2, layer=1
)
# New position after double arc
pos3 = pos3[0] + radius, pos3[1] + radius
total_structure += poly3

#---------------------------------------------#
#  Create Meandered Bragg Resonator Section   #
#---------------------------------------------#

# Meandered Bragg section with alternating CPWs
[poly4, pos4] = GM.CreateAlternateMeanderUp(
    pos3, length=quarterwave_resonator_length,
    no_meander=no_of_quarterwave_resonators,
    radius=radius, gap1=gap1, width1=width1,
    gap2=gap2, width2=width2, layer=1
)
total_structure += poly4

# Vertical CPW upward to complete meander length
[poly5, pos5] = GM.createCPWvertUp(pos4, length=length_straight, gap=gap, width=width, layer=layer)

# Full semicircle bend to return downward
[poly6, pos6] = GM.createSemiarc(
    (pos5[0] + radius, pos5[1]), radius=radius,
    gap=gap, width=width, angle=np.pi * 2, layer=layer
)

total_structure += poly5 + poly6

# Vertical CPW going down after arc
[poly7, pos7] = GM.createCPWvertDown(pos6, length=length_straight, gap=gap, width=width, layer=layer)
total_structure += poly7

# Final double arc to level back to horizontal feedline
[poly8, pos_8] = GM.createDoublearc(
    (pos7[0] + radius, pos7[1]), radius=radius,
    gap=gap, width=width, angle=-np.pi, layer=layer
)
# Adjusted position for next section
pos8 = (pos_8[0] + 2 * radius, pos_8[1] - radius)
total_structure += poly7 + poly8

#---------------------------------------------#
#        Create Right-Side Launchers          #
#---------------------------------------------#

# First dummy right launcher to compute relative positioning
[poly9, pos9] = GM.createLauncherRight(
    pos8, gap=gap, width=width, straight=straightLauncher,
    height=1000, taper=taperLauncher, gapL=gapLauncher, notch=True
)

# Adjust launcher position for symmetry
pos9 = [pos8[0] - pos9[0] + pos8[0] - radius, pos8[1]]

# Add final launcher (mirrored)
[poly10, pos10] = GM.createLauncherRight(
    pos9, gap=gap, width=width, straight=straightLauncher,
    height=1000, taper=taperLauncher, gapL=gapLauncher, notch=True
)
total_structure += poly10

# Estimate chip width from rightmost element
chip_width = pos8[0] + 2 * gapLauncher + taperLauncher + straightLauncher - radius

#---------------------------------------------#
#         Boolean Merge and Background        #
#---------------------------------------------#

# Create layout cell
merged_cell = gdspy.Cell('merge')

# Merge all geometry into a single structure
merged_path = gdspy.fast_boolean(total_structure, None, 'or', max_points=0)

# Either subtract from a metal ground or keep as-is
if write_gaps:
    final_merged = merged_path
else:
    chip_rect = gdspy.Rectangle((0, 0), (chip_width - gapLauncher, chip_height))
    subtracted_path = gdspy.fast_boolean(chip_rect, merged_path, 'not', layer=2)
    final_merged = gdspy.fast_boolean(subtracted_path, None, 'or', max_points=0)

# Add final structure to layout
merged_cell.add(final_merged)

#---------------------------------------------#
#         Export GDSII and View Layout        #
#---------------------------------------------#

# Write to GDSII file
gdspy.write_gds(filename + ".gds", unit=1.0e-6, precision=1.0e-9)

# Show layout in viewer
gdspy.LayoutViewer(cells=merged_cell)

print('Finished!')
