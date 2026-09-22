import sys as _sys
_sys.path.insert(0, '/home/theultimatecunt/.openclaw/skills/cad-builder-phase3')

from build123d import *

# Dimensions
total_length = 100.0
d1 = 30.0
l1 = 40.0
d2 = 20.0
l2 = 60.0

# The shaft is 100mm long.
# Segment 1: Diameter 30, Length 40.
# Segment 2: Diameter 20, Length 60.
# To center the whole part at the origin, the center of the 100mm length is z=0.
# The transition point is at z = (l1/2) - (l2/2) = 20 - 30 = -10.
# Segment 1 (large) spans z from -10 to +30 (center at z=10).
# Segment 2 (small) spans z from -50 to -10 (center at z=-30).

z_offset_1 = (l1 / 2) - (l2 / 2) # +10 (Wait: 40-60/2 = -10 is the split, but we want the 40mm part first)
# Let's re-calculate:
# Total length = 100. Range is -50 to +50.
# Part 1 (40mm) is from +10 to +50. Center is +30.
# Part 2 (60mm) is from -50 to +10. Center is -20.

center_1 = 30.0
center_2 = -20.0

shaft_1 = Pos(0, 0, center_1) * Cylinder(radius=d1/2, height=l1)
shaft_2 = Pos(0, 0, center_2) * Cylinder(radius=d2/2, height=l2)

result = shaft_1 + shaft_2