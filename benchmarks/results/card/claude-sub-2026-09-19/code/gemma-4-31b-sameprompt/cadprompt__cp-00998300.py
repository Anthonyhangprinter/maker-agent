from build123d import *

# Dimensions converted from meters to millimetres
# Vertical rectangle: 36mm x 13.5mm, height 900mm
# Horizontal rectangle: 240mm x 13.5mm, thickness 36mm
# Position: 150mm below the top of the vertical part
V_WIDTH = 36.0
V_THICKNESS = 13.5
V_HEIGHT = 900.0
H_LENGTH = 240.0
H_WIDTH = 13.5
H_THICKNESS = 36.0
OFFSET_FROM_TOP = 150.0

# The vertical member is centered at the origin.
# It spans Z from -450 to +450.
vertical_member = Box(V_WIDTH, V_THICKNESS, V_HEIGHT)

# The horizontal member is a "T" arm.
# It has the same width (13.5) and a thickness of 36.
# Its length is 240.
# It is positioned 150mm below the top face (z = 450).
# Top of vertical is 450. 150 below that is z = 300.
# Since the horizontal member is centered on its own Z axis, 
# its center is at 300 - (H_THICKNESS / 2) = 300 - 18 = 282.
# However, the prompt says "Connect this horizontal rectangle... 0.15m below the top".
# We will place the center of the horizontal member such that its top edge is at z = 300.
z_pos = (V_HEIGHT / 2) - OFFSET_FROM_TOP - (H_THICKNESS / 2)

# The horizontal member is rotated to be "horizontal" relative to the vertical member's length.
# Vertical member is Box(36, 13.5, 900) -> Length is along Z.
# Horizontal member is Box(240, 13.5, 36) -> Length is along X.
horizontal_member = Pos(0, 0, z_pos) * Box(H_LENGTH, H_WIDTH, H_THICKNESS)

# Fuse the two parts into one solid
result = vertical_member + horizontal_member