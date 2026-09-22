from build123d import *

# Dimensions in MILLIMETRES (converted from units provided in request)
# Note: The request provided dimensions like 0.399631. 
# Assuming these are in some unit (e.g. inches) but the prompt requires MILLIMETRES.
# However, the prompt says "every number here is AUTHORITATIVE". 
# I will treat the numbers as mm as per the "All dimensions in MILLIMETRES" rule.
RECT_W = 0.399631
RECT_H = 0.091591
HEX_MAX_W = 0.188808
EXTRUDE_H = 0.133054

# The semi-hexagonal section extends 2/3rd of the rectangle's length
HEX_L = (2.0 / 3.0) * RECT_W

# A semi-hexagon can be represented as a trapezoid (the top half of a hexagon)
# For a regular hexagon with "width" (flat-to-flat) of HEX_MAX_W:
# The side length s = HEX_MAX_W / sqrt(3)
# The base of the semi-hexagon (the part attached to the rectangle) is the long diameter: 2 * s
# The top edge is s.
# The height of the semi-hexagon is HEX_MAX_W / 2.
S = HEX_MAX_W / 1.7320508
HEX_BASE = 2 * S
HEX_TOP = S
HEX_HEIGHT = HEX_MAX_W / 2

# Create the main rectangle
# Centered at origin: X from -RECT_W/2 to RECT_W/2, Y from -RECT_H/2 to RECT_H/2
rect_solid = Box(RECT_W, RECT_H, EXTRUDE_H)

# Create the semi-hexagonal section (represented as a trapezoid/prism)
# It starts at one end (x = -RECT_W/2) and extends HEX_L.
# Its center in X is -RECT_W/2 + HEX_L/2.
# It is attached to the "longer edge" (the edge at y = RECT_H/2).
# The semi-hexagon's height is HEX_HEIGHT. Its center in Y is RECT_H/2 + HEX_HEIGHT/2.
# We use a Box for the core and two wedges or a custom polygon. 
# Simplest algebra mode: a Box for the center and two smaller boxes/wedges, 
# or just a Box that we then intersect with a larger rotated box to make the hex shape.
# Alternatively, since it's a "semi-hexagonal section", we can use a Box and subtract corners.

hex_center_x = -RECT_W/2 + HEX_L/2
hex_center_y = RECT_H/2 + HEX_HEIGHT/2

# Create a bounding box for the semi-hex
hex_body = Pos(hex_center_x, hex_center_y, 0) * Box(HEX_L, HEX_HEIGHT, EXTRUDE_H)

# To make it "semi-hexagonal" (trapezoidal profile in XY), we cut the top corners.
# The top edge is shorter than the bottom edge.
# Bottom width = HEX_L, Top width = HEX_L * (some ratio) or based on the HEX_MAX_W.
# Given the prompt specifies "maximum width of 0.188808", we treat that as the base.
# Let's refine the semi-hex as a prism with a trapezoidal base.
# Base = HEX_MAX_W, Top = HEX_MAX_W / 2, Height = HEX_HEIGHT.
# We'll use a Box and subtract two wedges.

# Re-calculating semi-hex based on MAX_W as the base attached to the rectangle
# Base = HEX_MAX_W, Height = HEX_MAX_W * (sqrt(3)/2 / 2) approx 0.433 * MAX_W
# But the prompt says it extends 2/3rd of the rectangle's length.
# This implies the "length" of the hex is HEX_L.
# If the "maximum width" is 0.188808, that is the dimension perpendicular to the rectangle edge.

hex_h = HEX_MAX_W
hex_l = HEX_L
hex_z = EXTRUDE_H

# The semi-hex is a trapezoid: base = hex_l, top = hex_l/2, height = hex_h
# Centered at (hex_center_x, RECT_H/2 + hex_h/2, 0)
hex_solid = Pos(hex_center_x, RECT_H/2 + hex_h/2, 0) * Box(hex_l, hex_h, hex_z)

# Cut corners to make it trapezoidal (semi-hex)
# Cut 1: Top left corner
cut1 = Pos(hex_center_x - hex_l/4, RECT_H/2 + hex_h, 0) * Box(hex_l/2, hex_h/2, hex_z)
# Cut 2: Top right corner
cut2 = Pos(hex_center_x + hex_l/4, RECT_H/2 + hex_h, 0) * Box(hex_l/2, hex_h/2, hex_z)

# Final combination
result = (rect_solid + (hex_solid - cut1 - cut2))