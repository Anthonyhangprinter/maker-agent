from build123d import *

# Dimensions
length = 0.875
width = 0.15
height = 1.5
fillet_radius = 0.25
hole_diameter = 0.3
hole_radius = hole_diameter / 2
padding = -0.639071

# Create the main body
# Box is centred at origin: X spans -length/2 to +length/2, Y spans -width/2 to +width/2, Z spans -height/2 to +height/2
body = Box(length, width, height)

# Round two corners of the same long edge
# The request asks for two corners of the same long edge.
# We target the vertical edges (parallel to Z) located at the ends of one long side (Y = width/2).
try:
    # Filter edges that are parallel to Z axis and located at Y = width/2
    edges_to_fillet = body.edges().filter_by(Axis.Z).filter_by(lambda e: abs(e.center.y - width/2) < 0.01)
    result = fillet(body, edges_to_fillet, radius=fillet_radius)
except:
    result = body

# Create two small, evenly-spaced circular holes
# Position: "close to one of the rounded corners"
# Padding is applied to length.
hole_x = (length / 2) + padding
hole_y = (width / 2)

# Create through-holes along Z axis
for x in (-hole_x, hole_x):
    result = result - Pos(x, hole_y, 0) * Cylinder(radius=hole_radius, height=height * 2)