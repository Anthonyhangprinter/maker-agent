from build123d import *

# Dimensions in MILLIMETRES (as per request units)
length = 0.875
width = 0.15
height = 1.5
fillet_radius = 0.25
hole_diameter = 0.3
hole_radius = hole_diameter / 2
padding = -0.639071

# Create the main body
# Box is centred at origin: X spans -length/2 to +length/2, Y spans -width/2 to +width/2
body = Box(length, width, height)

# Round two corners of the same long edge
# The long edges are parallel to X axis. 
# We select the edges on one side (e.g., Y = +width/2) that are perpendicular to the long edge (parallel to Y/Z)
# and the long edge itself to create the corner fillets.
# More reliably: filter edges by Axis.Y and select the ones at the ends of the long edge.
# The request asks for two corners of the same long edge.
# We target the edges at the ends of the rectangle on the top/bottom faces.
# However, the request says "extrude this sketch", implying the fillet is in the XY plane.
# So we fillet the vertical edges at the ends of one long side.
edges_to_fillet = body.edges().filter_by(Axis.Z).filter_by(lambda e: abs(e.center.y - width/2) < 0.01)
result = fillet(body, edges_to_fillet, radius=fillet_radius)

# Create two small, evenly-spaced circular holes
# Position: "close to one of the rounded corners"
# Padding is applied to length and height dimensions.
# Hole X positions: half length + padding
# Hole Y positions: Since they are on the "same side as rounded corners", Y = width/2
hole_x = (length / 2) + padding
hole_y = (width / 2)

# The request mentions padding for height as well, but holes are "circular holes" 
# usually implying through-holes in this context. 
# We place them at Z=0 and make them long enough to pierce the height.
for x in (-hole_x, hole_x):
    result -= Pos(x, hole_y, 0) * Cylinder(radius=hole_radius, height=height * 2)