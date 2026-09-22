from build123d import *

box_length = 0.75
box_width = 0.18
box_height = 0.18
fillet_radius = 0.02

cyl_length = 0.55
cyl_diameter = 0.16

prism = Box(box_length, box_width, box_height)

# Round the four edges running along the prism's long axis
long_edges = prism.edges().filter_by(Axis.X)
prism = fillet(long_edges, radius=fillet_radius)

# The cylindrical part removes material from the rectangular prism, bored
# along the prism's long axis and centred within it
bore = Rotation(0, 90, 0) * Cylinder(radius=cyl_diameter / 2, height=cyl_length)

result = prism - bore
