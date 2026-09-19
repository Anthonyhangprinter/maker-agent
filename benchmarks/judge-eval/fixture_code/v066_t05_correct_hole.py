from build123d import *

# Dimensions
length = 140.0
width = 90.0
height = 60.0
wall = 3.0
divider_thick = 2.5
divider_offset_from_end = 60.0
cable_hole_d = 12.0

# The outer box is centred at origin: X -70..70, Y -45..45, Z -30..30
outer_box = Box(length, width, height)

# The inner cavity is subtracted to create an open-top container.
# To keep the floor (thickness=wall), we offset the inner box UP by wall.
# Inner box dimensions are reduced by 2*wall in X and Y, and by wall in Z.
# The inner box spans Z from (-height/2 + wall) to (+height/2 + wall) to ensure it cuts through the top.
inner_cavity = Pos(0, 0, wall) * Box(length - 2 * wall, width - 2 * wall, height - wall)

# The divider is a thin box. 
# Position: 60mm from one end. 
# One end is at x = -70. 60mm from there is x = -10.
# The divider spans the internal width and internal height.
divider_x = -length / 2 + divider_offset_from_end + divider_thick / 2
divider_w = width - 2 * wall
divider_h = height - wall
# The divider must start from the floor (z = -height/2 + wall)
divider_z = -height / 2 + (divider_h / 2) + wall
divider = Pos(divider_x, 0, divider_z) * Box(divider_thick, divider_w, divider_h)

# Cable feed-through hole in the divider.
# The hole is circular, 12mm diameter, passing through the divider (X axis).
cable_hole = Pos(divider_x, 0, divider_z) * Rotation(0, 90, 0) * Cylinder(radius=cable_hole_d / 2, height=divider_thick * 2)

# Final assembly: (Outer - Cavity) + (Divider - Hole)
result = (outer_box - inner_cavity) + (divider - cable_hole)