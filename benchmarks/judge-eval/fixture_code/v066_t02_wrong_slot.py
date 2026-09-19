from build123d import *

# Dimensions
length = 140.0
width = 90.0
height = 60.0
wall = 3.0
divider_thick = 2.5
divider_offset_from_end = 60.0
cable_hole_d = 12.0

# Outer shell
outer_box = Box(length, width, height)

# Inner cavity to create open-top housing
# Outer box spans z = -height/2 to +height/2
# Inner box is offset up by 'wall' to leave the floor
# Inner box height is (height - wall) to leave the top open
inner_box = Pos(0, 0, wall) * Box(length - 2 * wall, width - 2 * wall, height - wall)

# The housing shell
shell = outer_box - inner_box

# Internal divider
# Positioned 60mm from one end. 
# End of box is at x = -length/2. 
# Divider center x = -length/2 + 60 + divider_thick/2
div_x = -length / 2 + divider_offset_from_end + divider_thick / 2
# Divider spans the inner width and inner height
divider_solid = Pos(div_x, 0, 0) * Box(divider_thick, width - 2 * wall, height - wall)
# Adjust divider z to sit on the floor (floor is at z = -height/2)
# The divider_solid is currently centered at z=0, spanning - (height-wall)/2 to + (height-wall)/2
# We shift it so the bottom is at -height/2
divider_solid = Pos(div_x, 0, (height - wall) / 2 - height / 2) * Box(divider_thick, width - 2 * wall, height - wall)

# Cable feed-through hole in the divider
# Centered in the divider's height and width
cable_hole = Pos(div_x, 0, 0) * Cylinder(radius=cable_hole_d / 2, height=divider_thick * 2)

# Final assembly
result = shell + (divider_solid - cable_hole)