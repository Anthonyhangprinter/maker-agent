from build123d import *

# Dimensions
length = 180.0
width = 130.0
height = 55.0
wall = 3.5
floor = 4.0
lip_width = 6.0
lip_depth = 2.0
boss_d = 9.0
boss_h = 14.0
hole_d = 3.2

# The outer shell is centered at origin. 
# Z spans from -height/2 to +height/2.
outer_box = Box(length, width, height)

# The inner cavity to make it a hollow enclosure.
# To leave a floor of 'floor' thickness, the inner box must be offset up.
# Bottom of outer box is at -height/2. 
# Bottom of inner box should be at -height/2 + floor.
# Inner box height = height - floor.
# Inner box center Z = (-height/2 + floor) + (height - floor)/2 = floor/2.
inner_box = Pos(0, 0, floor / 2) * Box(length - 2 * wall, width - 2 * wall, height - floor)

# The stepped recessed lip for the snap-on lid.
# This is a subtraction from the top rim.
# The lip is 6mm wide and 2mm deep.
# It removes material from the top edge (z = height/2).
# The lip box spans the inner cavity but is wider by lip_width on each side.
lip_cut = Pos(0, 0, height / 2 - lip_depth / 2) * Box(
    length - 2 * wall + 2 * lip_width, 
    width - 2 * wall + 2 * lip_width, 
    lip_depth
)

# Base enclosure
enclosure = outer_box - inner_box - lip_cut

# Corner bosses
# Bosses are 14mm tall, rising from the floor.
# Floor top is at z = -height/2 + floor.
# Boss center Z = (-height/2 + floor) + boss_h / 2.
boss_z = -height / 2 + floor + boss_h / 2
boss_x_pos = length / 2 - wall - boss_d / 2
boss_y_pos = width / 2 - wall - boss_d / 2

bosses = None
for x in (-boss_x_pos, boss_x_pos):
    for y in (-boss_y_pos, boss_y_pos):
        # Create the boss cylinder
        boss = Pos(x, y, boss_z) * Cylinder(radius=boss_d / 2, height=boss_h)
        # Create the through hole for the boss
        hole = Pos(x, y, boss_z) * Cylinder(radius=hole_d / 2, height=boss_h * 2)
        
        if bosses is None:
            bosses = boss - hole
        else:
            bosses += (boss - hole)

result = enclosure + bosses