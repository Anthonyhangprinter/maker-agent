from build123d import *

box_length = 0.45
box_width = 0.75
box_height = 0.375

cutout_length = 0.2796
cutout_width = 0.5272
cutout_height = 0.1965

cyl_diameter = 0.0269
cyl_height = 0.1065

rect2_length = 0.0638
rect2_width = 0.0305
rect2_height = 0.0315

main_box = Box(box_length, box_width, box_height)

# Cutout pocket in the top surface, aligned toward one of the longer (Y) edges
cutout_x = box_length / 2 - cutout_length / 2
cutout_y = box_width / 2 - cutout_width / 2
cutout_z = box_height / 2 - cutout_height / 2
cutout = Pos(cutout_x, cutout_y, cutout_z) * Box(cutout_length, cutout_width, cutout_height)

body = main_box - cutout

# Small cylindrical section on the top surface
cyl_boss = Pos(-box_length / 4, -box_width / 4, box_height / 2 + cyl_height / 2) \
    * Cylinder(radius=cyl_diameter / 2, height=cyl_height)

# Small rectangular section on the top surface, offset from the cylindrical one
rect_boss = Pos(box_length / 4, -box_width / 4, box_height / 2 + rect2_height / 2) \
    * Box(rect2_length, rect2_width, rect2_height)

result = body + cyl_boss + rect_boss
