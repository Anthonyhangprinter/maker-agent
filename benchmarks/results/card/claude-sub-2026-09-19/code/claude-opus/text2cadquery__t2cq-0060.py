from build123d import *

box_length = 0.45
box_width = 0.75
box_height = 0.375
pocket_length = 0.2796
pocket_width = 0.5272
pocket_depth = 0.1965
boss_diameter = 0.0638
boss_height = 0.1065
pad_length = 0.0638
pad_width = 0.0305
pad_height = 0.0315

top_z = box_height / 2
rim_centre_y = (pocket_width / 2 + box_width / 2) / 2

result = Box(box_length, box_width, box_height)
result -= Pos(0, 0, top_z - pocket_depth / 2) * Box(pocket_length, pocket_width, pocket_depth)

result += Pos(0, rim_centre_y, top_z + boss_height / 2) * Cylinder(radius=boss_diameter / 2, height=boss_height)
result += Pos(0, -rim_centre_y, top_z + pad_height / 2) * Box(pad_length, pad_width, pad_height)
