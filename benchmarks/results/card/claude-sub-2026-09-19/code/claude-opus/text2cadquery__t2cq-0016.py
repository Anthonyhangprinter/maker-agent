from build123d import *

block_length = 0.75
block_width = 0.18
block_height = 0.18
slot_length = 0.55
slot_width = 0.16
slot_depth = 0.0028
edge_radius = 0.02

slot_radius = slot_width / 2
slot_straight = slot_length - slot_width
slot_z = block_height / 2 - slot_depth / 2

cutter = Box(slot_straight, slot_width, slot_depth)
for x in (-slot_straight / 2, slot_straight / 2):
    cutter += Pos(x, 0, 0) * Cylinder(radius=slot_radius, height=slot_depth)

result = Box(block_length, block_width, block_height)
result -= Pos(0, 0, slot_z) * cutter

result = fillet(result.edges().filter_by(Axis.X), radius=edge_radius)
