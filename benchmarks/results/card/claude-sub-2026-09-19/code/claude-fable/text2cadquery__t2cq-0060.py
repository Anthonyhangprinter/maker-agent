from build123d import *

box_length = 0.45
box_width = 0.75
box_height = 0.375
cut_length = 0.2796
cut_width = 0.5272
cut_depth = 0.1965
slot_length = 0.0638
slot_width = 0.0269
slot_depth = 0.1065
pocket_length = 0.0638
pocket_width = 0.0305
pocket_depth = 0.0315
small_feature_y = 0.15

top_z = box_height / 2

body = Box(box_length, box_width, box_height)

# part 2: rectangular cutout sunk into the top surface, flush with the +X long edge,
# centred along the long (Y) direction
overshoot = 0.05
cut_x = box_length / 2 - cut_length / 2 + overshoot / 2
cut_z = top_z - cut_depth / 2 + overshoot / 2
body -= Pos(cut_x, 0, cut_z) * Box(cut_length + overshoot, cut_width, cut_depth + overshoot)

# remaining wide top land runs along the -X side; the two small features sit on it
land_x = -box_length / 2 + (box_length - cut_length) / 2

# part 3: round-ended (cylindrical) slot cut down from the top surface
slot_profile = SlotOverall(slot_length, slot_width)
slot_cut = Pos(land_x, small_feature_y, top_z - slot_depth) * extrude(
    slot_profile, amount=slot_depth + overshoot
)
body -= slot_cut

# part 4: small rectangular pocket cut down from the top surface
pocket_z = top_z - pocket_depth / 2 + overshoot / 2
body -= Pos(land_x, -small_feature_y, pocket_z) * Box(
    pocket_length, pocket_width, pocket_depth + overshoot
)

result = body
