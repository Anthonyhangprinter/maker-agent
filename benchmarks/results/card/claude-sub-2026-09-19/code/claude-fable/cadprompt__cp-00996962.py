from build123d import *

plate_length = 1.5
plate_width = 1.37143
plate_height = 0.078171
slot_width = 0.085714
slot_start = 0.685714

plate = Box(plate_length, plate_width, plate_height)

# slot sketched on the front (-Y) profile, full plate height, cut back to the plate centreline
slot_depth = plate_width / 2
slot_x = -plate_length / 2 + slot_start + slot_width / 2
overshoot = slot_width
slot_y = -plate_width / 2 + slot_depth / 2 - overshoot / 2

slot = Pos(slot_x, slot_y, 0) * Box(slot_width, slot_depth + overshoot, plate_height * 4)

result = plate - slot
