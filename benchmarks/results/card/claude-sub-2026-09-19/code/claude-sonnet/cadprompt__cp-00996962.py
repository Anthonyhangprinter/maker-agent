from build123d import *

plate_length = 1.5
plate_width = 1.37143
plate_height = 0.078171
slot_length = 0.085714
slot_start_from_left = 0.685714

plate = Box(plate_length, plate_width, plate_height)

# The slot starts 0.685714 from the left edge and cuts through the full
# depth (Y) and full height (Z) of the plate.
slot_x_start = -plate_length / 2 + slot_start_from_left
slot_x_center = slot_x_start + slot_length / 2
slot = Pos(slot_x_center, 0, 0) * Box(slot_length, plate_width * 2, plate_height * 2)

result = plate - slot
