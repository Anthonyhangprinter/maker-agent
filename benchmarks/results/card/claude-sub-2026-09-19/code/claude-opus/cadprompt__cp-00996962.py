from build123d import *

plate_length = 1.5
plate_width = 1.37143
plate_thickness = 0.078171
groove_width = 0.085714
groove_offset_from_left = 0.685714

groove_centre_x = -plate_length / 2 + groove_offset_from_left + groove_width / 2
cutter_depth = plate_width * 2
cutter_height = plate_thickness * 4

result = Box(plate_length, plate_width, plate_thickness)
result -= Pos(groove_centre_x, 0, 0) * Box(groove_width, cutter_depth, cutter_height)
