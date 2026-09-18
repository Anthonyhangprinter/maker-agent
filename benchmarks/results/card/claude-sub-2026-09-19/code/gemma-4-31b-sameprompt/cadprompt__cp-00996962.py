from build123d import *

# Dimensions provided in the request
plate_l = 1.5
plate_w = 1.37143
plate_h = 0.078171
slot_w = 0.085714
slot_start_from_left = 0.685714

# The plate is centered at the origin.
# X spans -0.75 to +0.75
# Y spans -0.685715 to +0.685715
# Z spans -0.0390855 to +0.0390855
plate = Box(plate_l, plate_w, plate_h)

# The slot is a rectangle of width slot_w and height plate_h.
# It is subtracted through the centerline of the larger rectangle.
# "Through the centerline" implies it cuts through the width (Y axis).
# The slot starts 0.685714 from the left edge.
# Left edge is at x = -plate_l / 2 = -0.75.
# Slot start x = -0.75 + 0.685714 = -0.064286.
# The slot is centered on its own width, so its center x is:
# slot_center_x = -0.064286 + (slot_w / 2) = -0.064286 + 0.042857 = -0.021429.
# The slot spans the entire height (Z) and the entire width (Y).

slot_center_x = (-plate_l / 2) + slot_start_from_left + (slot_w / 2)
slot_cutter = Pos(slot_center_x, 0, 0) * Box(slot_w, plate_w * 2, plate_h)

result = plate - slot_cutter