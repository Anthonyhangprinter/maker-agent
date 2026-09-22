from build123d import *

vertical_length = 0.036
vertical_width = 0.0135
vertical_height = 0.9

horizontal_length = 0.24
horizontal_width = 0.0135
horizontal_thickness = 0.036

drop_from_top = 0.15

post = Pos(0, 0, vertical_height / 2) * Box(vertical_length, vertical_width, vertical_height)

# The horizontal arm attaches to the side of the post, 0.15 below the top,
# which is what skews the shape away from a symmetric T.
arm_center_z = vertical_height - drop_from_top
arm_center_x = vertical_length / 2 + horizontal_length / 2
arm = Pos(arm_center_x, 0, arm_center_z) * Box(horizontal_length, horizontal_width, horizontal_thickness)

result = post + arm
