from build123d import *

post_length = 0.036
post_width = 0.0135
post_height = 0.9
arm_length = 0.24
arm_width = 0.0135
arm_thickness = 0.036
arm_drop_from_top = 0.15

arm_centre_z = post_height - arm_drop_from_top

post = Pos(0, 0, post_height / 2) * Box(post_length, post_width, post_height)
arm = Pos(0, 0, arm_centre_z) * Box(arm_length, arm_width, arm_thickness)

result = Pos(0, 0, -post_height / 2) * (post + arm)
