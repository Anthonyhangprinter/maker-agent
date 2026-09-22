from build123d import *

body_height = 0.493701
rhombus_pts = [
    (-0.215997, -0.071707),
    (0.283196 / 2 - 0.086266, -0.571139 / 2 + 0.128415),
    (0.283196 / 2 + 0.154673, 0.571139 / 2 - 0.103869),
    (0.283196 / 3 - 0.094331, 0.571139 - 0.45031),
]

with BuildSketch() as body_sk:
    with BuildLine():
        Polyline(*rhombus_pts, close=True)
    make_face()
body = extrude(body_sk.sketch, amount=body_height)

arm_length = 0.1323
arm_width = 0.78659
arm_height = 0.203642
arm_rotation_z = -41.7
arm_translate = (0.223543, -0.361526, 0.301748)

arm = Box(arm_length, arm_width, arm_height)
arm = Rotation(0, 0, arm_rotation_z) * arm
arm = Pos(*arm_translate) * arm

result = body + arm
