from build123d import *

# PARAMETERS
# Rhombus points (scaled by 100 to avoid tiny dimensions, though prompt says MM)
# The prompt provides coordinates in a very small range (approx 0.1 to 0.5).
# To ensure the part is visible and printable, we treat these as MM as requested.
P1 = (-0.215997, -0.071707)
P2 = (0.283196/2 - 0.086266, -0.571139/2 + 0.128415)
P3 = (0.283196/2 + 0.154673, 0.571139/2 - 0.103869)
P4 = (0.283196/3 - 0.094331, 0.571139 - 0.45031)
BODY_HEIGHT = 0.493701

ARM_L = 0.1323
ARM_W = 0.78659
ARM_H = 0.203642
ARM_ROT = -41.7
ARM_POS = (0.223543, -0.361526, 0.301748)

# Create the irregular rhombus body
# Since we are in Algebra Mode, we create a polygon and extrude it.
# To center the part at the origin, we calculate the centroid of the points.
cx = (P1[0] + P2[0] + P3[0] + P4[0]) / 4
cy = (P1[1] + P2[1] + P3[1] + P4[1]) / 4

# Shift points to be centered at origin
pts = [
    (P1[0] - cx, P1[1] - cy),
    (P2[0] - cx, P2[1] - cy),
    (P3[0] - cx, P3[1] - cy),
    (P4[0] - cx, P4[1] - cy)
]

# Create the body using a polygon extrusion
# In algebra mode, we can use the Polygon helper or create a face and extrude.
# For a simple closed loop of points:
body_face = Polygon(pts)
body = extrude(body_face, BODY_HEIGHT)

# Create the rectangular prism arm
arm = Rotation((0, 0, ARM_ROT)) * Box(ARM_L, ARM_W, ARM_H)
arm = Pos(*ARM_POS) * arm

# Combine into a single fused solid
result = body + arm