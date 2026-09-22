from build123d import *

# Dimensions from request
BASE_L = 1.10769
BASE_W = 0.474725
BASE_H = 0.18989

CYL_D = BASE_L / 2
CYL_H = BASE_W
CYL_R = CYL_D / 2

CONN_D = CYL_D
CONN_H = CYL_H / 2
CONN_R = CONN_D / 2

INNER_CYL_D = 0.194358
INNER_CYL_R = INNER_CYL_D / 2

INNER_BOX_L = INNER_CYL_D
INNER_BOX_W = INNER_CYL_D
INNER_BOX_H = 1.0  # Sufficient to cut through everything

# Arm polyline points
ARM_PTS = [(0, 0.316484), (-0.056, 0.483119), (BASE_L / 2, 0.613244), (BASE_L / 2, 0.434708)]

# 1. Base Box
base = Box(BASE_L, BASE_W, BASE_H)

# 2. Cylinder on top, partially protruding
cyl = Pos(0, 0, (BASE_H / 2) + (CYL_H / 2)) * Cylinder(radius=CYL_R, height=CYL_H)

# 3. Connector box (cylindrical shape based on "same diameter")
conn = Pos(0, 0, (BASE_H / 2) + CYL_H + (CONN_H / 2)) * Cylinder(radius=CONN_R, height=CONN_H)

# Combine additive parts
main_body = base + cyl + conn

# 4. Subtractions
inner_cyl = Cylinder(radius=INNER_CYL_R, height=2.0)
inner_box = Box(INNER_BOX_L, INNER_BOX_W, INNER_BOX_H)

# Apply subtractions
main_body = main_body - inner_cyl - inner_box

# 5. Arm shaped like a polyline extruded along the width
# Use BuildLine for polyline operations, then close it to create a face
with BuildLine(Plane.XZ) as bl:
    Polyline(ARM_PTS)
    Close()

# Create a face from the closed line and extrude it along Y axis
arm_face = bl.line.faces()[0]
arm = extrude(arm_face, BASE_W)

# Final assembly
result = main_body + arm