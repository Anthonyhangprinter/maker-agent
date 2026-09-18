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
# Positioned at z = (BASE_H/2 + CYL_H/2) to sit on top, 
# but request says "partially protrudes", implying it might be sunk.
# We will place it so its bottom is at the base's top face.
cyl = Pos(0, 0, (BASE_H / 2) + (CYL_H / 2)) * Cylinder(radius=CYL_R, height=CYL_H)

# 3. Connector box (cylindrical shape based on "same diameter")
# Aligned centrally on top of the cylinder
conn = Pos(0, 0, (BASE_H / 2) + CYL_H + (CONN_H / 2)) * Cylinder(radius=CONN_R, height=CONN_H)

# Combine additive parts
main_body = base + cyl + conn

# 4. Subtractions
# Inner cylinder (vertical through-hole)
inner_cyl = Cylinder(radius=INNER_CYL_R, height=2.0)

# Inner box (cutting through center)
inner_box = Box(INNER_BOX_L, INNER_BOX_W, INNER_BOX_H)

# Apply subtractions
main_body = main_body - inner_cyl - inner_box

# 5. Arm shaped like a polyline extruded along the width
# The polyline is in XZ plane (points are x, z)
# We create a face from the polyline and extrude it along Y (width)
with BuildSketch(Plane.XZ) as sk:
    Polyline(*ARM_PTS)
    # Close the loop to make a face
    Line(ARM_PTS[-1], ARM_PTS[0])
    
arm_face = sk.sketch.faces()[0]
# Extrude along Y axis (width of base)
arm = extrude(arm_face, BASE_W)

# Final assembly
result = main_body + arm