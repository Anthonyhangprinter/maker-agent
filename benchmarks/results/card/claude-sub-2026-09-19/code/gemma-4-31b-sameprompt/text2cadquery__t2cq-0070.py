from build123d import *

# Dimensions in MILLIMETRES (as per request units)
CYL_HEIGHT = 0.75
RECT_LENGTH = 0.0179
RECT_WIDTH = 0.0357
CYL_RADIUS = 0.01  # Proportional radius to fit the rectangular section

# The cylinder is centered at the origin.
# Top face is at z = CYL_HEIGHT / 2
# Bottom face is at z = -CYL_HEIGHT / 2
cylinder = Cylinder(radius=CYL_RADIUS, height=CYL_HEIGHT)

# The rectangular section is placed at the top of the cylinder.
# To sit ON TOP, its center z must be (CYL_HEIGHT / 2) + (RECT_HEIGHT / 2).
# Since no height for the rectangle was provided, we assume a small proportional thickness.
RECT_HEIGHT = 0.01
rect_z = (CYL_HEIGHT / 2) + (RECT_HEIGHT / 2)

# Create the rectangular solid
rectangle = Pos(0, 0, rect_z) * Box(RECT_LENGTH, RECT_WIDTH, RECT_HEIGHT)

# Combine the two solids into one fused body
result = cylinder + rectangle