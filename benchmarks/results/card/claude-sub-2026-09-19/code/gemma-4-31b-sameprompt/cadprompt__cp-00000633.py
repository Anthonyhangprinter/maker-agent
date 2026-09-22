from build123d import *

# Dimensions in MILLIMETRES (as per request units)
length = 0.75
width = 0.1875
height = 0.00391
hole_diameter = 0.12512

# Create the rectangular block centered at the origin
block = Box(length, width, height)

# Create the circular hole in the center
# Z is 0 (centered), height is larger than part thickness to ensure it pierces fully
hole = Cylinder(radius=hole_diameter / 2, height=height * 2)

# Combine using algebra mode
result = block - hole