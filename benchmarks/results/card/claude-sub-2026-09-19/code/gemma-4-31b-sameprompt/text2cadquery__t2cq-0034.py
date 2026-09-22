from build123d import *

# Dimensions based on the final resulting sizes provided in the request
# Length: 0.75, Width: 0.062, Height: 0.0117 (approx 0.012)
length = 0.75
width = 0.062
height = 0.0117

# The request describes a process of sketching, scaling, and extruding.
# In Algebra Mode, this is equivalent to a Box centered at the origin.
# Box(length, width, height) creates a solid from -length/2 to +length/2, etc.
result = Box(length, width, height)