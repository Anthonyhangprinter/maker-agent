from build123d import *

# Dimensions in MILLIMETRES (as requested by the prompt's authoritative units)
# The request specifies units as 0.75, 0.3875, 0.1826. 
# Since the system requires MILLIMETRES, these are treated as the absolute dimensions.
length = 0.75
width = 0.3875
height = 0.1826

# The request describes a process of sketching, scaling, and extruding.
# In Algebra Mode, this is represented by a Box primitive.
# Box(length, width, height) is centred at the origin.
result = Box(length, width, height)