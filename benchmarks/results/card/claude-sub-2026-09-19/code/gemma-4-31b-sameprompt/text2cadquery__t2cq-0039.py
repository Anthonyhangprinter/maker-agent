from build123d import *

# Dimensions provided in the request
# Note: Request specifies units as "units", treating them as mm for build123d
length = 0.7122
width = 0.1094
height = 0.75

# The part is described as a tall slender rectangular prism with a flat base and a pointed top.
# To achieve a "pointed top" on a rectangular prism, we can use a Cone or a Pyramid.
# Since the base is rectangular, we create the main body and a pyramid/cone top.
# However, the total height is 0.75. We will split this into a body and a point.

body_height = height * 0.7
point_height = height - body_height

# Main rectangular body (centred at origin)
# Box spans -body_height/2 to +body_height/2
body = Box(length, width, body_height)

# Pointed top: A pyramid-like shape. 
# In algebra mode, we can use a Cone with a small top radius or a custom shape.
# To make it a "pointed top" matching the rectangular profile, we use a Cone 
# that is scaled or simply a Cone that fits the profile.
# Positioned so its base is at z = body_height/2
top = Pos(0, 0, body_height/2 + point_height/2) * Cone(bottom_radius=length/2, top_radius=0, height=point_height)

# To ensure the "pointed top" matches the rectangular footprint better, 
# we intersect the cone with a box of the same width/length.
top_refined = top & Box(length, width, point_height)

# Combine the body and the pointed top
result = body + top_refined

# Ensure the result is a single fused solid
result.label = "slender_prism"