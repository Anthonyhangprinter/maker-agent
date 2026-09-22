from build123d import *

# Dimensions in MILLIMETRES (converted from provided units)
# User provided: L=0.75, W=0.18, H=0.18 (Rectangular)
# User provided: L=0.55, W=0.16, H=0.0028 (Cylindrical)
# Assuming units were in meters, converting to mm for build123d standard.
# If units were already mm, the part is microscopic. 
# Given the context of "complex assembly", these are likely meters.
RECT_L = 750.0
RECT_W = 180.0
RECT_H = 180.0

CYL_L = 550.0
CYL_W = 160.0
CYL_H = 2.8

# The request specifies the cylindrical part is used to remove material from the rectangular prism.
# Rectangular prism with rounded edges (fillets)
base_solid = Box(RECT_L, RECT_W, RECT_H)

# The "cylindrical shape" described as having "length 0.55, width 0.16, height 0.0028"
# implies a thin disc (Cylinder) where radius = width/2 and height = height.
# Length 0.55 is likely the axial length or a redundant dimension if it's a disc.
# However, if it's a cylinder of length 0.55, it's a long rod.
# "Circular bases" and "width 0.16" suggests radius = 0.08.
# "Height 0.0028" suggests it is a very thin washer/disc.
# "Cylindrical part being used to remove material" -> subtraction.
cutter = Cylinder(radius=CYL_W / 2, height=CYL_H)

# Perform subtraction
# Centered at origin as per rules
result = base_solid - cutter

# "Rectangular prism with rounded edges"
# Fillet all vertical edges (parallel to Z)
result = fillet(result.edges().filter_by(Axis.Z), radius=10.0)