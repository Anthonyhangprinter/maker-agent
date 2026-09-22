from build123d import *

# Dimensions in MILLIMETRES
HEIGHT = 0.75
DIAMETER = 0.1071
RADIUS = DIAMETER / 2

# The request mentions "two concentric circles" and "extrude", 
# implying a hollow cylinder (tube). 
# Since no wall thickness is provided, we assume a thin wall.
WALL_THICKNESS = RADIUS * 0.1
INNER_RADIUS = RADIUS - WALL_THICKNESS

# Create the outer cylinder
outer = Cylinder(radius=RADIUS, height=HEIGHT)

# Create the inner cylinder to subtract (making it a tube/hollow cylinder)
inner = Cylinder(radius=INNER_RADIUS, height=HEIGHT * 2)

# Combine using algebra mode
result = outer - inner

# The request mentions "rounded edges". 
# Fillets on very small parts often fail. Wrap in try/except as per repair hint.
try:
    # Filter for the circular edges at the top and bottom.
    # Using a very small radius to avoid the ValueError.
    result = fillet(result.edges().filter_by(GeomType.CIRCLE), radius=0.001)
except:
    # Keep the unfilleted solid on failure
    pass