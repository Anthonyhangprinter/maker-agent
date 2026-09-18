from build123d import *

# Dimensions in MILLIMETRES
# The request specifies units of 0.75 and 0.1071. 
# Assuming these are in a larger unit (like meters or inches) or literal mm.
# Following the rule "All dimensions in MILLIMETRES", we use the numbers provided.
HEIGHT = 0.75
DIAMETER = 0.1071
RADIUS = DIAMETER / 2

# The request mentions "two concentric circles" and "extrude", 
# implying a hollow cylinder (tube). 
# Since no wall thickness is provided, we assume a thin wall (e.g., 10% of radius).
WALL_THICKNESS = RADIUS * 0.1
INNER_RADIUS = RADIUS - WALL_THICKNESS

# Create the outer cylinder
outer = Cylinder(radius=RADIUS, height=HEIGHT)

# Create the inner cylinder to subtract (making it a tube/hollow cylinder)
inner = Cylinder(radius=INNER_RADIUS, height=HEIGHT * 2) # Height * 2 to ensure clean cut

# Combine using algebra mode
result = outer - inner

# The request mentions "rounded edges". 
# We apply a small fillet to the top and bottom circular rims.
# Filter for the circular edges at the top and bottom.
result = fillet(result.edges().filter_by(GeomType.CIRCLE), radius=0.01)