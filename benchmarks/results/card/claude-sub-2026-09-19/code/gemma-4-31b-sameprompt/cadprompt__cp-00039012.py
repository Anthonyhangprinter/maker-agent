from build123d import *

# Dimensions converted from inches to mm (1 inch = 25.4 mm)
# Inner diameter: 0.70248" -> 17.843 mm
# Outer diameter: 0.825" -> 20.955 mm
# Pipe height: 0.75" -> 19.05 mm
# Flange thickness: 0.01125" -> 0.28575 mm

inner_d = 17.843
outer_d = 20.955
pipe_h = 19.05
flange_t = 0.28575

# The request describes the pipe as "hollow" and mentions a "positive extrusion for the inner circle"
# and a "negative extrusion for the outer circle" to form a "thin flange".
# Interpreting the geometry: 
# 1. The main body is a cylinder of diameter inner_d and height pipe_h.
# 2. The flange is a thin disc of diameter outer_d and thickness flange_t at the base.
# 3. To make the pipe "hollow", we subtract a cylinder from the center.
# However, the prompt specifically says "positive extrusion for the inner circle" and 
# "negative extrusion for the outer circle" to form the flange. 
# This phrasing is slightly contradictory to standard CAD terminology.
# Standard interpretation for a "hollow pipe with a flange":
# Outer shell (Pipe) + Outer shell (Flange) - Inner core (Hole).

# Let's follow the dimensions provided:
# Main pipe body (the "inner circle" extrusion)
pipe_body = Cylinder(radius=inner_d / 2, height=pipe_h)

# The flange (the "outer circle" extrusion)
# Positioned at the base. Since the pipe is centered at Z=0, 
# the bottom face is at -pipe_h/2.
flange = Pos(0, 0, -pipe_h / 2) * Cylinder(radius=outer_d / 2, height=flange_t)

# Combine them
solid = pipe_body + flange

# To make it "hollow", we subtract the inner diameter from the center.
# Since the prompt says the "inner circle" is the positive extrusion, 
# and the "outer circle" is the negative one for the flange, 
# it implies the pipe wall is defined by these two.
# But a pipe with inner_d=17.8 and outer_d=20.9 is a standard pipe.
# Let's create the hollow pipe:
# Outer wall = outer_d, Inner wall = inner_d.
# The "positive extrusion" of 0.70248" is the hole.
# The "negative extrusion" of 0.825" is the flange.

# Corrected Logic based on "Hollow Pipe with Flange":
# 1. Outer Cylinder (Pipe wall)
# 2. Flange Disc at bottom
# 3. Subtract Inner Cylinder (The hole)

outer_pipe = Cylinder(radius=outer_d / 2, height=pipe_h)
flange_disc = Pos(0, 0, -pipe_h / 2) * Cylinder(radius=outer_d / 2, height=flange_t)
inner_hole = Cylinder(radius=inner_d / 2, height=pipe_h + flange_t * 2)

result = (outer_pipe + flange_disc) - inner_hole