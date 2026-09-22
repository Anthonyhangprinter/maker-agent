from build123d import *

# Dimensions in MILLIMETRES
D1 = 0.567213
H1 = 0.5625
D2 = 0.266022
H2 = 0.1875

# Base cylinder: centred at origin
# Box/Cylinder in algebra mode are centred. 
# To have the base cylinder span from z = -H1/2 to +H1/2
base = Cylinder(radius=D1/2, height=H1)

# Top cylinder: 
# It sits on the top face of the base.
# Base top face is at z = H1/2.
# Top cylinder height is H2. Its centre must be at z = H1/2 + H2/2.
top = Pos(0, 0, (H1 + H2) / 2) * Cylinder(radius=D2/2, height=H2)

# Fuse the two cylinders into a single solid
result = base + top