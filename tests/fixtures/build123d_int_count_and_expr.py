import sys as _sys
_sys.path.insert(0, '/home/theultimatecunt/.openclaw/skills/cad-builder-phase3')

from build123d import *
from b123d.domain import bolt_circle

# Parameters
stub_d = 60.0
stub_l = 110.0
bore_d = 44.0
rib_count = 6
rib_w = 2.5
rib_depth = 5.0
flange_d = 90.0
flange_t = 11.0
bolt_count = 6
bolt_d = 8.0
bolt_circle_d = 75.0

# Coordinate calculations to center the entire part at the origin
# Total height = 110 + 11 = 121
# Z range: -60.5 to +60.5
flange_z = -60.5 + (flange_t / 2)  # -55.0
stub_z = -60.5 + flange_t + (stub_l / 2)  # 5.5
bore_h = stub_l + flange_t

# Base components
stub = Pos(0, 0, stub_z) * Cylinder(radius=stub_d / 2, height=stub_l)
flange = Pos(0, 0, flange_z) * Cylinder(radius=flange_d / 2, height=flange_t)

# Combine main body
result = stub + flange

# Add external ribs
# Ribs are boxes that span the length of the stub and stick out from the surface
rib_geom = Box(rib_w, stub_d + 2 * rib_depth, stub_l)
for i in range(rib_count):
    result += Pos(0, 0, stub_z) * Rotation(0, 0, i * (360 / rib_count)) * rib_geom

# Subtract gas passage (bore) - centered at origin, spans full height
result -= Cylinder(radius=bore_d / 2, height=bore_h)

# Subtract bolt holes using bolt_circle helper
# Positioned at the flange center, height sufficient to pierce the flange
result -= Pos(0, 0, flange_z) * bolt_circle(bolt_count, bolt_circle_d, bolt_d, flange_t * 2)