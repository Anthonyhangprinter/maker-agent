from build123d import *

length = 1.10769
width = 0.474725
height = 0.18989

cyl_diameter = length / 2
cyl_height = width
connector_diameter = cyl_diameter
connector_height = cyl_height / 2
inner_cyl_diameter = 0.194358
inner_box_side = inner_cyl_diameter

base = Box(length, width, height)

# Cylinder mounted on the base so it partially protrudes above the top face
cyl_center_z = height / 2
cylinder = Pos(0, 0, cyl_center_z) * Cylinder(radius=cyl_diameter / 2, height=cyl_height)

# Connector box centred on top of the cylinder
connector_center_z = cyl_center_z + cyl_height / 2 + connector_height / 2
connector = Pos(0, 0, connector_center_z) * Box(connector_diameter, connector_diameter, connector_height)

body = base + cylinder + connector

# Bore through the centre of the cylinder and connector
bore_height = cyl_height + connector_height + height
inner_cylinder = Pos(0, 0, cyl_center_z) * Cylinder(radius=inner_cyl_diameter / 2, height=bore_height)
inner_box = Pos(0, 0, cyl_center_z) * Box(inner_box_side, inner_box_side, bore_height)

body -= inner_cylinder
body -= inner_box

# Arm profile: a closed polyline extruded along the base's width
arm_pts = [
    (0, 0.316484),
    (-0.056, 0.483119),
    (length / 2, 0.613244),
    (length / 2, 0.434708),
]
with BuildSketch(Plane.XZ) as arm_sketch:
    with BuildLine():
        Polyline(*arm_pts, close=True)
    make_face()
arm = extrude(arm_sketch.sketch, amount=width)
arm = Pos(0, -width / 2, 0) * arm

result = body + arm
