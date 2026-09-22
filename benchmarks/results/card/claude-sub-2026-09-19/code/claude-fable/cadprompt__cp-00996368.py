from build123d import *

rect_length = 0.399631
rect_height = 0.091591
max_width = 0.188808
hex_fraction = 0.6666667
extrude_height = 0.133054

# semi-hexagonal (trapezoid) section on the long top edge, starting at the left end
hex_base = rect_length * hex_fraction
hex_flat = hex_base / 2
hex_x0 = hex_base / 4
hex_x1 = hex_x0 + hex_flat

pts = [
    (0.0, 0.0),
    (rect_length, 0.0),
    (rect_length, rect_height),
    (hex_base, rect_height),
    (hex_x1, max_width),
    (hex_x0, max_width),
    (0.0, rect_height),
]

profile = make_face(Polyline(*pts, close=True).edges())
body = extrude(profile, amount=extrude_height / 2, both=True)

# centre the part on the origin
result = Pos(-rect_length / 2, -max_width / 2, 0) * body
