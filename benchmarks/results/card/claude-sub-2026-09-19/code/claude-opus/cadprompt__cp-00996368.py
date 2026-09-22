from build123d import *

rect_width = 0.399631
rect_height = 0.091591
hex_max_width = 0.188808
extrude_thickness = 0.133054
hex_span_fraction = 0.666667

hex_base_length = rect_width * hex_span_fraction
hex_inset = hex_base_length / 4

profile = Polygon(
    (0.0, 0.0),
    (rect_width, 0.0),
    (rect_width, rect_height),
    (hex_base_length, rect_height),
    (hex_base_length - hex_inset, hex_max_width),
    (hex_inset, hex_max_width),
    (0.0, rect_height),
    align=None,
)

body = extrude(profile, amount=extrude_thickness / 2, both=True)

result = Pos(-rect_width / 2, -hex_max_width / 2, 0) * body
