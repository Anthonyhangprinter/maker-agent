from build123d import *

rect_width = 0.399631   # the longer edge (X)
rect_height = 0.091591  # the shorter edge (Y)
hex_length = rect_width * 2 / 3
hex_max_width = 0.188808
thickness = 0.133054

base = Box(rect_width, rect_height, thickness)

# Semi-hexagonal section attached to the +Y long edge, starting at one end
# (the -X end) and extending 2/3 of the rectangle's length outward by
# hex_max_width at its widest point.
x0 = -rect_width / 2
y0 = rect_height / 2
hex_pts = [
    (x0, y0),
    (x0, y0 + hex_max_width * 0.4),
    (x0 + hex_length * 0.25, y0 + hex_max_width),
    (x0 + hex_length * 0.75, y0 + hex_max_width),
    (x0 + hex_length, y0 + hex_max_width * 0.4),
    (x0 + hex_length, y0),
]
with BuildSketch() as hex_sk:
    with BuildLine():
        Polyline(*hex_pts, close=True)
    make_face()
hex_bump = Pos(0, 0, -thickness / 2) * extrude(hex_sk.sketch, amount=thickness)

result = base + hex_bump
