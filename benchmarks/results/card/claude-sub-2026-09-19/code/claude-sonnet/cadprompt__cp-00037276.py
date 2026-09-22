from build123d import *

diameter_large = 0.485133
diameter_small = 0.157385
total_length = 0.993343
width = 0.243343

r_large = diameter_large / 2
r_small = diameter_small / 2
# Distance between circle centres, per the request's own formula
center_distance = total_length - (r_large + r_small)

# Two circles connected by their external tangent lines (a taut-chain / belt profile)
# is exactly the convex hull of the two circles.
with BuildSketch() as sk:
    with BuildLine():
        CenterArc((0, 0), r_large, 0, 360)
        CenterArc((center_distance, 0), r_small, 0, 360)
        make_hull()
    make_face()

result = extrude(sk.sketch, amount=width)
