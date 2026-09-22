from build123d import *
import math

large_d = 0.485133
small_d = 0.157385
total_length = 0.993343
extrude_width = 0.243343

R = large_d / 2
r = small_d / 2
# centre-to-centre distance so the overall length equals total_length
centre_dist = total_length - R - r

# external tangent points (large circle at origin, small circle at +X)
cos_t = (R - r) / centre_dist
sin_t = math.sqrt(1.0 - cos_t * cos_t)
p1 = (R * cos_t, R * sin_t)
p2 = (centre_dist + r * cos_t, r * sin_t)
p3 = (centre_dist + r * cos_t, -r * sin_t)
p4 = (R * cos_t, -R * sin_t)

web = make_face(Polyline(p1, p2, p3, p4, close=True).edges())
profile = Circle(R) + Pos(centre_dist, 0, 0) * Circle(r) + web

body = extrude(profile, amount=extrude_width / 2, both=True)

# centre the part on the origin along its length
x_shift = -(centre_dist + r - R) / 2
result = Pos(x_shift, 0, 0) * body
