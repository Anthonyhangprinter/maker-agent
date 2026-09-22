from build123d import *
import math

large_diameter = 0.485133
small_diameter = 0.157385
total_length = 0.993343
extrude_width = 0.243343

large_radius = large_diameter / 2
small_radius = small_diameter / 2
centre_distance = total_length - large_radius - small_radius

# unit normal of the external tangent lines between the two circles
nx = (large_radius - small_radius) / centre_distance
ny = math.sqrt(1.0 - nx * nx)

p_large_top = (large_radius * nx, large_radius * ny)
p_small_top = (centre_distance + small_radius * nx, small_radius * ny)
p_small_bot = (centre_distance + small_radius * nx, -small_radius * ny)
p_large_bot = (large_radius * nx, -large_radius * ny)

big_disc = Cylinder(radius=large_radius, height=extrude_width)
small_disc = Pos(centre_distance, 0, 0) * Cylinder(radius=small_radius, height=extrude_width)

web_profile = Polygon(p_large_top, p_small_top, p_small_bot, p_large_bot, align=None)
web = extrude(web_profile, amount=extrude_width / 2, both=True)

chain = big_disc + small_disc + web

centre_shift = (centre_distance + small_radius - large_radius) / 2
result = Pos(-centre_shift, 0, 0) * chain
