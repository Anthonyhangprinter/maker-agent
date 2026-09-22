from build123d import *

prism_length = 0.75
prism_width = 0.18
prism_height = 0.18
cut_length = 0.55
cut_width = 0.16
cut_depth = 0.0028
corner_radius = 0.02

body = Box(prism_length, prism_width, prism_height)

# shallow round-ended recess removed from the top face
recess_profile = SlotOverall(cut_length, cut_width)
recess = Pos(0, 0, prism_height / 2 - cut_depth) * extrude(recess_profile, amount=cut_depth * 2)
body = body - recess

# rounded edges: the four vertical corner edges (clear of the recess)
result = fillet(body.edges().filter_by(Axis.Z), radius=corner_radius)
