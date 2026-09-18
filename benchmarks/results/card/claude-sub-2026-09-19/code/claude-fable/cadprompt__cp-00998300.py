from build123d import *

# request is in metres; converted to millimetres (0.036 m = 36 mm, etc.)
post_length = 36.0
post_width = 13.5
post_height = 900.0
bar_length = 240.0
bar_thickness = 13.5
bar_depth = 36.0
bar_below_top = 150.0

# vertical post: 13.5 wide (X) x 36 deep (Y) x 900 tall (Z)
post = Box(post_width, post_length, post_height)

# horizontal crossbar: 240 long (X) x 36 deep (Y) x 13.5 thick (Z), 150 below the top
bar_z = post_height / 2 - bar_below_top
bar = Pos(0, 0, bar_z) * Box(bar_length, bar_depth, bar_thickness)

result = post + bar
