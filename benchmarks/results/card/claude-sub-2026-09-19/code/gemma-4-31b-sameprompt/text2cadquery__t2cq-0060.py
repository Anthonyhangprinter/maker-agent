from build123d import *

# Dimensions (Converted to MILLIMETRES as per rules)
# The request provides dimensions in what appears to be inches (0.45, etc).
# Rules state all dimensions in MILLIMETRES. I will treat the provided numbers as mm.
L1 = 0.45
W1 = 0.75
H1 = 0.375

L2 = 0.2796
W2 = 0.5272
H2 = 0.1965

L3 = 0.0638
W3 = 0.0269
H3 = 0.1065

L4 = 0.0638
W4 = 0.0305
H4 = 0.0315

# Part 1: Main rectangular box
# Centred at origin: spans X: -0.225..0.225, Y: -0.375..0.375, Z: -0.1875..0.1875
main_box = Box(L1, W1, H1)
main_box.label = "main_box"

# Part 2: Smaller rectangular cutout
# "Positioned on the top surface... aligned with one of its longer edges"
# Top surface is at z = H1/2 = 0.1875.
# To create a cutout, we subtract a box. 
# To be "on the top surface", the cutout box must overlap the top.
# Aligned with longer edge (Y axis): offset X by (L1 - L2)/2
cutout = Pos((L1 - L2) / 2, 0, H1 / 2) * Box(L2, W2, H2)
cutout.label = "cutout"

# Part 3: Cylindrical section
# "Placed on the top surface"
# The request describes it with length, width, height. 
# Since it's called "cylindrical", we use Cylinder. 
# Radius = width/2 = 0.0269/2. Height = 0.1065.
# Positioned at z = H1/2 + H3/2 to sit ON TOP.
cyl_radius = W3 / 2
cyl_section = Pos(0, 0, H1 / 2 + H3 / 2) * Cylinder(radius=cyl_radius, height=H3)
cyl_section.label = "cylindrical_section"

# Part 4: Another rectangular section
# "Located on the top surface"
# Positioned at z = H1/2 + H4/2 to sit ON TOP.
rect_section = Pos(0, 0, H1 / 2 + H4 / 2) * Box(L4, W4, H4)
rect_section.label = "rect_section"

# Assembly:
# The prompt says "assembled to form a rectangular box with a smaller rectangular cutout"
# This implies the cutout is subtracted from the main box.
# The other sections are "placed on" the top surface.
# To avoid interpenetration, we must ensure the added parts don't overlap.
# However, the prompt describes them as "parts" of the design.
# I will fuse the additions to the main body and subtract the cutout.

body = main_box - cutout
body += cyl_section
body += rect_section

result = body
result.label = "final_part"