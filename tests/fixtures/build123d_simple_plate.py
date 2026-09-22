import sys as _sys
_sys.path.insert(0, '/home/theultimatecunt/.openclaw/skills/cad-builder-phase3')

from build123d import *

# Parameters
plate_size = 55.0
plate_thickness = 3.0
cutout_size = 20.0
chamfer_size = 1.0

# Create the main plate
# Box is centred at origin: spans -27.5 to 27.5 in X and Y, -1.5 to 1.5 in Z
plate = Box(plate_size, plate_size, plate_thickness)

# Create the square cutout
# Centred at origin, height must be >= plate_thickness to ensure a through-cut
cutout = Box(cutout_size, cutout_size, plate_thickness * 2)

# Combine: subtract cutout from plate
result = plate - cutout

# Apply chamfers to the four outer vertical edges
# filter_by(Axis.Z) selects edges parallel to Z (the vertical corners)
result = chamfer(result.edges().filter_by(Axis.Z), length=chamfer_size)