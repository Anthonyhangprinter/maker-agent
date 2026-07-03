"""build123d CAD conventions — matching text-to-cad defaults."""

MM = 1.0            # all dimensions in mm
WALL_THICKNESS = 2.5   # mm — standard plastic enclosure wall
COSMETIC_FILLET = 1.5  # mm — default edge softening radius

# M-series clearance hole diameters (mm)
CLEARANCE_M3 = 3.4
CLEARANCE_M4 = 4.5
CLEARANCE_M5 = 5.5
CLEARANCE_M6 = 6.6
CLEARANCE_M8 = 9.0
CLEARANCE_M10 = 11.0
CLEARANCE_M12 = 13.5

# ISO coarse thread pitches (mm)
THREAD_PITCH = {
    1: 0.25, 1.2: 0.25, 1.6: 0.35, 2: 0.4, 2.5: 0.45,
    3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25, 10: 1.5,
    12: 1.75, 16: 2.0, 18: 2.5, 20: 2.5, 24: 3.0,
    30: 3.5, 36: 4.0, 42: 4.5, 48: 5.0,
}


def coarse_pitch(size_mm: float) -> float:
    """Return the ISO coarse thread pitch for a given nominal diameter."""
    # round(), not int(): truncation made the fractional keys (1.2/1.6/2.5) unreachable,
    # so e.g. M2.5 silently got M2's 0.4mm pitch instead of 0.45.
    return THREAD_PITCH.get(round(float(size_mm), 2), round(size_mm * 0.075, 2))
