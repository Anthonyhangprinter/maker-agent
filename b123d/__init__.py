"""build123d domain library for CAD Agent v3."""
from .domain import structural_section, spur_gear, hex_bolt
from .conventions import (
    MM, WALL_THICKNESS, COSMETIC_FILLET,
    CLEARANCE_M3, CLEARANCE_M4, CLEARANCE_M5, CLEARANCE_M6,
    THREAD_PITCH, coarse_pitch,
)
