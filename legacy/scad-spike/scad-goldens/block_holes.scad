// Golden (b) — the block.scad block with 4x Ø5mm through-holes, one near each
// corner, 10mm inset from both edges. Each cutter spans well beyond BOTH faces
// it pierces (height = thickness + 2*overshoot, centred on z=0) so it is a real
// through-hole, never a shallow/blind no-op. Simply-connected body with 4 clean
// through-holes -> topological genus 4 (same as a 4-holed torus).
$fn = 64;

length = 100;
width  = 60;
height = 20;
hole_d = 5;
inset  = 10;

module corner_holes() {
    for (x = [-(length/2 - inset), (length/2 - inset)])
        for (y = [-(width/2 - inset), (width/2 - inset)])
            translate([x, y, 0])
                cylinder(h = height + 4, r = hole_d / 2, center = true);
}

difference() {
    cube([length, width, height], center = true);
    corner_holes();
}
