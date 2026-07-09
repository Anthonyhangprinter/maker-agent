// Golden (c) — Ø80x10mm flange, Ø30 centre bore, 6x Ø8 bolt-circle holes at r=30mm.
// Every cutter overshoots both faces (height = thickness + 2*overshoot, centred on
// z=0) so all 7 holes are genuine through-holes, never blind no-ops.
//
// Genus, worked out (not assumed) in tests/test_scad_spike.py: a solid disc is
// genus 0 (simply connected, like a ball); the centre bore alone makes it an
// annulus/torus topology (genus 1); each of the 6 additional through-holes on the
// bolt circle adds one more independent handle (genus += 1 per hole, regardless
// of the base surface's existing genus, as long as the holes stay disjoint) ->
// genus 1 + 6 = 7 predicted. The test measures scad_mesh_gate's actual genus_sum
// and asserts whatever trimesh reports, commenting on any mismatch.
$fn = 64;

od          = 80;
thickness   = 10;
bore_d      = 30;
bolt_circle_r = 30;
bolt_hole_d = 8;
bolt_count  = 6;

module bolt_holes() {
    for (i = [0 : bolt_count - 1]) {
        a = i * 360 / bolt_count;
        translate([bolt_circle_r * cos(a), bolt_circle_r * sin(a), 0])
            cylinder(h = thickness + 4, r = bolt_hole_d / 2, center = true);
    }
}

difference() {
    cylinder(h = thickness, r = od / 2, center = true);
    cylinder(h = thickness + 4, r = bore_d / 2, center = true);
    bolt_holes();
}
