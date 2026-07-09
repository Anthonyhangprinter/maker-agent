// Golden (a) — plain 100x60x20mm block, centred at the origin. No cuts: genus 0,
// volume exactly 100*60*20 = 120000 mm^3. Baseline for the mesh gate's "clean part"
// path (loadable, watertight, correct bbox/volume/body-count, zero holes).
$fn = 64;

length = 100;
width  = 60;
height = 20;

cube([length, width, height], center = true);
