# lang-ab report

Run: langab-20260919-resident  
Specs: 40  Seed: 20260919  Arms: b123d, b123d-nofs, cadquery, openscad

## Per-arm summary

| arm | n | built % | match % | match-or-valid % | median codegen s | median build s | median tokens out |
|---|---|---|---|---|---|---|---|
| b123d | 40 | 77.5 | 10.0 | 25.0 | 96.83 | 2.29 | 326.5 |
| b123d-nofs | 40 | 80.0 | 17.5 | 30.0 | 81.17 | 2.27 | 331.5 |
| cadquery | 40 | 82.5 | 17.5 | 30.0 | 5.46 | 1.76 | 254.5 |
| openscad | 40 | 32.5 | 2.5 | 2.5 | 5.6 | 0.54 | 305.5 |

## Error classes (% of that arm's attempts)

| arm | api_misuse | import_or_name | kernel | no_code | other |
|---|---|---|---|---|---|
| b123d | 17.5 | 2.5 | 2.5 | 0.0 | 0.0 |
| b123d-nofs | 15.0 | 5.0 | 0.0 | 0.0 | 0.0 |
| cadquery | 15.0 | 0.0 | 2.5 | 0.0 | 0.0 |
| openscad | 0.0 | 0.0 | 0.0 | 42.5 | 25.0 |

## Paired comparison vs b123d (same specs only)

| arm | n pairs | match +improved | match -worsened | built +improved | built -worsened |
|---|---|---|---|---|---|
| b123d-nofs | 40 | 4 | 1 | 2 | 1 |
| cadquery | 40 | 4 | 1 | 6 | 4 |
| openscad | 40 | 1 | 4 | 3 | 21 |

## Paired comparison vs b123d, by tier

### b123d-nofs

| tier | n | match +improved | match -worsened |
|---|---|---|---|
| 0 | 40 | 4 | 1 |

### cadquery

| tier | n | match +improved | match -worsened |
|---|---|---|---|
| 0 | 40 | 4 | 1 |

### openscad

| tier | n | match +improved | match -worsened |
|---|---|---|---|
| 0 | 40 | 1 | 4 |

## Top error signatures per arm

### b123d

- (2x) ValueError: Unable to repositioned type <class 'NoneType'> with respect to local coordinates
- (1x) TypeError: Shape.translate() takes 2 positional arguments but 3 were given
- (1x) OCP.OCP.StdFail.StdFail_NotDone: BRep_API: command not done
- (1x) TypeError: Face.extrude() got an unexpected keyword argument 'amount'
- (1x) TypeError: Compound.extrude() got an unexpected keyword argument 'amount'
- (1x) AttributeError: 'Edge' object has no attribute 'pos'
- (1x) NameError: name 'Extrude' is not defined. Did you mean: 'extrude'?
- (1x) TypeError: BuildSketch.__init__() got an unexpected keyword argument 'origin'

### b123d-nofs

- (1x) AttributeError: 'BuildSketch' has no attribute 'extrude'. Did you intend '<BuildSketch>.sketch.extru
- (1x) NameError: name 'LineTo' is not defined. Did you mean: 'Line'?
- (1x) AttributeError: 'Polyline' object has no attribute 'to_edge'
- (1x) TypeError: Compound.extrude() got an unexpected keyword argument 'amount'
- (1x) NameError: name 'make_polygon' is not defined
- (1x) AttributeError: 'Vector' object has no attribute 'x'. Did you mean: 'X'?
- (1x) ValueError: Line requires two pts
- (1x) AttributeError: 'BuildSketch' has no attribute 'polygon'. Did you intend '<BuildSketch>.sketch.polyg

### cadquery

- (2x) AttributeError: 'Workplane' object has no attribute 'arcTo'
- (1x) AttributeError: type object 'Shape' has no attribute 'makePolygon'
- (1x) ValueError: Cannot find a solid on the stack or in the parent chain
- (1x) ValueError: Workplane object must have at least one solid on the stack to union!
- (1x) TypeError: Workplane.center() takes 3 positional arguments but 4 were given
- (1x) OCP.OCP.Standard.Standard_Failure: There are no suitable edges for chamfer or fillet

### openscad

- (17x) no_code: empty model reply
- (5x) Current top level object is empty.
- (3x) Can't parse file '/home/theultimatecunt/.openclaw/skills/cad-builder-phase3/benchmarks/lang-ab/resul
- (2x) Current top level object is not a 3D object.

