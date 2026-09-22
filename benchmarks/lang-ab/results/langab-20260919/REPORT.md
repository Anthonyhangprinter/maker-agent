# lang-ab report

Run: langab-20260919  
Specs: 40  Seed: 20260919  Arms: b123d, b123d-nofs, cadquery, openscad

## Per-arm summary

| arm | n | built % | match % | match-or-valid % | median codegen s | median build s | median tokens out |
|---|---|---|---|---|---|---|---|
| b123d | 40 | 85.0 | 15.0 | 32.5 | 18.26 | 2.22 | 410.0 |
| b123d-nofs | 40 | 87.5 | 15.0 | 25.0 | 16.14 | 2.21 | 407.5 |
| cadquery | 40 | 52.5 | 12.5 | 20.0 | 15.77 | 1.75 | 421.5 |
| openscad | 40 | 100.0 | 17.5 | 32.5 | 15.56 | 0.69 | 423.0 |

## Error classes (% of that arm's attempts)

| arm | api_misuse | import_or_name | kernel | other |
|---|---|---|---|---|
| b123d | 5.0 | 2.5 | 2.5 | 5.0 |
| b123d-nofs | 2.5 | 7.5 | 2.5 | 0.0 |
| cadquery | 42.5 | 0.0 | 5.0 | 0.0 |
| openscad | 0.0 | 0.0 | 0.0 | 0.0 |

## Paired comparison vs b123d (same specs only)

| arm | n pairs | match +improved | match -worsened | built +improved | built -worsened |
|---|---|---|---|---|---|
| b123d-nofs | 40 | 2 | 2 | 3 | 2 |
| cadquery | 40 | 1 | 2 | 2 | 15 |
| openscad | 40 | 4 | 3 | 6 | 0 |

## Paired comparison vs b123d, by tier

### b123d-nofs

| tier | n | match +improved | match -worsened |
|---|---|---|---|
| 0 | 40 | 2 | 2 |

### cadquery

| tier | n | match +improved | match -worsened |
|---|---|---|---|
| 0 | 40 | 1 | 2 |

### openscad

| tier | n | match +improved | match -worsened |
|---|---|---|---|
| 0 | 40 | 4 | 3 |

## Top error signatures per arm

### b123d

- (1x) Fix: add a final line  result = arm  (the FINISHED part, after all boolean cuts/joins).
- (1x) RuntimeError: BuildSketch doesn't have a Polyline object or operation (Polyline applies to ['BuildLi
- (1x) AttributeError: 'Vector' object has no attribute 'x'. Did you mean: 'X'?
- (1x) ValueError: Failed creating a fillet with radius of 0.05, try a smaller value or use max_fillet() to
- (1x) NameError: name 'Rect' is not defined
- (1x) TypeError: fillet() got multiple values for argument 'radius'

### b123d-nofs

- (1x) NameError: name 'tan' is not defined
- (1x) NameError: name 'atan2' is not defined
- (1x) AttributeError: type object 'SortBy' has no attribute 'X'
- (1x) NameError: name 'scaled' is not defined. Did you mean: 'scale'?
- (1x) ValueError: Failed creating a fillet with radius of 0.05, try a smaller value or use max_fillet() to

### cadquery

- (3x) ValueError: Cannot find a solid on the stack or in the parent chain
- (2x) AttributeError: 'Workplane' object has no attribute 'face'. Did you mean: 'faces'?
- (2x) AttributeError: 'Workplane' object has no attribute 'scale'
- (2x) ValueError: More than one wire or face is required
- (1x) AttributeError: 'BoundBox' object has no attribute 'x0'
- (1x) ValueError: No pending wires present
- (1x) TypeError: Workplane.rect() got an unexpected keyword argument 'fillet'
- (1x) TypeError: Workplane.rect() got an unexpected keyword argument 'center'
- (1x) TypeError: Multiplied(): incompatible function arguments. The following argument types are supported
- (1x) TypeError: Workplane.translate() takes 2 positional arguments but 3 were given

### openscad

(no failures)

