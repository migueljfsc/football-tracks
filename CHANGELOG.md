## v0.11.0 (2026-09-04)

### Feat

- **ball**: believe a ball placed on a restart spot

### Fix

- **ci**: type-check cleanly with and without the torch extra

## v0.10.0 (2026-09-04)

### Feat

- **pipeline**: make the board match the footage, not the metrics

## v0.9.0 (2026-09-03)

### Feat

- **calib**: train at 960x540, and make a long run resumable

### Fix

- **calib**: hold out the matches the benchmark clips come from

## v0.8.0 (2026-09-03)

### Feat

- **calib**: learn the pitch markings, so a fit needs no seed

## v0.7.0 (2026-09-03)

### Feat

- **bench**: one command, every clip, one table

## v0.6.0 (2026-09-03)

### Feat

- **refine**: snap the camera model onto the painted lines, off by default

## v0.5.0 (2026-09-03)

### Feat

- **tracks**: store one position per tenth of a second
- **seed**: anchor on every clicked frame, and refuse one that folds

## v0.4.2 (2026-09-03)

### Fix

- **calibrate**: measure drift where the camera is looking

## v0.4.1 (2026-09-02)

### Fix

- **calibrate**: score a carry against evidence, not against the carry

## v0.4.0 (2026-09-02)

### Feat

- **seed**: a panel behind the instructions, and markings drawn at their extent

## v0.3.8 (2026-09-02)

### Fix

- one bad click was dragging every other one with it

## v0.3.7 (2026-09-02)

### Fix

- crop chrome out of the picture, and leave the ball selection alone

## v0.3.6 (2026-09-02)

### Fix

- the crop was keeping 290 columns of black bar

## v0.3.5 (2026-09-02)

### Refactor

- remove dead code, and give the drift measurement a caller

## v0.3.4 (2026-09-02)

### Fix

- **ci**: a green local run is not a green CI

## v0.3.3 (2026-09-02)

### Fix

- tell the two kits apart properly, and stop the keeper poisoning it

## v0.3.2 (2026-09-02)

### Fix

- **ci**: the type checker needed a two-gigabyte GPU stack to look at a dataclass

## v0.3.1 (2026-09-02)

### Perf

- give up on a lost player quickly

## v0.3.0 (2026-09-02)

### Feat

- find the ball, for the one question a board asks of it

## v0.2.0 (2026-09-02)

### Feat

- the pipeline runs end to end on real broadcast television
- trace lines as well as click points when seeding
- seed a clip from its best frame, not its first
- ingest a real broadcast clip, and seed it by hand
- the automatic path, and the number it was built to produce
- carry a homography across gaps by tracking the ground plane
- stage 1 registration — pixels to pitch metres
- render and score, completing the ground-truth path
- soccernet fetch and ground-truth tracks.json
- pipeline scaffold and stage 0 segmentation

### Fix

- judge who is a PLAYER by a tighter margin than who is in a credible place
- decouple tracking from the homography, and correct D19

### Perf

- replace the detector with RT-DETR
- drop off-pitch people before tracking, not after
