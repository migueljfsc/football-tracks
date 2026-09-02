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
