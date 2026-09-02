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
