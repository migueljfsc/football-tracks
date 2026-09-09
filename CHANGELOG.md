## v0.23.0 (2026-09-09)

### Feat

- **calibrate**: say where the camera model is weakest, and what to click

## v0.22.1 (2026-09-09)

### Fix

- **seed**: stamp a seed with the picture it was clicked on

## v0.22.0 (2026-09-09)

### Feat

- **teams**: carry the kits the clip was played in

### Fix

- **tracking**: refuse an association whose kits plainly disagree

## v0.21.4 (2026-09-07)

### Fix

- **video**: a new clip must not inherit the old one's frames or caches

## v0.21.3 (2026-09-07)

### Fix

- **stitch**: read a velocity over seconds of track, not a count of samples

## v0.21.2 (2026-09-07)

### Fix

- **deps**: the vision extra was missing what RT-DETR itself imports

## v0.21.1 (2026-09-07)

### Fix

- **seed**: pressing d then t no longer crashes the click tool

## v0.21.0 (2026-09-07)

### Feat

- **seed**: move the diagram out of the way, or hide it

## v0.20.3 (2026-09-07)

### Fix

- **seed**: name the marking that is labelled with the wrong side

## v0.20.2 (2026-09-07)

### Fix

- **seed**: let a midfield view be traced, and refuse the fit it cannot support

## v0.20.1 (2026-09-07)

### Fix

- **ball**: assert the ball at 0.75, measured on the passes it draws

## v0.20.0 (2026-09-07)

### Feat

- **teams**: decline a side the kit does not settle, rather than flip a coin

## v0.19.0 (2026-09-07)

### Feat

- **bench**: judge a camera model where the players are, not at the probe points

## v0.18.0 (2026-09-07)

### Feat

- **stitch**: join a fragment to where the run was going, not to whatever is in reach

## v0.17.0 (2026-09-06)

### Feat

- **calib**: anchor a carried chain on the segmenter, a twentieth at a time

### Fix

- **schema**: declare source.intervalS, and check the source keys

## v0.16.0 (2026-09-06)

### Feat

- **teams**: name the officials rather than clustering them onto a side

## v0.15.0 (2026-09-05)

### Feat

- **teams**: cluster on the whole track's kit, not the tracker's rolling average

## v0.14.1 (2026-09-05)

### Fix

- **teams**: fall back to the least crowded split, not the best-scoring one

## v0.14.0 (2026-09-05)

### Feat

- **teams**: refuse a split that fields more players than a pitch holds

## v0.13.0 (2026-09-05)

### Feat

- **calib**: drop a fit that contradicts the frames before it

## v0.12.0 (2026-09-04)

### Feat

- **ball**: score the ball, and judge a still one by where it stands

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
