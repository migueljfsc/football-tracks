# The pipeline

How a broadcast clip becomes `tracks.json`. Each stage reads the previous stage's artefact
from `work/<clip>/` and writes its own, so any stage can be rerun without the ones before it.

The reasoning behind each choice is in [`decisions/`](./decisions); this describes what runs.

## The three invariants

1. **`tracks.json` is the only contract.** Nothing downstream of it touches video; nothing
   upstream of it knows about Pitchboard's schema. The seam is deliberate — the video half is
   the part most likely to be thrown away and rewritten, and it must be replaceable without
   the TypeScript half noticing.

2. **No pixels cross the boundary.** Positions in `tracks.json` are pitch metres, origin at
   the top-left corner of a 105 × 68 pitch — the same space and the same origin Pitchboard's
   `BoardDoc` uses. Image coordinates stay inside the stage that produced them.

3. **Every stage writes an artefact, and every stage has a picture.** A homography cannot be
   unit-tested; you look at it. Each stage drops its output in `work/<clip>/` and can render
   an overlay or a top-down video proving it did what it claims. A stage with no visual check
   is a stage you cannot debug.

## The pipeline

Each stage reads the previous stage's artefact from `work/<clip>/` and writes its own. They
run independently, so a failing stage 3 is re-run without redoing stage 1.

### Stage 0 — segment `segments.json`

Broadcast cuts constantly: replays, close-ups, crowd, dugout. Everything downstream assumes
one continuous view of the pitch from one camera, so the first job is to find that view and
throw away the rest.

Shot-boundary detection, then score each segment on duration, pitch-green fraction and frame-
to-frame motion. The main tactical camera is long, very green, and moves smoothly.

- *Check:* the reported segment boundaries match where the cuts are.
- *Difficulty:* low. This will not be what fails.

### Stage 1 — registration — **the solver works; the detector is being trained**

Per-frame 3 × 3 matrix mapping image pixels to pitch metres. Broadcast pans and zooms, so
this is solved per frame, not once.

The stage splits into a solver (named lines → homography) and a detector (frame → named
lines). **The solver is built and measured.** The detector is a learned segmenter, `calib.py`,
which solves 100% of frames from the picture alone but has not yet hit the accuracy bar — four
training runs in, the best sits at 0.67 m against a 0.5 m bar, and D36 is the whole account of
it. Nothing else in the module
changes when it lands: `calibration.homography` takes named polylines and does not care who
found them.

Until it does, a clip is registered from a human seed (`ft seed`), which works and is measured
below.

Measured on SNGS-147, feeding it the ground-truth lines. `carry` is how many frames a
homography may be propagated across gaps the solver cannot fill:

| carry | coverage | median | p90 | p99 |
|---|---|---|---|---|
| off | 606/750 (80.8%) | 0.67 m | 2.65 m | 12.14 m |
| 50 (default) | 710/750 (94.7%) | 0.83 m | 2.23 m | 10.94 m |
| uncapped | 750/750 (**100%**) | 0.90 m | 2.27 m | 10.36 m |

Carrying buys coverage for a quarter of a metre at the median, and it *improves* both
tails — a carried homography beats the marginal five-line fit that produced them.

That is the **ceiling** for the whole pipeline. It is measured by pushing ground-truth
bounding boxes through the fitted homography and comparing with the position SoccerNet
recorded for that same box, which holds detection and tracking fixed so the number is the
camera model's alone. No detector gets a position closer than this.

- *Check:* **reproject the pitch model back onto the video** (`ft calibrate --frame N`, or
  `--video`). Lines land on lines or they do not. Nothing else in this repo is as easy to
  verify or as easy to get subtly wrong — and it is what caught D16.
- *Difficulty:* the highest of any stage. The solver took the time; the detector is a model
  download and an adapter.

### Stage 2 — detect and track `detections.json`

Person detector per frame, then a tracker to string detections into tracks with stable ids.

The detection is solved and boring. **Identity persistence is the stage that decides whether
this works at all.** Trackers switch ids whenever two players cross, and one switch turns two
runs into two teleports — an error the reduction downstream cannot recover from because it
looks exactly like a fast run.

Mitigations, in order of cheapness: keep clips short (5–10s, fewer crossings), feed team
colour into the association cost, feed a resolved shirt number in as a re-id feature once
stage 5 exists.

- *Check:* boxes and ids overlaid on the video. Count the switches over 10s by eye.
- *Difficulty:* medium, and the risk is concentrated here.

### Stage 3 — teams `teams.json`

Cluster torso crops by colour: two outfield kits, two keepers, referees. Referees are dropped;
keepers are kept and tagged, because Pitchboard wants eleven a side.

- *Check:* crops grouped by assigned cluster, in a contact sheet. Obvious at a glance.
- *Difficulty:* low.

### Stage 4 — project `tracks.json`

Apply stage 1's per-frame homography to stage 2's tracks. Positions become metres. Clamp to
the pitch, drop tracks that spend most of their life outside it (crowd, dugout, cameraman).

**This is the proof.** Render the result as a top-down video of coloured dots. If the dots
move like a football team, the hard part is done and everything after is engineering. If they
jitter, swim or cross the touchline, the fault is upstream in stage 1 or 2 and this stage is
how you find out which.

- *Check:* the top-down dot video.
- *Difficulty:* low in itself; it is the integration test for everything before it.

### Stage 5 — numbers — **tried, and it does not work** (D32)

Best-effort shirt numbers. A player at broadcast 1080p is ~100 px tall and the number ~20 px,
so per-frame OCR is close to useless.

**Vote per track, never per frame.** A 6-second track yields ~150 torso crops. Upscale, OCR
all of them, take a confidence-weighted mode, and require a margin over the runner-up. A 15%
per-frame hit rate still resolves a number confidently. Players who never turn towards the
camera resolve to nothing, and that is the correct answer — they import as generic tokens.
Expect roughly half the squad to resolve. See D5.

- *Check:* resolved number against the crop that voted for it.
- *Difficulty:* medium; the failure mode is silence rather than a wrong answer, which is what
  makes it safe to ship at 70%.

### Not a stage — the ball

Deferred to v1. See D4.
