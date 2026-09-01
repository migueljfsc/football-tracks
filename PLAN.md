# football-tracks — implementation plan

Turn a broadcast football clip into player tracks in pitch metres, so Pitchboard can
import a play instead of the coach drawing it by hand.

## Mission

Input: a few seconds of broadcast footage of one passage of play.
Output: `tracks.json` — every player's position, per frame, in metres on a 105 × 68 pitch,
with a team and, where it can be read, a shirt number.

That file is the whole product. This repo does not know what a `BoardDoc` is, and Pitchboard
does not know what a video is. See D3.

Target accuracy for v0 is **70%** — good enough that a coach fixes the rest in the editor,
which is cheaper than drawing it from nothing. This is a proof, not a product.

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

### Stage 1 — registration `homography.json`

Per-frame 3 × 3 matrix mapping image pixels to pitch metres. Broadcast pans and zooms, so
this is solved per frame, not once.

Automatic first: a pitch-keypoint model detects line intersections, penalty-box corners, the
centre circle, and a solver fits the homography. Roboflow's `sports` package publishes a
keypoint model trained on broadcast football; verify it still works before building anything.

Fallback if that disappoints: seed frame 1 by hand — click four or more pitch landmarks —
and propagate the matrix forward by optical flow. Robust, ~2h of work, but needs a human per
clip. See D7.

- *Check:* **reproject the pitch model back onto the video.** Lines land on lines or they do
  not. Nothing else in this repo is as easy to verify or as easy to get subtly wrong.
- *Difficulty:* the highest of any stage. Budget most of the time here.

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

### Stage 5 — numbers `tracks.json` (annotated)

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

## What SoccerNet turned out to be

Measured, not assumed — `SN-GSR-2025`, split `test`, clip `SNGS-147`.

**It is not gated and needs no NDA password.** GSR-2025 is served from HuggingFace
ungated. The password matters only for the tasks still on SoccerNet's own mirror
(`tracking`, `calibration`). It stays in `.env` for those; nothing in the GSR path reads it.

**Clips are JPEG frames, not video.** 750 frames at 25 fps — 30 s — 1920 × 1080, one
continuous camera, already trimmed. **So stage 0 does not apply to SoccerNet at all: the
SoccerNet path enters at stage 1.** Stage 0 earns its place for arbitrary broadcast
footage, which is still where this has to work in the end.

**One clip costs ~150 MB, not 8.85 GB.** A zip's index sits at its end and HuggingFace
serves range requests, so the split's zip is opened remotely and a single clip read out
of it. `ft clips` lists 49 clips in `test` without downloading anything.

**The labels answer every stage at once:**

| field | what it grounds |
|---|---|
| pitch-line annotations, per frame | stage 1 |
| `track_id` | stage 2 |
| `attributes.role`, `attributes.team` | stage 3 |
| `bbox_pitch.{x,y}_bottom_middle` | stage 4 |
| `attributes.jersey` | stage 5 |

Two things fell out of reading it that change what we expect:

**Their origin is the centre spot, ours is the top-left corner.** `x + 52.5`, `y + 34`.
The open question in the first draft of this plan is answered: the conventions do *not*
match. The conversion lives in `soccernet.py` and nowhere else. The y *direction* is still
unverified — a flip mirrors the board, and the top-down render at M4 is what shows it.

**The ground truth has 230-metre outliers.** A homography extrapolates without bound for
anyone near the horizon, so SoccerNet's own positions run to x = −230, y = −430. This is
the same failure stage 1 will have, and it is why `tracks.on_pitch` drops rather than
clamps — see D13.

**The jersey ceiling is about 40%.** Nine of 22 tracks in SNGS-147 carry a shirt number,
and that is *human annotators with the whole clip in front of them*. Stage 5's OCR cannot
beat it and should not be measured as if it could. It also confirms the estimate this plan
started with: expect roughly half the squad, and generic tokens for the rest.

## Milestones

| # | done when | est. |
|---|---|---|
| M0 | scaffold, stage 0, and `ft truth` writing a real `tracks.json` | **done** |
| M1 | reprojected pitch lines sit on the real lines | 2–3 evenings |
| M2 | tracks survive 10s with few enough id switches to count | 1–2 evenings |
| M3 | teams cluster cleanly | 1 evening |
| M4 | **the top-down dot video looks like football** | 1 evening |
| M5 | numbers resolve for ~40% of tracks, matching the label ceiling | 1 evening |
| M6 | Pitchboard's `src/import/` turns `tracks.json` into a `BoardDoc` | 1–2 weeks, other repo |

**M6 no longer waits for anything.** `ft truth` emits a real, correct `tracks.json` from
ground truth with no CV in the loop, so the TypeScript reduction is built against genuine
30-second passages of play rather than hand-written fixtures — and its output can be looked
at in Pitchboard while stage 1 is still failing.

That also inverts how the CV is judged. Every later stage is scored against the same file
in the same format, so "70%" becomes a diff against a known-good baseline rather than a
feeling about a video.

## Decisions

**D1 — sibling repo, not `pitchboard/tools/`.** Pitchboard is strict TypeScript with a pure
engine and tight conventions. A Python venv, model weights and a CUDA-shaped dependency tree
inside it costs more than colocation is worth. The two repos meet at a file.

**D2 — Python pinned to 3.12.** The machine has 3.14. Torch, ultralytics and the OCR stack
all lag new releases by months; starting on 3.14 means an evening of build errors instead of
an evening of CV.

**D3 — `tracks.json` is the contract, and it is frozen before the CV works.** It is the one
artefact two repos share. Fixing it early lets the TypeScript reduction be written in
parallel against fixtures, and it means the entire video pipeline can be replaced — by a
different tracker, by a hand-annotation tool, by a vendor's data — without touching
Pitchboard.

**D4 — no ball in v0.** It is the hardest object in the frame: small, fast, motion-blurred,
occluded by legs, and frequently out of shot. Worse, it is the input to `carrier`, `shot` and
`hiddenRuns`, so its errors do not stay local — they corrupt the meaning of the board rather
than just its geometry. Players first; carriers get set by hand in the editor. `loft` is not
recoverable at all from one camera: a single homography assumes z = 0, so a chip and a ground
pass are the same measurement.

**D5 — shirt numbers are voted per track, never read per frame.** See stage 5. The
consequence that matters: an unresolved number is `null` and imports as a generic token. Never
guess. A wrong number silently attaches a run to the wrong player, which is worse than no name
at all, and the coach cannot see that it happened.

**D6 — SoccerNet first.** It is broadcast footage published with calibration and tracking
ground truth, so "70%" becomes a number you measure rather than a feeling you have about a
video. Arbitrary clips only once the numbers are known.

**D7 — automatic registration first, human-seeded as the fallback.** Automatic is the only
version that scales past a demo, and a published model already exists, so it is worth one
honest attempt. But the fallback is genuinely good — a human clicking four landmarks is more
accurate than any model — and reaching for it is not a failure. The decision to switch belongs
at the end of M1, judged on the reprojection picture.

**D8 — track samples are sparse.** A player occluded for twenty frames has no position for
those frames, and inventing one is a lie the reduction would fit a bezier to. Every sample
carries its own frame index; gaps are expected and the consumer interpolates or declines to.

**D9 — Ultralytics YOLO is AGPL-3.0.** Fine for a local proof that ships nothing. A problem
the moment this graduates into a product. RT-DETR and RF-DETR are Apache-licensed and are the
swap to make if that day comes — which is another reason the detector lives behind a stage
boundary and not in the middle of everything.

**D10 — SoccerNet enters at stage 1.** Its clips are single-camera JPEG sequences with no
cuts in them, so stage 0 has nothing to do on this path. Stage 0 is not dead code: it is
what the arbitrary-broadcast path needs, and that is the case this has to work on
eventually. But it must not sit in SoccerNet's way, and no later stage may assume it ran.

**D11 — a clip is fetched by range request, never a split download.** The zip index is at
the end of the file and HuggingFace serves ranges, so one clip costs ~150 MB against the
split's 8.85 GB. Members are read in header order rather than name order, which turns 750
random reads into a near-sequential scan.

**D12 — ground truth is written through the same writer as the CV path.** `ft truth` builds
`Track` objects and hands them to `tracks.write`, exactly as stage 4 will. Two consequences,
both wanted: the contract is exercised by real data before any vision exists, and every
later stage is scored by diffing two files of the same shape. A separate ground-truth
format would have made the comparison a translation exercise, which is where a scoring
harness quietly starts measuring itself.

**D13 — an off-pitch position is dropped, never clamped.** A position 200 m out is not a
player near the touchline, it is a homography that failed. Clamping launders that failure
into a plausible coordinate the reduction then fits a curve through, and the resulting run
looks like a real one. Rejecting loses a sample; clamping invents one. The guard is
`tracks.on_pitch`, so the CV path and the ground-truth path cannot disagree about what
counts as credible.

## Non-goals

Real-time. Multi-camera. Player identity across clips. Event detection (tackles, fouls).
Ball height. Anything that requires a GPU bigger than the laptop. A web service — this runs
locally, from a terminal, on files.

## Open questions

- Does SoccerNet's +y run the same way as Pitchboard's? The origin question is answered
  (centre spot vs top-left corner, handled in `soccernet.py`), but a sign flip mirrors the
  board and only the M4 render will show it. One constant either way.
- How short is short enough for stage 2? Measure id switches against clip length rather than
  guessing at 10s.
- Does the keypoint model cope with a half-pitch framing, or only wide shots?
