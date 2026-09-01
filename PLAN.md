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

### Stage 1 — registration — **the solver works; the detector is next**

Per-frame 3 × 3 matrix mapping image pixels to pitch metres. Broadcast pans and zooms, so
this is solved per frame, not once.

The stage splits into a solver (named lines → homography) and a detector (frame → named
lines). **The solver is built and measured**; the detector is still ground truth, and
swapping in a keypoint model is what remains. Nothing else in the module changes when it is:
`calibration.homography` takes named polylines and does not care who found them.

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
match. The conversion lives in `soccernet.py` and nowhere else.

**The y direction is measured, and it needs no flip.** Reading frame 1 of SNGS-147 against
its own labels, pitch y and image y move together — keeper 5.5 → 538 px, ball 8.1 → 595 px,
outfielders ~20 → ~1050 px. So higher SoccerNet y is nearer the camera, and since both axes
map with a positive scale the handedness survives: the top-down render is the pitch seen
from above with the broadcast camera at the bottom, not its mirror.

**The ground truth has 230-metre outliers.** A homography extrapolates without bound for
anyone near the horizon, so SoccerNet's own positions run to x = −230, y = −430. This is
the same failure stage 1 will have, and it is why `tracks.on_pitch` drops rather than
clamps — see D13.

**The jersey ceiling is about 40%.** Nine of 22 tracks in SNGS-147 carry a shirt number,
and that is *human annotators with the whole clip in front of them*. Stage 5's OCR cannot
beat it and should not be measured as if it could. It also confirms the estimate this plan
started with: expect roughly half the squad, and generic tokens for the rest.

### The ground-truth path is complete

Three commands, and between them they close invariant 3 and D12 for this stage:

* `ft truth` writes `work/<clip>/truth.json` — real positions, teams and numbers, no CV.
* `ft render` draws any tracks-format file as a top-down video of coloured dots. Not
  specific to ground truth: it is stage 4's proof reused, and it is how the y direction
  above was settled.
* `ft score` diffs a prediction against the truth — recall, precision, position error,
  team accuracy, identity purity, switches, and the jersey breakdown.

Scoring `truth.json` against itself returns a perfect card, which is the only check that
the harness measures what it claims. Scoring a deliberately degraded copy (12% of samples
dropped, 0.6 m of gaussian noise) returns 87.2% recall, 0.70 m median error and 110
switches — the numbers the noise implies.

### The automatic path, measured

`ft detect` then `ft auto --mode seed` runs the whole pipeline with **only frame one's
pitch lines** — everything a human clicking four corners once would give it — and `ft score`
diffs the result against ground truth. On SNGS-147:

| clip | recall | precision | error | purity | switches |
|---|---|---|---|---|---|
| 3 s | 96.6% | 80.8% | 0.60 m | 92.4% | 1 |
| 5 s | 97.2% | 79.5% | 0.69 m | 87.0% | 1 |
| **7 s** | **97.0%** | **74.4%** | **0.80 m** | **86.2%** | **2** |
| 10 s | 45.3% | 44.5% | 0.89 m | 88.0% | 8 |
| 30 s | 39.0% | 36.9% | 1.34 m | 76.7% | 67 |

**Up to about seven seconds, one seed is enough.** Past that the carried homography reaches
where drift turns sharp (D18 measured the corners going at frame ~150–200) and recall halves.
Seven seconds is a goal, a build-up, a press — the length this is for.

What does NOT work yet, and is not hidden by those numbers: shirt numbers resolve zero of
nine, exactly as predicted; precision sits at 74% because referees and touchline staff are
tracked as players; and the team split is near chance, because a fragmented track carries too
little colour to cluster on.

## Milestones

| # | done when | est. |
|---|---|---|
| M0 | scaffold, stage 0, and the ground-truth path: `ft truth`, `ft render`, `ft score` | **done** |
| M1 | reprojected pitch lines sit on the real lines | **solver done**; detector next |
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

**D19 — stage 1's accuracy is not what stage 2 needs.** Carried homographies score WELL on
stage 1's own metric — 0.90 m median, better tails than the solver alone — and they wreck
tracking: identity purity 62.9% with carrying on against 82.9% with it off, 145 switches
against 25.

The reason is that the two stages want different things from the same matrix. Stage 1 is
scored on absolute accuracy per frame. Tracking does not care where the pitch is, it cares
that it does not MOVE: a homography smoothly one metre off tracks perfectly, while one that
jitters two metres between frames throws every player at once and every track breaks
together. Frame-to-frame consistency is the property, and no number in `Registration`
measures it.

**Fixed, and the fix corrected the diagnosis.** Association moved to stabilised image space,
using the frame-to-frame transform, which is measured per pair and never accumulated.
Stage 2 now produces the *identical* 128 raw tracks whether carrying is on or off, so it is
genuinely independent of stage 1.

The score gap survived that (77.9% purity with carrying off against 66.2% on), which says
the original reading was half wrong. Carried homographies no longer damage the TRACKS — they
damage the POSITIONS, and a sample placed five metres out is matched to a different player
and counted as an identity error. What remains is a registration problem wearing a tracking
problem's clothes. `--carry 0` is still the better setting, for that reason and not the one
first recorded here.

**D25 — a seed propagates in BOTH directions.** A clip is rarely best seeded at its first
frame: the camera is often still finding the play, and on the Rio Ave clip frame 1 has the
goal half out of shot with almost no markings visible while frame 100 has the whole box.
Forward-only propagation would make the good frame useless. `fill` therefore runs a backward
pass as well, composing `h @ d` rather than `h @ inv(d)`.

**D23 — the seed is a file, not a UI.** `seed.json` holds clicked landmark
correspondences and nothing else. The click tool writes it, but so could a keypoint
model, and so could Pitchboard's own import view — which is where this ends up, so the
format is the interface and the OpenCV window is disposable.

**D26 — a human can TRACE a line as well as click a point.** A corner is one exact pixel
and is often out of shot; a long marking is easy to follow and says nearly as much once
several points are stacked. `calibration.fit` takes both in one DLT — a landmark
contributes two equations, a traced point one.

This came out of the first real clip. A tight goalmouth shot at night has faint, short box
lines and its corners off screen, and clicking them produced a seed whose points were
misidentified. The two long clear markings — the goal line and the penalty-box front — were
easy to trace and were there all along.

Two traps that come with it, both of which fit with ZERO residual and are therefore the most
convincing way to be wrong. **Two traced lines are always degenerate**: they cross
somewhere, and a homography sending the entire image to that crossing satisfies every
point-on-line constraint exactly. And **lines all running the same way pin down nothing**
about the direction across them. `_collapses` catches the first by checking that the fitted
map still covers ground; `_spans_two_directions` catches the second.

**D24 — four clicked landmarks are not enough, and the obvious four are degenerate.**
Both goalposts and both corners of a goal are the most natural things to click and ALL
FOUR SIT ON x = 0. A homography fitted to collinear points fits perfectly and describes
nothing, so `seed.homography` refuses it — D17's argument reaching the human.

The misclick threshold is in pitch METRES, not pixels, because that is the destination
space; the usual pixel default of 5 would be a five-metre tolerance. It is 0.5 rather
than 1.0 because with six points a homography has barely more constraints than degrees
of freedom, so at a loose threshold RANSAC prefers a warped fit that swallows a bad
click over one that rejects it. Measured on a deliberate 8 m misclick: 0.5 rejects it,
1.0 absorbs it and moves the centre spot sixteen metres.

**D22 — stage 2 does not depend on stage 1.** The gate is a speed — a footballer covers at
most MAX_SPEED metres in a second — and it reaches pixels through the only local scale that
needs no camera model: a detection box is about 1.8 m tall, so it says how many pixels a
metre is right there. Camera motion is removed with the frame-to-frame transform rather than
by projecting to the pitch. Both halves matter: raw pixels lose a panning camera, and pitch
metres inherit every wobble in the homography.

**D20 — the tracker is ours.** supervision's ByteTrack is deprecated and disappears in 0.31,
and this regime has a signal a general tracker does not use: a team wears one colour, which
is exactly what tells two crossing players apart. Association is greedy over a gate, in
metres, so the gate is a physical claim about how far a footballer runs rather than a claim
about how fast the camera pans.

**D21 — which cluster is "home" is decided by which end the side plays at**, never by
whichever labelling scores best — that would be fitting to the yardstick. It is a weak
discriminator over a long clip, where both sides cover the same ground, so `score` reports
the team split permutation-invariantly: the question worth measuring is whether the sides
were told apart, not whether they got SoccerNet's names.

**D18 — a homography is carried across gaps by tracking the ground plane.** Stage 1's
solver needs enough markings in shot, and real footage often has fewer. Features on the
grass move between consecutive frames by exactly the transform the camera's motion
induces, so tracking them gives a frame-to-frame matrix that composes with a known
homography to give the next one. It takes coverage on SNGS-147 from 80.8% to 100%.

Two properties, both measured rather than assumed. It **drifts**: every composition
multiplies in the last one's error. Carrying from frame 1 of SNGS-147, the pitch corners
stay inside 0.19 m after 10 frames, 1.39 m after 100 and 1.63 m after 120, then degrade
sharply. And it **cannot start itself** — something must supply the first homography,
which is the solver, or a keypoint model, or a human clicking four corners (D7).

Hence `DEFAULT_MAX_CARRY = 50`, two seconds, well inside where the measurement says drift
is still small. Uncapped happens to be fine on this clip because its gaps are short, but a
badly drifted matrix produces confident wrong positions and that is worse than a gap —
D13's argument once more. Features come only from the grass, so players and crowd, which
do not move with the ground plane, are never fed in.

**D16 — the homography is fitted from POINT-ON-LINE constraints, not from line
intersections.** Every annotated point is known to lie on a named pitch line, which gives
one linear equation `l · (H p) = 0`; stacking them is an ordinary DLT.

Intersections were the first approach and they fail on exactly the footage that matters.
Under an oblique camera two pitch lines meeting at a right angle project to nearly parallel
image lines, so their crossing flies off and a pixel of error becomes tens of metres. On
SNGS-147's most line-rich frame it produced four usable correspondences out of nine lines,
three of them on the same touchline — a degenerate configuration that fitted its own points
with **0.00 m residual** while placing players 100 m away. The residual could not see it;
only the reprojection picture could. Median error across the clip was 13 m.

Point-on-line uses every point of every visible marking, so a line seen edge-on contributes
what it can instead of being thrown away or, worse, being crossed with its neighbour.

**D17 — a frame must show MORE lines than the fit strictly needs.** Four lines is eight
constraints for eight degrees of freedom: exactly determined, fits perfectly whatever the
noise, and leaves nothing over to notice it is wrong with. Measured, four-line frames land
21.7 m out at the median while every over-determined configuration is inside 3 m. Refusing
them costs 7% of coverage and takes p90 from 19.1 m to 2.65 m.

This is D13's argument again one level up: a homography that cannot be checked is not a
cheaper homography, it is a wrong answer nobody can see.

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

**D14 — ground truth is `truth.json`, a prediction is `tracks.json`.** The same format
written by the same writer, given two names so that a stage cannot overwrite the yardstick
it is about to be measured against. `ft score` defaults to the `truth.json` beside its
argument.

**D15 — the scorecard separates the failures the averages hide.** Recall cannot see an
identity switch: a tracker that swaps two players halfway through still finds everybody, so
purity and the switch count are reported beside it. And a wrong shirt number is reported
apart from an unread one, because they are not the same mistake — unread imports as a
generic token and costs nothing, wrong attaches a run to the wrong player where no one
downstream can see it (D5). A single "jersey accuracy" percentage would average the free
error together with the expensive one.

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

- How short is short enough for stage 2? Measure id switches against clip length rather than
  guessing at 10s.
- Does the keypoint model cope with a half-pitch framing, or only wide shots?
