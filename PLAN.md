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

### What fragmentation turned out to be

A raw track count overstates it. On the Rio Ave clip, 50 tracks sounds like a dozen players
shattered — but 13 of them cover more than half the clip, which is about how many players are
in shot, and 17 are brief fragments any reduction can drop. Over 30 seconds it is genuinely
bad (2 tracks over half the clip, 69 under a tenth); over 7 seconds it is not.

The stage is worth measuring by how much of a player's life its best track covers, not by how
many tracks exist.

### Do the constants hold on clips they were not tuned on?

Every number here — the track age, the coverage floor, the carrier radius, the player
margin, the line minimums — was chosen against SNGS-147. One clip. So two more were
fetched and run untouched, both deliberately harder: a corner and a yellow card, at 13.5
and 13.3 players a frame against SNGS-147's 7.0.

| clip | tracks | recall | precision | error | purity | teams |
|---|---|---|---|---|---|---|
| SNGS-147 *(tuned on)* | 52 | 53.5% | 86.4% | 0.62 m | 80.5% | 87% |
| SNGS-116 | 58 | 59.2% | 83.4% | 0.42 m | **60.1%** | 78% |
| SNGS-121 | 38 | 43.4% | 85.0% | 0.53 m | 78.3% | 87% |

**Most of it generalises.** Precision holds within three points, position error is
actually BETTER on the clips nothing was fitted to, and the team split holds. The
detector, the off-pitch margin, the camera model and the kit split are not fitted to one
clip.

**Identity is the exception, and it fails by a different mechanism.** On the corner,
purity falls to 60.1%, and 93 of its 111 steals are between players BOTH VISIBLE at the
time — not after a gap, which was SNGS-147's problem and is fixed. Players there stand
2.13 m apart at the fifth percentile against 3.58 m on SNGS-147.

**And colour still does not fix it, which was worth finding out properly.** Weighting it
from 0.6 up to 5.0 moves purity on the corner by less than a point (60.1, 59.5, 60.5,
59.7). The reason is that 18% of boxes there overlap another by more than 30%, so the
torso crop contains two players and the kit signature blends — **appearance is least
reliable exactly when it is most needed.** A cue that fails in the case it exists for
cannot be tuned into working.

Crowded-scene identity is therefore a stated limitation, not an open task. Fixing it
needs an appearance model robust to partial occlusion — a learned re-identification
embedding — which is a real project of the same size as jersey OCR (D32). Turning up a
weight is not.

One methodological lesson worth keeping: **sweeping a constant on a clip where its
failure does not occur measures nothing.** Colour looked useless for most of this project
because it was swept on SNGS-147, whose steals happen after gaps where the right player
is simply absent and no colour could have helped.

### The automatic path, measured

`ft detect` then `ft auto --mode seed` runs the whole pipeline with **only frame one's
pitch lines** — everything a human clicking four corners once would give it — and `ft score`
diffs the result against ground truth. On SNGS-147:

| clip | recall | precision | error | purity | switches |
|---|---|---|---|---|---|
| 3 s | 95.1% | 78.6% | 0.57 m | 92.6% | 1 |
| 5 s | 96.3% | 81.2% | 0.68 m | 87.3% | 1 |
| **7 s** | **96.7%** | **83.1%** | **0.77 m** | **86.5%** | **2** |
| 10 s | 45.3% | 44.5% | 0.89 m | 88.0% | 8 |
| 30 s | 39.0% | 36.9% | 1.34 m | 76.7% | 67 |

**Up to about seven seconds, one seed is enough.** Past that the carried homography reaches
where drift turns sharp (D18 measured the corners going at frame ~150–200) and recall halves.
Seven seconds is a goal, a build-up, a press — the length this is for.

What does NOT work yet, and is not hidden by those numbers: shirt numbers resolve zero of
nine, exactly as predicted; precision sits at 74% because referees and touchline staff are
tracked as players; and the team split is near chance, because a fragmented track carries too
little colour to cluster on.

### It works on real broadcast footage

The whole point, finally tested. A sport.tv recording of a Rio Ave goal: screen-captured
and pillarboxed, 32 fps behind a container claiming 120, a night match with washed-out
markings, and no annotations of any kind. One seeded frame — nine landmarks and 28 traced
points along three lines — and:

```
3161 detections -> 52 tracks
frames solved   208/208
```

Judged objectively rather than by eye: project every DETECTED pitch marking through the
fitted camera and measure how far it lands from the real line it belongs to. Median
**0.11 m**, 78% inside half a metre, and not one pixel thrown off the pitch. The top-down
render puts ten players around the penalty area with the keeper on his line, which is what
the frame shows.

What that does not yet mean: 52 tracks for about a dozen people is heavy fragmentation, the
team split is unproven here, and there are no shirt numbers. The camera is solved; the rest
of the pipeline is where the remaining error is.

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

**D29 — the ball is found for ONE question: who has it.** Pitchboard models the ball as
`scene.carrier` and nothing else, which turns the intractable problem into an easy one.
Where the ball IS cannot be recovered — a ground homography assumes z = 0, so a ball in
flight lands metres from the truth — but who is NEAREST it can be, and that is the whole
question a board asks.

Measured against SoccerNet's own ball annotations, the nearest player is the right one
**99% of the time** within four metres, answering about half the frames and declining on
the rest. Declining is the point: a ball in flight belongs to nobody.

Two more things tried and rejected, both measured against SoccerNet's own ball. **Filtering
candidates by SIZE** — a 22 cm ball has a predictable apparent width at any point once the
camera is known — does not separate them: true sightings come in at 2.00x the predicted
width and false ones at 1.69x, and the tightest useful band keeps 74% of real balls while
still admitting 49% of the impostors. **Choosing the most confident candidate that lands ON
the pitch**, rather than the most confident anywhere, answers more frames and gets more of
them wrong: 630 frames against 589, of which 17 of the 41 extra answers are wrong, because
a weak false positive on the grass then wins a frame the ball was not in. Abstaining is the
better trade (D5).

Two things that had to be right. The detector reports about five "sports balls" a frame —
a head, a boot, a patch of hoarding — so the most confident one is taken and then MEDIAN
FILTERED over its neighbours, because what separates the real ball from the impostors is
that it moves smoothly. And picking the candidate NEAREST A PLAYER instead scores far
worse (55% against 99%), which is worth stating plainly: with five candidates a frame,
"nearest a player" reliably selects whichever false positive is standing beside somebody.

This supersedes D4's deferral. D4 was right that ball POSITION is the hardest thing in the
frame and would corrupt whatever it touched. It was wrong that the ball was therefore
out of reach, because it never asked the smaller question.

**D4 — no ball in v0 (superseded by D29).** It is the hardest object in the frame: small, fast, motion-blurred,
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

**D28 — the detector is RT-DETR, and the first one was deliberately a floor.**
torchvision's Faster R-CNN was chosen because it was BSD, already a dependency, and
certain to be beatable — so every number taken with it was a lower bound rather than a
best case. Once the pipeline was measurable enough to compare fairly, it was replaced:

    Faster R-CNN  conf 0.50   83.6% recall   1.6 spurious per frame   0.26 s/frame
    RT-DETR       conf 0.50   86.2% recall   0.5 spurious per frame   0.14 s/frame

Better on all three, and Apache-2.0, so the licence story stays clean (D9).

**The false-positive column matters as much as recall**, which is why the confidence
floor stays at 0.5 rather than dropping to 0.4 for four more points of recall. Everything
the detector invents competes for associations and spawns tracks. On the Rio Ave clip the
swap took 50 tracks to 35 and, more to the point, fragments covering under a tenth of the
clip from 17 to 5 — the twelve longest now run 205, 196, 185 and 183 frames out of 208.

Two things that did NOT lift recall and are not worth retrying: a larger input image (800
against 1333 changed nothing, because the misses are occlusions rather than small
players), and a lower confidence floor, which buys recall at about three spurious boxes
per real one.

**D33 — a carry can only be scored against evidence it did not produce, and on a
broadcast clip that evidence has to be clicked.** `ft calibrate --drift-from` was handed
the homographies the pipeline runs on. On a SoccerNet clip those are per-frame fits and
the measurement was roughly right; on a seeded clip every one of them IS the carry, so
the measurement compared the chain with itself and reported

    carried   corner error
         1f         0.00 m
        50f         0.00 m

for a homography whose reprojection at frame 903 visibly misses the painted lines it sits
on exactly at the seeded frame 853. Zero error is not a result a carry can produce, and it
read as the best possible one.

`direct` now holds only what was fitted from evidence — every frame on a labelled clip,
the one clicked frame on a seeded one — and drift is scored against that. Where there is
nothing to score against, the command refuses and says what would fix it rather than
printing a number. `ft seed <clip> --frame N --check` writes `seed.<frame>.json` as that
second piece of evidence without replacing the seed the pipeline runs from.

Fixing that exposed the second half of the same mistake. The error was measured at the
PITCH CORNERS, which are fixed points of the model and not of the picture. On the first
broadcast clip measured this way three of the four fall outside the frame - one of them
58,717 px out on a 2,774 px frame - so the number returned was the extrapolation error
twenty pitch-lengths beyond anything the camera saw:

    at the pitch corners                     25.44 m
    at the ten players actually detected      0.95 m median, 1.49 m max
    across the visible lower frame            1.76 m median

25.44 m was as wrong as 0.00 m had been, in the other direction, and both would have been
believed. `observed_error` probes a grid on the IMAGE and keeps the probes the true model
puts on grass, which is the question the pipeline actually has. It cross-checks against the
independent per-player measurement above at 1.74 m.

The corner metric had been flattering nothing and inflating everything, SoccerNet included.
The honest curves:

    SNGS-116 (wide, fixed camera)   1f 0.00   25f 0.12   50f 0.22   200f 0.94 m
    nottingham (broadcast, tight)                        50f 1.74 m

So DEFAULT_MAX_CARRY = 50 was derived from a number that was wrong by an order of
magnitude, and the cap it produced happens to be defensible for a different reason than
the one recorded: a fixed SoccerNet camera tolerates 200 frames comfortably, while
broadcast footage is already at 1.74 m by 50. There is no single right cap across footage
types, and 50 is a reasonable middle rather than a measured optimum. Bounding the
BACKWARD carry needs a second check seed early in the segment; the board built from this
clip is carried up to 456 frames back from its seed, and that distance is unmeasured.

**D32 — shirt-number OCR does not work on this footage, and the failures are confident.**
Measured on SNGS-147 with easyocr over the largest thirty sightings of each track, voting
by summed confidence exactly as this plan proposed:

    #20 -> 1   WRONG        #9  -> 9   RIGHT
    #23 -> 0   WRONG        #44 -> no answer
    #27 -> 1   WRONG        #4  -> no answer
    #50 -> 0   WRONG        #5  -> no answer
                            #14 -> no answer

One right, four wrong, four silent. Per crop: 3 right against 12 wrong.

The failure is not noise, which voting would survive. It reads ONE DIGIT out of a
two-digit number and is sure about it: `#20` scored `1` at 0.97 against `2` at 0.44. A
margin test does not save that, because the wrong answer wins by a mile.

So the number would be wrong four times as often as right, and D5 says plainly which way
that trade goes: an unread number imports as a generic token and costs nothing, a wrong
one attaches a run to the wrong player where nobody downstream can see it. Numbers stay
null.

Two things would have to change before this is worth revisiting: a detector-side crop
that finds the NUMBER rather than the middle of a torso, and a recogniser trained on
jersey digits rather than on printed text. Both are real projects. Trying a different
general-purpose OCR is not.

**D31 — the two kits are told apart on their axis of greatest variance, not by k-means.**
k-means was the obvious choice and it collapses. It minimises inertia, and the kits are
not cleanly bimodal — a dozen tracks of the same shirt vary more in light and pose than
two shirts differ from each other — so the cheapest split is one tight little cluster
against everybody else. Measured on SNGS-147: 44 tracks to 8, 70% right, and six spurious
tracks were enough to flip it.

Projecting the signatures onto their first principal component and cutting where the
between-class variance is greatest gives 26 and 26, 85% right. The `len(a) * len(b)` in
that score is exactly what stops one side swallowing the other.

**Goalkeepers are taken out first and put back after.** A keeper wears neither kit, and
left in he costs real accuracy — 83% against 93% on the same tracks. What identifies one
without being told is two things at once: a colour unlike either team AND standing near a
goal. Colour alone catches a player in odd light; position alone catches every defender on
a goal line.

Together these take the team split from 50.2% — pure chance — to **87.2%**.

**D30 — a track that has lost its player gives up quickly.** 88 of 98 identity changes
on SNGS-147 happened AFTER A GAP, at a median of ten frames — not during a visible
crossing, which is where I had assumed they were. A track that has lost its player coasts
on a stale prediction, its gate grows with the wait, and when detections resume it takes
whoever is nearest. Over half the time that was an opponent.

Cutting `MAX_AGE_S` from 0.8 to 0.24 takes purity from 76.9% to 80.5% and switches from
37 to 26, with the track count and recall unchanged: the player is picked up again either
way, and what is saved is a run stitched onto somebody else.

It also settles why weighting kit colour more heavily never helped, which had been an
open puzzle. At 0.8s the wrong candidate is reachable and colour is asked to talk the
tracker out of it; shorten the wait and it was never reachable, and the two weightings
score identically.

**Which measurement chose the number matters here.** Run fidelity compares a board with
the TRACKS it was built from, so it cannot see a steal — the board faithfully draws
whatever the tracker believed. Only ground truth can, and a shorter age looked WORSE by
fidelity while being better by truth. Do not tune this against a clip with no answers.

**D27 — anyone standing off the pitch is dropped BEFORE tracking, not after.** Two fifths
of what the detector finds on SoccerNet is crowd, dugout staff and ballboys behind the
hoardings. They were always discarded at the end, but until then they were competing for
associations and spawning tracks of their own. Filtering first takes 123 tracks to 69,
lifts identity purity from 77.8% to 79.3% and drops switches from 44 to 38, at no cost to
recall.

This is the one place stage 2 consults stage 1, and it is a deliberate exception to D22:
only as a FILTER. The association still never sees a homography, so a drifting camera can
change which detections are considered and cannot change the identities.

**Four things that did NOT work, recorded so they are not retried.** Associating in pitch
metres rather than stabilised pixels; optimal assignment instead of greedy (worth keeping
on its own merits once the junk was gone — 69 tracks against 72 — but it fixed nothing);
raising the weight on kit colour, which is genuinely discriminative (same-team pairs sit at
0.32, opposing at 0.67) and still moved nothing; and stitching fragments back together
afterwards, which reunited one player for every two players it wrongly welded into one. The
last was deleted rather than tuned: a fragment loses a run, but a bad join invents one, and
nothing downstream can tell.

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
