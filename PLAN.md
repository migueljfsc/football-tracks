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

## Where this stands — 4 September 2026

**Nothing is running.** Four segmenter training runs are done and measured, plus a fitter
change, a carry sweep and a quality gate.

**The headline is a negative result and it is the most useful thing here.** All of it improved
per-frame accuracy — `observed_error` 1.33 m to 0.57 m, precision 40.4% to 91.4% — and all of
it made the actual Pitchboard board worse: 19 players down to 10, and 15 m of player travel
down to 4.9 m. The shipping `--mode seed` pipeline still makes the best board. Stage 1 is not
what is limiting the product; stage 2's identity fragmentation is.

Read this section, then D36 — starting from *"Measured through PITCHBOARD"*, which is the part
that changes what to do next.

### The four runs

| run | resolution | trained on | SNGS-147 | SNGS-116 | SNGS-121 |
|---|---|---|---|---|---|
| 1 | 640×360 | 5 matches | 0.84 m | 3.76 m | 1.54 m |
| 2 | 960×540 | 5 matches | 1.13 m | 3.20 m | 0.69 m |
| 3 | 960×540 | 350 matches | 0.90 m | 5.19 m | 1.13 m |
| **4** | **1280×720** | **5 matches** | **0.67 m** | **2.87 m** | **0.67 m** |
| | | *bar* | *0.5 m* | *0.5 m* | *0.5 m* |

Medians of `observed_error`. No run has yet cleared the bar.

**The "100% solved" in that column means something narrower than it reads.** `calib-eval`
scores only frames that carry ground-truth line annotations, sampled at `--stride 25`. Put
every frame of a clip through the same fitter and run 4 solves 83% of SNGS-147, 81% of
SNGS-116 and **51% of SNGS-121** — it refuses the rest rather than guessing, which is correct
behaviour and a very different number. Any claim about the solve rate has to say which
population it counted.

**Run 4 is the best on every clip and on every p90** (147: 1.70 m, 116: 5.37 m, 121: 1.20 m),
and it is the only run that improved all three at once. It is also the only controlled
comparison in the table: runs 2 and 4 share a training set, a holdout and a validation set
byte for byte, so the difference between them is resolution and nothing else. Run 3 changed
two things at once and is not comparable with either.

The weights on disk, kept so the table above stays reproducible rather than remembered:

    work/calib/segmenter.pt            run 4  1280×720, 5 matches   <- current
    work/calib/segmenter.350match.pt   run 3   960×540, 350 matches
    work/calib/segmenter.5match.pt     run 2   960×540, 5 matches

### What the four runs actually established

- **Resolution is the lever.** Every clip improved when it went up and nothing else moved.
- **Diversity is not.** Run 3 multiplied the matches by 70 and made two clips of three worse.
  That was the stated binding constraint after run 2 (see D36) and it did not survive testing.
- **Five matches overfits at epoch 2**, at both 960×540 and 1280×720. Resolution moved the
  floor, not the onset. A run needs ~4 epochs to find its best checkpoint; the other ten are
  spent confirming it.
- **Validation loss does not predict `observed_error`.** Run 3 had a worse loss than run 2 and
  a better median on 147. Judge on `calib-eval`, never on the loss.

### What to do next

**0. Seeding was the biggest single defect in the pipeline and it is fixed (D55).** Recall
72.6 / 66.0 / 71.6% across the three clips, against 41.3 / 67.3 / 15.8% this morning. Two
measurement bugs were found underneath it -- read D55 before trusting any number in this
document that predates it.

**Then: stage 2 fragmentation is partly fixed; stage 1 is still not the thing to work on.**
`stage2_stitch.py` joins fragments and is on by default (D53) — SNGS-116's board went from a
5.8 s passage to 13.5 s. What remains is that a player is only in SHOT for about a third of a
clip, which no amount of joining repairs and which sets the roster.

**Was: nothing on stage 1 until stage 2 is fixed.** The segmenter work of 4 September improved
every per-frame metric and made the Pitchboard board WORSE (D36). The board is the product;
`observed_error` is not. Every variant, ground truth included, shatters 22 players into 43-88
fragments and only five or six survive `MIN_COVERAGE` — so the binding constraint on a usable
board is tracking identity, not the camera model. `ft auto --mode seed` remains what to ship.

Anything below this line was the plan BEFORE that was measured. It is kept because the
reasoning is still sound about stage 1 in isolation, and stage 1 in isolation is no longer
the thing to work on.

**1. A run at 1920×1080.** GSR is natively 1080p, so this is the ceiling for the data with
nothing resampled anywhere, and the trend says it is where the remaining gain is. It is
cheaper than it sounds — the best checkpoint arrives at epoch 2, so it wants ~4 epochs, not
14, which is about 2.5 hours rather than eight:

```sh
# set WIDTH, HEIGHT = 1920, 1080 and LINE_PX = 8 in calib.py first -- see the note there
nohup uv run ft calib-train --stride 10 --epochs 4 --batch 2 --no-extra \
  --holdout-games "7,8" > work/calib/train4.log 2>&1 &
```

`--no-extra` is not optional above 960×540: SN-Calibration-2023 is natively 960×540, so
including it upsamples 82% of the set and trains the model to expect blur it will not meet at
inference. Batch has to come down as the resolution goes up or MPS runs out of memory.

**2. Diagnose SNGS-116 instead of throwing pixels at it.** It has sat at 2.87–5.19 m across
four configurations while the other two clips swung by a factor of two. That is a property of
the clip, not of the model, and no amount of resolution has touched it. Look at which frames
fail before assuming the next run fixes it.

Then re-score, always all three, against the bar set before any of this was trained — **beat
0.5 m median `observed_error` and solve 80% of frames**:

```sh
uv run ft calib-eval SNGS-147 --stride 25
uv run ft calib-eval SNGS-116 --stride 25
uv run ft calib-eval SNGS-121 --stride 25
```

**If it clears the bar**, stage 1 needs no seed at all, drift and cut-detection stop being
separate problems, and the manual path becomes the fallback D7 always intended it to be.

**If it does not**, the honest retreat is to ship the segmenter as a seed *proposal* rather
than a solver: it puts landmarks on the frame and a human drags the wrong ones. That is worth
having at 0.67 m, and it is a smaller claim than the one D36 set out to make. Say which
happened; do not quietly redefine the bar.

### The three clips this is scored on

`SNGS-147`, `SNGS-116`, `SNGS-121` — held out by match, never by clip (116 and 121 are both
game 7, so a clip-level split would leak). `--holdout-games "7,8"` is what keeps them out.

### State of the tree

Uncommitted, and deliberately so — nothing here has been committed without being asked:

    src/football_tracks/calib.py      index_calibration(), the trailing-space fix, 1280×720
    src/football_tracks/cli.py        --extra / --extra-stride on calib-train
    src/football_tracks/config.py     CALIB_DATA
    tests/test_calib.py               three tests; 118 pass

`work/` and `*.pt` are gitignored, so trained weights cannot be committed to a public repo by
accident.

### Getting the training data back

GSR-2025 comes down with `ft fetch`. SN-Calibration-2023 has no command yet — it was fetched
with this, which is worth turning into one if it is ever needed twice:

```python
from football_tracks.detect import trust_certifi

trust_certifi()  # macOS python.org builds have no wired CA bundle
from SoccerNet.Downloader import SoccerNetDownloader

dl = SoccerNetDownloader(LocalDirectory="data/calib2023")
dl.downloadDataTask(task="calibration-2023", split=["train", "valid"])
```

2.9 GB, and it needs `uv sync --extra data`. No password: see the note in AGENTS.md. Note that
it is only useful at 960×540 or below.

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
| M1 | reprojected pitch lines sit on the real lines | **solver done**; detector at 0.67 m vs a 0.5 m bar (D36) |
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

**D36 — the camera model is learned, because the missing thing was never the paint but
the NAME of it.** `refine.line_pixels` finds markings to a median of 0.00 m under a correct
homography. What it cannot do is say which marking a white pixel belongs to: it infers that
from the homography it is trying to fix, which is the circularity that gave it a two-metre
capture radius and lost it the benchmark (D35).

A segmenter answers that one question. Every pixel arrives already named, so a
correspondence is a fact rather than an inference, and a homography can be fitted per frame
from nothing at all -- no seed, no carry, and therefore no drift. Manual seeding, drift and
the cut-detection problem are one problem wearing three hats, and this is the hat.

DeepLabv3 on a MobileNetV3 backbone, 27 classes. Trained on SN-GSR-2025 -- broadcast footage
carrying per-frame line annotations in the format `lines_of` already reads -- at 640x360, then
960x540, then 960x540 with SN-Calibration-2023 added, and finally at 1280x720 on GSR alone.
Four runs, and the last is the best on every clip.
Both are training data ONLY: at inference the model sees the user's own clips and SoccerNet is
never involved. Weights are gitignored rather than committed, which is also the answer to what
the data licence permits.

Three things this is built around:

- **The split is by MATCH, never by clip or frame.** SNGS-116 and SNGS-121 are both game 7,
  so a clip-level split puts the same stadium, camera and kit on both sides and reports a
  generalisation that was never tested. `split_by_game` is the whole guard and
  `test_calib.py` pins it.
- **Background is 95% of the pixels**, so plain cross-entropy scores 95% by predicting
  nothing. The background class is weighted to 0.05.
- **A predicted class the fitter cannot name is wasted supervision, not a bug.** Circles and
  goalposts are labelled and learned because they teach the network what a pitch looks
  like; only the 17 straight markings become correspondences. A test asserts those 17 are
  exactly `PITCH_LINES`, because the fitter silently ignores anything else.

The fit is DLT first -- it needs no starting guess, which is the entire point -- then the
geometric least squares from `refine`, because a DLT is biased by how many pixels each
marking happens to contribute (D35 again).

Kill criterion, set before training: a per-frame fit must beat 0.5 m median `observed_error`
on a held-out MATCH and solve 80% of frames. A good human seed is 0.15-0.3 m, so anything
worse is not worth replacing seeding with.

**That bar was FAILED, and the verdict stands.** Four runs, best 0.67 m. The segmenter does
not replace the human seed and this document does not claim it does.

**A second, different question is now open, and it needs its own bar.** The first bar asked
*can this replace a human?* — measured against a human's 0.15-0.3 m, on frames the fit was
attempted on. What it never asked is *is this better than what the pipeline actually does
today?* The pipeline does not have a human's 0.15-0.3 m: it has ONE seeded frame carried
through the clip by tracking the grass, and the carry drifts. D19 measured the cost of that
drift at 62.9% identity purity against 82.9%. So the two questions have different answers,
and end-to-end measurement says the second is the useful one.

The second bar, stated before the numbers came in, in the terms `ft bench` already prints:

> The segmenter path must beat the `--mode seed` baseline **on every clip**, on precision and
> on position error, without costing recall on any of them.

Three things about that. It is measured end to end on what reaches `tracks.json`, not on
`observed_error`, because a camera model that is better by its own metric while the tracks
get worse is precisely the D35 failure and this stage has now walked into it once. It is a
comparison against the pipeline as it stands rather than against an absolute, because
"better than what we ship" is the decision actually being made. And it is deliberately
strict on recall: the segmenter's headline gains come partly from refusing frames, and a bar
that ignored coverage would reward refusing more of them.

Failing the FIRST bar is not evidence about the second, and passing the second does not
retire the first. The honest summary of both is: this cannot replace seeding, and it may
still be the better thing to ship.

**First run: solves everything, and is not accurate enough.** Trained at 640x360 on games
4, 6 and 9 (4,275 frames), evaluated on the held-out matches:

    clip        solved   median   p90     with refine chained on
    SNGS-147      100%    0.84 m  2.76 m         1.32 m
    SNGS-116      100%    3.76 m  8.39 m         0.72 m
    SNGS-121      100%    1.54 m  8.42 m         0.38 m

The solve rate is the part worth noticing: 100% of the ANNOTATED frames, from the picture
alone, with no seed and nothing carried. The accuracy fails the bar. (That 100% was read as
"of every frame" for four runs and it is not — across whole clips run 4 solves 51-83%. See
the end-to-end results below, which is where the difference finally showed up.)

It is NOT a naming problem, which is what it was built to fix and what it did fix. Under
the true homography only 3-9% of predicted pixels sit more than 2 m from the line they
claim. What they are is imprecise: the median predicted pixel is 0.26 m from its line near
the camera and 1.45 m from it far away, because at 640x360 a 3 px line upscales to a 9 px
band and a band that wide is worth over a metre at the far touchline. So the limit is
resolution, and the retrain is at 960x540.

Two things not to repeat. Chaining `refine` after the segmenter helps enormously on two
clips and wrecks the third (p90 2.76 m -> 17.65 m), so it cannot simply be switched on.
And choosing between the two fits by which better explains the segmenter's own pixels does
not work, for a reason worth remembering: the mask fit was fitted to minimise exactly that
quantity, so the test is rigged for it and picked it 45 times out of 65. Selecting on the
data you fitted on is not selection.

**Second run: 960×540, the same five matches, and it settles what was actually missing.**
Validation loss reached its best at epoch 2 and then rose for twelve consecutive epochs.
Two and a half hours of compute for a checkpoint taken inside the first thirty minutes.

    clip        solved   median   p90      640×360 median
    SNGS-147      100%    1.13 m  2.27 m         0.84 m
    SNGS-116      100%    3.20 m  6.94 m         3.76 m
    SNGS-121      100%    0.69 m  1.35 m         1.54 m

Better on two clips, worse on one, still nowhere near half a metre. Resolution was a real
limit — 121 more than halved and 116's p90 came in by a fifth — but it was not the *binding*
one. **Five matches is.** A network shown three stadiums learns those three stadiums, and at
960×540 it learns them faster. The overfitting curve is the evidence: nothing after epoch 2
was learning about pitches, it was learning about those pitches.

*That conclusion was wrong, and the third run is what disproved it. It is left standing
because it is why the third run was worth doing, and because the reasoning still looks sound
from here — which is the point. Read on.*

**Third run: 345 matches instead of five.** SN-Calibration-2023 is 19,675 annotated frames
across six leagues and three seasons, natively 960×540 — exactly the training size, so
nothing is resampled on the way in. It carries no video and no tracks, so the segmenter is
the only thing in this repo that can read it.

Three things made it usable rather than merely large:

- **`match_info.json` names the fixture behind every image.** Without it these frames would
  have arrived with no match tag, `split_by_game` would have had nothing to hold out, and the
  leakage guard would have gone on passing while guarding nothing — the same failure as the
  clip-level split, arriving by a different door. The dataset ships its own match-disjoint
  train/valid split (290 matches against 55, none shared), so it is added on either side of
  ours rather than re-split.
- **The labels have a trailing space in them.** `"Goal left post left "` is written with one,
  and an exact `INDEX` lookup returns `None` for it, so 1,101 instances would be dropped as an
  unknown marking and that class would train on nothing at all. It fails as *silence*, not as
  an error. The lookup strips, at the one site where names resolve.
- **It is public.** calibration-2023 is not behind the NDA — the `.env` password is in fact
  rejected for it (401 where the library's public default gets 200), which is how this was
  found out. The licence question the run seemed to raise was moot.

GSR clips stay in the mix, because the benchmark is GSR footage and the model should see some.

**The caveat that cannot be closed.** GSR identifies a clip's match only as `game_id: 7`/`8`
— no league, no fixture, no date — so it is impossible to prove *by name* that the three
benchmark matches are absent from calibration-2023's 345. The season ranges do not overlap
(2014–17 against a 2025 capture), so the risk is low, but low is not zero and this is
unverifiable rather than verified. Say so beside any number this run produces.

One thing to keep an eye on: validation is now dominated by calibration-valid (3,212 frames)
over the held-out GSR games (225), and the checkpoint is chosen on the combined mean. That is
a slight mismatch with a benchmark that is entirely GSR footage — the saved checkpoint is the
best on *pitches in general*, not the best on this broadcast camera.

**And it did not work.** Seven hours, 70x the matches, and two clips of three got worse:

    clip        solved   median   p90      960×540/5-match median
    SNGS-147      100%    0.90 m  2.29 m         1.13 m
    SNGS-116      100%    5.19 m  8.47 m         3.20 m
    SNGS-121      100%    1.13 m 15.61 m         0.69 m

121's p90 is the alarming number — 1.35 m to 15.61 m, a tail of frames that are not merely
worse but wrong, on the one clip that had been nearly good.

The obvious mechanical explanation was checked and is NOT the cause: if calibration-2023
named its markings differently from GSR, `INDEX.get(name.strip())` would drop them silently
and 82% of the training frames would carry masks with lines missing — teaching the model to
suppress exactly what it needs. The two datasets share a vocabulary almost exactly: 27,497
label hits against 6 misses, all of them a stray `"Line unknown"`.

So the conclusion after run 3 is the uncomfortable one. **Diversity was not the binding
constraint, and the claim at the end of the second run was wrong.** More than that: the
scatter *between* runs was as large as the differences *between* conditions, three runs in,
which means none of the three had separated its hypothesis from noise. The first two runs
each moved two variables, and the third moved two more.

**Fourth run: 1280×720, and the first controlled experiment in the series.** One variable.
Same 4,275 training frames, same 225-frame validation set, same holdout as run 2 — the log
header is byte-identical — with only the resolution changed. GSR is 1920×1080 on disk, so
960×540 had been discarding half the linear resolution of the only footage the benchmark is
scored on.

    clip        solved   median   p90      960×540 median   960×540 p90
    SNGS-147      100%    0.67 m  1.70 m         1.13 m        2.27 m
    SNGS-116      100%    2.87 m  5.37 m         3.20 m        6.94 m
    SNGS-121      100%    0.67 m  1.20 m         0.69 m        1.35 m

Best on every clip and every p90, and the only run to improve all three at once. Against a
directly comparable validation loss it is better too — 0.1376 against run 2's 0.1534 — which
is the one place in this series where the loss and the metres agreed.

Two things it settles, and one it does not:

- **Resolution is the lever.** It is now measured against a controlled baseline rather than
  inferred from runs that moved several things.
- **Overfitting onset is a property of the data, not the resolution.** Both 960×540 and
  1280×720 peak at epoch 2 on five matches and climb for every epoch after. Resolution moved
  the floor and left the onset alone. Practically: a run of this shape wants ~4 epochs, and
  the other ten only confirm the checkpoint already on disk.
- **It still misses the bar**, at 0.67 m against 0.5 m. 147 and 121 are close; 116 is not, and
  it has never been — 2.87 to 5.19 m across four configurations while the other two swung by a
  factor of two. That is a property of the clip and no resolution has touched it. It wants a
  look at which frames fail, not another run.

Above 1280×720 there is a constraint worth stating outright, because it is invisible and it
would quietly poison the next run: **SN-Calibration-2023 is natively 960×540.** Training the
combined set any higher upsamples 82% of it, which adds no detail and teaches the model to
expect blur that inference will not supply. Above 960×540, train on GSR alone (`--no-extra`)
or not at all. The constants in `calib.py` carry this note beside them.

**End to end, it is better than the pipeline it would replace — on two clips of three.**
`ft auto --mode segmenter` registers every frame from the learned lines, with no seed, no
carry and no ground truth. Against `--mode seed` (what the pipeline does today: frame one's
lines, carried) and `--mode truth` (every frame's real lines, the ceiling):

    clip      mode        recall  precision  position error   solved
    SNGS-147  seed         41.3%     40.4%     1.33 m         100%
              segmenter    52.0%     80.9%     0.65 m          83%
              truth        81.1%     81.8%     0.77 m         100%
    SNGS-116  seed         67.3%     74.9%     0.54 m         100%
              segmenter    48.8%     68.7%     0.59 m          81%
              truth        66.4%     73.4%     0.48 m         100%
    SNGS-121  seed         15.8%     15.2%     1.30 m         100%
              segmenter    44.9%     87.8%     0.50 m          51%
              truth        53.1%     51.2%     0.60 m         100%

On 147 and 121 it is not close: precision doubles on one and goes 15% to 88% on the other,
while the position error halves. On 116 it is a small regression, which is the same clip the
segmenter has never been good on.

**Read the medians with the solve rate beside them, or they lie.** The segmenter's 0.50 m on
121 is measured over the 51% of frames it accepted, having refused the rest; `truth`'s 0.60 m
is over all of them. It is not more accurate than ground truth, it is more SELECTIVE than
ground truth. What the mode really buys is precision — the samples it writes are far more
likely to be real — and it pays in recall.

**Which makes refusals, not accuracy, the binding constraint now.** Half of SNGS-121 is
declined. The obvious next move is a SHORT carry to bridge the gaps — `--carry 5` rather than
the unbounded chain D19 condemned — which should recover most of the recall while keeping the
drift bounded to a few frames. That is untested, and it is the cheapest experiment left.

**What the refusals actually are: a midfield view, one line short.** The gaps are not
scattered hard frames, they are contiguous passages — SNGS-121 refuses frames 0-367 in one
block and then solves nearly everything after; SNGS-116 refuses 136 frames from 614; SNGS-147
has blocks of 54 and 44. Every one of them is a MIDFIELD camera. Counting what the segmenter
names in SNGS-121:

    frame 100  (refused)   Big rect. right main 2260 px   Big rect. right top 2914 px
                           Middle line 3139 px            Side line top 17629 px
                           Circle central 11765 px  <- DISCARDED
                           -> 4 usable straight lines, and MIN_LINES is 5

    frame 500  (solved)    9 usable straight lines, box and six-yard box both in shot

Half of that clip is refused for want of ONE line, while the second-largest marking in the
frame — a confidently segmented centre circle — is thrown away by `fit_from_mask`, which
speaks only `PITCH_LINES`. The model is doing its job; the FITTER is what refuses.

This looks like the thing D35 says not to retry, and it is not quite. What was measured and
rejected there was circle *pixels* as correspondences: a pixel on the circle says only that it
lies somewhere on a 57 m curve, which is a point-to-curve constraint, and 45 of them carried
enough systematic bias (0.26 m) to take a 0.16 m fit to 1.03 m. Two things differ here. The
frames in question produce NO fit at all, so the comparison is against nothing rather than
against 0.16 m. And there is a construction with no such ambiguity: the halfway line runs
through the circle's centre, so it cuts the circle at exactly two points — (52.5, 24.85) and
(52.5, 43.15) — and a penalty arc meets its box line at two more. Those are exact
correspondences, not point-on-curve ones. Whether they are enough is unmeasured; what is
measured is that the current gate refuses half a clip while looking at 11,765 pixels of
usable geometry.

Do not let this become the D35 mistake in reverse. The bar is `ft bench`, end to end, on all
three clips — not the number of frames that stop being refused.

**Built, measured, and it misses the second bar by one cell of nine.** `CURVE_CROSSINGS` in
`calibration.py` names the three places a curve meets the line that cuts it; `calib._crossings`
finds them in the mask and hands `calibration.fit` exact point correspondences. Per-frame
accuracy on the annotated frames:

    clip        before   after    p90 before -> after
    SNGS-147    0.67 m   0.67 m      1.70 -> 1.70
    SNGS-116    2.87 m   1.20 m      5.37 -> 3.39
    SNGS-121    0.67 m   0.67 m      1.20 -> 1.20

SNGS-116 more than halves. That is the clip that would not move for resolution, for 70x the
matches, or for anything else tried across four training runs — and it was never a training
problem. Its box views put a penalty arc across the box line, and those two exact spots were
being thrown away.

End to end against the `--mode seed` baseline the second bar names:

    clip       recall           precision        position error
    SNGS-147   41.3 -> 52.5     40.4 -> 75.9     1.33 -> 0.66 m
    SNGS-116   67.3 -> 62.5     74.9 -> 86.6     0.54 -> 0.51 m
    SNGS-121   15.8 -> 45.1     15.2 -> 81.7     1.30 -> 0.50 m

Precision and position error clear on all three. **Recall on SNGS-116 does not** — 62.5%
against 67.3% — so the bar as written is missed. Eight cells of nine is not the bar; the bar
said every clip and no recall cost. It was written before these numbers and is not being
adjusted after them.

What the crossings cost is worth stating separately, because it is the same trade the whole
mode makes. Against the segmenter WITHOUT them, 116 gains 13.7 points of recall and 17.9 of
precision, while 147 and 121 each LOSE about 6 points of precision for a fraction of a point
of recall. Newly-admitted frames are the ones that were being refused, and they are harder
than average; admitting them raises coverage and lowers the average quality of what is
admitted. On 116 that is overwhelmingly worth it. Elsewhere it is close to neutral.

**The conic was the wrong tool and is gone.** `cv2.fitEllipse` on a clipped arc — 3,174 px of
penalty arc cut off by the frame edge — returns a 46x153 sliver, an unconstrained
five-parameter surface through a stub, and intersecting it puts crossings wherever the algebra
lands. Angular coverage does not tell those fits from good ones either: the sliver scores 69%
where a healthy circle scores 50%, so that guard was measured and rejected rather than shipped.

`_touching` needs no fit at all. A penalty arc IS the part of a circle outside the box, so it
ends ON the box line; the halfway line runs through the centre spot, so it cuts the centre
circle radially, at 90 degrees. Take the curve's own pixels within a line width of the line,
sort them along it and split at the widest gap: two clusters are two crossings, one cluster is
a clipped arc and is refused. It cannot invent a crossing, because a pixel centroid is by
definition where pixels are — which also made the `_near` guard dead code, so it went.

One bias worth knowing before it is chased: labels compete for the pixels where two markings
overlap, so a curve's own pixels stop about a line width SHORT of the true crossing. At 90
degrees that costs nothing, which is the centre-circle case. A penalty arc meets the box line
at 53 degrees, so both its endpoints are pushed the same way and the fit absorbs most of it.

**Measured, and it is better construction rather than a better outcome.** `calib-eval` is
identical to the conic's (0.67 / 1.20 / 0.67), and so is end to end on SNGS-147 and SNGS-116.
The whole difference is SNGS-121:

    SNGS-121         unsolved   recall   precision   error
    no crossings        368      44.9%     87.8%     0.50 m
    conic               339      45.1%     81.7%     0.50 m
    touching            288      46.5%     74.5%     0.51 m

It rescues 51 frames the conic could not, so the clipped-arc handling does work — and they are
BAD frames: 1.4 points of recall for 7.2 of precision. Precision falls monotonically as
coverage rises, which is the evidence that the frames still refused are refused correctly.

The nuance that matters more than the change: **crossings are not a uniform win.** They improve
SNGS-116 on every axis (0.59 to 0.51 m, precision 68.7 to 86.5%, recall 48.8 to 62.5%) and
mildly hurt SNGS-121. They are the fix for BOX views with a penalty arc, which is what 116 is
made of, and near-neutral elsewhere. The second bar is still missed, still on SNGS-116's
recall.

**The bar was set below the noise floor of the ruler.** The 0.5 m criterion was written
before anything was trained, from the reasoning that a human seed is 0.15-0.3 m. Nobody
checked what the GROUND TRUTH is worth, and it is worth less than the bar:

    clip        held-out marking lands this far from where the rest of the frame puts it
    SNGS-147    median 0.318 m   p90 1.441 m   (4,410 held-out fits)
    SNGS-116    median 0.348 m   p90 1.067 m   (3,912)
    SNGS-121    median 0.512 m   p90 1.630 m   (3,204)

Leave one marking out, fit from the others, and project the held-out marking's own annotated
points: they miss its pitch line by a third of a metre typically, and by half a metre on
SNGS-121 — which is the bar exactly. Every `observed_error` in this document is measured
against a reference carrying that much disagreement with itself, so a run at 0.67 m is nearer
its ceiling than the raw number suggests, and part of what four training runs were chasing was
annotation noise.

Two honest limits on that. It measures self-CONSISTENCY, not accuracy: a systematic error the
whole annotation shares is invisible to it. And a held-out marking residual is not the same
quantity as `observed_error`, so the two do not subtract cleanly. What it does establish is
that 0.5 m was never a safe target on this data, and that a fifth training run chasing 0.17 m
would have been chasing something the measurement cannot resolve.

The first thing to do with this is NOT to move the bar. It is to notice that the ceiling was
never measured before the bar was set, and that measuring it cost under an hour and no GPU.

**Measured through PITCHBOARD, the whole thing is a regression. Read this before doing more
of it.** Every number above is a proxy. The artefact this repo exists to produce is a board,
and `src/import/` in the sibling repo is the only thing that makes one. Running every variant
of SNGS-147 through `boardFromTracks`:

    variant                players     scenes  window   x range    max travel  curves
    seed (shipping)        19 (8H/11A)   6      2.9 s   36-79 m     15.0 m      10
    truth (ground truth)   14 (11H/3A)   4      8.2 s    6-68 m     12.1 m      18
    touch / cross / r0     10 (8H/2A)    4      6.9 s    3-35 m      4.9 m      15
    r0.54 (residual gate)  10 (1H/9A)    4      6.2 s    4-37 m      6.6 m      13
    r0.48                   8            2      2.8 s    4-38 m      3.3 m       8

**The shipping seed-and-carry pipeline makes the best board by a distance** -- 19 players
against 10, and 15 m of travel against 4.9 m. Every segmenter variant confines the board to a
third of the pitch with players that barely move. `observed_error` fell from 1.33 m to 0.57 m
and precision rose from 40.4% to 91.4% across the same series.

The residual gate is the sharpest version of the error. "Refuse the frames you fit worst" and
"refuse the frames looking at the far end of the pitch" are the SAME instruction on a panning
camera, so the gate bought its precision by discarding the wide views -- and `chooseWindow`
then had no well-covered window except one where the camera sat still. Hence 1 home player,
9 away, and a clump.

The lesson is D35's, at a larger scale and after D35 was written: a metric that improves while
the output degrades is not a metric to optimise against. Four training runs, a fitter change,
a carry sweep and a quality gate were all judged on per-frame quantities, and the one
measurement that mattered took twenty minutes and was never run until the end.

**What this does NOT say** is that the segmenter is worthless. Its per-frame accuracy is real
and so is SNGS-116's 2.87 -> 1.20 m. What it says is that per-frame accuracy was never the
binding constraint on a BOARD, and the binding constraint is now visible in the same table:
every variant, ground truth included, shatters 22 players into 43-88 fragments lasting 2-4% of
the clip, and only five or six survive `MIN_COVERAGE`. That is stage 2, not stage 1. A perfect
camera model would still produce a ten-player board.

**D53 — stage 2 fragments every player, and joining the pieces afterwards is safe where
lengthening the tracker's memory was not.** The measurement that redirected the work: against
ground truth on SNGS-147, a player is in shot about 239 frames of 750 and comes out as a
median of 4 predicted fragments — roughly 48 frames each, which is 6% of the clip against the
importer's `MIN_COVERAGE` of 30%. That, not the camera model, is why a 22-player clip becomes
a ten-player board.

`MAX_AGE_S` is 0.24 s deliberately (see its comment): a longer wait lets a track coast on a
stale prediction and take an opponent when detections resume. The fragments are the price that
was knowingly paid for the steals. So the fix is not to raise it.

`stage2_stitch.py` joins fragments afterwards instead, and the asymmetry is the whole point:
the tracker must decide AT the gap with nothing after it to go on, while this runs when both
sides are known and can require two fragments to be each other's best continuation. 39% of
identity changes have a gap of one frame or less — the player never disappears, the tracker
merely renumbers them — and 70% are inside 12 frames.

Three rules, each answering a way this could put one player's run on another's shirt:

- **Mutual best, not greedy.** A fragment ending in a crowd has several plausible successors
  and taking the cheapest is exactly how a track teleports. Requiring the choice to be
  returned makes an ambiguous join fail into two honest halves, which the importer survives.
- **A speed gate in METRES**, which is why this is a separate pass and not a change to stage
  2's gate. Stage 2 works in image pixels on purpose, so it cannot inherit the camera model's
  errors (D19); whether two fragments are one person is a question about m/s and needs the
  pitch.
- **Colour breaks ties and never repeals the speed limit.** Same role and value as the
  tracker's own `COLOR_WEIGHT`.

Measured on all three clips, `--mode seed`, against no stitching:

    clip        tracks      identity purity    switches
    SNGS-147    88 -> 67    73.6 -> 77.2%      59 -> 48
    SNGS-116    85 -> 63    64.4 -> 67.9%      271 -> 261
    SNGS-121    50 -> 42    61.3 -> 61.3%      50 -> 50

Purity RISING is the evidence the joins are right: joining two fragments of one player raises
it, and joining two different players would lower it. Recall, precision and position error are
unchanged to the digit, as they must be — stitching regroups samples without altering one.

And through `boardFromTracks`, which is the measurement that counts:

    clip        players   scenes    window        max travel     curves
    SNGS-147    19 = 19    6 -> 5   2.9 -> 2.8s   15.0 -> 19.7m  10 -> 14
    SNGS-116    22 = 22    7 -> 12  5.8 -> 13.5s  26.7 -> 40.0m  26 -> 49
    SNGS-121    21 = 21    9 =  9  17.4 -> 13.2s  18.3 -> 14.7m  37 -> 37

SNGS-116 roughly doubles: a 13.5 s passage rather than 5.8 s, and 49 curved runs rather than
26. SNGS-147 gains travel and curves. **SNGS-121 is a mild regression** — longer tracks change
the coverage landscape and `chooseWindow` settles somewhere shorter. Two clear wins and one
small loss; it is on by default, and `--no-stitch` turns it off.

What it does NOT do is add players. The roster is set by how long each player is in shot, and
joining fragments cannot put a player on camera. That ceiling is the next thing in the way.

**D54 — the detector calls the penalty spot a ball, and the board believed it.** Reported
from a Pitchboard board built off SNGS-116: a shot that was never taken, a ball already in the
six-yard box instead of the corner being delivered, and a keeper holding it to the end. All
three are one bug.

`ball_path` took the most confident sighting per frame and asserted it, with no continuity and
no way to abstain. The detector fires about five ball candidates a frame at a 0.15 floor,
spread over a thousand pixels. On SNGS-116 its favourite was pixel (1069, 612) — a white
circle painted on grass, which projects to 93.5, 33.7 m. The right-hand penalty spot is at
94, 34.

Measured against SoccerNet's OWN ball annotations, which is the only honest way to judge this
and had never been done:

    conf floor   frames given a ball   of those, within 20 px of the real ball
    0.15 (was)          74%                          25%
    0.35                27%                          48%
    0.55                12%                          74%
    0.65 (now)          10%                          84%
    0.75                 9%                          94%

**The ball can be frequent or right, not both**, because the detector finds it at all in only
41% of frames — an oracle that always picked the best available candidate would still be blind
more than half the time. No selection rule beats that ceiling.

Three changes, in the order they were tried, and the two that did nothing are worth keeping on
the record:

- **A continuity gate: no measurable effect.** Following the ball rather than re-choosing it
  each frame sounds right and changed 746 frames to 719, because the false positives are
  spatially clustered near the goal — continuity is happy to sit on them. Worse, the first
  version let the gate grow with the gap, which is precisely the failure stage 2 documents at
  `MAX_AGE_S`. It is capped now and it is still nearly free.
- **A static-position filter: the real fix for the penalty spot.** In PITCH metres a painted
  mark has one position all clip and a ball has a new one every second, so anything occupying
  a square metre for a third of the frames is scenery. No hardcoded pitch geometry, so it also
  catches litter and whatever else a ground has painted on it.
- **A confidence floor of 0.65: the change that mattered.** Median ball error 11.8 m to 3.3 m,
  within 3 m 22% to 50%, false positives 164 to 95.

On the board, the twelve scenes of SNGS-116 went from six holders and SEVEN handovers — seven
passes that never happened — to two holders and one. A corner delivered, possession changing
once.

**Correction, from watching the clip: the corner is NOT off-camera.** That was written from
looking at frame 88 and failing to spot the ball, and it was wrong. SoccerNet annotates the
ball there at image (465.5, 387.5), which projects to (105.2, -0.4) -- the corner flag. The
ball is in view, on the ground, waiting to be taken. What is true is that the DETECTOR finds
it in 8 frames of the 90 the corner occupies: a small, stationary, low-contrast ball at the
corner flag is close to invisible to it. Never explain a defect by what the footage does not
show until the annotations have been asked.

Two further bugs found by pursuing it, both in the smoothing rather than the selection:

- **A median of one value is that value.** `ball_path` emitted a position for every frame with
  ANY sighting within +/-5, so a single detection filled eleven frames -- at up to five frames'
  remove from the only evidence for it. That is how SNGS-116's board asserted a carrier at a
  scene where our ball was 25 m from the real one. `MIN_SMOOTH_SAMPLES` now needs three.
- **A corner ball is off the pitch.** `on_pitch`'s player margin is 0.05 m and a corner is
  taken from ON the line, so SoccerNet's own corner annotation at x = 105.2 is rejected -- 7%
  of all true ball positions with it. `BALL_MARGIN_M` is 1.5, still tight enough to reject an
  airborne ball's projection, which on this clip reaches (135.5, -16.6).

Measured against the ball annotations:

    clip        ball frames   median error   within 3 m
    SNGS-116        136          1.1 m          97%
    SNGS-121        493          0.6 m          75%
    SNGS-147        483          5.4 m          33%

SNGS-116's twelve scenes now carry NO ball at all, which is the honest answer and what the
clip was reported for: the board previously showed a possession change from one team to the
other, built from a ball position 25 m out at the only scene frame it existed. Ground truth
says the ball is 4.6 to 9.5 m from the nearest player through most of that passage -- a loose
ball in a crowded box, outside Pitchboard's 4 m `CARRIER_RADIUS_M`, held by nobody.

SNGS-147's ball remains poor at 5.4 m and 33%, so this is not a general fix. The ball is
reliable on two clips of three and no global threshold makes it reliable on the third.

**D55 — seed mode was seeding from the FIRST solvable frame, and the first frame is the
worst one.** SNGS-121 scored 15.8% recall where SNGS-116 scored 67.3%, and the gap had been
sitting in every table unexplained. It is not the clip.

The chain: its first 369 frames are midfield views carrying at most four usable markings, so
nothing could register them; `--mode seed` therefore seeded at frame 370 and carried the fit
BACKWARDS across a camera pan to cover half the clip, at 10.73 m median camera error with 94%
of frames worse than 2 m. Against a 2 m match radius nothing matched.

Making those frames solvable made it WORSE, which is the instructive part. `curve_crossings`
rescues them at a 0.385 m residual against 0.123 m at frame 370 — they are by construction the
fits the fitter was least sure of — so seeding on the earliest put the weakest fit in the clip
into every frame of it, and recall fell to 9.4%.

The fix is to seed from the best-EVIDENCED frame, counting visible markings:

    clip        seeded            recall          precision       position error
    SNGS-147    frame 1 -> 288    41.3 -> 72.6%   40.4 -> 72.4%   1.33 -> 0.69 m
    SNGS-116    frame 1 -> 162    67.3 -> 66.0%   74.9 -> 74.5%   0.54 -> 0.59 m
    SNGS-121    frame 1 -> 405    15.8 -> 71.6%   15.2 -> 69.1%   1.30 -> 1.00 m

Two clips transform and one is a shade worse. SNGS-121's board gains most: a 20.2 s passage
against 13.2 s, 26.9 m of travel against 14.7 m, and 57 curved runs against 37, at the cost of
three players.

Counting MARKINGS rather than scoring each fit's own residual, deliberately: a fit is chosen
to minimise that residual, so a barely-solvable frame scores well on it for exactly the reason
it is fragile. That is D35's rigged-selection trap, and the crossing-rescued frames demonstrate
it -- 0.343 m residual and useless as seeds.

It also models the intended human better. Seed mode stands for "a coach clicks four corners
once"; a person doing that picks a view where they can see the pitch, and taking whatever comes
first models a worse human than the one being modelled.

**Two measurement bugs found underneath this, both worse than the thing they were hiding.**
`ft truth` wrote the yardstick through `tracks.write`'s 0.1 s default, so the file every score
is measured against held two fifths of the samples of the 25 fps runs being judged -- SNGS-116's
precision read 37.5% instead of 74.9% with no pipeline code changed. It now defaults to 0, the
way `ft bench` already argued for its own interval. And the `truth.json` files in `work/` were
of unknown provenance, generated by some earlier version and never regenerated; every baseline
in this document had been measured against them.

**D56 — the tracker smoothed the kit colour and not the velocity, and velocity was doing the
harder job.** SNGS-116 carried 266 identity switches against SNGS-147's 56. Splitting them by
kind is what made it tractable:

    clip        switches   steal (id also serves another player)   fragment   at a gap <=1 frame
    SNGS-147        56                  52%                          48%            38%
    SNGS-116       266                  93%                           7%            68%
    SNGS-121       119                  83%                          17%            70%

`MAX_AGE_S`'s comment records that 88 of 98 switches on SNGS-147 happen AFTER a gap, and that
is still true of 147. SNGS-116 is the opposite failure: 93% are steals with NO gap, both
players continuously visible, the tracker simply swapping them. Shortening the coasting window
fixed 147 and can do nothing here, because nothing is coasting.

They are crossings. At a steal the nearest other player is 0.73 m away on SNGS-116 against
3.12 m for a typical sample -- and that clip is a corner, so 41% of all its samples have
somebody inside two metres. Kit colour is silent between team-mates and both observations sit
inside both gates, so the position prior is the only thing left.

And the position prior was noise. Velocity came from a SINGLE frame's displacement: a player at
5 m/s covers 13.6 px between frames at this scale, and the detector's box centre wanders a few,
so a quarter of it was jitter. The kit colour three lines below already had a rolling average,
on the stated reasoning that one frame of shadow should not redefine a kit -- the noisier
quantity, doing the harder job, was the raw one. Swept on SNGS-116:

    smoothing   identity purity   switches
    1.0 (none)      67.3%           266
    0.5             70.5%           256
    0.3             70.7%           246

Recall and precision do not move at any setting: this changes which track a sample lands on,
never whether it is found. Tightening `MIN_GATE_BOXES` was swept alongside and is the wrong
lever -- 0.20 reaches 71.8% purity with 292 switches, because a tighter gate tears tracks
rather than keeping them straight, which is what that constant says it exists to prevent.

**And it exposed a metric that had been read backwards all day.** SNGS-147's board appeared to
LOSE movement, 18.5 m of travel down to 7.6 m. The track responsible moved 22.4 m across the
window while covering THREE different ground-truth players at 76% purity; after smoothing it
covers one, at 100%, and moves 14.8 m. A track that hops between players covers more ground
than any real player can, so "max travel" rewards precisely the failure being removed. Judge a
board by median travel and by whether its longest run belongs to ONE player.

**D57 — the ball was three pixels wide, and everything else about the ball was downstream of
that.** `RTDetrImageProcessor` resizes any input to 640x640. A 1920x1080 frame therefore
arrives at a third of its width, and the ball -- 16 px across in the original, 11 px while it
waits at a corner flag -- reaches the detector as three or four pixels. Every ball fix before
this one (a confidence floor, the painted-spot filter, a continuity gate, the median-of-one
bug) was selection logic operating on a candidate pool that did not contain the ball.

`detect.py` now runs a second, TILED pass for the ball only: 3 x 2 crops, which divide
1920 x 1080 exactly, at native resolution. People are not re-detected -- they are large,
already found reliably, and slicing would cut them across seams.

    clip        ball detectable in a frame     candidates per frame
    SNGS-147          80% -> 88%                     5 -> 10
    SNGS-116          39% -> 74%                     3 -> 15
    SNGS-121          79% -> 87%                     5 -> 13

SNGS-116's corner goes from 1 of 70 frames to 67 of 70.

**A shortest path over the candidates was then written, measured and reverted.** With 15
candidates a frame, picking the most confident stops working -- the real ball scores about
0.21 and something else usually scores more -- so a global path with an emission cost from
confidence and a transition cost from distance is the natural answer. It is worse: 31% within
3 m on SNGS-116 against the conservative selector's 73%. At frame 110 the filtered candidates
include the real ball at (105.1, -0.3) scoring 0.18 and a false positive at (105.9, 10.9)
scoring 0.33. Both are STATIONARY, so continuity separates nothing; both fall under the
static filter's occupancy floor, so that separates nothing either. The path then follows the
confident one for the whole clip where the conservative selector abstains. Tuning the emission
weight from 1.2 down to 0.05 does not move it, which is the evidence that it is not a
weighting problem.

Kept: tiling, with the confidence-and-continuity selector.

    clip        untiled          tiled
    SNGS-147    483 fr, 33%      643 fr, 43%
    SNGS-116    136 fr, 97%      269 fr, 73%
    SNGS-121    493 fr, 75%      545 fr, 75%

And on the board, where SNGS-116 was reported for showing a possession change that never
happened: it had six holders and SEVEN handovers, then none at all once the false positives
were filtered, and now one holder across eleven scenes with no handover. The ball is back and
it invents nothing.

**Where this leaves the set-piece idea, which prompted the work.** It was proposed to fix the
corner and could not have: a classifier cannot promote detections that do not exist, and
before tiling the corner had three frames of evidence in a hundred and twenty. It now has
sixty-seven, and it is the discriminator the selector is missing -- the two candidates it
cannot choose between are 11 m apart and one of them is ON THE CORNER ARC. The labels for it
are already in the clips: `info.action_class` and `info.action_position`, 60 clips, 19 of them
set pieces (7 Corner, 6 Direct free-kick, 6 Kick-off), each with the exact frame.

**D58 — the ball's margin was a share of the pitch and was read as metres, so the gate was
157 m wide.** `tracks.on_pitch(x, y, margin)` scales its margin BY THE PITCH -- `mx = 105 *
margin` -- and D54 introduced `BALL_MARGIN_M = 1.5` to let a corner sit on the line, named in
metres and passed straight into it. 1.5 became 157 m and 102 m, which is every projection the
detector can produce. The constant's own comment says it is "narrow enough to still reject an
airborne ball's projection, which lands tens of metres away", and SNGS-116 was emitting a ball
at (137.1, -33.4) while it said so.

The ball has its own metre-space check now (`_ball_near_pitch`); `on_pitch` keeps its fraction
for players, where every caller already means a fraction. It is a pure removal of wrong balls:

    clip        within 3 m of the real ball    median error
    SNGS-116        44.5% -> 47.6%             3.56 -> 3.37 m
    SNGS-121        72.9% -> 74.7%             1.77 -> 1.76 m
    SNGS-147        38.5% -> 44.4%             4.84 -> 4.67 m

A unit that lives in a name and not in a type is worth one look per use. This one survived a
review that quoted the comment back approvingly.

**D59 — a set piece is the one moment the ball's position is known before it is seen, and
that is worth a rule of its own.** A ball waiting to be struck is small, still and far away,
so it scores about 0.2 and never clears `BALL_ASSERT_CONF`. SNGS-116 asserted NO ball at all
across the whole 157-frame corner that opens the clip -- the board therefore began with the
ball already in the box, which is what was reported.

Three things had to be measured before it could be built, and two of them contradicted the
obvious design.

*`action_position` marks the EXECUTION, not the placement.* At SNGS-116's labelled corner
frame the ball projects to (107.5, -3.9): already struck, airborne, and off the pitch. Every
set-piece clip looks like this. The stationary ball is BEFORE the labelled frame, and in 12 of
13 corner and kick-off clips it rests within 2 m of a canonical point for the entire
pre-action period -- 150 frames, six seconds.

*Only corners and kick-offs have a canonical position.* Kick-offs land on the centre spot to
within 1.2 m. Direct free-kicks are taken at (21.6, 7.2), (78.6, 10.4), (7.0, 54.6) -- nowhere
in particular. A positional prior covers 13 of the 19 set pieces and cannot be stretched to
the rest, so it does not try.

*Position alone is not enough, and this is what nearly shipped a regression.* Within 2 m of a
restart spot on SNGS-116, 89 of 91 candidates are the real ball, and the scores separate
nothing -- the true ones run 0.15 to 0.37 and the two false ones score 0.18 and 0.32, which is
exactly why the confidence gate could never find this ball. But the SAME region on clips with
no restart in them holds only false positives: 32 of 32 on SNGS-121, at the corner flag, 25 m
from the real ball. A low floor near a restart spot, on its own, is a regression.

What makes it safe is the VETO: the pass speaks only where the pipeline would emit no ball at
all. On SNGS-121 the ball is being tracked throughout, so it never speaks. And the veto has to
be read from the SMOOTHED output rather than the raw sightings -- SNGS-116's confident pass
fires twice before the corner, at frames 98 and 100, and both are wrong by over 30 m. Two
isolated blips are not a tracked ball, `MIN_SMOOTH_SAMPLES` already says exactly that, and
letting them veto costs 66 frames of a corner that is really there.

Radius 1.5 m, and a run of at least 10 frames. 2.0 m finds four more frames of SNGS-116's
corner and puts eight fabricated ones into SNGS-121. The run length is doing as much work as
the radius: at 4 frames the free-kick clip SNGS-066 gained 12 fabricated frames on the centre
spot, and every run the pass gets WRONG across eleven clips is short and transient -- 5 frames
on a centre spot, 8 at a corner just after it was taken, 4 more on a centre spot -- while
every run it gets right is 16 to 80 frames of a ball genuinely sitting there. Ten frames is
0.4 s, which is the shortest thing that can be called placed.

    clip        [action]            ball frames   within 3 m      median
    SNGS-116    Corner               252 -> 336   47.6 -> 59.6%   3.37 -> 1.94 m
    SNGS-067    Corner               232 -> 356   60.3 -> 74.2%   2.23 -> 1.03 m
    SNGS-110    Corner               333 -> 423   18.0 -> 36.1%   6.22 -> 4.83 m
    SNGS-075    Corner               319 -> 373   90.6 -> 91.9%   1.06 -> 1.45 m
    SNGS-060    Kick-off             620 -> 641   92.4 -> 92.7%   0.89 -> 0.88 m
    SNGS-069    Kick-off             336 -> 336        unchanged
    SNGS-151    Kick-off             547 -> 547        unchanged
    SNGS-066    Direct free-kick     342 -> 342        unchanged
    SNGS-100    Direct free-kick     200 -> 200        unchanged
    SNGS-121    Yellow card          515 -> 515        unchanged
    SNGS-147    Clearance            552 -> 552        unchanged

Five improved, six untouched, none worse. It adds 460 pre-action frames across the five and
two of them are wrong. SNGS-075's median rises while its within-3 m improves: the frames it
adds are all correct and looser than the tight ones already there, which moves a median
without putting a wrong ball anywhere.

The penalty spots are deliberately NOT restart spots. A painted white disc on grass is the
detector's favourite false positive -- it is what `_painted_spots` was built for -- and a
penalty is the one restart this footage never contains.

**And it barely reaches the board, which is the thing that actually ships.** Through
`boardFromTracks` the boards for SNGS-067, SNGS-075, SNGS-110 and SNGS-060 are IDENTICAL
before and after. Only SNGS-116's changed, where the carrier sequence went from `home-5` for
seven scenes then `away-1` to `home-10`, `home-5`, `away-1` -- the taker, the header, and the
keeper claiming it, which is the sequence in the footage.

The cause is `chooseWindow` in the Pitchboard repo. It maximises the number of player tracks
at or above `MIN_COVERAGE` and never looks at the ball, so it reliably picks the open play
AFTER a set piece over the set piece itself -- during a corner the players are bunched in the
box occluding each other, their tracks fragment, and coverage drops.

    clip        set piece at   window chosen
    SNGS-067       f172          f311-437     outside
    SNGS-110       f158          f416-682     outside
    SNGS-060        f22          f221-434     outside
    SNGS-075       f156          f34-405      inside, ball was already right
    SNGS-116       f157          f88-424      inside, board changed

So the ball is now right in frames the board never opens. The next move for set pieces is not
in this repo: it is teaching `chooseWindow` that a ball resting on a restart spot is worth
starting at. That signal needs no labels -- it is sitting in our own tracks.json, a ball still
on a corner arc for 80 frames.


**D60 — a stationary false positive is only distinguishable from a placed ball by WHERE it
is standing.** `_painted_spots` calls a square metre scenery when a "ball" holds it for a third
of a clip. That floor cannot go lower, because a ball placed for a corner holds one for a fifth
of a clip — so the filter that catches paint would call every set piece scenery. Two clips paid
for it: SNGS-147 asserted a ball on all 163 pre-action frames with NONE within 3 m of the real
one, and SNGS-151 the same on 84. Both are a stationary false positive a few metres from a
genuinely stationary real ball, holding its bin for about 21% of the clip — under the floor.

    clip        ours                truth               error
    SNGS-147    (10.1, 44.3) frozen  (5.3, 42.1)         ~5 m for 160 frames
    SNGS-151    (55.2, 28.9) frozen  (52.5, 34.0) spot   ~5.5 m

D59's restart geometry is the discriminator, used the other way round: a ball sitting still AT
a restart spot is legitimate, and one sitting still anywhere else is not. So the floor is 0.33
on a restart cell and 0.20 off it.

0.20 rather than lower, and the bound is real rather than cautious. At 0.15 SNGS-066 gains 24
points and SNGS-121 LOSES 8, because a stoppage leaves the ball sitting still off any spot —
SNGS-121 is a Yellow card, and the ball waits on the grass while the referee books somebody.
"Nothing legitimate is stationary away from a restart spot" is false, and 0.20 is what fits
between a placed ball and a mark that never moves.

    clip        within 3 m       median
    SNGS-147    44.4 -> 59.3%    4.67 -> 1.75 m
    SNGS-116    59.6 -> 58.5%    1.94 -> 1.95 m
    the other nine             unchanged

**And it moved no board at all.** All eleven are identical through `boardFromTracks`, for the
same reason four were after D59: SNGS-147's window is f195-264 and everything that improved is
in f1-163. Kept anyway — it removes balls that are systematically wrong, and the window will
not always miss them — but it ships as a source fix, not a board fix.

**Ball accuracy is now in `ft score`.** `ft truth` writes SoccerNet's `category_id` 4 into
truth.json's ball, and scoring is a diff of two files in one format, the way D12 set it up for
players. Six throwaway scripts measured the ball across the two decisions above; none of them
survives, and the next ball change would have been judged by eye.

**D61 — the kit signature separates the two teams almost perfectly, and gating on it still
does not fix the id switch. Written, measured, reverted.** Two recorded beliefs were wrong and
worth correcting before the next attempt repeats them.

*The switches are not team-mates.* `stage2_track` says a steal happens "where kit colour says
nothing because they are team-mates". Measured against ground truth, on the clips where
switches are worst, they are mostly OPPOSITE kits:

    clip        opposite   same   immediate (1-frame gap)   separation
    SNGS-116      166       48           70%                  0.8 m
    SNGS-110      174       32           60%                  1.1 m
    SNGS-121       56       12           70%                  1.6 m
    SNGS-067       47       51           45%                  1.6 m

*And they are immediate, not after a gap.* D19's "88 of 98 happened AFTER A GAP" is a fact
about SNGS-147, the clip it was measured on, and 147 is the outlier: 36% of its switches follow
a gap of more than six frames against 15% on SNGS-116. `MAX_AGE_S` is therefore not the lever
on the crowded clips.

*The signature is excellent.* Sampled over 2,400 detection pairs on SNGS-116 with ground-truth
teams, same-team pairs run to 0.62 at p90 and opposite-team pairs begin at 0.68 at p10 — no
overlap at all, and NO opposite-team pair looks more alike than the median team-mate pair.

So the signal is there, it is clean, and the switches are exactly the kind it should catch.
Raising `COLOR_WEIGHT` from 0.6 to 4.0 does nothing (SNGS-116 purity 71.9 -> 70.6 -> 71.8 ->
70.6, recall unmoved), which is what the existing comment already claimed. A hard GATE that
refuses an association outright rather than pricing it does slightly better:

    clip        purity           switches      recall
    SNGS-067    63.5 -> 66.0%    157 -> 149    unchanged
    SNGS-110    56.7 -> 58.3%    268 -> 264    unchanged
    SNGS-147    76.2 -> 76.4%     59 ->  51    unchanged
    SNGS-116    71.9 -> 71.5%    250 -> 251    unchanged

And through `boardFromTracks` it is a net regression: observed player-seconds across eleven
clips fall from 114.7 to 111.9, four boards worse, two better, five identical. SNGS-116 loses
almost six seconds of window and SNGS-066 goes from 92 curved runs to 65. The one real gain is
SNGS-069, whose home side goes from 0 players to 2.

Reverted. The board is the test (D35), and a metric that improves while the board does not is
not a reason to ship.

**Why it does not work, which is the useful part.** The track count barely moves — 56 to 55 on
SNGS-116 — so the gate is almost never the thing that refuses a match. At the moment of a steal
the track's own colour has already been pulled toward the thief by the rolling average
(`0.8 * prior + 0.2 * seen` reaches a new kit in about five frames), so what the gate compares
against is a blend rather than an opponent.

**Holding the colour still was then tried, and is worse than either.** Freezing a track's
signature after its first few observations should have left the gate something clean to test.
It costs purity everywhere instead:

    clip        rolling   freeze 5   freeze 10   freeze 5 + gate
    SNGS-116     71.9%     67.4%      68.4%        66.5%
    SNGS-067     63.5%     59.3%      58.3%        64.2%
    SNGS-121     71.8%     70.1%      69.2%        70.3%

and SNGS-116's switches go from 250 to 281, 263 and 291. The rolling average is not a bug to
be removed: a kit signature genuinely changes with pose, shadow and a turned back, so a frozen
one stops matching THE SAME PLAYER.

That is the real finding, and it closes this line of attack. Adapting to a player and
discriminating between players are the same mechanism pulling opposite ways, and 0.2 is already
a reasonable place to stand between them.

**And the evidence a steal needs is present, which rules out the two remaining excuses.** At the
frame a switch happens, the player who should have been taken IS detected -- 98% of the time on
SNGS-116, 93 to 95% on the other crowded clips, 80% on SNGS-147. So the detector is not the
constraint and the tracker is choosing wrongly with the right answer in front of it.

Nor is the crop spoiled by the occlusion that caused the crossing. Measured against per-team
reference signatures, at 120 of SNGS-116's switch frames the correct detection's kit sits at
0.45 from its OWN team and 0.87 from the other, and only 16% look more like the other team.

    at a switch    the right player is detected     98%
                   its kit still reads as its own   84%
                   the two kits are separable       no overlap at all

The detection exists, its appearance is right, and the signature discriminates. The failure is
therefore in the ASSIGNMENT rather than in any of the evidence it is given, and no amount of
work on the appearance model will reach it. That is where the next attempt has to start:
whether the correct pairing is inside the distance gate at all, and whether the optimal
assignment is sacrificing it to a track that wants it more.

**D35 — snapping the camera model onto the painted lines improves the camera model and
makes the tracks worse. Off by default.** Propagation is open-loop, so `refine.py` closes
the loop: project the model's markings into the frame, find the paint they should be lying
on, and refit. Against SoccerNet's per-frame ground truth it does what it claims --

    carry only      median 0.22 m   worst 1.10 m   at 200 frames 0.94 m
    carry + refine  median 0.22 m   worst 0.65 m   at 200 frames 0.16 m

-- and end to end, on the same clips with the interval held at 0 so recall is comparable,
it is a plain regression:

    clip        snap    recall   precision
    SNGS-147    off      41.3%      40.4%
    SNGS-147    on        8.4%      97.8%
    SNGS-116    off      67.3%      74.9%
    SNGS-116    on       61.1%      67.6%

97.8% precision on a recall of 8.4% is the on-pitch filter throwing nearly everything away:
the model is plausible enough to pass its own guard and wrong enough to put the players off
the grass. Feeding each snap back as the basis for the next carry made it worse still --
one bad fit poisons the rest of the chain rather than costing one frame -- and carrying
from the unrefined chain instead recovered 147 from 8.4% to 26.7%, which is still below
doing nothing.

So the code stays, behind `--snap`, and the default is off. Three things went wrong on the
way to it and all three are the same mistake in different clothes:

- a DLT minimises an ALGEBRAIC residual, and applied to a homography that was already
  exactly right it moved it half a metre, because lines carrying 48 snapped points outvote
  lines carrying 6 and the weight per constraint varies with depth. Geometric least squares
  in metres fixed it.
- snapping to the NEAREST painted pixel always finds the near edge of a line several pixels
  wide, so it under-corrected by half a line width every pass and converged to being wrong.
  The centre of the stripe fixed it.
- it was built as a pass over the finished chain, on the reasoning that it therefore could
  not compound. It also cannot PREVENT compounding: snapping has a capture radius of about
  two metres, so it refused 384 of the 695 frames on the clip whose chain had wandered 44.

And one thing that is simply not worth retrying: adding the centre circle and penalty arcs
as constraints, on the reasoning that a mid-pitch frame has almost no straight paint. Their
correspondences carry a 0.26 m systematic bias where the lines carry none, and 45 of them
were enough to take a fit from 0.16 m to 1.03 m.

The lesson worth keeping is the shape of it: the camera model got measurably better by the
measurement built to judge camera models, and the thing the pipeline actually produces got
worse. A metric that improves while the output degrades is not a metric to optimise against.

`ft bench` exists because of this. One command, fixed settings, every clip, one table --
so the next change can be judged against something reproducible rather than against a
number nobody can regenerate. The baseline it prints today:

    clip             tracks  recall  precis    error  purity  teams
    SNGS-116             85   67.3%   74.9%   0.54 m   64.4%    75%
    SNGS-121             50   15.8%   15.2%   1.30 m   61.3%    55%
    SNGS-147             88   41.3%   40.4%   1.33 m   73.6%    72%
    geny_rioave          29       no truth: 208 frames, home=23 away=4 gkHome=2
    nottingham           77       no truth: 695 frames, home=43 away=32 gkHome=2

A caveat on the numbers above, because it matters for anyone comparing them with the
cross-validation table earlier in this file: those two sets do not agree. `ft auto --mode
seed` plus `ft score` gives SNGS-121 15.8% recall where the table records 43.4%, and
`--mode truth` gives 53.1% recall at 51.2% precision where the table records 85.0%
precision. The difference is NOT this work -- the last commit before any of it scores the
same -- so the table was produced by a bespoke sweep rather than by these two commands, and
it should not be read as a baseline these commands reproduce. The snap-on against snap-off
comparison is internally consistent and is the one that decided the default.

**D34 — a seed is checked against the frame it claims to describe, not against its own
clicks; and one seed does not cross a broadcast clip.** Carried 453 frames, the Nottingham
seed lands 44 m from where the players actually are (129 m at worst). Nothing downstream
can tell: the tracks and the board share the wrong coordinate frame, so Pitchboard's
fidelity score stays excellent while the play happens in the wrong half. The single-seed
run also dropped ZERO detections as off-pitch where a correct model drops 4,358 — the
drifted homography was mapping the crowd and the dugout onto the grass, and "nothing to
filter" read as a clean clip.

So the pipeline now anchors on every clicked frame. `fill` already prefers a direct fit and
carries only the gaps, so more seeds shorten every chain rather than adding a mechanism.

That immediately made things worse, which is the real lesson here. A seed clicked across a
BAND of the frame is unconstrained in depth: three points along the top of frame 400 fit
their own clicks to 0.27 m median and put the horizon a third of the way DOWN the picture,
so two thirds of the frame mapped behind the camera and players landed at x = -93 m. Every
number the fit reported about itself was excellent. Anchoring on it is worse than having no
anchor there, because it is not wrong in proportion to distance - it is wrong AT the anchor.

    seed        points  traced  own residual   frame behind camera
    853             11      38   0.18 m med          0%
    903             10      31   0.21 m med          0%
    400              3      22   0.27 m med         33%   <- refused

`behind_camera` is the check the residuals cannot make. `h[2] . p` is the homogeneous
scale, so where it changes sign the ground plane has passed through infinity. A fit whose
frame straddles that line does not describe its own picture, whatever it says about the
points it was given. The margin is 0% against 33%, so the 25% threshold is not a boundary
anyone has to defend.

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
- Does the segmenter cope with a half-pitch framing, or only wide shots? Every frame it has
  been trained on is a wide tactical camera.
- ~~Is the accuracy ceiling the *labels* rather than the model?~~ **MEASURED, and largely
  yes.** Leave one marking out of a ground-truth frame, fit from the rest, and the held-out
  marking's own annotated points land 0.318 m (SNGS-147), 0.348 m (SNGS-116) and 0.512 m
  (SNGS-121) from where that fit says its line is. The annotation does not agree with ITSELF
  to half a metre — on SNGS-121 its self-disagreement IS the bar. See D36.
- **Why is SNGS-116 stuck?** 2.87–5.19 m across four configurations, immovable while the other
  two clips halved. Nothing has looked at which frames fail.
- Does 1920×1080 keep the gain going, or is 1280×720 where resolution saturates? 121 barely
  moved between 960 and 1280 (0.69 → 0.67) while 147 and 116 did, which reads like different
  clips hitting the ceiling at different points.
- Is diversity worth revisiting at 1280×720 or above? Run 3 tested it only at 960×540, and
  tested it badly — confounded with a dataset change. A diverse set captured at 1080p would
  be a real test; SN-Calibration-2023 cannot be one, because it is 960×540.
