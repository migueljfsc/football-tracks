# football-tracks — where this stands

Turn a broadcast football clip into player tracks in pitch metres, so Pitchboard can import a
play instead of the coach drawing it by hand.

**Target for v0 is 70%** — good enough that a coach corrects the board instead of drawing it
from nothing. A comparison, not an absolute. This is a proof, not a product.

This file is the current state and the next move. Everything durable lives in [`docs/`](docs):
[what and why](docs/overview.md), [the pipeline](docs/pipeline.md),
[the benchmark](docs/benchmark.md), and [the decisions](docs/decisions) — fifty of them, cited
from source comments by number, and most recording something that was measured and abandoned.

## Where this stands — 6 September 2026

**Every one of the eleven benchmark clips produces a full board.** `ft auto --mode seed` is
what ships, and through Pitchboard's importer it gives 18-21 players of 22, a handful of
scenes marking possession changes and real movement, and windows of 3 to 29 seconds. 225
fielded players across the eleven, 190 of them on the right side.

**A coach has now looked at the boards, and the verdict is the useful thing here.** They read
as football; the curved runs are right; wrong teams are annoying rather than fatal. What is
wrong is that passes and players go MISSING -- a centre back and the pass to him absent from
SNGS-151, a clearance drawn as a player carrying the ball. Everything shipped on 5-6 September
came out of watching a board beside its clip: the officials, the keeper, the taker of a
restart, a ball handed to a player who was not on the pitch. Two of those were invisible to
every metric in this repo.

### The constraint is registration coverage, and it is not what five training runs measured

Traced through the whole funnel on the clip a coach called bad:

    stage                                   SNGS-151   SNGS-060
    detector finds a visible player           96.7%      97.3%
    survives tracking                         90%        100%
    lands within 5 m of a real player         77.4%       95.9%
    dropped off the pitch                      997          25

**The detector is finished.** It finds 94-98% of visible players on every clip, measured in
image space where no camera model is involved. Training a bigger one buys two to four points.

**The loss is the camera model, and the two registrations fail in opposite directions.**

    SNGS-151   seed        750 frames solved   p50 1.24 m   p90 8.08 m   >5 m 22.6%
               segmenter   227 frames solved   p50 1.21 m   p90 4.06 m   >5 m  6.2%
    SNGS-116   seed        750 frames solved   p50 0.74 m   p90 7.86 m   >5 m 12.7%
               segmenter   594 frames solved   p50 0.56 m   p90 2.03 m   >5 m  3.1%

Seeding propagates a single human fit through every frame, so its coverage is total and it
drifts. The segmenter fits each frame on its own, so it is three to four times cleaner and
silent wherever the markings are too few. Neither has both, and a board needs both: the
segmenter's SNGS-151 board is 18 correct players over a six-second window against seed mode's
20 over eighteen seconds (D67).

**That is why the five runs "failed".** They were scored on `observed_error` -- accuracy on the
frames the model could already solve -- while the binding constraint is how many frames it
solves at all. The 0.5 m bar measured the wrong axis, and D62 already established that the
frames the segmenter refuses are ones where ground-truth LINES cannot fit either, so no amount
of training recovers them by fitting alone.

### The four runs

| run | resolution | trained on | SNGS-147 | SNGS-116 | SNGS-121 |
|---|---|---|---|---|---|
| 1 | 640×360 | 5 matches | 0.84 m | 3.76 m | 1.54 m |
| 2 | 960×540 | 5 matches | 1.13 m | 3.20 m | 0.69 m |
| 3 | 960×540 | 350 matches | 0.90 m | 5.19 m | 1.13 m |
| 4 | 1280×720 | 5 matches | 0.67 m | 2.87 m | 0.67 m |
| **5** | **1920×1080** | **5 matches** | **0.71 m** | **0.70 m** | **0.70 m** |
| | | *bar* | *0.5 m* | *0.5 m* | *0.5 m* |

Medians of `observed_error`. No run has yet cleared the bar.

**The "100% solved" in that column means something narrower than it reads.** `calib-eval`
scores only frames that carry ground-truth line annotations, sampled at `--stride 25`. Put
every frame of a clip through the same fitter and run 4 solves 83% of SNGS-147, 81% of
SNGS-116 and **51% of SNGS-121** — it refuses the rest rather than guessing, which is correct
behaviour and a very different number. Any claim about the solve rate has to say which
population it counted.

Medians of `observed_error` against a bar set before any of them were trained. **No run cleared
it, and D67 explains why that was the wrong question**: `observed_error` is conditioned on the
frames a model already solves, so it rewards refusing the hard ones. The full account of the
five runs is in [`docs/decisions/registration.md`](docs/decisions/registration.md) under D36 and
D67; what they established is in [`docs/benchmark.md`](docs/benchmark.md).

### What to do next — registration that keeps segmenter accuracy at seed coverage

This is the months-scale project and it is the only one with measured reason to expect a gain.
Nothing else on this list is close.

**The goal.** Every frame registered, and registered within about a metre. Today one approach
gives the first and the other gives the second. The shape of the answer is to propagate through
frames with no markings and RE-ANCHOR on segmenter fits wherever the pitch is visible, so drift
is corrected as it appears rather than accumulating to eight metres by the end of a clip.

**Measure `registered within N metres, as a share of ALL frames`.** Not `observed_error`, which
is conditioned on the frames a model already solves and therefore rewards refusing the hard
ones. Every run so far reports the wrong number, which is how a segmenter three times cleaner
than the shipping path was concluded to be worse than it.

**Re-anchoring has been tried once and made things worse, so it has to be gated.** `--snap`
refits each carried homography onto the painted lines it can see and is a plain regression on
tracks (D35). What is different now is `winnow` (D62), which judges a fit by whether the
PREVIOUS fit walked forward by measured motion agrees with it, separates good from bad by
twenty to one, and took SNGS-147's identity purity from 69.3% to 90.3%. An anchor that has to
pass winnow is not the anchor D35 measured.

**What training would then be for.** Raising the share of frames the segmenter can fit at all,
which is a coverage problem and not an accuracy one. The evidence on how:

- **Resolution works.** Run 5 at native 1080p took SNGS-116 from 2.87 m to 0.70 m, and it was
  the one clip four earlier configurations could not move.
- **More matches do not.** Run 3 multiplied them by seventy and made two clips of three worse.
- **A bigger backbone is untried.** DeepLabv3 on MobileNetV3 was chosen to train on a laptop.

**Not this:** another detector. Another segmenter run scored on `observed_error`. Another
attempt on the tracker's colour -- identity purity has resisted seven (D61). Another pass at
the ball -- its three faults are diagnosed and two are closed (D66).

**Before any of it, one evening.** The stated bar for v0 is 70%: good enough that a coach
corrects the board instead of drawing it. A coach has now seen four boards and the answer is
"better, still not useful". Getting that judgement on a clip where registration is GOOD --
SNGS-060 scores 95.9% of players within 5 m against SNGS-151's 77.4% -- would say whether this
project is one fix away or several, and it costs nothing.

## Milestones

| # | done when | est. |
|---|---|---|
| M0 | scaffold, stage 0, and the ground-truth path: `ft truth`, `ft render`, `ft score` | **done** |
| M1 | reprojected pitch lines sit on the real lines | **the binding constraint** (D67); the bar it was scored against measured the wrong axis |
| M2 | tracks survive 10s with few enough id switches to count | **partly**; stitching ships, purity 57-86% and stuck (D61) |
| M3 | teams cluster cleanly | **done** (D63); 85% of fielded players on the right side |
| M4 | **the top-down dot video looks like football** | `ft render` exists; never judged by eye |
| M5 | numbers resolve for ~40% of tracks, matching the label ceiling | **abandoned** (D32) |
| M6 | Pitchboard's `src/import/` turns `tracks.json` into a `BoardDoc` | **done**; all eleven clips make a board |

**M6 no longer waits for anything.** `ft truth` emits a real, correct `tracks.json` from
ground truth with no CV in the loop, so the TypeScript reduction is built against genuine
30-second passages of play rather than hand-written fixtures — and its output can be looked
at in Pitchboard while stage 1 is still failing.

That also inverts how the CV is judged. Every later stage is scored against the same file
in the same format, so "70%" becomes a diff against a known-good baseline rather than a
feeling about a video.

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
