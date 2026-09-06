# The benchmark

What this is measured on, how, and what the numbers mean. Nothing in this repo is believed
without a measurement, and several things that measured well are recorded as failures because
they did not survive the board.

### The three clips this is scored on

`SNGS-147`, `SNGS-116`, `SNGS-121` — held out by match, never by clip (116 and 121 are both
game 7, so a clip-level split would leak). `--holdout-games "7,8"` is what keeps them out.

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
