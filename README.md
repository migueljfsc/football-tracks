# football-tracks

Broadcast football clip in, player tracks in pitch metres out.

The output is `tracks.json` — every player's position per frame on a 105 × 68 pitch, with a
team and, where it can be read, a shirt number. [Pitchboard](../pitchboard) imports that file
and turns it into a board, so a coach corrects a play instead of drawing one.

This repo knows nothing about Pitchboard's schema, and Pitchboard knows nothing about video.
They meet at `schema/tracks.schema.json`.

**Read [PLAN.md](PLAN.md).** It has the stages, what each one has to prove before the next is
worth starting, and the decisions behind the shape of all this.

## Running it

```sh
uv sync --extra data                 # stage 0 plus the SoccerNet fetcher

uv run ft clips --split test         # 49 clips, downloads nothing
uv run ft fetch SNGS-147             # one clip, ~150MB out of an 8.85GB split
uv run ft truth SNGS-147             # ground truth -> work/SNGS-147/truth.json
uv run ft render work/SNGS-147/truth.json          # top-down mp4 of coloured dots
uv run ft render work/SNGS-147/truth.json --still 220
uv run ft score work/SNGS-147/tracks.json          # diff a prediction against the truth

uv run ft calibrate SNGS-147         # stage 1: fit per frame, and measure the error
uv run ft calibrate SNGS-147 --carry 0     # solver only, no propagation across gaps
uv run ft calibrate SNGS-147 --frame 288   # the overlay - do lines land on lines?
uv run ft calibrate SNGS-147 --video

uv run ft detect SNGS-147            # stage 2a: find people, cached
uv run ft auto SNGS-147 --mode seed  # the whole pipeline from ONE seeded frame
uv run ft score work/SNGS-147/tracks.json

# YOUR OWN broadcast clip, which has no annotations
uv run ft frames my_goal.mov         # -> data/clips/my_goal/img1/, bars removed
uv run ft seed my_goal               # click pitch landmarks on frame 1
uv run ft calibrate my_goal --frame 1      # check: do lines land on lines?
uv run ft detect my_goal
uv run ft auto my_goal --mode seed         # -> work/my_goal/tracks.json
uv run ft render work/my_goal/tracks.json  # the top-down proof

uv run ft segment data/clips/foo.mp4 # stage 0, for arbitrary broadcast footage
```

`ft truth` produces a real tracks file with no CV in the loop. It is the yardstick every
stage is scored against, and it is what Pitchboard's importer is built against today.
Ground truth is `truth.json` and a prediction is `tracks.json` — same format, two names, so
a stage cannot overwrite what it is about to be measured against.

Artefacts land in `work/<clip>/`, one directory per clip. Everything in there is reproducible
from the clip plus a stage, so `make clean` is always safe.

Stages after 0 need heavier dependencies, installed when that stage is built:

```sh
uv sync --extra vision   # detection, tracking, pitch keypoints
uv sync --extra ocr      # shirt numbers
```

## Status

| stage | | |
|---|---|---|
| — | SoccerNet fetch, ground truth, render, score | **done** |
| 0 | segment — find the tactical camera | built, unproven |
| 1 | registration — pixels to metres | solver done, detector next |
| 2 | detect and track | RT-DETR; purity 86% and precision 83% at 7s |
| 3 | teams | clustering near chance — needs work |
| 4 | project — **the proof** | working |
| 5 | shirt numbers | not started |
| — | the ball's holder | 99% right within 4m; declines when it is in flight |

From one seeded frame, a 7-second clip comes out at 97% recall, 0.80 m median error and
86% identity purity. Past ~7s the carried homography drifts and recall halves.

SoccerNet clips are single-camera and already trimmed, so they enter at stage 1 — stage 0
is for arbitrary broadcast footage (D10).

**It runs end to end on real broadcast TV.** On a sport.tv recording of a Rio Ave goal —
screen-captured, pillarboxed, night match, faint markings, no annotations of any kind —
one seeded frame produces a top-down reconstruction of the play. Objectively: every
detected pitch marking projects onto the pitch, with a median of **0.11 m** from the real
line it belongs to, and 78% inside half a metre.

## Requirements

`ffmpeg`, and Python 3.12 via `uv` — not the system 3.14, whose wheels the vision stack does
not have yet.
