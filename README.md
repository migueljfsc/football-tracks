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
uv sync                              # stage 0 deps only
uv run ft segment data/clips/foo.mp4 # or: make segment CLIP=data/clips/foo.mp4
```

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
| 0 | segment — find the tactical camera | built, unproven |
| 1 | registration — pixels to metres | not started |
| 2 | detect and track | not started |
| 3 | teams | not started |
| 4 | project — **the proof** | not started |
| 5 | shirt numbers | not started |

## Requirements

`ffmpeg`, and Python 3.12 via `uv` — not the system 3.14, whose wheels the vision stack does
not have yet.
