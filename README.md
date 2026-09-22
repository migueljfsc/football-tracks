# football-tracks

Broadcast football clip in, player tracks in pitch metres out.

The output is `tracks.json` — every player's position per frame on the pitch, with a team and,
where it can be read, a shirt number. [Pitchboard](../pitchboard) imports that file and turns it
into a board, so a coach corrects a play instead of drawing one.

This repo knows nothing about Pitchboard's schema, and Pitchboard knows nothing about video.
They meet at `schema/tracks.schema.json`.

**Where it stands, and what to do next: [PLAN.md](PLAN.md).** What each stage does:
[`docs/pipeline.md`](docs/pipeline.md). Why -- and every attempt that failed:
[`docs/decisions/`](docs/decisions/README.md).

## A coach's clip

```sh
uv sync --extra vision

# The first clip of a match: extract, set the pitch size, click landmarks on a few frames.
# The clicks also fit where the match's camera stands.
uv run ft run ~/Downloads/clip.mov --game milan-benfica

# Every other clip of the same match: no clicks where the pitch lines can be read (D96).
uv run ft run ~/Downloads/clip_2.mov --game milan-benfica

# Then, in ../pitchboard: the board a coach will see, in one line.
pnpm board ../football-tracks/work/clip_2/tracks.json --scenes
```

`ft run` asks for what it needs and picks a clip up where it was left. The same steps one at a
time, for when a stage has to be redone on its own:

```sh
uv run ft frames clip.mov                       # -> data/clips/clip/img1/, bars removed
uv run ft pitch clip --length 105 --width 68    # BEFORE seeding, or landmarks lie (D89)
uv run ft seed clip                             # click landmarks on a frame
uv run ft camera clip --game milan-benfica      # fit the match's camera from the seeds
uv run ft detect clip                           # people and ball candidates, cached
uv run ft reid clip                             # how each player looks, cached
uv run ft auto clip --mode camera --game milan-benfica   # -> work/clip/tracks.json
uv run ft render work/clip/tracks.json          # the top-down dot video
```

A clip is about ten minutes of compute on a laptop, most of it detection.

## The benchmark

Eleven SoccerNet GSR clips with ground truth for every player, the ball and the pitch lines.

```sh
uv sync --extra data --extra vision
uv run ft fetch SNGS-147                 # one clip, ~150 MB by range request
uv run ft truth SNGS-147                 # ground truth -> work/SNGS-147/truth.json
uv run ft detect SNGS-147 && uv run ft reid SNGS-147
uv run ft bench --mode camera            # every clip end to end, one table
uv run ft reg-eval SNGS-147 --mode camera   # registration, over ALL frames (D67)
uv run ft score work/SNGS-147/tracks.json   # a prediction against the truth

# in ../pitchboard: possession frame by frame against the truth board (D101)
pnpm board ../football-tracks/work/SNGS-*/tracks.json --truth
```

`--mode camera` needs the match's camera, fitted from its OTHER clips so a benchmark clip is
never judged by a camera that saw it: `uv run ft camera <other clips> --game sngs-4 --truth`.
`--mode seed` simulates one click on the best frame; `--mode truth` holds registration fixed so
the score is tracking and teams alone.

Ground truth is `truth.json` and a prediction is `tracks.json` — same format from the same
writer, two names, so a stage cannot overwrite what it is about to be measured against (D14).
Artefacts land in `work/<clip>/` and are all reproducible, so `make clean` is always safe.

## The pitch-line segmenter

The match camera reads its lines from a segmenter trained on SoccerNet (D36). The weights are
never committed (`work/calib/segmenter.pt`); to train them again:

```sh
uv run ft calib-train --stride 10 --epochs 6 --batch 6 --holdout-games "7,8"
```

Held out by MATCH, never by clip: SNGS-116 and SNGS-121 are both game 7, so a clip-level split
puts the same stadium, camera and kit on both sides of it. SN-Calibration-2023 is added if it
has been fetched into `data/calib2023`; D36 found it did not help.

## Requirements

`ffmpeg`, and Python 3.12 via `uv` — not the system 3.14, whose wheels the vision stack does
not have yet.
