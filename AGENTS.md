# football-tracks — broadcast clip to player tracks

Working conventions for this repo. The stages, what each has to prove, and the reasoning
behind every choice live in [`PLAN.md`](PLAN.md).

## Mission

A few seconds of broadcast football in, `tracks.json` out — every player's position per
frame in metres on a 105 × 68 pitch, with a team and, where it can be read, a shirt number.
[Pitchboard](../pitchboard) imports that file so a coach corrects a play instead of drawing
one from nothing.

Target accuracy is 70%. This is a proof, and it runs locally from a terminal on files.

## The three invariants

1. **`tracks.json` is the only contract.** Nothing downstream of it touches video; nothing
   upstream of it knows Pitchboard's schema. The video half is the part most likely to be
   thrown away and rewritten, and it must be replaceable without the TypeScript half
   noticing.

2. **No pixels cross the boundary.** Positions in `tracks.json` are pitch metres, origin at
   the top-left of a 105 × 68 pitch — the same space and origin `BoardDoc` uses. Image
   coordinates stay inside the stage that produced them. `config.py` holds the dimensions so
   no stage invents its own.

3. **Every stage writes an artefact, and every stage has a picture.** A homography cannot be
   unit-tested; you look at it. Each stage drops output in `work/<clip>/` and renders an
   overlay proving what it claims. A stage with no visual check is a stage you cannot debug.

## Repository layout

```
PLAN.md                     stages, kill criteria, decisions D1-D9, non-goals
schema/tracks.schema.json   the contract, frozen before the CV works (D3)
src/football_tracks/
  config.py                 pitch dimensions and work paths — the one source
  stage0_segment.py         cuts -> score by green and motion -> the tactical camera
  soccernet.py              GSR fetch and ground truth -> Tracks
  tracks.py                 the tracks.json writer, shared by every producer
  calibration.py            named pitch lines -> a homography
  stage1_register.py        fit per frame, and measure what it costs
  overlay.py                the markings reprojected onto a frame - stage 1's picture
  pitch.py                  the markings in metres; `model()` is the one description
  render.py                 a tracks file -> a video of coloured dots
  score.py                  a prediction diffed against ground truth
  cli.py                    one command per stage
tests/                      the pure helpers only
data/clips/                 source video, never committed
work/<clip>/                every stage's artefacts, all reproducible
```

## Conventions

- uv, Python 3.12 pinned. mypy strict, ruff.
- Conventional Commits, enforced by commitizen in `commit-msg` and by CI on PRs.
- `pre-commit install` after cloning.
- Tests are pytest, numerical helpers only — no tests that need a video file. The visual
  check is the test for anything touching a frame.
- Stage dependencies are extras (`--extra vision`, `--extra ocr`), not base deps. Torch is
  ~2GB and stage 0 does not need it.

## Known traps

- **A `null` shirt number is an answer, not a gap** (D5). Unread imports as a generic token;
  a *wrong* number silently attaches a run to the wrong player, and nothing downstream can
  see that it happened. Never guess, never fill in a plausible value.
- **Track samples are sparse** (D8). A player occluded for twenty frames has no position for
  them. Inventing one gives the reduction a bezier to fit to a lie. Every sample carries its
  own absolute frame index; gaps are expected.
- **Frame indices are absolute, in the source video's numbering** — not relative to the
  segment. Everything after stage 0 works on a trimmed clip, so an off-by-N here is silent
  and shows up as tracks that do not line up with the video.
- **`ffmpeg -c copy` cuts only on keyframes.** Extraction re-encodes for that reason: a copy
  puts the boundary somewhere other than the cut and leaves frames of the wrong shot at the
  front of the segment.
- **An id switch looks exactly like a fast run.** The tracker crossing two players produces
  a position sequence the reduction happily fits a curve to. This is the failure the whole
  pipeline is most likely to die of, and it is invisible in the numbers — only the top-down
  dot video shows it.
- **A single homography assumes z = 0.** Ball height is not recoverable from one camera, so
  a chip and a ground pass are the same measurement. Pitchboard's `loft` cannot be filled in
  from this data, and guessing it is worse than leaving it false.
- **Ground truth is `truth.json`, a prediction is `tracks.json`** (D14). Same format, two
  names, so a stage cannot overwrite the yardstick it is about to be scored against.
- **Recall cannot see an identity switch.** A tracker that swaps two players still finds
  everybody. Purity and the switch count are what show it, which is why `score` reports
  them beside recall rather than folding everything into one number (D15).
- **A homography fitted from four lines is exactly determined, so it has no residual and
  cannot be checked** (D17). It fits its own points perfectly whatever the noise. Never
  treat a small residual as evidence a fit is good without first asking whether the system
  was over-determined.
- **Two perpendicular pitch lines project to nearly parallel image lines under an oblique
  camera**, which is what makes their intersection useless and why the fit is point-on-line
  (D16). Anything reaching for line crossings is reaching for the approach that produced a
  0.00 m residual and a 100 m error.
- **A goal post is not on the ground plane.** `PITCH_LINES` lists only ground markings, and
  crossbars and posts are deliberately absent — a ground homography puts them metres from
  where they are. Circles are absent too, for the different reason that they are not lines.
- **`pitch.model()` is the one description of the markings.** `draw` renders it top-down and
  `overlay` reprojects it onto a frame. A second copy is two answers that drift.
- **`pitch.py` is for looking, never for measuring.** It exists to draw a picture. The
  moment anything reads geometry out of it there are two answers to where the penalty spot
  is, and they drift the way preview and export do in Pitchboard.
- **The stands are green too, sometimes.** The pitch test leans on the saturation floor, not
  the hue: grey has no meaningful hue, so hue alone calls it green.
- **Ultralytics YOLO is AGPL-3.0** (D9) while this repo is MIT. Fine for a local proof that
  distributes nothing; a real question the moment anything ships. RT-DETR and RF-DETR are
  Apache and are the swap — which is why the detector sits behind a stage boundary.

## Credentials

The SoccerNet password is under their NDA. It lives in `.env` (gitignored) and is read from
the environment. It is never committed, never hard-coded, and never pasted into a chat
transcript — including to an assistant, which does not need to see it to write code that
reads `SOCCERNET_PASSWORD`.

## Definition of done

`uv run ruff check . && uv run mypy && uv run pytest` clean — and, for anything touching a
frame, the stage's own picture, looked at.

## Git

Never create branches, commits, or PRs unless explicitly asked. "Fix X" means prepare the
change, not commit it.
