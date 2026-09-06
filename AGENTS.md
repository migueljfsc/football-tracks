# football-tracks — broadcast clip to player tracks

Working conventions for this repo. Where it stands and what to do next is [`PLAN.md`](PLAN.md);
what it is and how it works is [`docs/`](docs), and the reasoning behind every choice — fifty
numbered decisions, cited from source comments — is [`docs/decisions/`](docs/decisions).

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
PLAN.md                     where this stands, and the next move
docs/                       what it is, the pipeline, the benchmark, the decisions
schema/tracks.schema.json   the contract, frozen before the CV works (D3)
src/football_tracks/
  config.py                 pitch dimensions and work paths — the one source
  stage0_segment.py         cuts -> score by green and motion -> the tactical camera
  soccernet.py              GSR fetch and ground truth -> Tracks
  tracks.py                 the tracks.json writer, shared by every producer
  calibration.py            named pitch lines -> a homography
  calib.py                  the LEARNED detector — frame -> named lines, no seed (D36)
  refine.py                 snap a homography onto the painted lines. Off by default (D35)
  stage1_register.py        fit per frame, and measure what it costs
  stage1_propagate.py       carry a homography across gaps by tracking the grass
  video.py                  a recording -> the numbered-JPEG layout, bars removed
  seed.py                   clicked landmarks -> a homography; seed.json is the format
  seedui.py                 the click tool. Disposable: the FILE is the interface (D23)
  detect.py                 stage 2a, RT-DETR (Apache) - see D28 for why not the others
  stage2_track.py           stage 2b, association in stabilised pixels
  stage3_teams.py           kit clustering, and which end each side plays at
  auto.py                   the whole automatic path, frames in and tracks.json out
  overlay.py                the markings reprojected onto a frame - stage 1's picture
  pitch.py                  the markings in metres; `model()` is the one description
  render.py                 a tracks file -> a video of coloured dots
  score.py                  a prediction diffed against ground truth
  cli.py                    one command per stage
tests/                      the pure helpers only
data/clips/                 source video, never committed
data/calib2023/             SN-Calibration-2023, training data for the segmenter only
work/<clip>/                every stage's artefacts, all reproducible
work/calib/                 segmenter weights and training logs. Gitignored — see below
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

Each of these cost a day. Where one names a decision, the full account is in
[`docs/decisions/`](docs/decisions).

### The contract

- **Track samples are sparse** (D8). A player occluded for twenty frames has no position for
  twenty frames. Gaps are expected and must never be invented — a consumer interpolates.
- **Frame indices are absolute**, in the source video's numbering, never relative to a window.
- **Ground truth is `truth.json`, a prediction is `tracks.json`** (D14), same format from the
  same writer. Regenerate truth before trusting any score, and only with `--interval-s 0`.
- **A `null` shirt number is an answer, not a gap** (D5). Unread imports as a generic token; a
  guessed one attaches a run to the wrong player and nothing downstream can tell.

### Geometry

- **A homography assumes z = 0.** Ball height is not recoverable from one camera, so a ball in
  flight lands metres from where it is — 8-26% of SoccerNet's own ball annotations project off
  the pitch for this reason. Aerial play cannot be drawn (D66).
- **A fit from four lines is exactly determined**, so it has no residual and cannot be checked
  (D17). Ask for more lines than the fit needs.
- **Two perpendicular pitch lines project to nearly parallel image lines** under an oblique
  camera, which is degenerate however clean the click. Prefer a structural guard to a numerical
  one.
- **RANSAC's threshold is in the DESTINATION space** — metres here, not pixels.
- **`carry` composes as `h @ inv(d)`, not `d @ h`.** Both produce a plausible matrix.
- **A goal post is not on the ground plane.** `PITCH_LINES` lists ground markings only.
- **FAR and NEAR, never left and right.** Which post is "left" depends on the camera.
- **`pitch.py` is for looking, never for measuring.** It draws; `pitch.model()` is the one
  description of the markings.
- **A carried homography drifts without bound and never announces it** (D18).

### Tracking

- **Association happens in STABILISED PIXELS; projection comes after** (D22). The other order
  makes stage 2 inherit every wobble in stage 1.
- **An id switch looks exactly like a fast run.** Two crossed players produce one plausible
  track, so recall cannot see it (D15) — only ground truth can.
- **Retire stale tracks BEFORE associating.** The gate grows with the gap, so a track that is
  already too old would otherwise match anyway.
- **Off-pitch people are dropped before tracking, not after** (D27).
- **A raw track count overstates fragmentation.** What matters is how much of a player's time
  a track covers, which is what the importer asks.
- **A bad camera frame throws every player at once**, so registration failures look like
  tracking failures (D62).

### Teams and the ball

- **k-means collapses on kit colours** (D31). It minimises inertia, so the cheapest split is one
  tight cluster and one holding everybody else. The axis of greatest variance and an Otsu cut.
- **A goalkeeper is not a third team**, and leaving him in the clustering costs both sides.
  Neither is an official (D64).
- **The ball's POSITION is not usable and its HOLDER is** (D29).
- **Picking the ball candidate nearest a player is worse than picking the most confident one**
  — there are a dozen candidates a frame and "nearest a player" selects whichever false
  positive stands beside somebody.

### Measuring

- **Judge a change by `boardFromTracks`, never by a per-frame metric alone** (D36). Six separate
  per-frame wins have failed to reach the board.
- **`observed_error` is conditioned on the frames a model already solves**, so it rewards
  refusing the hard ones (D67). For registration, measure the share of ALL frames.
- **One variable per run.** Runs 1-3 each moved two and none can be read.
- **Sweeping a constant on a clip where its failure does not occur measures nothing.**
- **Validation loss does not predict `observed_error`.** Judge on the eval, never the loss.
- **`ft truth` drops referees unless asked.** Any accuracy measured without `--referees` cannot
  see an official fielded as a player (D64).

### Environment and video

- **CI has base dependencies and no ffmpeg.** Verify against that, not against a laptop.
- **Never train the combined set above 960×540.** SN-Calibration-2023 is natively 960×540, so
  anything higher upsamples 82% of it (D36).
- **`ffmpeg -c copy` cuts only on keyframes.** Extraction re-encodes for that reason.
- **A container's frame rate is not the clip's.** A screen recording claimed 120fps.
- **Re-cropping moves every pixel a seed holds.** `seed.json` is in cropped coordinates.

## Credentials and weights

The SoccerNet password is under their NDA. It lives in `.env` (gitignored) and is read from
the environment. It is never committed, never hard-coded, and never pasted into a chat
transcript — including to an assistant, which does not need to see it to write code that
reads `SOCCERNET_PASSWORD`.

**Not every SoccerNet task needs it, and one of them rejects it.** calibration-2023 is public:
the NDA password gets a 401 there where the library's own public default gets a 200. Neither
dataset used here is actually gated — GSR-2025 comes ungated from HuggingFace — so the
password has never been exercised. If a download 401s, check whether the task is public before
assuming the credential is wrong.

**Trained weights are never committed.** `work/` and `*.pt` are gitignored, which is what keeps
a 42 MB checkpoint derived from licensed data out of a public MIT repo. That is a structural
guard, not a habit — do not add a path that escapes it.

## Definition of done

`uv run ruff check . && uv run mypy && uv run pytest` clean — and, for anything touching a
frame, the stage's own picture, looked at.

## Git

Never create branches, commits, or PRs unless explicitly asked. "Fix X" means prepare the
change, not commit it.
