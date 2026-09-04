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

- **Shirt-number OCR was measured and rejected** (D32). A general-purpose reader gets one
  digit out of a two-digit number and is confident about it, so numbers come out wrong
  four times as often as right. Do not reach for a different OCR package; the crop and the
  recogniser both need to be jersey-specific.
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
- **A point-on-line constraint admits a solution that collapses the image to a point.** Two
  traced lines cross somewhere, and mapping everything to that crossing satisfies both
  exactly, with zero residual (D26). Never read a small residual as a good fit without
  asking whether the constraints could be satisfied trivially.
- **RANSAC keeps whichever subset it can fit, not the one worth fitting.** On the first real
  seed it kept five points along the goal line, fit them to 0.1 m, and rejected exactly the
  points that pinned down the pitch's depth. The inlier set is checked for degeneracy now,
  not just the input.
- **A least-squares fit spreads one bad click over every other point.** Never conclude
  from "all the residuals are large" that all the clicks are bad — drop the worst, refit,
  and look again. The single-pass version dropped all eleven landmarks of a real seed and
  returned the fit it was repairing.
- **Trimming repairs bad clicks, not bad geometry.** If every traced marking runs the same
  way, the lines pin down depth and nothing pins down across, so two landmarks swapped
  across the pitch fit as well as the truth and nothing can prefer one. The answer there is
  a line that crosses the others, not a better solver.
- **The four most natural landmarks to click are collinear** (D24). Both posts and both
  corners of a goal all sit on x = 0, and a homography fitted to them fits perfectly and
  describes nothing. `seed.degenerate` refuses that set.
- **RANSAC's threshold is in the DESTINATION space.** For every homography fitted here
  that is pitch METRES, so the usual pixel default of 5 means five metres and accepts
  almost any error. It also cannot be loose: with six points there is barely more data
  than there are degrees of freedom, so a loose threshold buys a warped fit that swallows
  the bad point instead of rejecting it.
- **CI has base dependencies and no ffmpeg.** Verify against that, not against a laptop
  with everything installed — twice now a green local run has pushed a red CI. The heavy
  imports are inside the functions that need them for the same reason.
- **Prefer a structural guard to a numerical one.** Two traced lines are ALWAYS degenerate
  and that can be counted; checking whether the resulting fit "collapses" is at the mercy
  of which machine ran the SVD, and passed locally while failing on CI.
- **A container's frame rate is not the clip's.** A screen recording claimed 120fps while
  holding 208 frames across 6.4 seconds. `video.probe` derives it from duration and count,
  because these numbers become scene durations.
- **The video is the longest CONTIGUOUS run of content, not the outermost bright columns.**
  A recorder's own furniture — a subscribe button, a watermark — sits out in the bar, and
  taking first-to-last swallows everything between it and the picture. That is how a
  3354px frame came back uncropped with 290 black columns in it.
- **Sample the crop across frames and take a MEDIAN, never a union.** One flash, fade or
  overlay widens a unioned crop for the whole clip.
- **Re-cropping moves every pixel coordinate a seed holds.** `seed.json` is in cropped
  frame space, so changing the crop invalidates it — shift it by the delta rather than
  asking for it to be clicked again.
- **Pillarbox bars are not black enough to ignore.** Compression noise lifts them over the
  grass mask's value floor, so they register as pitch and the optical flow tries to track
  them. That is what `video.content_box` is for, and it uses a max rather than a mean: one
  bright pixel anywhere in a column means that column is content.
- **Sweeping a constant on a clip where its failure does not occur measures nothing.**
  Weighting kit colour more heavily looked useless for most of this project, because it
  was swept on a clip whose steals happened after gaps, where the right player was simply
  absent and no colour could have helped. In a crowd the wrong candidate is an opponent
  standing right there. Check which failure a clip actually has before concluding from it.
- **Tracking fails differently in a crowd.** On SNGS-147 (7 players a frame) steals follow
  gaps; on SNGS-116 (13.5 a frame, players 2.13 m apart) 93 of 111 are between players both
  visible at the time. A fix for one is not a fix for the other.
- **Kit colour cannot solve a crossing, and no weight makes it.** 18% of boxes in a crowded
  box overlap another by over 30%, so the torso crop blends two players and the signature
  is worthless in precisely the case it exists for. Crowded identity needs a learned
  re-identification model; it is a stated limitation, not a tuning problem.
- **A homography good enough for stage 1 can be useless for stage 2** (D19). Tracking needs
  frame-to-frame CONSISTENCY, not absolute accuracy, and nothing in `Registration` measures
  that. Carrying improves stage 1's card and halves identity purity.
- **Track association happens in STABILISED PIXELS, and projection comes after** (D22). Raw
  pixels lose a panning camera; pitch metres inherit every wobble in the homography. The gate
  is a speed converted through the box height, so stage 2 needs no camera model at all.
- **k-means collapses on kit colours** (D31). It minimises inertia, and same-kit tracks
  vary more in light than two kits differ, so the cheapest split is one tiny cluster
  against everyone. Split on the first principal component by between-class variance,
  whose `len(a) * len(b)` term is what keeps the two sides comparable in size.
- **A goalkeeper is not a third team, and leaving him in the clustering costs both.** He
  is removed by being an odd colour AND near a goal — either test alone is wrong.
- **A raw track count overstates fragmentation.** What matters is how much of a player's
  life their best track covers. Fifty tracks where thirteen cover half the clip is a usable
  reconstruction; the rest are fragments a reduction drops.
- **The ball's POSITION is not usable and its HOLDER is** (D29). A ground homography
  assumes z = 0, so a ball in flight lands metres away. Never write a ball position into a
  board; derive the carrier and let the board draw the pass.
- **Picking the ball candidate nearest a player is worse than picking the most confident
  one** — 55% against 99%. With five candidates a frame, "nearest a player" selects
  whichever false positive is standing beside somebody. Confidence, then a median over
  neighbouring frames, because the real ball is the one that moves smoothly.
- **A detector's false positives cost as much as its misses** (D28). Everything it invents
  competes for associations and spawns a track, so the confidence floor stays at 0.5 even
  though 0.4 buys four points of recall. Judge a detector on both columns.
- **Off-pitch people are dropped before tracking, not after** (D27). Two fifths of the
  detector's output is crowd and staff, and until they are filtered they compete for
  associations. This is the one place stage 2 consults stage 1, and only as a filter.
- **A bad homography now shows up as an identity error, not a tracking error.** The tracker
  is independent of it, but a sample projected five metres out matches a different player and
  scores as a switch. Do not read a purity drop as a tracker regression without checking what
  the positions did.
- **Retire stale tracks before associating, not after** — the gate grows with the gap, so the
  other order lets an already-dead track match anyway, and whether it does depends on whether
  the caller's frame list is contiguous.
- **A carried homography drifts without bound, and never announces it** (D18). Every
  composition multiplies in the last one's error; the failure is a matrix that still looks
  like a matrix. `DEFAULT_MAX_CARRY` is the guard, and it is a measured number, not a
  round one.
- **FAR and NEAR, never left and right.** Which post is "left" depends on where the camera
  stands and is ambiguous on a screen. The broadcast camera sits on one touchline, so near is
  always the bottom of the frame — and that is the same convention the pitch y axis uses.
- **A seed propagates both ways** (D25). The best frame to seed is rarely the first one.
- **A chain cannot start itself.** Propagation needs a first homography from somewhere —
  the solver, a keypoint model, or a human. Anything assuming `fill` alone can calibrate a
  clip has missed that it takes `direct` as input.
- **`carry` composes as `h @ inv(d)`, not `d @ h`.** Both produce a plausible matrix and
  the wrong one drifts the wrong way; no type catches it, so `test_propagate` pins the
  direction with a point that must land on the same metre in both frames.
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
- **A label lookup that does not `.strip()` fails as silence.** SN-Calibration-2023 writes
  `"Goal left post left "` with a trailing space, and `INDEX.get(name)` returns `None` for it
  — 1,101 instances dropped as an unknown marking, and a class that then appears in no label
  at all. Nothing raises, nothing warns, the class simply never learns. The strip lives at the
  single lookup site in `rasterise`, not at each caller.
- **A training frame with no match tag silently disables the leakage guard.** `split_by_game`
  can only hold a match out if every `Frame` knows which match it came from, and a dataset
  read without its `match_info.json` would tag them all alike. The split would go on passing
  and testing nothing. Any new source of frames owes a real match key before it owes anything
  else.
- **Regenerate `work/*/truth.json` before trusting a score, and only with `--interval-s 0`**
  (D55). `ft score` counts samples, so a yardstick written on a 0.1 s grid holds two fifths of
  what a 25 fps run holds and halves every precision figure without any pipeline change. The
  files in `work/` outlive the code that made them and record nothing about their provenance.
- **Max travel on a board is a SWITCH detector, not a quality metric** (D56). A track that
  hops between players covers more ground than a player can, so the boards with the most
  movement are often the ones with the worst identities. Check that the longest run belongs to
  one ground-truth player before calling it a good board.
- **Split identity switches into steals and fragments before fixing them** (D56). They have
  opposite cures -- a steal wants a better position prior, a fragment wants a longer or
  re-joined track -- and the ratio differs per clip: 93% steals on SNGS-116, 48% fragments on
  SNGS-147. A single "switches" number hides which one you have.
- **Seed from the best-evidenced frame, never the earliest** (D55). Count visible markings, not
  the fit's own residual -- a barely-solvable frame minimises that residual for the same reason
  it is fragile. Seeding SNGS-121 on its first solvable frame instead of its best cost 56 points
  of recall.
- **Check the detector's INPUT SIZE before tuning anything downstream of it** (D57). The
  processor resizes to 640x640, so a 1920x1080 frame loses two thirds of its width and the
  ball arrives three pixels across. Four separate ball fixes were made before anyone asked
  whether the ball was in the candidate pool at all.
- **`info.action_class` names the set piece and `info.action_position` gives its frame**, in
  every GSR clip -- 19 set pieces across the 60 on disk. An evaluation set for whether a board
  tells the right story, without training anything.
- **Ask the ANNOTATIONS before blaming the footage** (D54). "The corner is off-camera" was
  written from looking at a frame and missing a ball that SoccerNet annotates at the corner
  flag. `category_id` 4 is the ball and it is labelled in 724 of SNGS-116's 750 frames.
- **The ball is right or frequent, never both** (D54). The detector sees it in 41% of frames
  and calls the PENALTY SPOT a ball — a white circle on grass is exactly what it was trained
  to find. At its 0.15 floor the asserted ball is the real one 25% of the time. Anything
  loosening `BALL_ASSERT_CONF` must be measured against SoccerNet's category-4 annotations,
  not judged by how often a ball appears.
- **`on_pitch`'s margin is a SHARE of the pitch, not metres** (D58). `mx = PITCH_LENGTH *
  margin`, so a "1.5 m" margin is 157 m and admits every projection the detector can make.
  The ball has `_ball_near_pitch` for metre-space; players keep `on_pitch`. A unit that lives
  in a name and not in a type is worth one look per use.
- **`action_position` is when the set piece is TAKEN, not when the ball is placed** (D59). At
  the labelled frame the ball is already airborne and projects off the pitch — SNGS-116's
  corner lands at (107.5, -3.9). The still ball is in the frames BEFORE it.
- **The restart prior is safe because of its VETO, not its radius** (D59). Within 1.5 m of a
  corner arc or the centre spot, 89 of 91 candidates are the real ball on a clip with a corner
  in it — and on a clip WITHOUT one, the same region is 32 false positives at the corner flag.
  What separates them is that the pass only speaks where the pipeline would emit no ball at
  all. Read that veto from the SMOOTHED output: two isolated wrong sightings are not a tracked
  ball, and letting them veto costs 66 frames of a real corner.
- **Only corners and kick-offs have a canonical position** (D59). Kick-offs sit on the centre
  spot to 1.2 m; direct free-kicks are taken wherever the foul was. A positional prior covers
  13 of the 19 set pieces in this dataset and cannot be stretched to the rest.
- **A restart run's LENGTH is as load-bearing as its radius** (D59). Every run the prior gets
  wrong is short and transient — 5 frames on a centre spot during a free kick, 8 at a corner
  just after it was taken — and every run it gets right is 16 to 80. At `RESTART_MIN_FRAMES`
  of 4 the free-kick clip SNGS-066 gained 12 fabricated frames; at 10 it gains none.
- **A better ball does not mean a better board, and here it mostly did not** (D59). Pitchboard's
  `chooseWindow` maximises trackable PLAYERS and never looks at the ball, so it picks the open
  play after a set piece over the set piece itself — players bunched in a box occlude each
  other and their coverage drops. Four of five improved clips produced a byte-identical board.
  Measure a ball change through `boardFromTracks` before believing it shipped anything.
- **Judge a change by `boardFromTracks`, never by `observed_error` alone** (D36). The
  4 September segmenter work improved every per-frame metric in this repo and made the board
  in the sibling repo strictly worse — 19 players to 10, 15 m of travel to 4.9 m. Per-frame
  accuracy is a proxy; the board is the product. Run the variant through `src/import/` before
  believing any of it, which costs twenty minutes.
- **A confidence gate on a panning camera is a spatial filter in disguise.** "Refuse the frames
  you fit worst" and "refuse the frames looking at the far end" are the same instruction, so
  gating on fit residual quietly restricted the board to a third of the pitch and left
  `chooseWindow` no window but a static one.
- **Resolution is the segmenter's binding constraint, not match diversity** (D36). The
  opposite was written here after run 2 and it was wrong: run 3 raised the matches from five
  to 350 and made two benchmark clips of three WORSE, while run 4 raised the resolution alone
  and improved all three. Trust that ordering only as far as it was measured — it comes from
  the one run in the series that changed a single variable.
- **A training run that moves two variables cannot be read** (D36). Runs 1-3 each changed
  resolution and data together, and the scatter between them turned out to be as large as the
  differences they were supposed to demonstrate. Change one thing and keep the log header
  byte-identical to its baseline; run 4 is the pattern.
- **Never train the combined set above 960×540.** SN-Calibration-2023 is natively 960×540, so
  a higher `WIDTH, HEIGHT` upsamples 82% of the frames — no added detail, and the model learns
  to expect a blur that inference will not supply. GSR is 1920×1080, so above 960×540 pass
  `--no-extra` and train on GSR alone. `LINE_PX` scales with the resolution for the same
  reason: it is a dilation of a line whose apparent width grows with the image.
- **The validation loss does not predict `observed_error`.** Run 3 had the worse loss and a
  better median on SNGS-147; the two only agreed once, between runs 2 and 4. The verdict is
  `calib-eval` on all three clips. Losses from runs with different validation sets are not
  comparable at all.
- **Five matches overfits at epoch 2**, at every resolution tried. Only the best checkpoint is
  kept, so a long run is not wrong, merely wasteful — budget ~4 epochs to find the checkpoint
  and treat the rest as confirmation.
- **Do not chain `refine` after the segmenter by default.** It rescues two benchmark clips
  and wrecks the third — p90 2.76 m to 17.65 m (D36). And do not try to choose between the two
  fits by which better explains the segmenter's own pixels: the mask fit was fitted to minimise
  exactly that quantity, so the test is rigged and picks it 45 times out of 65. Selecting on
  the data you fitted on is not selection.
- **Ultralytics YOLO is AGPL-3.0** (D9) while this repo is MIT. Fine for a local proof that
  distributes nothing; a real question the moment anything ships. RT-DETR and RF-DETR are
  Apache and are the swap — which is why the detector sits behind a stage boundary.

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
