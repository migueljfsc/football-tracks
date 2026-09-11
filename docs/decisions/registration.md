# Decisions — registration

Stage 1 — turning a frame into a camera model. The longest-running problem here.

Every decision here was measured before it was made. They are kept because the code
cites them by number, and because the failures are worth as much as the successes.

**D7 — automatic registration first, human-seeded as the fallback.** Automatic is the only
version that scales past a demo, and a published model already exists, so it is worth one
honest attempt. But the fallback is genuinely good — a human clicking four landmarks is more
accurate than any model — and reaching for it is not a failure. The decision to switch belongs
at the end of M1, judged on the reprojection picture.

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

**D23 — the seed is a file, not a UI.** `seed.json` holds clicked landmark
correspondences and nothing else. The click tool writes it, but so could a keypoint
model, and so could Pitchboard's own import view — which is where this ends up, so the
format is the interface and the OpenCV window is disposable.

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

**D25 — a seed propagates in BOTH directions.** A clip is rarely best seeded at its first
frame: the camera is often still finding the play, and on the Rio Ave clip frame 1 has the
goal half out of shot with almost no markings visible while frame 100 has the whole box.
Forward-only propagation would make the good frame useless. `fill` therefore runs a backward
pass as well, composing `h @ d` rather than `h @ inv(d)`.

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

**D62 — the camera model is the biggest lever in the pipeline, and the segmenter's problem is
neither coverage nor accuracy but a tail of confidently wrong fits.** Measuring where
ground-truth players are lost, rather than following the last visible defect:

    clip        detected in the image   projected within 2 m   the homography loses
    SNGS-116          92.0%                   71.3%                  22.6%
    SNGS-110          83.2%                   44.3%                  46.8%
    SNGS-147          91.1%                   69.3%                  23.9%

Detection is not the constraint. Swapping in ground-truth registration and changing nothing
else is worth 12 to 21 points:

    clip        seed recall/precision   truth recall/precision
    SNGS-110       42.5 / 53.3             60.8 / 74.9
    SNGS-147       72.8 / 72.5             85.3 / 85.8
    SNGS-116       66.0 / 74.6             70.0 / 77.4

**Two obvious explanations are both wrong, and measuring them first would have saved a day.**
The segmenter is NOT short of coverage: it solves 615 frames on SNGS-116 where the ground-truth
LINES solve 613, 428 against 419 on SNGS-110, 645 against 652 on SNGS-147. The frames neither
can solve are views with too few markings to fit anything, so training could recover one to
nine frames a clip. And it is not inaccurate: on the frames both solve it matches or beats the
annotations, 92.7% of players within 2 m against 89.0% on SNGS-116.

What it has is a TAIL. On SNGS-147, 84 of 645 fitted frames put players more than 3 m out, with
a p90 of 10.81 m against the annotations' 2.41 m. Those frames wreck recall, and bridging them
by carrying makes it worse rather than better -- carry 5 adds 36 frames to SNGS-147 and costs
eight points of precision, because a carry from a bad anchor is bad immediately.

**The fit's own residual cannot find them, and its docstring says why.** It is in-sample: the
fit was chosen to minimise roughly that quantity, so a frame with few constraints is
confidently wrong and scores well. Measured, good frames sit at 0.470 and bad at 0.756, and a
gate at 1.0 m keeps 100% of the good and 87% of the bad. Nor is the homography self-evidently
wrong: good frames also project image corners to absurd distances, because the horizon maps to
infinity.

**The previous fit, walked forward by measured motion, is independent of this frame's fit and
separates them by twenty times.**

    clip        neighbour disagreement, good   bad      gate 2.0 m keeps
    SNGS-147          0.17 m median            4.44 m   96% of good, 35% of bad
    SNGS-116          0.34 m                   4.80 m   99% of good, 50% of bad
    SNGS-110          0.31 m                   3.22 m   96% of good, 25% of bad

`stage1_propagate.winnow` applies it, chained so a rejected fit never becomes the standard its
neighbours are judged against, and with the reference expiring after `DEFAULT_MAX_CARRY` because
a stale carry drifts and starts refusing good fits. On segmenter mode it is a large per-frame
win:

    SNGS-147   recall 52.5 -> 48.7%   precision 75.6 -> 94.1%   purity 69.3 -> 90.3%

90.3% identity purity is the best figure in this repo, and it came from stage 1. Five direct
attempts on the tracker could not move purity at all (D61); removing the frames where the
camera model throws every player at once did. That is the lesson worth keeping.

**And the board does not care.** Within segmenter mode the gate improves two of three boards and
loses the third, for a net of -0.7 observed player-seconds. Segmenter mode still loses to seed
mode overall, because 48.7% recall fields thin rosters -- SNGS-147 comes out with one home
player. So the gate is kept as a strict improvement to a mode that is not the default, and it
does NOT change which mode to use. The 12-to-21-point prize from truth-grade registration is
still unclaimed, and neither carry, retraining, nor this gate claims it.

**D67 — the two registrations fail in opposite directions, and the board needs both halves.**
Run on the clip a coach called bad and on one of the benchmark three, with everything after
stage 1 held fixed:

    clip        mode        frames solved   p50      p90      >5 m from any player
    SNGS-151    seed          750 of 750    1.24 m   8.08 m         22.6%
                segmenter     227 of 750    1.21 m   4.06 m          6.2%
    SNGS-116    seed          750 of 750    0.74 m   7.86 m         12.7%
                segmenter     594 of 750    0.56 m   2.03 m          3.1%

Seeding propagates one human fit through every frame: coverage is total and it drifts, so a
fifth of SNGS-151's players end up somewhere nobody is. The segmenter fits each frame on its
own: three to four times cleaner, and silent wherever too few markings are in shot.

**Cleaner registration makes a WORSE board, and the reason is coverage.** Through the importer,
SNGS-151 comes out as 20 correct players over an 17.8 s window from the seed and 18 over 6.0 s
from the segmenter, because refusing 519 frames collapses the passage there is anything to
build from. SNGS-116 is 18 correct over 21.6 s against 17 over 13.6 s.

**So `observed_error` was the wrong number all along.** It is conditioned on the frames a model
already solves, which rewards refusing the hard ones -- and every judgement about the segmenter
in this document rests on it, including the 0.5 m bar that five runs were killed against. The
number that matters is the share of ALL frames registered within a metre or so, and no run has
ever reported it.

**And the detector is not involved.** In image space, where no camera model can interfere, it
finds 96.7% of visible players on SNGS-151 and 97.3% on SNGS-060 -- the clip that produces the
best board and the one that produces the worst. Whatever separates those two boards, it is not
detection.

**D68 — the two registrations were joined, and the join makes no board better.** D67 asked for
one thing: segmenter accuracy at seed coverage. `--mode hybrid` is it. The segmenter's fits are
winnowed (D62), then refused if they sit further than five metres from where the seed's own
chain says the camera is, then bled into that chain at a twentieth of the difference per frame
rather than replacing it. `anchor_chain` is the filter; the carry supplies the motion and the
anchors supply the position.

Measured with `ft reg-eval`, which is the number D67 said nobody had ever reported — the share
of ALL frames the ground truth can judge, with a frame that has no homography counted as a miss
rather than skipped:

    clip       registered           within 1 m / 2 m / 5 m of the annotated camera
               seed  seg  hybrid    seed            segmenter       hybrid
    SNGS-147   100%  82%   100%     55  62  97      64  81  81      63  86  96
    SNGS-116   100%  97%   100%     39  51  99      48  68  96      46  59  98
    SNGS-121   100%  80%   100%     85  85  85      62  74  74      84  84  85
    SNGS-060   100%  85%   100%     96 100 100      69  79  83      75  89 100
    SNGS-151   100%  64%   100%     73  85  93      21  49  63      24  73  93

**It does what it was built to do on the two clips whose seed chain drifts** — SNGS-147 gains
24 points inside two metres and SNGS-116 gains eight — and it cannot help the three where the
seed is already better than the segmenter. There is no way to tell those apart from inside the
pipeline: the near-seed disagreement between the two sources, which ought to say which is
wrong, is 0.50 m on the clip where anchoring helps most and 1.16 m on the other one where it
helps, against 0.69 m on the clip it hurts most.

**Anchoring by REPLACEMENT is a per-frame win and a per-track disaster, and the mechanism is
worth keeping.** Every anchor moves the whole camera model at once, so a standing player takes
a step. Measured as the p90 metres a fixed point moves between adjacent frames:

    anchor rate   jitter p90        SNGS-147 tracks
    1.00 (hard)   0.27 - 0.85 m     precision 74.2 -> 58.9%, teams 79.4 -> 60.2%
    0.05          0.02 - 0.05 m     precision 74.2 -> 71.2%, teams 79.4 -> 61.0%
    seed          0.00 m            precision 74.2%,         teams 79.4%

A twentieth per frame removes the jitter and keeps most of the registration gain. It does not
recover the tracks, which is the finding.

**Through the tracks and then through the board, it is neutral at best.** `ft auto --interval-s
0` for both, scored the same way — and the interval matters: comparing a file written at 0.1 s
against one written at 0 measures the grid and not the pipeline, which cost an hour here:

    clip       recall        precision     position error   purity
    SNGS-147   70.1 -> 66.4  74.2 -> 71.2  0.71 -> 0.78 m   77.2 -> 75.7%
    SNGS-116   65.4 -> 64.1  78.2 -> 75.0  0.61 -> 0.51 m   72.9 -> 74.5%
    SNGS-121   71.7 -> 55.3  74.1 -> 57.1  1.03 -> 0.63 m   71.9 -> 74.5%
    SNGS-060   89.7 -> 89.6  92.8 -> 92.6  0.55 -> 0.74 m   79.6 -> 79.4%
    SNGS-151   53.3 -> 50.8  59.2 -> 56.3  0.70 -> 1.08 m   85.3 -> 87.0%

And through `pnpm board`, in observed player-seconds — coverage times duration, the thing the
board is actually built from: SNGS-060 347 -> 348, SNGS-116 212 -> 212, SNGS-147 38 -> 35,
SNGS-151 179 -> 159, SNGS-121 **303 -> 117**. Nothing gained, and one clip lost two thirds of
its passage.

**So `--mode seed` still ships**, and hybrid is kept the way `--snap` is: implemented, measured,
off. What would change the answer is a segmenter that beats the drift on every clip rather than
two of five — its per-clip accuracy runs from 0.35 m to 1.3 m and nothing in the file says
which clip you are on. That is the coverage-and-accuracy training problem D67 describes, and
this closes the question of whether the plumbing around it was what was missing. It was not.

**Two measurement bugs found on the way, both older than this experiment.** `max_carry=None`
meant "uncapped" in `fill` and "carry nothing" in the segmenter branch of `auto.homographies`,
and `--carry -1` is the CLI default — so every segmenter number ever recorded, D67's included,
carried nothing whatever was asked for, and asking for a carry was impossible. And
`schema/tracks.schema.json` never declared `source.intervalS`, which the writer has emitted and
Pitchboard has read for eleven releases: every shipped file was invalid against its own
contract, because `test_contract.py` checked the top level and the tracks and never `source`.

**D70 — a camera model is judged where the PLAYERS are, and on two clips the ground truth
cannot judge it at all.** D68 left a contradiction: the hybrid registered far more of SNGS-147
within two metres and made the tracks worse, while pushing the same detections through the
ground-truth camera made them much better. Both cannot be true of one quantity, and they are
not: `observed_error` probes the visible pitch, and players stand in a band across the middle
of the frame. `ft reg-eval` now reports both, the second by pushing every annotated box's
bottom edge through the fitted camera and comparing with the position SoccerNet recorded for
that same box:

    clip       model     probes <2 m   players <2 m   p50 at players   thrown off pitch
    SNGS-147   seed          62%           73%           0.98 m               40
               hybrid        86%           69%           1.11 m               23
               truth        100%           90%           0.58 m                0
    SNGS-116   seed          51%           78%           0.76 m              196
               hybrid        59%           76%           0.64 m              124
               truth        100%           83%           0.53 m                0
    SNGS-060   seed         100%           94%           0.57 m                0
               hybrid        89%           97%           0.82 m                0
               truth        100%           99%           0.31 m                0
    SNGS-121   seed          85%           75%           1.44 m                0
               hybrid        84%           53%           1.74 m                0
               truth        100%           48%           2.33 m                0
    SNGS-151   seed          85%           57%           1.30 m              593
               hybrid        73%           53%           1.68 m              567
               truth        100%           50%           1.80 m              204

**The two metrics disagree, and the probe one is the one that misleads.** SNGS-147's hybrid
gains 24 points at the probes and loses four at the players. Every registration judgement in
this repo before today was made on the probe metric, including the five training runs and the
0.5 m bar.

**And on SNGS-121 and SNGS-151 the annotation disagrees with itself.** A camera fitted from the
ground-truth LINES puts the players further from the ground-truth POSITIONS than the shipping
seed does -- 2.33 m against 1.44 m, and 1.80 m against 1.30 m. Those clips cannot rank two
fitters: the yardstick's own floor sits above the error being measured. It also explains the
recall table, where `--mode truth` scores 51.7% on SNGS-121 against seed mode's 71.7% and
nothing about the pipeline changed.

**What that leaves for the camera model, clip by clip.** Headroom to a perfect fit is 17 points
of players-within-two-metres on SNGS-147 and 5 on SNGS-116; it is 5 on SNGS-060, and on
SNGS-121 and SNGS-151 there is none to measure. The lever is real on two clips of five and
unmeasurable on two others, which is a smaller and better-understood target than "the camera
model is the constraint".

**And it answers the question this project has circled since D36.** The learned segmenter,
judged at the players over every annotated box, does not beat one human seed carried through
the clip -- on any clip:

    players within 2 m     seed   segmenter   a perfect fit
    SNGS-147                73%      47%          90%
    SNGS-116                78%      74%          83%
    SNGS-060                94%      66%          99%
    SNGS-121                75%      39%          48%
    SNGS-151                57%      28%          50%

**Its accuracy is not the problem and never was.** On the frames it solves it puts players at a
0.53-1.20 m median, which is the seed's range. What it does not do is answer: the boxes it
misses are on the frames it refuses. Six training runs have now been scored on accuracy, and
the axis that decides is how much of a clip it will commit to. Anything trained from here is
judged on `players within 2 m` over ALL boxes, or it is measuring the same wrong thing again.

**It also settles the hybrid.** At the players it loses on four clips of five -- the exception
being SNGS-116, where it takes the median from 0.76 m to 0.64 m and the boxes thrown off the
pitch from 196 to 124. `--mode seed` stays what ships, now for a reason measured on the
quantity that matters.

**D71 — the segmenter's coverage cannot be trained, because the frames it refuses carry no
paint to read.** D70 set the bar (players within two metres, over every box) and named coverage
as the axis. This is what happened when that was taken seriously enough to look at the refusals
one by one, and it closes the segmenter as a line of work.

**Every refusal is a frame with one direction of paint, and the annotation has no more of it
than the model does:**

    clip       solved   one direction only   too few markings   model saw   annotation carries
    SNGS-147     86%           12%                  2%          2 (p90 4)      2 (p90 4)
    SNGS-116     82%           17%                  1%          2 (p90 2)      2 (p90 2)
    SNGS-121     62%           27%                 12%          3 (p90 4)      3 (p90 4)
    SNGS-151     48%           50%                  2%          2 (p90 3)      2 (p90 2)

A frame showing two touchlines and nothing crossing them is underdetermined however well it is
read. **No training run on this data can register those frames**, and six of them have now been
spent on the assumption that one could.

**And the frames it refuses are the crowded ones**, which is worse than the frame count reads.
On SNGS-147 the refused frames carry 9.6 MORE players each than the solved ones -- players
occlude the paint, so the moments a board is made of are exactly the moments the model goes
quiet. 86% of frames solved is 70% of the boxes.

**Coverage can be had by carrying, and it buys nothing.** Filling every gap from the frames the
segmenter does solve reaches 100% of frames and moves players-within-two-metres from 51% to 51%
on SNGS-147, 75% to 75% on SNGS-116, 47% to 55% on SNGS-121. It is not short of answers; the
answers it has are not good enough:

    players within 2 m      seed (ships)   segmenter   + carry to 100%   a perfect fit
    SNGS-147                    73%           51%            51%             90%
    SNGS-116                    78%           75%            75%             83%
    SNGS-060                    94%           68%            71%             99%
    SNGS-121                    75%           47%            55%             48%
    SNGS-151                    57%           32%            37%             50%

**The failure is a TAIL, not a median.** On the frames it solves the segmenter puts players at a
0.53-1.20 m median, which is the seed's own range -- and with full coverage a third of all boxes
still land more than five metres out (64% within 5 m on SNGS-147 against seeding's 89%). A
confidently wrong fit from two markings and a mislabelled third is worth less than no fit.

**What the whole line of work could have been worth, at best.** A segmenter as good as the
annotation it is trained on would reach the "perfect fit" column: +17 points on SNGS-147, +5 on
SNGS-116, +5 on SNGS-060, and less than nothing on the two clips where the annotation disagrees
with itself (D70). Six runs have not moved SNGS-147 past 51%.

**Winnow is a net cost at the players and should not be read as a safety net** (D62 justified it
on identity purity, which is a different measurement). Across the five clips it drops 3-16 points
of frame coverage to gain nothing: SNGS-147 51% -> 47% within two metres, SNGS-121 47% -> 39%,
SNGS-151 32% -> 28%. It refuses frames and improves the median of what survives, which is the
same conditioning trap as `observed_error`.

**So: `--mode seed` ships, and the learned detector is finished as a line of work.** What would
register a two-marking frame is not a better segmenter but a model that does not need paint --
regressing a camera directly from the picture, grass texture and horizon included. That is a
different architecture and a real project, and D70's table says it is worth 17 points on one
benchmark clip. It is not the next thing to do.

**D74 — more seeds do not fix the camera; only stronger ones help, and only a little.** With the
segmenter closed (D71) and the camera model sized at the players (D70), the obvious remaining
lever was the human: if one clicked frame drifts, click five. Measured as the share of annotated
boxes placed within two metres:

    SNGS-147                <1 m   <2 m   <5 m   thrown off
    one seed (ships)         51%    73%    89%       40
    three, spread            50%    73%    91%       76
    nine, spread             44%    79%    94%       15
    only strong frames (6)   44%    79%    93%       25

    SNGS-151                <1 m   <2 m   <5 m   thrown off
    one seed (ships)         41%    57%    71%      593
    three, spread            44%    58%    71%      592
    five, spread             28%    37%    68%      421
    only strong frames (4)   47%    58%    71%      592

**Spreading seeds naively makes it WORSE** -- SNGS-151 falls from 57% to 37% of boxes within two
metres with five of them. This is D55 again at a different scale: the extra anchors are the
frames the fitter was least sure of, because a slice of the clip with little paint still
contributes its best frame, and one bad anchor is believed absolutely by every frame it reaches.

**Seeded only where the evidence is strong** -- refusing any frame carrying under 80% of the
markings the best frame has -- it is a modest gain: six points of boxes inside two metres on
SNGS-147 and the boxes thrown off the pitch cut from 40 to 25, one point on SNGS-151. The
one-metre band gets WORSE on SNGS-147 (51% to 44%), so this buys the tail and sells the middle.

**What that settles.** The gap between what ships and a perfect fit -- 73% against 90% on
SNGS-147 -- is not reachable by asking the human for more clicks, any more than by the segmenter.
It is the carry itself: a chain that is right at its anchor and wrong a hundred frames later,
which is D18, and neither more anchors nor better anchors have moved it. What has never been
tried is correcting the chain against something that is not a fit at all -- the players' own
motion, or the grass, measured over the whole clip rather than frame to frame.

**And the multi-seed path already exists** for real clips (`seed.*.json`, D34), so this is
guidance rather than code: click frames where a lot of paint is visible, and do not click one
just because a stretch of the clip has none.

**D80 — a second anchor only helped the frames after it, because the chain walked one way.**
`fill` carried forward from each direct fit and ran backwards only to cover the frames before
the FIRST one. So two anchors did not halve anything: everything between them chained off the
earlier one, however far away it was.

Found on a coach's clip, seeded at frame 382 where the goalmouth markings are, then seeded again
at 56 on his own initiative after the opening came out fifteen metres deep. It fixed the opening
and broke the end — *"the run before the shot is very much on top of the penalty box, now it's
more near the centre line"* — because frames 57 to 381 now chained off 56, up to 325 frames of
carry, while the exact fit at 382 sat one frame away.

Two chains are built now, one with the clip and one against it, and every frame between two
anchors takes a MIX of them weighted by how far each has been carried. Mixed rather than
switched at the midpoint, and that distinction is the whole decision:

    taking the nearer anchor      taking a weighted mix
    board density   50%           72%
    fielded         16 players    21
    window          4.9 s         9.2 s
    tracks cut      every one at the join   none

At the frame where the nearer anchor changes, the two chains disagree by whatever they have
drifted, and swapping between them moves every player at once — so Pitchboard's
`splitImpossible` cut EVERY track in the clip at that one frame, no passage was left that
spanned it, and the importer picked a short honest window on one side of the join. That is D68's
finding arriving from the other direction: a per-frame improvement that steps the whole picture
is a per-track loss. `blend` already existed for it.

The three benchmark clips are unchanged to the digit, which is the check that matters here: with
one anchor the backward chain covers only the frames before it and the forward chain only those
after, so nothing is mixed and the behaviour is exactly what it was.

**This also puts a caveat under D74.** Those runs spread three to nine seeds across a clip and
concluded that more anchors buy almost nothing — measured through a chain that walked forward
only, so half of each anchor's reach was never used. Whether spreading seeds is worth it is
worth asking again now, on the same clips.

**D82 — a seed is stamped with the picture it was clicked on, because a frame number is not an
identity.** D34 said a seed is checked against the frame it claims to describe, and the check it
got was geometric: does this fit fold its own frame over the horizon? That catches a bad seed.
It does not catch a seed that belongs to a different clip, and `work/<clip>/` is keyed by name,
so a coach who exports two clips as "Untitled" gets the first one's anchors on the second.

It happened, on the second clip in a row: `seed.56.json` from Monday's match was still in the
folder, `seed_paths` returns every `seed.*.json` as an anchor, and both were used. Frames 56
onward were registered from landmarks clicked on another stadium. The board was, in the coach's
words, *"horrible, it has nothing to do with the actual clip"* — and nothing in the pipeline
said a word, because a wrong coordinate frame is self-consistent: 41 tracks, zero detections
dropped off the pitch, a full roster, a plausible window.

Two guards, and they are the same guard at two ranges:

- **The seed carries a fingerprint of its frame** — a difference hash, sixty-four bits of "is
  this cell brighter than the one to its right", which survives compression and exposure and
  does not survive a different match. A seed whose frame no longer looks like the one it was
  clicked on is refused by name, with the reason.
- **`ft frames` moves every seed aside**, not just `seed.json`, and says so. It used to print a
  note about the primary seed alone — which is the one a coach re-clicks anyway. The extras are
  the dangerous ones precisely because nobody thinks about them.

Old seeds carry no stamp and are still trusted: the fingerprint is checked when it is there.
That is the compatible half of the fix, and the reason the second guard exists.

**D83 — the chain now says where it is weakest, because that is the only thing a coach can
act on.** D18 established that a carried homography drifts without bound; D80 made a second
anchor reach backwards as well as forwards. Neither told anybody WHERE to click. On two clips
in a row the answer came from a coach noticing that the play was in the wrong place — once
fifteen metres deep, once after a stale seed put it in another stadium — and the second seed
went where the error happened to be visible rather than where the chain is worst supported.

`ft calibrate <clip>` now reports it, from what building both chains already costs:

    frames solved     256/256  (100.0%)
    weakest           frame 256, carried 255 frames from the nearest seed
    seed it           ft seed Untitled --frame 256 --check

Two numbers, answering different questions. **How far a frame is from its nearest anchor** is
available on every clip and is the only guide when there is one seed. **Where two anchors reach
the same frame from opposite directions, their disagreement in metres is drift MEASURED** —
the two chains accumulated it independently, so the gap between them is the error they have
built up, not an estimate of it. With two seeds the report names that frame instead, because a
measured six metres is worth more than a counted hundred frames.

And where the chain has no answer at all, the report says so and where: `no homography 458-464
(7 frames) - a cut, a whip pan, or no grass in shot`. That is the same signal the Manchester
United clip's shot change produced, and nothing had ever surfaced it.

**D86 — the chain broke for want of corners, not for want of grass.** A coach's clip
registered 278 of its 634 frames and the board stopped dead halfway through the play. The
report blamed a cut, a whip pan or no grass in shot (D83) and all three were wrong: the
grass share never left 70–79%, the flow tracked 43 of 44 features it was given, and the
frame after the break solved fine.

What actually happened is that `goodFeaturesToTrack` returned 39 corners on a 2761 × 1551
frame with a cap of 800. `qualityLevel` is RELATIVE to the strongest corner inside the
mask, so a painted line junction or a bright shadow edge in shot raises the bar for every
patch of plain grass behind it. At 0.01 the feature count on this clip swung between 39
and 106 with the grass share flat, and 54 pairs came back with 19–24 RANSAC inliers
against a threshold of 25 — missing the bar by ones.

The cost of each refusal is not one frame. `fill` cannot step over a missing link, so the
first of those 54 ended the chain for all 356 frames after it:

    frames solved     278/634  (43.8%)     ->  634/634  (100.0%)

Dropping `QUALITY` to 0.003 recovers all 54, at a median of 246 candidates and 94
inliers — four times the threshold, so these are not marginal rescues. RANSAC is the
arbiter and `MIN_INLIERS` is the guard, which is why the bar for a CANDIDATE can be this
low: more features to choose from cannot fabricate an agreement among 94 of them.

Two alternatives were measured and refused. **Straddling the gap** — fitting frame f-1 to
f+1 when the pair between them fails — is worse, not better: the six straddles tried came
back with 15–23 inliers where the consecutive pairs had 19–24, because the camera moves
twice as far. **Another seed** cannot help either; the holes run 279 to 467, so a chain
started anywhere inside them dies within two frames.

On the three clips with ground truth it is a wash, and it has to be: all three already
registered 100% of their frames, so they can only show harm.

    at the players, within 2 m     147  73% -> 72%     116  78% -> 77%     121  75% -> 75%
    p50                                0.98 -> 0.96 m      0.76 -> 0.76 m      1.44 -> 1.45 m
    thrown off the pitch                 40 -> 72            196 -> 237            0 -> 0

A tenth of a percent more boxes land off the pitch, against 56% of a real broadcast clip
that had no camera model at all. **A clip that never fails cannot measure a fix for
failure** — the benchmark is the harm test here, and the coach's clip is the only one
that shows the benefit.

What the fix does NOT do is make the tail accurate. The clip now carries 633 frames from
one seed, and the overlay at frame 600 puts the lines metres off the painted ones — 13% of
positions after frame 450 land past a goal line on a 105 m pitch. Coverage and accuracy are
separate problems and this is the first one; `ft calibrate` names the frame for the second.

**D88 — a pitch is symmetric END TO END as well as side to side, and only one of those
was guarded.** `orientation` catches a swapped far/near, and catches it arithmetically
because the overlay cannot: a y-mirrored model draws onto the real markings perfectly
while every position is flipped. The other axis had nothing, and it cost a clip. The
coach seeded frame 634 of Sporting–Galatasaray without pressing `e`, so nine clicks and
thirty-two traced points describing the FAR goal were all written as the near one. The
fit was clean, every residual was small, and the board came out mangled:

    anchors disagree  by 111.5 m at frame 633

Which is one pitch length, and that is the whole tell — but only because a second anchor
existed to disagree with. The number was already being printed and still needed a person
to interpret it.

The check that does not need interpreting is **handedness**. A camera cannot get
underneath a football pitch, so every frame of a clip sees the ground plane from the same
side and the image-to-pitch map keeps the same handedness however the camera pans, zooms
or tilts. Labelling the clicks with the wrong end is a REFLECTION, and a reflection
reverses it. So the seeds of one clip must agree, and one that does not is on the other
goal.

It has to be judged across a clip rather than within a seed, which is the difference from
`orientation`: handedness depends on which touchline the camera sits on, and that is a
property of the broadcast rather than of the pitch. nottingham's two seeds are both
left-handed and agree with each other, which is all the check asks.

Between them the two guards are complete. A pitch has three non-identity symmetries and
they divide as:

    reflect y       orientation catches it      handedness does not
    reflect x       orientation does not        handedness catches it
    reflect both    orientation catches it      handedness does not - two reflections
                                                are a rotation

`ft seed` flips the end to match the clip, the way it already flips far/near, because
that is where a person is present to read the message. `usable_seeds` REFUSES rather than
flips: reinterpreting a file at pipeline time is the silent kind of fix this guard exists
to prevent, and by then nobody is watching.

It found a second one immediately. `geny_rioave/seed.as-clicked.json` is a backup of an
earlier clicking that still matches `seed.*.json`, so it had been loading as a live
anchor, mirrored, for as long as it had been sitting there.

**D89 — a pitch is 105 x 68 only in elite competition, and the seed is written in metres.**
The Laws fix the MARKINGS and leave the FIELD variable: a goal is 7.32 m wide, a penalty
spot 11 m out and a penalty box 16.5 deep on every ground in the world, while the field
itself may be anything from 90 x 45 to 120 x 90. UEFA pins 105 x 68 for the Champions
League, which is why the coach's Sporting clip was exactly right and most football is not.

It matters because the dimensions are not an annotation on the output, they are an INPUT.
`seed.landmarks` writes "goal post far" as `(0, W/2 - 3.66)`, so clicking it with the wrong
W puts the fit a metre or two out across the whole pitch. Nothing catches that afterwards:
the clicks agree with whatever they were told they meant, so the residuals are small, the
overlay draws onto the real markings, and every fidelity score stays good while the
touchline is not where the touchline is. It is D34's failure exactly, in the other axis.

So `Pitch` is carried rather than assumed. `ft pitch <clip> --length --width` records it
beside the seed -- human knowledge the video does not contain, so `ft frames` must not
throw it away with the derived artefacts -- and it flows to the three places that need it:

* **the landmarks and traceable lines**, which is where it enters the fit at all;
* **`on_pitch`**, which decides what is a player and what is a spectator;
* **`tracks.json`'s `pitch` field**, which Pitchboard has always read and which has always
  been told 105 x 68.

Three places deliberately keep the constant. `calibration.PITCH_LINES` names SoccerNet's
own ground truth, which is 105 x 68 by their convention and not ours to reinterpret; the
two `s = float(PITCH_LENGTH)` in the solvers are numerical conditioning, where the only
requirement is that the pitch scales to roughly unit size; and `refine` is off by default
(D35) and still assumes it.

The order matters and the command says so. A seed already clicked was written against
whatever was in force then, and setting the size afterwards does not go back and change
it -- so `ft pitch` warns when the clip is already seeded, and the range check refuses
anything that is not a football pitch rather than accepting a typo that would land every
player somewhere plausible.
