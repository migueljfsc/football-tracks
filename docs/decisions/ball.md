# Decisions — ball

The ball. Not a stage: it is looked for, never tracked.

Every decision here was measured before it was made. They are kept because the code
cites them by number, and because the failures are worth as much as the successes.

**D4 — no ball in v0 (superseded by D29).** It is the hardest object in the frame: small, fast, motion-blurred,
occluded by legs, and frequently out of shot. Worse, it is the input to `carrier`, `shot` and
`hiddenRuns`, so its errors do not stay local — they corrupt the meaning of the board rather
than just its geometry. Players first; carriers get set by hand in the editor. `loft` is not
recoverable at all from one camera: a single homography assumes z = 0, so a chip and a ground
pass are the same measurement.

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

**D66 — the ball has three separate faults, and the camera model is not one of them.**
Measured across the eleven clips, ball error varies 4.6x while player error barely moves:

    clip        player error   ball error   within 3 m   beyond 10 m
    SNGS-060       0.55 m        0.88 m       92.7%         0.0%
    SNGS-066       0.60 m        3.90 m       45.0%         2.9%
    SNGS-110       0.77 m        4.09 m       40.6%        16.9%
    SNGS-116       0.61 m        1.95 m       58.5%        20.7%
    SNGS-121       1.03 m        1.76 m       74.8%         7.4%

**Registration is not the constraint.** If it were, the two columns would move together and
they do not -- SNGS-066 has the second-best homography and the second-worst ball. The mean
PLAYER offset is (+0.01, -0.03) m on that clip over ten thousand matched samples, so the
camera is not shifted; the mean BALL offset is (+1.45, -1.56). That is height. A ground
homography assumes z = 0 and an airborne ball lands metres away, which is what `ball_path`
and `tracks.ts` both say and what the tight, directional error confirms. No selector fixes
it. Estimating the ball's height from its apparent size is the obvious answer -- a football
is a known 0.22 m across -- and it was measured before being built. It does not work.

The camera itself recovers cleanly: with the principal point assumed central, one
homography gives the focal length from the orthonormality of its first two columns, and
decomposing it puts SNGS-060's camera at (52.4, 88.3, 10.9) m -- on the halfway line,
across the touchline, eleven metres up, which is where a broadcast camera stands. That part
is reusable for anything needing 3D.

What fails is the size. Against SoccerNet's own ball annotations on 378 well-conditioned
frames, apparent width correlates with the width geometry predicts at only +0.37, and
inverting size for depth is 20.9 m out on a true depth of 53 m -- 39%, which is metres of
height error to remove a two-metre bias. The annotated boxes also run 55% wider than
geometry predicts (14.0 px against 9.0), which is motion blur and a generous annotator: a
ball at 25 m/s smears a metre a frame, and it does that most while airborne, which is the
only case this was for.

**Nor is detection.** There are 7,000 to 10,800 ball candidates over 750 frames -- nine to
fourteen a frame -- and 327 to 641 are kept. The ball is in there; the choice is the problem.

**The heavy tails are a different fault from the bias.** SNGS-110 and SNGS-116 put 17-21% of
accepted sightings beyond ten metres, where SNGS-066's error is clustered tightly at four.
A metre-space speed gate on acceptance -- the rule `splitImpossible` applies to a player,
which the existing gate does not because it is in PIXELS and cannot see perspective -- was
written and measured. It works on its own terms: the tail falls to 12.3% on SNGS-116 and
12.8% on SNGS-110, p90 from 14.46 to 10.19 m, and accuracy improves on six clips of seven.

**And it does not reach the board, which is why it is not here.** Carrier precision is
unchanged, 75.7% to 75.9% on the same 1,700 frames: the sightings it removes are ones where
the nearest player was already outside `CARRIER_RADIUS_M`, so they were being declined
anyway. On SNGS-060 it is a REGRESSION -- 125 fewer asserted frames bought 1.1 points of
accuracy that clip did not need, and the board loses a handover because a holder with no
sighting to contradict him stands for longer. Few and right beats many and wrong, but only
where the many were being believed.
