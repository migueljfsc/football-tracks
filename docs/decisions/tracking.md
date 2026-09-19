# Decisions — tracking

Stage 2 — finding people and following them.

Every decision here was measured before it was made. They are kept because the code
cites them by number, and because the failures are worth as much as the successes.

**D8 — track samples are sparse.** A player occluded for twenty frames has no position for
those frames, and inventing one is a lie the reduction would fit a bezier to. Every sample
carries its own frame index; gaps are expected and the consumer interpolates or declines to.

**D13 — an off-pitch position is dropped, never clamped.** A position 200 m out is not a
player near the touchline, it is a homography that failed. Clamping launders that failure
into a plausible coordinate the reduction then fits a curve through, and the resulting run
looks like a real one. Rejecting loses a sample; clamping invents one. The guard is
`tracks.on_pitch`, so the CV path and the ground-truth path cannot disagree about what
counts as credible.

**D20 — the tracker is ours.** supervision's ByteTrack is deprecated and disappears in 0.31,
and this regime has a signal a general tracker does not use: a team wears one colour, which
is exactly what tells two crossing players apart. Association is greedy over a gate, in
metres, so the gate is a physical claim about how far a footballer runs rather than a claim
about how fast the camera pans.

**D22 — stage 2 does not depend on stage 1.** The gate is a speed — a footballer covers at
most MAX_SPEED metres in a second — and it reaches pixels through the only local scale that
needs no camera model: a detection box is about 1.8 m tall, so it says how many pixels a
metre is right there. Camera motion is removed with the frame-to-frame transform rather than
by projecting to the pitch. Both halves matter: raw pixels lose a panning camera, and pitch
metres inherit every wobble in the homography.

**D27 — anyone standing off the pitch is dropped BEFORE tracking, not after.** Two fifths
of what the detector finds on SoccerNet is crowd, dugout staff and ballboys behind the
hoardings. They were always discarded at the end, but until then they were competing for
associations and spawning tracks of their own. Filtering first takes 123 tracks to 69,
lifts identity purity from 77.8% to 79.3% and drops switches from 44 to 38, at no cost to
recall.

This is the one place stage 2 consults stage 1, and it is a deliberate exception to D22:
only as a FILTER. The association still never sees a homography, so a drifting camera can
change which detections are considered and cannot change the identities.

**Four things that did NOT work, recorded so they are not retried.** Associating in pitch
metres rather than stabilised pixels; optimal assignment instead of greedy (worth keeping
on its own merits once the junk was gone — 69 tracks against 72 — but it fixed nothing);
raising the weight on kit colour, which is genuinely discriminative (same-team pairs sit at
0.32, opposing at 0.67) and still moved nothing; and stitching fragments back together
afterwards, which reunited one player for every two players it wrongly welded into one. The
last was deleted rather than tuned: a fragment loses a run, but a bad join invents one, and
nothing downstream can tell.

**D28 — the detector is RT-DETR, and the first one was deliberately a floor.**
torchvision's Faster R-CNN was chosen because it was BSD, already a dependency, and
certain to be beatable — so every number taken with it was a lower bound rather than a
best case. Once the pipeline was measurable enough to compare fairly, it was replaced:

    Faster R-CNN  conf 0.50   83.6% recall   1.6 spurious per frame   0.26 s/frame
    RT-DETR       conf 0.50   86.2% recall   0.5 spurious per frame   0.14 s/frame

Better on all three, and Apache-2.0, so the licence story stays clean (D9).

**The false-positive column matters as much as recall**, which is why the confidence
floor stays at 0.5 rather than dropping to 0.4 for four more points of recall. Everything
the detector invents competes for associations and spawns tracks. On the Rio Ave clip the
swap took 50 tracks to 35 and, more to the point, fragments covering under a tenth of the
clip from 17 to 5 — the twelve longest now run 205, 196, 185 and 183 frames out of 208.

Two things that did NOT lift recall and are not worth retrying: a larger input image (800
against 1333 changed nothing, because the misses are occlusions rather than small
players), and a lower confidence floor, which buys recall at about three spurious boxes
per real one.

**D30 — a track that has lost its player gives up quickly.** 88 of 98 identity changes
on SNGS-147 happened AFTER A GAP, at a median of ten frames — not during a visible
crossing, which is where I had assumed they were. A track that has lost its player coasts
on a stale prediction, its gate grows with the wait, and when detections resume it takes
whoever is nearest. Over half the time that was an opponent.

Cutting `MAX_AGE_S` from 0.8 to 0.24 takes purity from 76.9% to 80.5% and switches from
37 to 26, with the track count and recall unchanged: the player is picked up again either
way, and what is saved is a run stitched onto somebody else.

It also settles why weighting kit colour more heavily never helped, which had been an
open puzzle. At 0.8s the wrong candidate is reachable and colour is asked to talk the
tracker out of it; shorten the wait and it was never reachable, and the two weightings
score identically.

**Which measurement chose the number matters here.** Run fidelity compares a board with
the TRACKS it was built from, so it cannot see a steal — the board faithfully draws
whatever the tracker believed. Only ground truth can, and a shorter age looked WORSE by
fidelity while being better by truth. Do not tune this against a clip with no answers.

**D53 — stage 2 fragments every player, and joining the pieces afterwards is safe where
lengthening the tracker's memory was not.** The measurement that redirected the work: against
ground truth on SNGS-147, a player is in shot about 239 frames of 750 and comes out as a
median of 4 predicted fragments — roughly 48 frames each, which is 6% of the clip against the
importer's `MIN_COVERAGE` of 30%. That, not the camera model, is why a 22-player clip becomes
a ten-player board.

`MAX_AGE_S` is 0.24 s deliberately (see its comment): a longer wait lets a track coast on a
stale prediction and take an opponent when detections resume. The fragments are the price that
was knowingly paid for the steals. So the fix is not to raise it.

`stage2_stitch.py` joins fragments afterwards instead, and the asymmetry is the whole point:
the tracker must decide AT the gap with nothing after it to go on, while this runs when both
sides are known and can require two fragments to be each other's best continuation. 39% of
identity changes have a gap of one frame or less — the player never disappears, the tracker
merely renumbers them — and 70% are inside 12 frames.

Three rules, each answering a way this could put one player's run on another's shirt:

- **Mutual best, not greedy.** A fragment ending in a crowd has several plausible successors
  and taking the cheapest is exactly how a track teleports. Requiring the choice to be
  returned makes an ambiguous join fail into two honest halves, which the importer survives.
- **A speed gate in METRES**, which is why this is a separate pass and not a change to stage
  2's gate. Stage 2 works in image pixels on purpose, so it cannot inherit the camera model's
  errors (D19); whether two fragments are one person is a question about m/s and needs the
  pitch.
- **Colour breaks ties and never repeals the speed limit.** Same role and value as the
  tracker's own `COLOR_WEIGHT`.

Measured on all three clips, `--mode seed`, against no stitching:

    clip        tracks      identity purity    switches
    SNGS-147    88 -> 67    73.6 -> 77.2%      59 -> 48
    SNGS-116    85 -> 63    64.4 -> 67.9%      271 -> 261
    SNGS-121    50 -> 42    61.3 -> 61.3%      50 -> 50

Purity RISING is the evidence the joins are right: joining two fragments of one player raises
it, and joining two different players would lower it. Recall, precision and position error are
unchanged to the digit, as they must be — stitching regroups samples without altering one.

And through `boardFromTracks`, which is the measurement that counts:

    clip        players   scenes    window        max travel     curves
    SNGS-147    19 = 19    6 -> 5   2.9 -> 2.8s   15.0 -> 19.7m  10 -> 14
    SNGS-116    22 = 22    7 -> 12  5.8 -> 13.5s  26.7 -> 40.0m  26 -> 49
    SNGS-121    21 = 21    9 =  9  17.4 -> 13.2s  18.3 -> 14.7m  37 -> 37

SNGS-116 roughly doubles: a 13.5 s passage rather than 5.8 s, and 49 curved runs rather than
26. SNGS-147 gains travel and curves. **SNGS-121 is a mild regression** — longer tracks change
the coverage landscape and `chooseWindow` settles somewhere shorter. Two clear wins and one
small loss; it is on by default, and `--no-stitch` turns it off.

What it does NOT do is add players. The roster is set by how long each player is in shot, and
joining fragments cannot put a player on camera. That ceiling is the next thing in the way.

**D56 — the tracker smoothed the kit colour and not the velocity, and velocity was doing the
harder job.** SNGS-116 carried 266 identity switches against SNGS-147's 56. Splitting them by
kind is what made it tractable:

    clip        switches   steal (id also serves another player)   fragment   at a gap <=1 frame
    SNGS-147        56                  52%                          48%            38%
    SNGS-116       266                  93%                           7%            68%
    SNGS-121       119                  83%                          17%            70%

`MAX_AGE_S`'s comment records that 88 of 98 switches on SNGS-147 happen AFTER a gap, and that
is still true of 147. SNGS-116 is the opposite failure: 93% are steals with NO gap, both
players continuously visible, the tracker simply swapping them. Shortening the coasting window
fixed 147 and can do nothing here, because nothing is coasting.

They are crossings. At a steal the nearest other player is 0.73 m away on SNGS-116 against
3.12 m for a typical sample -- and that clip is a corner, so 41% of all its samples have
somebody inside two metres. Kit colour is silent between team-mates and both observations sit
inside both gates, so the position prior is the only thing left.

And the position prior was noise. Velocity came from a SINGLE frame's displacement: a player at
5 m/s covers 13.6 px between frames at this scale, and the detector's box centre wanders a few,
so a quarter of it was jitter. The kit colour three lines below already had a rolling average,
on the stated reasoning that one frame of shadow should not redefine a kit -- the noisier
quantity, doing the harder job, was the raw one. Swept on SNGS-116:

    smoothing   identity purity   switches
    1.0 (none)      67.3%           266
    0.5             70.5%           256
    0.3             70.7%           246

Recall and precision do not move at any setting: this changes which track a sample lands on,
never whether it is found. Tightening `MIN_GATE_BOXES` was swept alongside and is the wrong
lever -- 0.20 reaches 71.8% purity with 292 switches, because a tighter gate tears tracks
rather than keeping them straight, which is what that constant says it exists to prevent.

**And it exposed a metric that had been read backwards all day.** SNGS-147's board appeared to
LOSE movement, 18.5 m of travel down to 7.6 m. The track responsible moved 22.4 m across the
window while covering THREE different ground-truth players at 76% purity; after smoothing it
covers one, at 100%, and moves 14.8 m. A track that hops between players covers more ground
than any real player can, so "max travel" rewards precisely the failure being removed. Judge a
board by median travel and by whether its longest run belongs to ONE player.

**D69 — the stitcher was refusing most of what there is to join, and reach was the wrong
gate.** Nothing had ever asked WHY a join was refused. Grouping the fragments five clips
produce by the real player each one tracks, and putting every consecutive pair through the
stitcher's own gates:

    refusal                     SNGS-147   SNGS-060
    gap longer than MAX_GAP_S     55%        64%
    fragments overlap in time     17%        15%
    not mutual best               14%         3%
    too fast                      10%         3%
    joined                         3%        15%

**The gap limit is most of it, and it was set against the wrong population.** 0.5 s was chosen
because 70% of identity CHANGES happen inside twelve frames -- but the breaks that cost the
roster are what is left after the tracker has already given up, and those run to a median of
2.2-2.8 s. Half a second was refusing most of what there is to join.

**Reaching further on a speed limit does not work, and that is why the constant was short.**
`MAX_SPEED` is 12 m/s, so a reach gate at three seconds admits 38 metres -- most of the pitch,
and any two players in one kit. The gate is now a PREDICTION: where the first fragment was
going, walked forward, against where the second came from, walked back, with a tolerance of
2.5 m plus 1.5 m per second of gap. Judged on the joins whose identity ground truth can
confirm, over five clips:

    gate                     joins    wrong
    0.5 s reach (was)         5-15     41%
    3.0 s reach              16-20     38%
    3.0 s predict 1.5 m/s     7-17     26%

More joins and fewer of them wrong, which no length of reach gate could offer. A margin rule
on top -- refuse where the runner-up is nearly as good -- buys nothing: the wrong joins are
confident rather than ambiguous.

**Through the boards, in observed player-seconds:** 1934 across the eleven SoccerNet clips
becomes 2531. SNGS-067 goes 117 -> 259, SNGS-121 196 -> 321 and the Nottingham clip 58 -> 147,
mostly by holding a window twice as long. Identity purity rises on SNGS-060 (79.6 -> 81.6%) and
SNGS-121 (71.9 -> 74.4%) and the team split holds within two points everywhere. Recall,
precision and position error do not move at all, by construction: this changes which track a
sample belongs to, never whether it exists.

**What it costs is two real players in seventy-seven.** Measured at a FIXED window, so the
comparison is not flattered by the longer passages this buys: a board fields 77 distinct real
players before and 75 after, because a wrong join merges two people into one shirt. The rate
was 1.5 m/s rather than 2.0 for the same reason -- teams are clustered on a whole track's kit,
so a track holding two players holds a blend of two kits, and at 2.0 the team split on
SNGS-147 falls from 79.4% to 62.3%.

**And the ceiling on this stage is now measured, which matters more than the change.** Join
every fragment of a player perfectly -- an oracle, from ground truth -- and the median player's
best track holds 47% of their life on SNGS-151, 52% on SNGS-116, 60% on SNGS-147, 67% on
SNGS-121 and 91% on SNGS-060. Today's stitcher already reaches 47%, 43%, 53%, 50% and 79%. On
three of the five clips there is almost nothing left to join: the missing half of a player is
not in another fragment at all.

Where it went, by match radius, is the funnel to read next:

    clip       recall @1 m   @2 m    @5 m    @10 m
    SNGS-060      73.7%     89.7%   93.1%   93.3%
    SNGS-147      48.5%     70.1%   82.1%   89.7%
    SNGS-121      35.0%     71.7%   92.3%   95.8%
    SNGS-116      51.9%     65.4%   75.0%   79.2%
    SNGS-151      39.2%     53.3%   68.2%   79.8%

On SNGS-121 and SNGS-147 the samples exist and land 2-10 m from the player: that is the camera
model, not the tracker. Pushing the same detections through the ground-truth camera takes
SNGS-147 from 70.1% to 82.8% at two metres -- and takes SNGS-121 from 71.7% DOWN to 51.7%,
because its seed chain is already better than a per-frame fit from the four markings that clip
shows. Which is D68's finding from the other side: the camera model is worth 13 points of
recall where it is bad, and the segmenter is not reliably better than what ships.

**D76 — two tracks alive at once are two players, not one player twice.** With gap-bridging
settled (D69) the remaining fragmentation is in pairs the stitcher cannot reach by construction:
fragments that OVERLAP in time, where the tracker appears to have started a second track while
the first was still running. Merging those would be free coverage, so it was worth checking
whether they are what they look like.

They are not. Across four clips, of every pair of tracks sharing at least five frames:

    clip        overlapping pairs   really the same player   agree within 2 m
    SNGS-060           748                    3                      0
    SNGS-116           561                    3                      7  (1 of them the same)
    SNGS-121           394                    0                      0
    SNGS-147           374                    1                      0

Overlapping tracks are two different players standing near each other, which is what a football
match consists of. The handful that are one player do not agree in position either -- one of the
two is mis-projected, which is why it became a second track. **There is no duplicate-merging win
here**, and a rule that merged on proximity would join teammates.

**What is left of fragmentation is 6 to 18 points**, measured as the difference between what a
real player's best track holds and what all their tracks hold together: 9 on SNGS-147, 11 on
SNGS-060, 18 on SNGS-121, 6 on SNGS-151. That is the ceiling for joining, and the two mechanisms
that could reach it are both spent -- the prediction gate takes what geometry can take (D69),
and colour has resisted seven attempts (D61). Closing it needs an appearance model robust to
occlusion, which the benchmark has called a project of its own since SNGS-116 first showed 18%
of boxes overlapping another by more than a third.

**D77 — a velocity read over a fixed number of SAMPLES measures the frame rate.** The
prediction gate (D69) is only as good as the heading behind it, and that heading was read over
the last five samples of a fragment. Five samples is half a second on a 25 fps clip stored at a
tenth of a second, and an eighth of a second on a 33 fps clip stored at full rate -- and an
eighth of a second turns 0.7 m of position noise into 5.7 m/s of sprinting sideways.

Found on a coach's own clip, and visible on the board without any metric: *"red player number 8
just chilling in the penalty area then magically gets the ball -- the player that makes the run
is the same player that shoots"*. He was right. The runner's track ended at frame 114 and a new
one began at 125, their ends 2.1 m apart, and the noise-built prediction missed by 3.3 m against
a 3.0 m tolerance. Two shirts, and the ball handed to the one standing still.

Read over `VELOCITY_S` of track instead -- 0.4 s, whatever that is in frames -- and the same
pair joins at a cost of 0.53. Below `MIN_VELOCITY_S` of track there is no heading to read, so
the prediction falls back to "stays where it was", which the tolerance absorbs and a wrong
heading does not.

    SNGS-121/060/116/147     before        after
    recall, precision        unchanged     unchanged
    identity purity          72.2-83.9%    74.4-83.9%   (SNGS-116 +2.5, SNGS-147 +0.3)
    id switches              58-254        56-252

A wash on 25 fps footage, because half a second and a fifth of a second are both long enough
there. The whole cost fell on the clip nobody had run: this repo's benchmark is eleven clips at
one frame rate, and a constant counted in samples is invisible until somebody brings their own
video. Pitchboard's D52 is the same fault on the other side of the seam, found the same way.

**D78 — a colour WEIGHT is a preference, and a preference loses when the right player is
missing.** Found on a coach's own clip, the same way D77 was: *"by scene 3 it falls apart, it
shows that the away team held possession but it is not true, a home player made a run on the
left and passed it to the second post for the goal"*.

The board was drawing a yellow-shirted player in the opponent's colour. One track held two
people — the yellow attacker up to frame 89, a white-shirted opponent after it — and a track's
team is clustered on its whole kit (D63), so the average landed on the white side and the runner
came out as the other team. Cropping the track's own boxes at its own frames is what showed it:
three yellow, then three white, in one id.

`COLOR_WEIGHT` was supposed to prevent exactly this and cannot, because it is a preference and
preferences only decide between candidates that exist. The player's own detection was missed for
six frames, the gate grows with the wait, and the cheapest thing left inside it was an opponent
— at which point 0.6 of colour cost is still cheaper than going unmatched. So the association is
refused outright above `KIT_VETO`, whatever the geometry says. Above 0.5 by construction: an
unreadable kit scores exactly 0.5 (`color_distance` neither attracts nor repels), so the veto can
only ever refuse two colours that were both read and disagree.

It is a rare event and a cheap guard — 0.5% of the associations on that clip are above 0.6, and
the bad one is in the worst four of 2392.

    veto      SNGS-147 purity/switches/teams   SNGS-116            SNGS-121
    off       77.5%  56  59%                   74.7%  252  70%     74.4%  103  86%
    0.7       78.0%  47  59%                   72.3%  253  74%     73.8%  100  85%
    0.6       78.6%  43  77%                   73.5%  257  73%     73.9%  104  86%
    0.55      78.5%  45  77%                   73.2%  264  71%     73.8%  108  85%

Recall, precision and position error do not move at any setting — this only changes WHICH track
a sample lands on. **The metric it pays out in is the team split, not identity purity**, which is
the point: a switch between team-mates costs a shirt number nobody reads, and a switch across
kits costs the colour of the pass. SNGS-147 gains eighteen points of team accuracy for one point
of purity.

**D79 — the stitcher makes the same claim across a longer gap, so it needs the same veto.** With
D78 in place, the joins five clips actually make still include several whose kits disagree by
0.6 to 0.88 — and this file's own `PREDICT_DRIFT_MS` note records what a joined track holding
two kits costs: seventeen points of team split. The same guard, in the same units, on the colours
the stitcher already carries:

    stitch veto   SNGS-147             SNGS-116             SNGS-121
    off           78.6%  43  77%       73.5%  257  73%      73.9%  104  86%
    0.6           78.6%  43  77%       73.4%  257  78%      73.6%  104  86%

Five points of team split on the crowded clip, nothing anywhere else, recall and precision
unchanged. Vetoing on `kit_mean` instead of the fragment's end colour was measured at the same
time and is a wash (116 +3, 121 +2, 147 unchanged), so the shipped guard is the one that needs
no new signature.

**What neither of them fixed, on the clip that found them.** The board now names the white-shirt
carrier correctly through the first two thirds of the passage and gets the last third wrong,
because the ball is not detected at all between frames 121 and 178 — 1.8 s covering the run —
and where it IS seen, at 108 to 121, it lands 1.9 m from a defender and 3.8 m from the player
who actually has it. That is D66 (a ball in flight is metres from where z = 0 puts it) and D75
(recall cannot be bought by lowering confidence), not the labelling. A runner-up margin in the
importer was tried against it and measured worse: at 1.5 m the board stops naming anybody and
possession collapses onto whoever held it first.

**D90 — two tracks in the same place at the same time are one player, and the detector says
which.** `stitch` asks whether one fragment CONTINUES another and refuses anything that
overlaps in time (`gap <= 0`), so the case it cannot see is the tracker renumbering a player
it never actually lost. Both ids stay live and alternate frames.

It is not a spare track. The two halves are labelled independently, so a player is home for
one stretch and away for the next, and that reaches the board as a turnover nobody played.
A coach found it the other way round: his striker was tracked from the halfway line through
the run that won a penalty, and that half was declined as `unknown` while a second id
starting 190 frames later was named, fielded, and HELD -- parking him in an offside position
for nine seconds of a thirteen-second clip.

Distance alone cannot decide it, and this is the trap. Across fourteen clips the overlapping
pairs cluster at 0.6-1.5 m and then climb steadily from 2 m into the thousands, so a cut
under the tracker's own 1.7 m resolution looks safe -- and it is not, because a striker and
the man marking him run a metre apart for the whole move. Merging those destroys two players
to fix nothing.

**What separates them is the detector.** It finds a player once, so two tracks on one man
have to take turns, while two tracks on two men each get a box of their own every frame.
Measured on the coach's clip the populations do not touch:

    t23 + t36   1.4 m apart   9% of frames shared   one player  (the striker)
    t35 + t38   0.8 m          5%                   one player
    t42 + t43   0.6 m          0%                   one player
    t19 + t36   2.0 m         81%                   two players (the marker and the striker)
    t28 + t46   1.9 m        100%                   two players

The absorbed track's shirt readings move with it, so the survivor is named on both halves'
evidence rather than on whichever half was longer.

**And that exposed the second half of this.** Merging made the striker LESS certain of his
own side, because `side_mean` counted every sighting equally and his distant ones outnumbered
his close ones -- sixty pixels of a player is mostly grass and reads like neither kit. So the
side signature is now weighted by how big he looked, which is the one shirt question that
should care: associating wants every pixel and painting wants a mean colour, but "which of
these two kits" is only as good as the look at the shirt (D87's split, one level deeper).

    teams            147  89% -> 92%      116  77% -> 77%      121  83% -> 83%
    asserted right        3402 -> 3486         unchanged            unchanged
    declined               104 -> 0            unchanged            unchanged

Tracks, recall, precision, error and purity are identical to the digit on all three: the
merge fires on none of them, because their close pairs share frames and are real players.

What it does NOT fix is the striker. Merged and weighted he is still declined, at own 0.270
against a bar of 0.251 -- clustered with the right side, 8% short of the confidence to say
so. He carries the ball at frames 506-573 and the board cannot draw him, because a track the
kit will not settle is never fielded (D72). That is the next thing, and it is a question
about the margin rather than about him.

**D93 — the ball cannot be recovered from this detector by any selection rule, and nearness
to a player is evidence AGAINST it.** A coach's clip opens with thirty frames his team spends
losing possession, and the board draws none of it: the ball is not asserted until frame 43.
It is not invisible there -- the detector finds it on the grass at 0.44 to 0.53, under
`BALL_ASSERT_CONF`, which D73 measured at 0.75 and which buys two phantom passes of four
across six boards. Lowering it undoes that, so the question is whether a weak sighting can be
CORROBORATED instead.

Two corroborations were built and measured. Both fail, and the second fails in an
instructive direction.

**Continuity, walking back from the first confident sighting.** It recovers a path, and the
path is made of 0.15 to 0.26 detections that converge on a point the best-scoring blob of the
same frames disagrees with. There are several stationary candidates in that opening and
continuity cannot separate them -- which is the same wall `ball_path` documents from the
shortest-path experiment.

**A tracked player carrying it**, on the argument that a fixed artefact cannot follow a man
across the grass. It found 201 candidates at somebody's feet in the first 44 frames, and 595
of the clip's 603 frames have one, which is already a warning: a test that fires everywhere
is not a test. Measuring what is actually at a player's feet says why:

    band                   n     within 2 m of a player    median gap
    confident >= 0.75    290              37%                 2.7 m
    middling 0.5-0.75    222              59%                 1.6 m
    faint < 0.5         5440              58%                 1.6 m

**The faint sightings are MORE likely to be at a player's feet than the real ball is.** They
are boots, socks and the white of a shirt -- the false positives live ON the players, which
is exactly where this test looked. Proximity to a player selects against the ball.

So no rule over these candidates gets the opening, and D73's line is the honest summary made
sharper: the ball can be frequent or right and not both, because the detector finds it at all
in 41% of frames and its errors are not scattered -- they are on the people. What is left is
a ball-specific detector, which is a model rather than a rule.

Nothing shipped. The measurement is the result.

**D94 — a track breaks where players touch, and of four ways to rejoin it three ship.** A
coach's clip, Benfica against Gil Vicente, ten seconds that end in a save. Two complaints: two
Benfica defenders stand in their own box for the first four seconds while the play comes at
them, and the board ends with a Gil Vicente attacker on the ball the Benfica keeper caught.

Both are fragmentation, and every break is a contact. The defenders' tracks end in a tackle,
where the detector draws one box over tackler and carrier; the keeper's end in his dive and
again on the ground. Looked at on the frames, each pair below is one man:

    join       refused by
    10 -> 27   prediction 3.93 m against a tolerance of 3.01
    19 -> 28   prediction 2.78 m against 2.59, and kit 0.67 against the 0.6 veto
    20 -> 34   nothing -- 20's best continuation was 35, the second half of 34's own chain
    35 -> 41   prediction 3.62 m against 3.31

The importer did the rest. It holds a player at his first sighting until his track begins, so
a defender first tracked at frame 97 stands in the box from frame 1; and it fields ONE keeper
track per side, so it kept the one that ended before the save and gave the ball to whoever
stood nearest.

Four changes, measured one at a time on the eleven benchmark clips and three coach clips.

**The keeper is a role, and his fragments are joined once `assign` has named them**
(`stage2_stitch.keepers`): the best-supported run of keeper tracks that never overlap, each
within reach of the last and each on the field, so a man behind the goal in an odd kit is not
folded in. It fires on five clips of fourteen, moves purity on two (SNGS-075 59.1% -> 60.3%,
SNGS-100 64.3% -> 65.6%) and nothing else scored, and adds 35 observed player-seconds across
the boards, 17 of them on 075. On the coach's clip the ball is now the keeper's.

**Mutual best is repeated until nothing more joins.** A player broken twice has a first
fragment whose best continuation is his third, which prefers his second, so one round joins
those two and strands the first. Alone it lifts purity on five clips and SNGS-110's board by 20
player-seconds -- and costs the team split: SNGS-075 moves 886 correctly-sided samples to
declined or wrong, and SNGS-066 gains 331 on the wrong side. The joins a second round makes are
the marginal ones, and the veto was judging them on a colour that did not describe the track.

**So the stitcher's kit veto reads `kit_mean`, the whole track, rather than the rolling
`color`.** A fragment ends in contact, its last frames read two shirts, and the rolling average
is mostly those frames. On the coach's clip:

                                  rolling color   whole track
    10 -> 27, one player              0.48           0.29
    19 -> 28, one player              0.67           0.37
    between the two sides           0.90-1.00      0.81-0.93

On top of the second change it puts 1,460 samples on the right side and takes 1,431 off the
wrong one. The three together, against the baseline:

    eleven clips      right +638   wrong -998   declined +360   recall, precision, error identical
    fourteen boards   +107 observed player-seconds

What they cost is recorded rather than averaged away: SNGS-066's board loses 16
player-seconds and declines 428 more samples, SNGS-151 fields 19 players rather than 20, and
SNGS-060's worst scene falls from 36% witnessed to 23%.

**The fourth was built and refused: more position slack where a box holds two men.** It is the
only one of the four that rejoins the defenders. Each join the prediction gate refused ends or
begins on a box another covers by 0.30-0.43 of its area, and a metre of slack at such an end
admits all three. On ground truth it admits sixteen joins, judged by which real player holds
each whole fragment:

    same player   team-mate   other side   cannot judge
         3            6            6            1

Team split +914 wrong for 45 player-seconds of board. Nothing available separates the three
from the twelve. The cross-kit joins sit at 0.30-0.59 on the whole-track kit, under the veto,
and a bound tight enough to refuse them refuses both of the coach's joins as well (0.29, 0.37).
A join with no rival candidate is not safer either: of the six judged, none was the same man --
what `PREDICT_DRIFT_MS` already records for the ordinary gate, that the wrong joins are
confident. The confident continuation of a tackle is the other man.

So the two defenders are still two fragments each. What would rejoin them is knowing which of
two men a merged box belongs to, which a position cannot say: the other man's track running on
through the contact might, or appearance.

**D95 — which of two men a merged box belongs to is a question about how they look, and
answering it exposed a fault in every join.** D94 left the coach's two defenders split: their
tracks break in a tackle, the break's box holds both men, and a metre of slack there joined the
other man twelve times in fifteen. So the slack is now spent only where the two ends LOOK like
one man.

**The look is OSNet-AIN**, a person re-identification network (`reid.py`; the architecture
vendored from Torchreid in `osnet.py`, MIT; the MSMT17 weights downloaded on first use, checked
against a pinned SHA-256 before every load, loaded `weights_only`, never committed). `ft reid`
embeds every detection; a fragment end's look is the mean of up to eight crops nobody else's box
covers. Measured before any of it was built, on 306 ground-truth candidate joins with clean crops
at both ends:

    separation (AUC)          same v team-mate   same v opponent   same v either
    kit signature                   0.74               0.90              0.83
    OSNet x1.0                      0.81               0.88              0.84
    OSNet-AIN x1.0                  0.88               0.92              0.90

On the joins only contact slack would make, the ones D94 refused, AIN separates the same man
from an opponent perfectly (1.00) and from a team-mate at 0.87. `LOOK_APART` is the 95th
percentile of the same-player distances over ALL those joins, not the contact ones it is judged
on; the pipeline reproduces the measured distances to the third decimal.

**Gated like that, it still cost the team split, and the joins were not why.**

    appearance alone, against D94      right    wrong   declined   observed p·s
    LOOK_APART 0.172                   -511     +843     -332         +20
    LOOK_APART 0.219                   -511     +391     +120         +21

On SNGS-066 both new joins were the same player and kept his side, and a 604-sample track that
is 65% one player flipped from declined to the wrong side. On SNGS-075 the join matched the end
of a track the TRACKER had already switched from a home player to an away one. The second is
D61's; the first is a fault older than any of this. `stitch` folded a fragment's samples into a
track and left its shirt readings behind, so every join took evidence out of the side clustering
and moved the cut for everybody else -- the thing `duplicates` has absorbed readings to avoid
since D90.

**So the readings move with the samples, on every join**, and that is the bigger change:

    against D94                        right    wrong   declined   observed p·s
    readings absorbed                  +4538    -2295    -2243         +47
    ... and appearance at 0.219        +4775    -2379    -2396         +65

Recall, precision and position error identical on all eleven clips. SNGS-151 fields 22 players
again, where D94 had cost it one. Appearance on top of the absorb is +237 right, -84 wrong, purity
up on six clips and switches down on eight; 0.172 and 0.219 make identical joins on every
benchmark clip, and 0.219 is the one that also joins the coach's second defender (0.191). What
it costs: SNGS-069's board 18 player-seconds, nottingham's 13, and SNGS-067 three more switches,
all from the absorb.

On the coach's clip both defenders now run with the play from their first sighting, and the
keeper still takes the ball at the end.

**D97 -- with the camera fixed, a player's missing half is in pieces; appearance cannot say which
pieces are his, and a veto built on it breaks the tackles it was meant to protect.** D96 put
the benchmark's positions within 2 m for 98% of players, and that changes D76's arithmetic. A
player's tracks together now hold 85% of his life and the best one 59%, so 26 points are lost
to splits -- up from 20, because the half that used to be mis-projected is now in the right
place and visibly in pieces. `ft score` prints both numbers now, and a third: TRACK purity, the
share of a predicted track's samples that are its main player's. Identity purity asks whether
a player stayed in one track and cannot see two players joined into one; track purity can.

Why the splits are not joined, by the life each join would have added, on the camera runs:

    refused because                                   share
    gap over 3 s, player OUT OF SHOT (camera away)     ~28%
    gap over 3 s, player in shot but untracked         ~16%
    prediction gate (median miss 1.5x tolerance)        34%
    both alive at once                                  16%
    lost to a better candidate                           5%

And 19% of the links the stitcher does make are wrong (38 of 200).

**Appearance cannot open a gate.** OSNet's same-player and different-player distances overlap
too far: across long gaps a threshold that admits 159 of one man's pieces admits 591 of other
people's. Short handovers are not a lever either -- 28 of 2,855 are one player.

**It looked like it could close one, and on the benchmark it did.** Among the joins made, the
right ones sit at a median look distance of 0.09 and the wrong ones at 0.18. A veto at 0.18 on
every join took track purity 91.4% -> 92.1% on the camera runs and 87.4% -> 88.7% on the seed
runs, and paired with a longer reach (5 s, 2.5 m/s) the boards gained 4.5% and 3.2% of watched
time with the team split up and SNGS-147 -- D69's casualty -- at 93.9% from 91.6%.

**It failed on the first clip that played no part in choosing it.** On the nottingham coach
clip the veto alone cost 8% of the board's watched time and added three turnovers the clip does
not have: a red defender who goes to ground in a sliding tackle does not look, lying down, like
himself standing up, so his track is split exactly where the tackle happens -- and the loose
half, near the ball, is named the carrier of a ball he never won. That moment is where tracks
break in the first place, so a veto on appearance fires on the joins that matter most. Both
constants were reverted; the measurement stays.

**What the benchmark could not show and a coach clip did** is the rule this repeats: a gate
tuned on the clips it is scored on has to be checked on one it was not, and the check is the
board with its scenes, not the score.

**The camera can say a player left the picture; it cannot say who walked back in.** The largest
share of the loss is a player out of shot while the camera looks elsewhere, and D96's camera sees
that happen: projecting a track's last point a few frames on shows whether it left the frame. Of
141 long gaps between one player's pieces, 79 (56%) are exactly that -- his track ends as he
walks out of the picture and the next begins as he walks back in. So the event is detectable.
The identity is not. Matching each exit to a later entry on the same side, with the TRUE teams:

    rule                                  made   right
    nearest entry                          182    85  (47%; his return often never comes)
    mutual nearest, within 10 m             65    48  (74%)
    one global assignment, within 8 m       67    45  (67%)

Team-mates leave and come back together, a back line a few metres apart, so position picks the
right man about seven times in ten -- below the 81% of the stitcher's own links, with the side
known perfectly, which the pipeline does not know. Not built.

**A and B fail for one reason.** Neither geometry nor this appearance model can say which of two
team-mates a piece of track belongs to, and every remaining join needs exactly that. What would
answer it is an identity signal that survives a tackle and a pan: a readable shirt number, or a
re-id model trained on football rather than on pedestrians.

**D98 -- a re-id network trained on football tells team-mates apart on clean crops and barely at
the ends of tracks, which is where the stitcher asks.** D97 left every remaining join needing an
identity signal, so OSNet-AIN was fine-tuned from its pinned weights on SoccerNet GSR crops:
eighteen clips of three matches from the valid split, batch-hard triplet with each batch drawn
from ONE clip and mostly ONE team, so the negatives are team-mates in the same kit. Chosen on a
fourth match (test split, game 11), judged on the benchmark once.

**The benchmark leaked, and it showed.** SoccerNet's clips are Swiss league: St. Gallen plays in
two training matches and in benchmark match 4, whose clips jumped to 0.93-0.96 AUC -- memory, not
skill. Scored team by team with St. Gallen set apart, a player against his own team-mates on
ground-truth crops goes from about 0.72 to about 0.91 on clubs the network never saw.

**At the stitcher it is 0.77 -> 0.82.** Measured on the fragments the pipeline actually makes,
players the network never saw, same player against team-mate:

    fragment ends, unseen players                       old     football
    AUC, same player vs team-mate                      0.765     0.818
    long gaps: others admitted keeping 75% of his       848       625
    re-entry, mutual best on position + look        76% right  78% right

The difference is WHERE the look is taken. A fragment ends because its player was lost -- in
contact, at the frame's edge, behind somebody -- and the crops there are the worst of him. Three
wrong players per right one at a usable threshold is not a join rule, and re-entry stays under
the stitcher's own 81%. Not integrated; the weights are in work/reid/football.pt.

**A look over the whole track is better, and it is still not identity.** Averaged over a
fragment's clean crops -- full height, clear of the frame's edge, not overlapped by another box --
rather than its last eight, the football network reaches 0.854 same-player-vs-team-mate at the
stitcher (from 0.765 today) and halves the wrong players a long-gap threshold lets in. Paired
with position for re-entry (D97), mutual best on distance plus look:

    re-entry matching                     made   right
    position alone                          67    51  (76%)
    + whole-track football look             73    61  (84%)   best-track coverage 58.3 -> 62.4%
      of those, clubs it never saw          49    37  (76%)
      of those, St. Gallen                  24    24  (100%)

Every point above position alone is St. Gallen, whose players it trained on. On clubs it never
saw it matches position and stays under the stitcher's own 81%. Stopped here, by the rule set
before it started.

The flip side is worth writing down. A network that has seen a side's players picks them out
perfectly -- and a coach analyses his OWN team, in every clip, every week. A model fitted to one
club's players would be the identity signal this stage lacks. What it would need that this
repo does not have is labels for that club, which a coach correcting boards is making anyway.

**D100 -- tagging a coach's own players names them within a match about as well as the stitcher
joins them, and tags from earlier matches do not carry to the next.** D98 ended on a hope: a
network that had seen a club's players picked them out, and a coach analyses one club every week.
So the coach tags his players -- one click per player per clip, the shirt number typed -- and the
fragments of a new clip are named against those tags, and same-named fragments joined. Simulated
on SoccerNet from ground truth, scored on the pipeline's own fragments of eight benchmark clips,
with a fragment's side taken from ground truth and a tag labelling a clean window: both favour
the idea.

Within one match, five clubs the network never saw, three clips tagged, fine-tuned on the tags:

    look                        named   right   joins right/wrong   best-track coverage
    stitcher alone                 --      --        --                  57.3%
    football, gallery (d1/d2<=0.7) 54%     83%      45/7                 63.2%
    fine-tuned on the tags         76%     81%      62/16                64.0%
    ... every other clip tagged    85%     88%      76/17                65.9%

Names are right about as often as the stitcher's own links (81%), so they can neither join
fragments the stitcher refused nor be shown -- one shirt in five wrong is what D5 exists to
prevent. The gain is a third of the 26 points D97 lost to splits, at the stitcher's precision.

**Earlier matches do not help, and a kit change is why.** St. Gallen is in four SoccerNet
matches, found by squad number (36, 44 and 50 recur together; unrelated sides share a median
0.18 of their numbers). A network retrained without them, tagged four clips per earlier match in
white, tested on a match in green:

    tags                                       right   joins right/wrong   coverage (61.1% alone)
    three clips of today's match               86-88%     83/2-83/4          70.4-70.7%
    one to three earlier matches, none today   51-73%     poor               63-65%
    trained on earlier, gallery from today         94%     71/2               68.9%

A gallery from other matches does not name a player in another shirt. Training on them and naming
against today's tags gives the one figure above 90%, and it cannot be credited to the earlier
matches: the same untrained network against two draws of today's tags names 86% and 95% right, a
spread as large as the gain. St. Gallen is also an easy side -- 86-88% within its own match
against 81% for the five others -- which is most of what D98 read as the network having seen them.

**Not built.** Identity from a coach's tags would need the look to survive a change of shirt,
which is exactly what a network trained on team-mates in one kit does not learn, or a signal
that is not the look at all -- the number itself, read (D32).
