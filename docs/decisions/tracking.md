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
