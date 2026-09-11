# Decisions — teams

Stage 3 — which side each track is on.

Every decision here was measured before it was made. They are kept because the code
cites them by number, and because the failures are worth as much as the successes.

**D21 — which cluster is "home" is decided by which end the side plays at**, never by
whichever labelling scores best — that would be fitting to the yardstick. It is a weak
discriminator over a long clip, where both sides cover the same ground, so `score` reports
the team split permutation-invariantly: the question worth measuring is whether the sides
were told apart, not whether they got SoccerNet's names.

**D31 — the two kits are told apart on their axis of greatest variance, not by k-means.**
k-means was the obvious choice and it collapses. It minimises inertia, and the kits are
not cleanly bimodal — a dozen tracks of the same shirt vary more in light and pose than
two shirts differ from each other — so the cheapest split is one tight little cluster
against everybody else. Measured on SNGS-147: 44 tracks to 8, 70% right, and six spurious
tracks were enough to flip it.

Projecting the signatures onto their first principal component and cutting where the
between-class variance is greatest gives 26 and 26, 85% right. The `len(a) * len(b)` in
that score is exactly what stops one side swallowing the other.

**Goalkeepers are taken out first and put back after.** A keeper wears neither kit, and
left in he costs real accuracy — 83% against 93% on the same tracks. What identifies one
without being told is two things at once: a colour unlike either team AND standing near a
goal. Colour alone catches a player in odd light; position alone catches every defender on
a goal line.

Together these take the team split from 50.2% — pure chance — to **87.2%**.

**D61 — the kit signature separates the two teams almost perfectly, and gating on it still
does not fix the id switch. Written, measured, reverted.** Two recorded beliefs were wrong and
worth correcting before the next attempt repeats them.

*The switches are not team-mates.* `stage2_track` says a steal happens "where kit colour says
nothing because they are team-mates". Measured against ground truth, on the clips where
switches are worst, they are mostly OPPOSITE kits:

    clip        opposite   same   immediate (1-frame gap)   separation
    SNGS-116      166       48           70%                  0.8 m
    SNGS-110      174       32           60%                  1.1 m
    SNGS-121       56       12           70%                  1.6 m
    SNGS-067       47       51           45%                  1.6 m

*And they are immediate, not after a gap.* D19's "88 of 98 happened AFTER A GAP" is a fact
about SNGS-147, the clip it was measured on, and 147 is the outlier: 36% of its switches follow
a gap of more than six frames against 15% on SNGS-116. `MAX_AGE_S` is therefore not the lever
on the crowded clips.

*The signature is excellent.* Sampled over 2,400 detection pairs on SNGS-116 with ground-truth
teams, same-team pairs run to 0.62 at p90 and opposite-team pairs begin at 0.68 at p10 — no
overlap at all, and NO opposite-team pair looks more alike than the median team-mate pair.

So the signal is there, it is clean, and the switches are exactly the kind it should catch.
Raising `COLOR_WEIGHT` from 0.6 to 4.0 does nothing (SNGS-116 purity 71.9 -> 70.6 -> 71.8 ->
70.6, recall unmoved), which is what the existing comment already claimed. A hard GATE that
refuses an association outright rather than pricing it does slightly better:

    clip        purity           switches      recall
    SNGS-067    63.5 -> 66.0%    157 -> 149    unchanged
    SNGS-110    56.7 -> 58.3%    268 -> 264    unchanged
    SNGS-147    76.2 -> 76.4%     59 ->  51    unchanged
    SNGS-116    71.9 -> 71.5%    250 -> 251    unchanged

And through `boardFromTracks` it is a net regression: observed player-seconds across eleven
clips fall from 114.7 to 111.9, four boards worse, two better, five identical. SNGS-116 loses
almost six seconds of window and SNGS-066 goes from 92 curved runs to 65. The one real gain is
SNGS-069, whose home side goes from 0 players to 2.

Reverted. The board is the test (D35), and a metric that improves while the board does not is
not a reason to ship.

**Why it does not work, which is the useful part.** The track count barely moves — 56 to 55 on
SNGS-116 — so the gate is almost never the thing that refuses a match. At the moment of a steal
the track's own colour has already been pulled toward the thief by the rolling average
(`0.8 * prior + 0.2 * seen` reaches a new kit in about five frames), so what the gate compares
against is a blend rather than an opponent.

**Holding the colour still was then tried, and is worse than either.** Freezing a track's
signature after its first few observations should have left the gate something clean to test.
It costs purity everywhere instead:

    clip        rolling   freeze 5   freeze 10   freeze 5 + gate
    SNGS-116     71.9%     67.4%      68.4%        66.5%
    SNGS-067     63.5%     59.3%      58.3%        64.2%
    SNGS-121     71.8%     70.1%      69.2%        70.3%

and SNGS-116's switches go from 250 to 281, 263 and 291. The rolling average is not a bug to
be removed: a kit signature genuinely changes with pose, shadow and a turned back, so a frozen
one stops matching THE SAME PLAYER.

That is the real finding, and it closes this line of attack. Adapting to a player and
discriminating between players are the same mechanism pulling opposite ways, and 0.2 is already
a reasonable place to stand between them.

**And the evidence a steal needs is present, which rules out the two remaining excuses.** At the
frame a switch happens, the player who should have been taken IS detected -- 98% of the time on
SNGS-116, 93 to 95% on the other crowded clips, 80% on SNGS-147. So the detector is not the
constraint and the tracker is choosing wrongly with the right answer in front of it.

Nor is the crop spoiled by the occlusion that caused the crossing. Measured against per-team
reference signatures, at 120 of SNGS-116's switch frames the correct detection's kit sits at
0.45 from its OWN team and 0.87 from the other, and only 16% look more like the other team.

    at a switch    the right player is detected     98%
                   its kit still reads as its own   84%
                   the two kits are separable       no overlap at all

**Two more attempts, after `kit_mean` existed, and both fail the same way.** The rolling
average blending toward the thief was D61's stated cause, so comparing the association
against the whole-track mean instead should have left something clean to match on. Switches
rise on every clip tried -- SNGS-116 250 -> 268, SNGS-075 343 -> 353, SNGS-066 277 -> 287 --
because over a long track the mean behaves like the frozen signature that already failed.

Repairing a switch AFTERWARDS was the other shape worth trying, since that reasoning is what
makes `stage2_stitch` work. The structure is wrong for it. Of the tracks that hold more than
one true identity, only a quarter hold exactly two -- 6 of 24 on SNGS-116, 6 of 31 on
SNGS-075, 6 of 26 on SNGS-110 -- so there is no single boundary to cut at; the rest flip
repeatedly. And the best split of a track's kit sequence lands within three sightings of a
true boundary 26-37% of the time against about 14% for guessing. A trace, not a lever.

Seven attempts now. The one thing that HAS moved purity is stage 1: `winnow` took SNGS-147
from 69.3 to 90.3% (D62) by removing frames where the camera model throws every player at
once. Purity should be attacked from there, not from the tracker.

The detection exists, its appearance is right, and the signature discriminates. The failure is
therefore in the ASSIGNMENT rather than in any of the evidence it is given.

**Instrumenting the cost matrix says which part of the assignment**, and it is none of the
suspects. Every cost enquiry on SNGS-116 was logged and joined against ground truth:

    correct pairing gated out or never offered      1%
    reachable, priced, and passed over             87%
    right player genuinely undetected              12%

    of those passed over: the wrong one was priced LOWER   91%
                          the solver sacrificed it          9%
    median cost   correct 0.290   taken 0.300

So the gate is not it and the global assignment is not it. **The cost function is simply
indifferent** -- a 3% margin between the right answer and the wrong one, decided by noise.

**And the colour term contributes nothing to that margin, for a circular reason.** At those
same moments the track sits 0.22 from the correct player and 0.24 from the one it takes, even
though 59 of the 77 are OPPOSITE kits. A track's colour is learned from the identity it exists
to verify: after a switch it follows the wrong player through dozens of unambiguous frames and
correctly learns their kit. Two more attempts died on that:

* **Refusing to learn from a contested frame** (a margin the chosen pairing must beat before
  the kit updates) changes nothing at all, at any threshold from 0.05 to 0.4. The corruption
  does not happen in the contested frame; it happens in the clear ones afterwards.
* **A global team reference** -- kits fitted over every detection in the clip with
  `split_kits`, each track carrying a running MAJORITY side rather than a rolling average, and
  a penalty for pairing across it -- moves purity by less than two points in either direction
  and makes switches consistently WORSE: SNGS-116 goes 250, 253, 258, 275, 277 as the penalty
  rises, because a track that refuses an opponent dies and respawns instead.

Five attempts, no gain, and they rule out the whole appearance family: weighting, gating,
freezing, gating the learning, and replacing the signature with a global one. What is left is
the term that actually decides these matches. `dist / gate` is the dominant cost, the two
candidates are under a metre apart, and the prediction that separates them carries its own
error -- which is why VELOCITY_SMOOTHING was worth 3.4 points (D56) when none of this was worth
anything. A better motion model, or an association that defers the decision across frames
instead of committing every frame, is where the next attempt belongs.

**D63 — a labelling that fields nineteen players on one side is wrong whatever the kit
colours say, and a pitch knows it with no ground truth at all.** The boards that came out
badly were not small, they were LOPSIDED, which is a different defect and points somewhere
else:

    clip        board       home tracks / away tracks in the file
    SNGS-067    11 v 1            65 / 5
    SNGS-060    11 v 2            58 / 5
    SNGS-069     2 v 11            5 / 35
    SNGS-151     6 v 11           11 / 44
    SNGS-066    11 v 10           36 / 41   (healthy)

Against ground truth the team split on SNGS-067 is 51.1%, which is a coin flip, and nothing
downstream could tell: `unknown` was emitted on zero tracks across eleven clips, so a
collapsed split and a good one look identical to the importer.

**The collapse is provable from the file alone.** Counting tracks that hold a sample at one
frame — fragments of one player never overlap in time, so that count is a count of people:

    clip        concurrent home / away, sampled through the clip
    SNGS-067        [10, 15, 19, 10, 13] / [2, 1, 1, 1, 1]
    SNGS-069        [ 1,  1,  1,  1,  1] / [16, 18, 15, 8, 7]
    SNGS-066        [11, 10, 13, 10,  7] / [10, 6, 8, 11, 9]   (healthy)

Nineteen players of one side are not on a football pitch. `split_kits` is the thing that
failed, and it failed the way its own docstring says k-means did before it. Between-class
variance RESISTS one side swallowing the other -- that is what the `len(a) * len(b)` term is
for -- but it does not FORBID it: a handful of tracks far enough along the axis outscore an
even cut, and the light and pose within one kit vary more than two kits differ.

So the cut is now chosen as the best-scoring one that a pitch allows, and only the cut that
labels the teams is constrained -- the earlier one that hunts odd kits is left alone, or
keeper detection moves with it. The cap is twelve rather than eleven: a keeper that was not
pulled out as an outlier, somebody on the touchline, and two fragments either side of a
one-frame break each read as one player more than there is, and the failure being caught is
at nineteen, where the extra slack costs nothing.

    clip        team split, collapse rule -> and searching the axes
    SNGS-067       51.1 -> 69.2 -> 78.8%
    SNGS-147       70.5 -> 81.4 -> 81.4%
    SNGS-110       70.3 -> 70.3 -> 70.3%
    SNGS-116       71.6 -> 71.6 -> 71.6%
    SNGS-121       86.0 -> 86.0 -> 86.0%

**The rule needs somewhere to look, and one component was not enough.** On SNGS-060 with
the kit signature averaged over the whole track, no cut on the FIRST component leaves fewer
than twenty players on one side -- so the rule has nothing to choose and falls back to the
collapsed answer it exists to refuse. The largest axis of variance is the kits only when
the kits are what varies most; when it is the light, every cut along it is lopsided.
`split_kits` now searches the top `KIT_AXES` components and takes the best-scoring feasible
cut across all of them. Scores are compared raw, so the first component still wins wherever
it has a feasible cut. Without a feasibility test the search does not run at all -- there
would be nothing to tell a kit axis from a lighting one -- so the unconstrained answer is
byte for byte what it was.

**And when nothing fits, the fallback matters.** Falling back to the highest-scoring cut
hands back the collapse the rule exists to refuse -- which is what SNGS-060 got. The
feasibility test is now a COST, how many players over a team's worth the worst moment puts
on one side, and the least crowded cut stands when none reaches zero. With the shipping kit
signature nothing reaches the fallback and the output is identical; it is a floor under a
case that has already happened once.

SNGS-067 gains 9.6 points from the axis search and no clip loses any. Its board reaches ELEVEN A SIDE,
the first complete one in this repo, and SNGS-069 gains two players; nine boards are
untouched. Across the eleven that is 222 to 226 players and 2108.9 to 2087.1 player-seconds
-- more players, one percent fewer observed seconds, because a fuller roster is available
over a shorter passage. Player-seconds counts how much was watched and not whether it was
attributed to the right side, so it cannot see this change's point.

Recall, precision and identity purity are unchanged to the digit on all five, which is the
check that matters: the constraint moves team labels and touches nothing else.

**Four other things were tried and three of them failed**, which is worth as much as the
rest of this. Team split measured on the five clips with ground truth, against a 69.2 /
70.3 / 71.6 / 86.0 / 81.4 baseline:

    change                                        team split      wrong side   outcome
    grass masked out of the torso crop            net -0.6        34 of 210    dropped
    plus a saturation split for achromatic kits   net -2.0        46 of 208    dropped
    classify each sighting and vote per track     worse or equal  not run      dropped
    kit averaged over the track, not an EMA       net +5.2        32 of 211    SHIPPED

The wrong-side column was measured afterwards, rebuilt on the shipping code, because the
team split column is the one that had already misled this section twice. It does not rescue
either feature change: grass masking is a wash and costs three correct players, and the
saturation split costs seventeen. Per-sighting voting was never judged on team split -- it
lost to the track mean at the ceiling, 90.0 against 92.0 on SNGS-067 and 75.5 against 79.6
on SNGS-110 -- so it needs no re-reading.

The last one shipped, and only on the third measurement. Averaging over the whole track is
the better estimator: the ceiling, as nearest-centroid against ground-truth centroids, goes
86.0 -> 92.0 on SNGS-067, 92.1 -> 97.4 on SNGS-116 and 81.4 -> 86.0 on SNGS-147. `color` is
an exponential average over about five sightings and answers "which kit is this track
wearing NOW" -- the right question for the next frame's match and the wrong one for which
team it is on, so `kit_mean` answers that one instead.

It was rejected twice first. On its own it collapses SNGS-060, which the least-crowded
fallback then fixes; and on the five clips that had ground truth it still looked like a
regression, which is what the next paragraph is about. It fields three players fewer and
5% fewer player-seconds, and puts twelve fewer of them on the wrong side. A player on the
wrong side is worse than an absent one: it is an assertion the footage contradicts, and a
coach has to notice it before they can correct it.

**`team split` is not what the board sees, and neither is the player count.** `ft score`
measures team accuracy over samples across EVERY track, and a board fields the twenty or so
best-covered ones -- so relabelling fragments that never reach a board moves the metric and
not the product. `player-seconds` has the opposite blind spot: it counts how much was
watched, never whether it was attributed to the right side, so a board that swaps three
wrong players for three right ones scores identically. The measure that answers the
question is FIELDED PLAYERS ON THE WRONG SIDE, and it needs ground truth for every clip
that has a board:

    configuration                fielded   wrong   correct
    cap + axis search              214       44      170
    plus the whole-track mean      211       32      179

**Measured on five clips this reverses, and the five say the opposite of the eleven.** On
SNGS-067/110/116/121/147 alone the mean scores 16 of 89 against 15 of 92 and looks like a
regression; across all eleven it removes twelve wrong players for three fielded ones. Six
clips had no `truth.json` for no better reason than that `ft truth` had never been run on
them, though every one had its `Labels-GameState.json` on disk the whole time. The mean was
rejected twice on that subset before the missing six were generated. Judge team assignment
on the whole benchmark or not at all.

15% is the honest size of what is left: two or three players a board on the wrong side.

**And this one reaches the board**, which is what four of the last five per-frame wins did
not do. Eleven clips, same flags both sides:

    clip        before            after
    SNGS-060    13 (11 v 2)       21 (10 v 11)
    SNGS-067    12 (11 v 1)       20 (11 v 9)
    SNGS-069    11 ( 0 v 11)      19 (11 v 8)
    SNGS-151    17 ( 6 v 11)      21 (10 v 11)
    SNGS-147    18 (10 v 8)       18 (10 v 8), window 3.2 -> 8.6 s

Five improved, six were untouched, none lost a player. Observed player-seconds across the
eleven go 1906.6 to 2108.9.

**One board trades honestly and it is worth naming.** SNGS-067 gains eight players and loses
player-seconds, 153.9 to 116.7, because a fuller roster is available over a shorter passage
and `chooseWindow` takes it. By D52's measure that is a regression; a board with one opponent
on it is not a tactics board, so it is accepted. Average coverage per player is 0.43 either
way -- the window halved, the observation did not thin.

**D64 — an official is an odd kit that is not standing in a goal, and until 6 September one
reached nearly every board.** Found by watching a board next to its clip rather than by any
metric, which is the point: `ft truth` drops referees unless asked for them, so every accuracy
figure in this document written before this was measured against a ground truth containing no
officials at all. They were not counted as errors. They were invisible.

    boards                    fielded   officials   wrong side   correct players
    before                      223        11           30           182
    after                       222         6           31           185

`assign` never emitted `referee` although the schema has always had the label, so an official
was clustered onto whichever kit he sat nearer. The rule that finds a keeper already had the
shape: an outlier in colour, tested against position. A keeper is one that stands in a goal;
an official is one that does not. Both need the position test, because colour alone also
catches a player in strange light -- and `MAX_REFEREES` caps it at three, so an odd kit is
only read as an official while a slot remains.

Two things were measured and both changed the design. **Removing officials from the
clustering costs more than it saves**: three fewer tracks moves the axis and the cut, and it
took four correct players off the boards to take four officials off -- so they are named from
the split rather than held out of it. And **the outlier test must run against the SETTLED
split, not the rough one that finds the keepers**: measured against the final cluster centres
it finds six of eleven rather than four, because an outlier test is only as good as the model
it measures distance from.

**The kick-off was a separate defect and it is fixed in Pitchboard, not here (D65).** The
window and the restart detection were both right; the board handed the ball to a player who
was not on the pitch yet. Two wrong diagnoses were written before the measurement that
settled it, and both are worth knowing about. It is NOT that no scene falls at the handover
-- seeding one there was built, measured and reverted, because it moved a single scene by ten
frames and improved nothing. And it is NOT that the receiver is unfielded, which was inferred
from "no fielded player is within four metres of the ball" -- a kick-off may be played
anywhere in one's own half, so a ball far from everyone is what a pass in flight looks like
and proves nothing at all.

**D72 — a side the kit split is not sure of is declined, because a wrong colour is a pass that
never happened.** A coach watching SNGS-067 said the board mixed up which team made the pass.
It did: the team split is 79.7% there, so one shirt in five is the wrong colour, and the
importer draws the pass between whatever shirts the carrier lands on. A wrong side is not a
cosmetic error, it is an invented turnover.

The same rule as an unread shirt number (D5). A track whose kit sits nearly as close to the
other side's colours as to its own is labelled `unknown` and the importer drops it, so the
board fields fewer players and the ones it fields are right. Measured against ground truth, per
sample:

    clip        before          after                        declined
    SNGS-067    79.7% right     90.8% of what it asserts      1298
    SNGS-151    84.0%           90.6%                         1203
    SNGS-060    79.2%           85.7%                         1202
    SNGS-116    80.5%           81.8%                          383
    SNGS-121    85.3%           85.3%                            0
    SNGS-147    78.2%           76.0%                          377

Across the six: 30% fewer wrong-coloured samples for 6% fewer correct ones. SNGS-121 has no kit
ambiguity to find and SNGS-147's errors are CONFIDENT -- its mislabelled tracks sit squarely in
the wrong cluster, which is an identity problem upstream rather than a colour one, and no
margin on this measurement can see it.

**Judged leave-one-out, which is the whole difference.** A track compared with a centre it
helped compute drags that centre towards itself, and the closer to the cut it sits the more it
flatters itself: with the naive centres SNGS-147 declined nothing at all.

**And `ft score` now reports teams the way it reports shirt numbers** -- right, WRONG, declined
-- because a single accuracy figure counts a declined side as an error and would have scored
this change as a regression. That is the same trap as `observed_error` (D67) in a third place:
a metric that cannot see a refusal will always reward guessing.

**D81 — the board wears the kits, because a coach reads his own clip in them.** Pitchboard
draws `home` red and `away` blue, and `home` is whichever side defends the nearer goal (D63).
On a coach's own clip that made Manchester United, in red, the blue team — and every sentence
after it had to be translated: *"the blue team (which is man united) loses the ball to a red
player"*. Three rounds of this repo's own analysis went past each other for the same reason.

The measurement was already there. Every track carries a kit signature, which is what the two
sides are told apart on — but a signature is a histogram and a histogram is not a colour, so a
second and much simpler statistic rides along: the mean BGR of the same torso crop. The median
of those across a side, written into `tracks.json` as `kits`, is what the board paints.

Three things it does NOT do:

- **It does not average a side into existence.** The MEDIAN across the tracks, so one track
  holding two players (D78) or a keeper labelled as an outfielder is one shirt among a dozen
  rather than a fifth of the answer.
- **It does not offer two colours that look the same.** Where the sides' medians are closer
  than `KIT_TONE_APART`, the field is absent and a consumer keeps its own palette -- two greys
  is a worse board than two arbitrary colours that at least differ.
- **It does not invent a hue.** A torso crop averages the shirt with shadow, skin and grass, so
  a red kit measures as dull salmon; the hue survives that mixing and the rest does not, so the
  hue is kept and the saturation and brightness are restated at what a kit actually has. Below
  `KIT_ACHROMATIC` there is no hue to keep -- white, grey and black kits measure noise -- and
  those get a light or dark neutral instead of whatever colour the noise pointed at.

On the coach's clip: Everton `#3a81d1`, United `#d1493a`. The contract carries it as optional,
so every file written before this one still imports, and a consumer that ignores it is correct.

**D84 — two ways to catch a track that changes shirt, both measured, both refused.** On a Porto
possession highlight the board never showed the pass out of defence, because the player who
received it is a track holding two people: Porto's stripes to frame 87, a Manchester City shirt
after it. Stage 3 sees one kit sitting exactly between the two sides — own 0.24, other 0.24 —
declines it (D72), and the board cannot field him. D78's veto did not catch the switch: the two
histograms disagree by 0.53 at the frame it happens, under the 0.6 that ships.

**A tone veto — the same crop's mean colour, which a hue histogram is blind to.** The switch is
obvious in it: [170, 166, 157] to [75, 120, 110], a step of 62 where the histogram moved 0.53.
It is also wrong:

    tone veto      SNGS-147 purity/switches/teams     SNGS-116
    off            78.6%  43  77%                     73.4%  257  78%
    80             78.6%  44  73%                     72.6%  269  74%
    50             75.5%  57  65%                     72.9%  290  76%
    40             67.9%  81  87%                     67.6%  290  73%

A tone is the shirt mixed with the light on it, and a player crossing into shadow moves thirty
or forty without changing shirts — so every setting that catches the switch tears real tracks,
and the team split falls with the purity. This is D81's warning arriving from the other side:
the mean is for SHOWING a kit, and it is too blunt to decide anything.

**Splitting a track whose kit changes halfway.** Cut each track at the point that maximises the
difference between the two halves' mean kits, and refuse the join above a threshold. There is
no threshold. Across three clips the declined tracks — the population this is for — split at
0.12 to 0.71, and the tracks whose side the split was SURE of run to 0.63, 0.69 and 0.83. The
distributions sit on top of each other, so any cut that catches the Porto/City switch at 0.53
also cuts a dozen tracks that are one player.

**What that leaves.** A switch between two kits that a hue histogram cannot separate needs an
appearance model that is not a colour histogram, which is D32's and D61's answer in a third
place. What ships instead is on the consumer's side and much smaller: a track nobody could name
BLOCKS the ball rather than being stepped over (Pitchboard D78), so the ball stops being handed
to the nearest player who happens to have a side, which on this clip was an opponent.

**D85 — a DECLINED track may be cut where its shirt changes, and only a declined one.** D84
looked for a threshold that says "this track holds two players" and found none: across three
clips the declined tracks split at 0.12 to 0.71 and the confidently named ones run to 0.83, so
any cut deep enough to catch a real switch also cuts a dozen tracks that are one man.

The asymmetry it missed is that a declined track is ALREADY THROWN AWAY. Pitchboard fields
nobody it cannot name (D72), so for that population the question is not "is this suspicious
enough" but "does cutting it produce two halves the ordinary test is SURE of, one on each
side". If it does, two players come back that were otherwise lost. If it does not, the track
stays declined and nothing is worse. There is no threshold to defend, because the test is the
same `KIT_MARGIN` that declined the whole track in the first place.

On the coach's Porto clip the man who received the goalkeeper's pass was exactly this: Porto's
stripes to frame 87, a Manchester City shirt after it, one kit sitting midway between the sides
at own 0.24 against other 0.24. Cut at 87 it is two players, each named with the margin to
spare, and the move he was in the middle of can be drawn. Across the benchmark it costs nothing
-- SNGS-147 and SNGS-121 are unchanged to the digit, SNGS-116 gains one track with the same
recall, precision, purity and team split.

What it does not do is prevent the switch. The tracker still walked from one player to the
other; this only reads the evidence it left behind.

**D87 — a shirt answers three questions, and only one of them is about hue.** D81 split
DECIDING from SHOWING: a histogram tells two kits apart and paints nothing, a mean BGR
paints a shirt and cannot tell red from blue. The coach's Sporting–Galatasaray clip found
a third question hiding inside the first.

Two of Sporting's hooped shirts came out on Galatasaray. Not marginally: the leave-one-out
test was SURER of them than of two genuinely red tracks (0.58 and 0.74 against 0.69 and
0.67), so no threshold reaches this. The split itself had them, and it had them because
their signatures really did sit nearer the red centre:

    hue bin          h0     h1     h2    ...    h9    h10    h11
    hoops, near     .155   .351   .354         .024   .034   .046
    red             .440   .125   .097         .000   .037   .301
    t13             .150   .301   .117         .089   .239   .104
    t14             .220   .271   .103         .047   .224   .135

t13 and t14 keep the hoops' green peak at h1 and carry 40% of their mass at h9-h11, where
the near-side hooped tracks carry 10%. That mass is their WHITE. A pixel with no
saturation still has a hue and it is noise, and the noise is not evenly spread -- a warm
floodlight lands it at the pink end, next door to red. Far-side players are small and
blurred, so proportionally more of their crop is that noise, which is why this hit the two
deepest players and nobody else.

So the side is now read from `side_mean`: the same hue/value grid with the colourless
pixels gathered into one bin instead of spread across twelve.

**Gathered, not dropped**, and the difference is a whole clip. Discarding them scored
better on the clip that found the bug and collapsed on SNGS-116, which is WHITE against
red -- a white shirt with its colourless pixels thrown away is a signature of trim and
skin. Colourlessness is a kit, not a gap. One bin says what the shirt is; twelve say what
it is not.

    teams               147  77% -> 89%      116  78% -> 77%      121  86% -> 83%
    asserted right           2919 -> 3402        5410 -> 5328        6564 -> 6395
    asserted WRONG            366 ->  301        1185 -> 1628        1097 -> 1266
    declined                  522 ->  104         361 ->    0           0 ->    0
    the coach's clip    11 of 13 right, 2 WRONG  ->  14 of 14 right

And the tracker does not see it. Associating asks a different question -- is this the same
player next frame -- and there a white hoop or a black sleeve is as much a part of what
somebody looks like as anything else. Applied to `kit` as well, the floor starved the
tracker and fragmented it: SNGS-116 went 57 tracks to 79 and its purity 74.9% to 69.4%,
SNGS-121 38 to 58 and 73.7% to 63.0%. Kept to the side question, recall, precision, error
and purity are identical to the digit on all three clips.

The sides also came out named the way a coach names them, which is a consequence rather
than a goal: with the two deep defenders back among their own team, Sporting's mean x
includes its own keeper and the board calls them home.

**D91 — the margin is silence, and silence costs more for the man on the ball.** D72 declines
a side the kit will not settle, because a wrong colour reaches the board as a pass between
the wrong shirts. That price is right for the twenty-one players who are not on the ball and
wrong for the one who is, and the difference is not a matter of degree: Pitchboard fields
nobody it cannot name, so declining an outfielder leaves a board with ten men, while
declining the CARRIER leaves the move undrawn.

The coach watched it happen. His striker was tracked from the halfway line through the run
that won a penalty, carried the ball for frames 506-573, and came out `unknown` at own 0.270
against a bar of 0.251 -- clustered with the right side, 8% short of the confidence to say
so. What the board drew in his place was a player first seen forty frames from the end, held
at the position he would eventually reach, standing offside for nine seconds of a thirteen
second clip.

So a track the ball went through is named on the plain comparison, without the margin.
**Still only where the kit agrees**: nearer its own side than the other is the whole claim,
and a carrier that fails it stays unknown. The exemption is one track, not a looser
threshold -- naming a carrier against his own kit is exactly the invented turnover D72
exists to prevent, and this does not do it.

It is a surgical rule and the measurement says so. Across the three coach clips it renames
ONE track:

    Untitled      ball through 4 tracks   declined 4 -> 3   rescued t23 (the striker)
    nottingham    ball through 3 tracks   declined 6 -> 6   rescued nothing
    geny_rioave   ball never located      declined 0 -> 0   rescued nothing

The benchmark cannot judge it, and that is worth saying rather than hiding behind: after
D90's weighting, SNGS-147, SNGS-116 and SNGS-121 decline NOTHING, so there is nothing for
this to rescue and the numbers are identical to the digit. It is the same shape as D86 --
the clips with ground truth are the harm test, and the harm they can show is zero because
the condition never arises on them. What validates the rule is that it fires nine times out
of ten on nobody, and the tenth is a striker the coach could see.

The board now draws the move he described: `keeper -> midfielder -> striker -> a Galatasaray
defender after the foul`, with the striker first drawn at (47.6, 43.2), beside the man who
receives rather than twenty-five metres past him.
