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
