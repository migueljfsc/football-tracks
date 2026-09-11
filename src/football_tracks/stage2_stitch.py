"""Fragments of one player, joined back into one track.

Stage 2 tracks in IMAGE space on purpose: its gate is a multiple of the detection box's
height, so it needs no homography and cannot inherit the camera model's errors. That is
the right call for tracking and the wrong one for this, because whether two fragments are
the same person is a question about metres per second, and only the pitch knows those.
So this is a separate pass over the pitch-space samples rather than a change to the gate.

**Why fragments exist at all.** `MAX_AGE_S` is 0.24 s, and deliberately so: a track that
outlives its player coasts on a stale prediction, and when detections resume it takes
whoever is nearest -- an opponent, over half the time. Shortening the wait fixed the
steals and accepted the fragments. Measured on SNGS-147 against ground truth, a player is
in shot about 239 frames of 750 and comes out as a median of 4 fragments, so roughly 48
frames each. Pitchboard's importer wants 30% of its window covered before a track becomes
a player, and 48 frames is 6%. That is why a 22-player clip becomes a ten-player board.

**And what joining them is worth, which is less than it looks.** Join every fragment of a
player perfectly -- the ceiling, from ground truth -- and the median player's best track
holds 47% of their life on SNGS-151, 52% on SNGS-116, 60% on SNGS-147, 67% on SNGS-121 and
91% on SNGS-060. The rest is not in another fragment: it was never tracked, or the camera
model put it more than the match radius away. This stage cannot reach it (D69).

**Why joining them afterwards is safe where lengthening the wait was not.** The tracker
must decide at the moment of the gap, with nothing after it to go on. This runs when both
sides are known, so it can ask whether the two fragments are mutually each other's best
partner, and where each was GOING rather than merely how far apart they are. 39% of the
breaks have a gap of one frame or less -- the player never disappears, the tracker simply
renumbers them -- and 70% are inside 12 frames.

The failure this must not cause is a track that teleports between two players, which is
worse than two honest halves: it puts one player's run on another's shirt. Hence the speed
gate, the mutual-best rule, and no chaining beyond what each link earns on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .stage2_track import MAX_SPEED
from .stage2_track import color_distance as kit_distance
from .tracks import Sample

# How long a gap may be and still be bridged.
#
# It was 0.5 s, chosen because it reaches 70% of the identity CHANGES on SNGS-147 -- but
# the breaks that cost the roster are a different population, and they are much longer.
# Grouping fragments by the real player they track and asking why each consecutive pair
# was refused, the gap limit is 55-64% of every refusal and the gaps it refuses run to a
# median of 2.2-2.8 s. Half a second was rejecting most of what there is to join.
#
# What makes three seconds safe is the gate below rather than the length itself: reaching
# further on a plain speed limit admits most of the pitch, which is exactly why the
# original constant was short.
MAX_GAP_S = 3.0

# Slack on the position gate, in metres, for the camera model's own error. Position error
# is 0.5-1.3 m per sample depending on the registration, so two samples of one stationary
# player can sit two metres apart with nobody having moved.
POSITION_SLACK_M = 2.5

# How far a player may deviate from where they were going, in metres per second of gap.
#
# The gate is a PREDICTION rather than a reach: where the first fragment was heading, met
# by where the second came from. A reach gate grows at 12 m/s, so at three seconds it
# admits 38 metres and any two players in one kit; this grows at two, which is what a
# player can do by changing their mind rather than by running.
#
# Measured on the fragments five clips actually produce, joining pairs whose real identity
# the ground truth can confirm:
#
#     gate                     joins    wrong of those judged
#     0.5 s reach (was)         5-15          41%
#     3.0 s reach              16-20          38%
#     3.0 s predict 2.0 m/s    10-19          27%
#     3.0 s predict 1.5 m/s     7-17          26%
#
# The prediction is only as good as the velocity behind it, and that was read over a fixed
# number of SAMPLES until a 33 fps clip showed what it costs -- see VELOCITY_S.
#
# The prediction gate is the only one that makes MORE joins and gets FEWER of them wrong,
# which is the trade a reach gate could not offer at any length. Refusing a join whose
# runner-up is nearly as good buys nothing on top -- the wrong ones are confident, not
# ambiguous -- so there is no margin rule here.
#
# 1.5 rather than 2.0 because of what a wrong join does downstream. Teams are clustered on
# a whole track's kit, so a track holding two players holds a blend of two kits: at 2.0 the
# team split on SNGS-147 falls from 79.4% to 62.3%, and at 1.5 it holds at 78.2% for most
# of the same gain (2531 observed player-seconds across eleven boards against 2640, from a
# baseline of 1934).
PREDICT_DRIFT_MS = 1.5

# How much TRACK a fragment's velocity is read over, in seconds.
#
# Seconds and not samples. It was five samples, which is half a second on a 25 fps clip
# stored at a tenth of a second and an eighth of a second on a 33 fps one stored at full
# rate -- and an eighth of a second turns 0.7 m of position noise into 5.7 m/s of sprinting
# sideways. Measured on a real clip: the player who made the run and the player who shot
# were one man, his two fragments ended 2.1 m apart, and the prediction built from that
# noise missed by 3.3 m against a 3.0 m tolerance. The board drew the shot as a pass to
# somebody standing in the box.
#
# The same fault Pitchboard's D52 records on its side of the seam: a speed measured across
# a short window is a position error divided by a small number.
VELOCITY_S = 0.4

# Below this much track there is no direction to read, only noise, so the prediction falls
# back to "stays where it was" -- which the tolerance can absorb and a wrong heading cannot.
MIN_VELOCITY_S = 0.15

# What disagreeing kit costs, as a fraction of the distance budget. Same role and the same
# value as the tracker's own: enough to break a tie between two candidates, not enough to
# override a plain speed violation.
COLOR_WEIGHT = 0.6

# Kit distance beyond which a join is refused outright, whatever the prediction says.
#
# The same guard the tracker makes at association time (`stage2_track.KIT_VETO`) and for the
# same reason, because a join is the same claim made across a longer gap: a weight only
# decides between candidates that exist, and where a player's own fragment is missing the
# best continuation on offer can be an opponent standing where he was heading. Measured on
# the joins five clips actually make, several disagree on kit by 0.6 to 0.88 -- and a joined
# track holding two kits is the case this file's own PREDICT_DRIFT_MS note records as
# costing seventeen points of team split.
#
# In the tracker's units, not this file's `_color_distance`, so that one number means one
# thing on both sides of the gap.
KIT_VETO = 0.6


@dataclass(slots=True)
class Fragment:
    id: int
    samples: list[Sample]
    color: np.ndarray | None = None

    @property
    def first(self) -> Sample:
        return self.samples[0]

    @property
    def last(self) -> Sample:
        return self.samples[-1]


def _color_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Nothing is known about a missing kit, so it costs nothing rather than everything."""
    if a is None or b is None:
        return 0.0
    return float(np.linalg.norm(a - b))


def _velocity(samples: list[Sample], fps: float, *, at_end: bool) -> tuple[float, float]:
    """Metres per second at one end of a fragment, capped at what a footballer can run.

    Read over `VELOCITY_S` of track, so the answer means the same thing whatever the frame
    rate and whatever interval the file was written at.
    """
    edge = samples[-1] if at_end else samples[0]
    window = [s for s in samples if abs(s.f - edge.f) <= VELOCITY_S * fps]
    if len(window) < 2:
        return (0.0, 0.0)
    ss = window if at_end else window[::-1]
    dt = abs(ss[-1].f - ss[0].f) / fps
    if dt < MIN_VELOCITY_S:
        return (0.0, 0.0)
    vx, vy = (ss[-1].x - ss[0].x) / dt, (ss[-1].y - ss[0].y) / dt
    speed = float(np.hypot(vx, vy))
    if speed > MAX_SPEED:
        vx, vy = vx * MAX_SPEED / speed, vy * MAX_SPEED / speed
    return (vx, vy)


# How close two tracks must sit, in metres, before they might be one player seen twice,
# and over how many samples.
SAME_PLAYER_M = 1.5
MIN_SHARED_SAMPLES = 5

# And the test that actually decides it: what share of the overlap both tracks have a
# sample on.
#
# Distance alone cannot separate a duplicate from a striker and the man marking him -- they
# run a metre apart for as long as the move lasts, and merging them destroys two players to
# fix nothing. What separates them is the DETECTOR: it finds a player once, so two tracks
# on one man have to take turns, while two tracks on two men each get a box every frame.
# Measured on the coach's clip the split is unmissable -- the duplicates share 0%, 5% and
# 9% of their frames, the real pairs 81% and 100% (D90).
TOGETHER_MAX = 0.25


def _at(samples: list[Sample], f: int) -> tuple[float, float] | None:
    """Where a track was at a frame, interpolated INSIDE its span and nowhere else."""
    if f < samples[0].f or f > samples[-1].f:
        return None
    before = [s for s in samples if s.f <= f]
    after = [s for s in samples if s.f >= f]
    lo, hi = before[-1], after[0]
    if hi.f == lo.f:
        return (lo.x, lo.y)
    r = (f - lo.f) / (hi.f - lo.f)
    return (lo.x + r * (hi.x - lo.x), lo.y + r * (hi.y - lo.y))


def duplicates(positions: dict[int, list[Sample]], fps: float) -> dict[int, int]:
    """Absorbed track id -> the track it is a duplicate of.

    The case `stitch` cannot see, because it asks whether one fragment CONTINUES another
    and refuses anything that overlaps in time (`gap <= 0`). Two tracks running alongside
    each other a metre apart are not a continuation and not two players either; they are
    one man the tracker renumbered without ever losing.

    It matters more than a spare track. The two halves are labelled separately, so a
    player who is home for the first half of a clip is away for the second: 30 of these
    pairs across the benchmark disagree about the SIDE, which reaches the board as a
    turnover that never happened. And a coach found it the other way round -- his striker
    was tracked from the halfway line as one id and the run was thrown away as `unknown`,
    while a second id starting 190 frames later was fielded and held, parking him offside
    for nine seconds.
    """
    ids = sorted(positions)
    same: dict[int, int] = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            here, there = positions[a], positions[b]
            if not here or not there:
                continue
            lo = max(here[0].f, there[0].f)
            hi = min(here[-1].f, there[-1].f)
            if hi <= lo:
                continue
            apart = [
                float(np.hypot(s.x - o[0], s.y - o[1]))
                for s in here
                if lo <= s.f <= hi
                for o in [_at(there, s.f)]
                if o is not None
            ]
            if len(apart) < MIN_SHARED_SAMPLES:
                continue
            mine = {s.f for s in here if lo <= s.f <= hi}
            theirs = {s.f for s in there if lo <= s.f <= hi}
            together = len(mine & theirs) / max(1, len(mine | theirs))
            if together > TOGETHER_MAX:
                continue  # both had a detection of their own: two players, not one twice
            if float(np.median(apart)) < SAME_PLAYER_M:
                # The longer track survives, so the shorter one's samples fill its gaps
                # rather than the other way round.
                keep, gone = (a, b) if len(here) >= len(there) else (b, a)
                while keep in same:
                    keep = same[keep]
                if keep != gone and gone not in same:
                    same[gone] = keep
    return same


def merge(positions: dict[int, list[Sample]], same: dict[int, int]) -> dict[int, list[Sample]]:
    """Fold every duplicate into the track it duplicates, keeping one sample per frame."""
    out = {i: list(ss) for i, ss in positions.items()}
    for gone, keep in same.items():
        if gone not in out or keep not in out:
            continue
        held = {s.f for s in out[keep]}
        out[keep] = sorted(out[keep] + [s for s in out[gone] if s.f not in held], key=lambda s: s.f)
        del out[gone]
    return out


def _cost(a: Fragment, b: Fragment, fps: float) -> float | None:
    """What it costs to call `b` the continuation of `a`, or None if it cannot be.

    Both ends have to agree. `a` is walked forward from where it was going and `b` back
    from where it came from, and the join is refused on whichever disagrees more -- so a
    fragment that merely happens to be within reach is not a continuation, and one that
    lands where the run was heading is, even seconds later.
    """
    gap = b.first.f - a.last.f
    if gap <= 0 or gap > MAX_GAP_S * fps:
        return None
    gap_s = gap / fps
    ax, ay = _velocity(a.samples, fps, at_end=True)
    bx, by = _velocity(b.samples, fps, at_end=False)
    tolerance = POSITION_SLACK_M + PREDICT_DRIFT_MS * gap_s
    ahead = float(
        np.hypot(b.first.x - (a.last.x + ax * gap_s), b.first.y - (a.last.y + ay * gap_s))
    )
    behind = float(
        np.hypot(a.last.x - (b.first.x + bx * gap_s), a.last.y - (b.first.y + by * gap_s))
    )
    worst = max(ahead, behind)
    if worst > tolerance:
        return None  # not where either of them was going; these are two different people
    if kit_distance(a.color, b.color) > KIT_VETO:
        return None  # two kits: whoever this is, it is not the same player
    return worst / tolerance + COLOR_WEIGHT * _color_distance(a.color, b.color)


def stitch(
    positions: dict[int, list[Sample]], colors: dict[int, Any], fps: float
) -> dict[int, list[Sample]]:
    """Join fragments that are each other's best continuation, and nobody else's.

    Mutual best rather than greedy-global: a fragment that ends in a crowd has several
    plausible successors and picking the cheapest is how one player's run lands on
    another's shirt. Requiring the choice to be returned makes an ambiguous join fail
    into two honest fragments, which is the outcome the importer can survive.
    """
    frags = {
        i: Fragment(id=i, samples=sorted(ss, key=lambda s: s.f), color=colors.get(i))
        for i, ss in positions.items()
        if ss
    }
    if len(frags) < 2:
        return positions

    order = sorted(frags.values(), key=lambda f: f.first.f)
    best_next: dict[int, tuple[float, int]] = {}
    best_prev: dict[int, tuple[float, int]] = {}
    for a in order:
        for b in order:
            if a.id == b.id:
                continue
            c = _cost(a, b, fps)
            if c is None:
                continue
            if a.id not in best_next or c < best_next[a.id][0]:
                best_next[a.id] = (c, b.id)
            if b.id not in best_prev or c < best_prev[b.id][0]:
                best_prev[b.id] = (c, a.id)

    successor = {a: b for a, (_c, b) in best_next.items() if best_prev.get(b, (0.0, -1))[1] == a}

    # Walk each chain from its head, so a run of fragments collapses into one track and
    # keeps the id it started with -- the id a caller may already have written down.
    tails = set(successor.values())
    out: dict[int, list[Sample]] = {}
    joined: set[int] = set()
    for head in sorted(frags):
        if head in tails:
            continue
        samples: list[Sample] = []
        node: int | None = head
        while node is not None and node not in joined:
            joined.add(node)
            samples.extend(frags[node].samples)
            node = successor.get(node)
        out[head] = sorted(samples, key=lambda s: s.f)
    # A fragment inside a cycle would otherwise vanish; there should be none, but losing
    # a player silently is not an acceptable way to find that out.
    for i, f in frags.items():
        if i not in joined:
            out[i] = f.samples
    return out
