"""Stage 3 - which side is each track on.

Two kits, told apart on the shirt signature the tracker already maintains. Written out
rather than imported: the input is a dozen vectors, and sklearn is a large dependency to
add for thirty lines.

Which cluster is `home` is NOT decided by looking at the answer. It is decided by mean
pitch x - the side whose players average nearer x=0 defends the left goal - which is
SoccerNet's own left/right convention and is deterministic. Picking the labelling that
happens to score best would be fitting to the yardstick.

A keeper wears neither kit, and left in the clustering they cost real accuracy: on
SNGS-147 the two sides come out 83% right with keepers included and 93% without. So
they are taken out and put back. What identifies one without being told is the two
things at once - a colour unlike either team, and standing near a goal.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from .config import PITCH_LENGTH
from .stage2_track import Track
from .tracks import TeamLabel

Vec = npt.NDArray[np.float64]
# How far from its cluster's centre a kit may sit, as a multiple of the median distance,
# before it is treated as neither team's. A goalkeeper is the case this exists for.
OUTLIER_RATIO = 2.0

# How near a goal line a track's average position must be, in metres, for an odd-coloured
# track to be read as that goal's keeper rather than a player in a strange light.
KEEPER_ZONE_M = 22.0

# Most officials that may be picked out of one clip. A match has three on the pitch and
# rarely more than two in shot, and the cap is what stops a player in strange light being
# deleted: an odd kit is only read as an official while there is a slot left for one.
MAX_REFEREES = 3

# Most tracks of one side that may be in shot at once before the split is read as having
# collapsed. A team fields eleven; the slack covers a keeper that was not pulled out as an
# outlier, somebody on the touchline, and two fragments either side of a one-frame break.
MAX_CONCURRENT_PER_SIDE = 12

# Principal axes searched for a cut a pitch allows. The first is the direction of greatest
# variance, which is the kits when the kits are what varies most and the light when they
# are not. Only reached when a feasibility test is supplied, because without one there is
# nothing to tell those two cases apart.
KIT_AXES = 3


def split_kits(
    points: Vec,
    crowding: Callable[[npt.NDArray[np.int_]], int] | None = None,
) -> npt.NDArray[np.int_]:
    """Two groups of kits: project onto the axis they differ along, and cut.

    NOT k-means, which was the first version and collapses. k-means minimises inertia,
    and when the data is not cleanly bimodal — which it is not, because a dozen tracks
    of the same kit vary more in light and pose than two kits differ — the cheapest
    split is one tight little cluster and one holding everybody else. Measured on
    SNGS-147 it put 44 tracks on one side and 8 on the other, 70% right, and six
    spurious tracks were enough to flip it.

    Instead the kits are projected onto their axis of greatest variance, which is the
    direction the two teams differ along, and cut where the between-class variance is
    greatest. That is Otsu's method, and the `len(a) * len(b)` in its score is exactly
    what stops one side swallowing the other. Same data: 26 and 26, 85% right.

    Exhaustive over cut points and so reproducible. A pipeline that relabels the teams
    on a rerun is one nobody can check.

    `crowding` scores a labelling on something colour cannot see: how many players over
    what a pitch holds the worst moment puts on one side. Zero is allowed, and the
    best-scoring cut that manages it wins. Between-class variance resists one side
    swallowing the other but does not forbid it -- a handful of tracks far enough along
    the axis outscores an even cut, and the result is every player on one team.

    When no cut manages zero, the LEAST crowded one stands. Falling back to the
    highest-scoring cut instead returns the collapsed answer this exists to refuse,
    which is what SNGS-060 got.

    The search runs over the top `KIT_AXES` components rather than the first alone. The
    largest axis of variance is the kits only when the kits are what varies most; when it
    is the light instead, EVERY cut along it is lopsided and choosing between them cannot
    help -- measured on SNGS-060, where no cut on the first component left fewer than
    twenty players on one side. Scores are compared raw across axes, so the first
    component wins wherever it has a feasible cut and the others are reached only when it
    does not.
    """
    n = len(points)
    if n < 2:
        return np.zeros(n, dtype=np.int_)

    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    # Without a feasibility test there is no way to prefer one axis over another, so the
    # unconstrained answer stays what it always was: the first component.
    axes = min(KIT_AXES, len(vt)) if crowding is not None else 1

    orders, scored = [], []
    for ax in range(axes):
        projected = centred @ vt[ax]
        order = np.argsort(projected)
        orders.append(order)
        for i in range(1, n):
            a, b = projected[order[:i]], projected[order[i:]]
            score = len(a) * len(b) * (float(a.mean()) - float(b.mean())) ** 2
            scored.append((score, ax, i))
    # Ties keep the earliest axis and the lowest cut, which is what the running maximum
    # this replaced did.
    scored.sort(key=lambda s: (-s[0], s[1], s[2]))

    def labelling(ax: int, cut: int) -> npt.NDArray[np.int_]:
        labels = np.zeros(n, dtype=np.int_)
        labels[orders[ax][cut:]] = 1
        return labels

    if crowding is None:
        return labelling(scored[0][1], scored[0][2])

    best, over = None, None
    for _, ax, cut in scored:
        labels = labelling(ax, cut)
        crowd = crowding(labels)
        if crowd == 0:
            return labels
        if over is None or crowd < over:
            best, over = labels, crowd
    return best if best is not None else labelling(scored[0][1], scored[0][2])


def _oddness(points: Vec, labels: npt.NDArray[np.int_]) -> Vec:
    """How far each kit sits from its own group, as a multiple of the median distance.

    Returned as the ratio rather than a verdict because two things are decided from it and
    they need different amounts of it: a goalkeeper is whichever odd kit stands in a goal,
    and an official is the oddest few that do not.
    """
    centres = np.array(
        [
            points[labels == k].mean(axis=0) if np.any(labels == k) else points.mean(axis=0)
            for k in (0, 1)
        ]
    )
    d = np.linalg.norm(points - centres[labels], axis=1)
    median = float(np.median(d))
    if median <= 0:
        return np.zeros(len(points), dtype=np.float64)
    return np.asarray(d / median, dtype=np.float64)


def _crowding(
    tracks: list[Track], frames: dict[int, list[int]] | None
) -> Callable[[npt.NDArray[np.int_]], int] | None:
    """How many players over a team's worth a labelling puts on one side at once.

    Fragments of one player never overlap in time -- the tracker ends one and starts the
    next -- so the number of tracks carrying a sample at a frame is a number of people.
    Twenty of them on one side is not a close call about kit colour; it is the cut having
    collapsed, and it is visible without any ground truth.

    Frames come from the stitched pitch samples where the caller has them, because a
    joined track's own observations are only the half the tracker kept.
    """
    seen = (
        [set(frames.get(t.id, ())) for t in tracks]
        if frames is not None
        else [{o.f for o in t.observations} for t in tracks]
    )
    every = sorted(set().union(*seen)) if seen else []
    if not every:
        return None

    at = {f: i for i, f in enumerate(every)}
    present = np.zeros((len(tracks), len(every)), dtype=np.int_)
    for i, fs in enumerate(seen):
        for f in fs:
            present[i, at[f]] = 1

    def crowd(labels: npt.NDArray[np.int_]) -> int:
        worst = max(int(np.max(present[labels == k].sum(axis=0), initial=0)) for k in (0, 1))
        return max(0, worst - MAX_CONCURRENT_PER_SIDE)

    return crowd


# How much closer to its own side's kit than to the other's a track has to be.
#
# A kit sitting between the two is a coin flip, and a coin flip reaches the board as a
# player in the wrong colour -- which a coach reads as the wrong team making the pass,
# because that is exactly what it draws. The same rule as a shirt number nobody could
# read (D5): the answer is that there is no answer, and the importer already drops a
# track whose side could not be told.
KIT_MARGIN = 0.8


def assign(
    tracks: list[Track],
    mean_x: dict[int, float],
    frames: dict[int, list[int]] | None = None,
) -> dict[int, TeamLabel]:
    """Track id -> team label.

    Clustered on `kit_mean` and not `color`: the tracker's rolling average is about the
    next frame's match, and which team a track is on is about the whole track.

    `mean_x` is each track's average position along the pitch, in metres, which is what
    decides which cluster is which. Tracks with no colour signature at all come back
    as "unknown" rather than being guessed into a side (D5's rule, applied to teams).

    Keepers are excluded from the clustering and then placed by the goal they stand in,
    which is both more accurate and the only way to label them `gkHome`/`gkAway` at all.

    `frames` is the frames each track holds a sample at, which is what tells a collapsed
    split from a real one. It is optional only so a caller with nothing but tracks still
    gets a labelling; the observations are a poorer answer once fragments are stitched.
    """
    usable = [t for t in tracks if t.kit_mean is not None and t.id in mean_x]
    if len(usable) < 2:
        return {t.id: "unknown" for t in tracks}

    points = np.array([t.kit_mean for t in usable], dtype=np.float64)
    first = split_kits(points)

    # A kit far from both teams, standing near a goal, is that goal's keeper. Both
    # halves matter: colour alone catches a player in odd light, and position alone
    # catches every defender on a goal line.
    oddness = _oddness(points, first)
    odd = oddness > OUTLIER_RATIO
    in_a_goal = np.array(
        [
            mean_x[t.id] <= KEEPER_ZONE_M or mean_x[t.id] >= PITCH_LENGTH - KEEPER_ZONE_M
            for t in usable
        ]
    )
    keeper = odd & in_a_goal

    # An odd kit standing where no keeper stands is an official. They are the one thing on
    # the pitch wearing neither team's colours and not tied to a goal, and left unnamed
    # they reach the board as a player a coach has to delete. Only the oddest few, because
    # colour alone also catches a player in strange light -- the same reason the keeper
    # test asks for position as well.
    #
    # They stay in the clustering they are named out of. Removing three tracks moves the
    # axis and the cut, and measured that way it cost four correct players to remove four
    # officials; naming them afterwards leaves every other track's side exactly as it was.
    outfield = [t for i, t in enumerate(usable) if not keeper[i]]
    if len(outfield) >= 2:
        labels = split_kits(
            np.array([t.kit_mean for t in outfield], dtype=np.float64),
            _crowding(outfield, frames),
        )
    else:
        outfield, labels = usable, first

    # Named against the FINAL split rather than the rough one that found the keepers: an
    # outlier test is only as good as the model it measures distance from, and the first
    # cut is a single unconstrained axis.
    settled = _oddness(np.array([t.kit_mean for t in outfield], dtype=np.float64), labels)
    at_a_goal = {t.id for i, t in enumerate(usable) if in_a_goal[i]}
    candidates = [
        i
        for i in np.argsort(-settled)
        if settled[i] > OUTLIER_RATIO and outfield[i].id not in at_a_goal
    ]
    referee = {outfield[i].id for i in candidates[:MAX_REFEREES]}

    sides = {}
    for k in (0, 1):
        xs = [mean_x[t.id] for t, lab in zip(outfield, labels, strict=True) if lab == k]
        sides[k] = float(np.mean(xs)) if xs else 0.0
    left = 0 if sides[0] <= sides[1] else 1

    # Which tracks the split is actually sure of. Distance to its own side's kit against
    # distance to the other's, in the same colour space the cut was made in.
    kits = np.array([t.kit_mean for t in outfield], dtype=np.float64)
    sure = np.ones(len(outfield), dtype=bool)
    for k in (0, 1):
        mine, theirs = labels == k, labels != k
        if not bool(mine.any()) or not bool(theirs.any()):
            continue
        # Leave-one-out: a track judged against a centre it helped compute drags that
        # centre towards itself, and the closer to the cut it sits the more it flatters
        # itself. Measured on SNGS-147, that bias alone was the difference between
        # declining nothing and declining a quarter of the clip.
        total, n = kits[mine].sum(axis=0), int(mine.sum())
        without = (total - kits[mine]) / (n - 1) if n > 1 else kits[mine]
        own = np.linalg.norm(kits[mine] - without, axis=1)
        other = np.linalg.norm(kits[mine] - kits[theirs].mean(axis=0), axis=1)
        sure[mine] = own <= KIT_MARGIN * other

    out: dict[int, TeamLabel] = {t.id: "unknown" for t in tracks}
    for i, (t, lab) in enumerate(zip(outfield, labels, strict=True)):
        if sure[i]:
            out[t.id] = "home" if lab == left else "away"
    for t in outfield:
        if t.id in referee:
            out[t.id] = "referee"
    for i, t in enumerate(usable):
        if keeper[i]:
            # The keeper of the goal they are standing in, whichever side that is.
            out[t.id] = "gkHome" if mean_x[t.id] <= PITCH_LENGTH / 2 else "gkAway"
    return out
