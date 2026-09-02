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


def split_kits(points: Vec) -> npt.NDArray[np.int_]:
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
    """
    n = len(points)
    if n < 2:
        return np.zeros(n, dtype=np.int_)

    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ vt[0]
    order = np.argsort(projected)

    best, cut = -1.0, n // 2
    for i in range(1, n):
        a, b = projected[order[:i]], projected[order[i:]]
        score = len(a) * len(b) * (float(a.mean()) - float(b.mean())) ** 2
        if score > best:
            best, cut = score, i

    labels = np.zeros(n, dtype=np.int_)
    labels[order[cut:]] = 1
    return labels


def _outliers(points: Vec, labels: npt.NDArray[np.int_], ratio: float) -> npt.NDArray[np.bool_]:
    """Which kits sit too far from their own group to belong to either team."""
    centres = np.array(
        [
            points[labels == k].mean(axis=0) if np.any(labels == k) else points.mean(axis=0)
            for k in (0, 1)
        ]
    )
    d = np.linalg.norm(points - centres[labels], axis=1)
    median = float(np.median(d))
    if median <= 0:
        return np.zeros(len(points), dtype=np.bool_)
    return np.asarray(d > ratio * median, dtype=np.bool_)


def assign(tracks: list[Track], mean_x: dict[int, float]) -> dict[int, TeamLabel]:
    """Track id -> team label.

    `mean_x` is each track's average position along the pitch, in metres, which is what
    decides which cluster is which. Tracks with no colour signature at all come back
    as "unknown" rather than being guessed into a side (D5's rule, applied to teams).

    Keepers are excluded from the clustering and then placed by the goal they stand in,
    which is both more accurate and the only way to label them `gkHome`/`gkAway` at all.
    """
    usable = [t for t in tracks if t.color is not None and t.id in mean_x]
    if len(usable) < 2:
        return {t.id: "unknown" for t in tracks}

    points = np.array([t.color for t in usable], dtype=np.float64)
    first = split_kits(points)

    # A kit far from both teams, standing near a goal, is that goal's keeper. Both
    # halves matter: colour alone catches a player in odd light, and position alone
    # catches every defender on a goal line.
    odd = _outliers(points, first, OUTLIER_RATIO)
    keeper = np.array(
        [
            bool(odd[i])
            and (mean_x[t.id] <= KEEPER_ZONE_M or mean_x[t.id] >= PITCH_LENGTH - KEEPER_ZONE_M)
            for i, t in enumerate(usable)
        ]
    )

    outfield = [t for i, t in enumerate(usable) if not keeper[i]]
    if len(outfield) >= 2:
        labels = split_kits(np.array([t.color for t in outfield], dtype=np.float64))
    else:
        outfield, labels = usable, first

    sides = {}
    for k in (0, 1):
        xs = [mean_x[t.id] for t, lab in zip(outfield, labels, strict=True) if lab == k]
        sides[k] = float(np.mean(xs)) if xs else 0.0
    left = 0 if sides[0] <= sides[1] else 1

    out: dict[int, TeamLabel] = {t.id: "unknown" for t in tracks}
    for t, lab in zip(outfield, labels, strict=True):
        out[t.id] = "home" if lab == left else "away"
    for i, t in enumerate(usable):
        if keeper[i]:
            # The keeper of the goal they are standing in, whichever side that is.
            out[t.id] = "gkHome" if mean_x[t.id] <= PITCH_LENGTH / 2 else "gkAway"
    return out
