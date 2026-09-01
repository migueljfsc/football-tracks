"""Stage 3 - which side is each track on.

Two kits, clustered on the shirt signature the tracker already maintains. k-means with
k=2, written out rather than imported: the input is a dozen vectors, and sklearn is a
large dependency to add for twenty lines.

Which cluster is `home` is NOT decided by looking at the answer. It is decided by mean
pitch x - the side whose players average nearer x=0 defends the left goal - which is
SoccerNet's own left/right convention and is deterministic. Picking the labelling that
happens to score best would be fitting to the yardstick.

Goalkeepers and referees are a known gap. A keeper wears neither kit and a referee
wears a third, so both land in whichever cluster is least unlike them. They are
measured as errors rather than special-cased, because a special case that has not been
measured is a guess with extra steps.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .stage2_track import Track
from .tracks import TeamLabel

Vec = npt.NDArray[np.float64]
ITERATIONS = 25


def kmeans2(points: Vec, *, iterations: int = ITERATIONS) -> npt.NDArray[np.int_]:
    """Two clusters. Seeded with the two most distant points, so the result does not
    depend on a random draw and is reproducible across runs."""
    n = len(points)
    if n < 2:
        return np.zeros(n, dtype=np.int_)

    gram = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    a, b = np.unravel_index(int(np.argmax(gram)), gram.shape)
    centres = points[[a, b]].copy()

    labels = np.zeros(n, dtype=np.int_)
    for _ in range(iterations):
        d = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new = np.asarray(np.argmin(d, axis=1), dtype=np.int_)
        if np.array_equal(new, labels):
            break
        labels = new
        for k in (0, 1):
            if np.any(labels == k):
                centres[k] = points[labels == k].mean(axis=0)
    return labels


def assign(tracks: list[Track], mean_x: dict[int, float]) -> dict[int, TeamLabel]:
    """Track id -> team label.

    `mean_x` is each track's average position along the pitch, in metres, which is what
    decides which cluster is which. Tracks with no colour signature at all come back
    as "unknown" rather than being guessed into a side (D5's rule, applied to teams).
    """
    usable = [t for t in tracks if t.color is not None and t.id in mean_x]
    if len(usable) < 2:
        return {t.id: "unknown" for t in tracks}

    labels = kmeans2(np.array([t.color for t in usable], dtype=np.float64))

    sides = {}
    for k in (0, 1):
        xs = [mean_x[t.id] for t, lab in zip(usable, labels, strict=True) if lab == k]
        sides[k] = float(np.mean(xs)) if xs else 0.0
    left = 0 if sides[0] <= sides[1] else 1

    out: dict[int, TeamLabel] = {t.id: "unknown" for t in tracks}
    for t, lab in zip(usable, labels, strict=True):
        out[t.id] = "home" if lab == left else "away"
    return out
