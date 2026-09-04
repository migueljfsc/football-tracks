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

**Why joining them afterwards is safe where lengthening the wait was not.** The tracker
must decide at the moment of the gap, with nothing after it to go on. This runs when both
sides are known, so it can ask whether the two fragments are mutually each other's best
partner and refuse when they are not. 39% of the breaks have a gap of one frame or less --
the player never disappears, the tracker simply renumbers them -- and 70% are inside 12
frames.

The failure this must not cause is a track that teleports between two players, which is
worse than two honest halves: it puts one player's run on another's shirt. Hence the speed
gate, the mutual-best rule, and no chaining beyond what each link earns on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .stage2_track import MAX_SPEED
from .tracks import Sample

# How long a gap may be and still be bridged. 0.5 s reaches 70% of the identity changes
# measured on SNGS-147; past that the position prior is too weak to tell two players in
# the same shirt apart, and the speed gate alone would admit most of the pitch.
MAX_GAP_S = 0.5

# Slack on the speed gate, in metres, for the camera model's own error. Position error is
# 0.5-1.3 m per sample depending on the registration, so two samples of one stationary
# player can sit two metres apart with nobody having moved.
POSITION_SLACK_M = 2.5

# What disagreeing kit costs, as a fraction of the distance budget. Same role and the same
# value as the tracker's own: enough to break a tie between two candidates, not enough to
# override a plain speed violation.
COLOR_WEIGHT = 0.6


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


def _cost(a: Fragment, b: Fragment, fps: float) -> float | None:
    """What it costs to call `b` the continuation of `a`, or None if it cannot be."""
    gap = b.first.f - a.last.f
    if gap <= 0 or gap > MAX_GAP_S * fps:
        return None
    reach = MAX_SPEED * (gap / fps) + POSITION_SLACK_M
    moved = float(np.hypot(b.first.x - a.last.x, b.first.y - a.last.y))
    if moved > reach:
        return None  # nobody runs that fast; these are two different people
    return moved / reach + COLOR_WEIGHT * _color_distance(a.color, b.color)


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
