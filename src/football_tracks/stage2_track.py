"""Stage 2b - string detections into tracks with stable ids.

**Association happens in STABILISED IMAGE SPACE, and does not touch the homography.**

Two forces pull in opposite directions here. A broadcast camera pans, and a pan moves
every player across the frame at once by more than they move themselves, so raw image
pixels are the wrong space. But pitch metres are worse: they inherit every wobble in
the homography, and a carried one wobbles enough to throw all the players at the same
instant, which breaks every track together. That was measured - purity 62.9% tracking
in metres against a drifting homography, 82.9% with carrying switched off (D19).

So the camera's motion is removed WITHOUT going through the homography, using the
frame-to-frame transform from `stage1_propagate.motions`. That transform is measured
per pair and never accumulated, so it does not drift. The gate is still a physical
claim - a footballer covers at most MAX_SPEED metres in a second - converted to pixels
through the only local scale that needs no camera model at all: a player is about
PLAYER_HEIGHT_M tall, and their box says how many pixels that is right there.

The result is that stage 2 no longer depends on stage 1 at all, which is the real
lesson of D19.

Written here rather than taken from a library for two reasons. supervision's ByteTrack
is deprecated and disappears in 0.31, and more usefully, this regime has a signal a
general tracker does not use: a football team wears one colour. Two players crossing
are the classic id switch, and if they are on opposite sides their shirts say so.

Association is greedy by ascending cost rather than optimal assignment. With a dozen
observations and a tight gate the two agree almost always, and greedy can be read - a
switch traces to a pair rather than to a solver.

THE FAILURE THIS STAGE HAS is the id switch, and it is invisible in a detection count:
every player is still found. `score.identity_purity` is what sees it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .detect import Detection

# Metres per second. Usain Bolt peaks near 12; a footballer with the ball does not.
# Generous on purpose - the gate is there to reject a jump across the pitch, not to
# adjudicate sprint speed.
MAX_SPEED = 11.0

# What a detection box's height is taken to mean, in metres. The local pixels-per-metre
# scale comes from this and nothing else, which is what keeps the gate free of the
# homography: near players have tall boxes and move more pixels, and the ratio holds.
PLAYER_HEIGHT_M = 1.8

# Floor on the gate, in units of box height, so a one-frame step is not gated to
# nothing at high frame rates and detector jitter does not tear a stationary player's
# track in half.
MIN_GATE_BOXES = 0.35

# How much disagreeing kit colour costs, relative to a full gate of distance. High
# enough to break a tie between two players crossing, low enough that a shadow or a
# turned back does not tear a track in half.
COLOR_WEIGHT = 0.6

# Seconds a track survives unmatched before it is closed. Long enough to ride out an
# occlusion, short enough that its id is not handed to somebody else later.
MAX_AGE_S = 0.8

MIN_TRACK_LENGTH = 5


def warp(x: float, y: float, motion: np.ndarray | None) -> tuple[float, float]:
    """Where a point on the grass lands in the next frame if nobody moves but the camera."""
    if motion is None:
        return (x, y)
    out = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), motion)
    return (float(out[0, 0, 0]), float(out[0, 0, 1]))


@dataclass(slots=True)
class Observation:
    """One detection, in image pixels, at the point the player meets the grass."""

    f: int
    x: float
    y: float
    det: Detection

    @classmethod
    def of(cls, d: Detection) -> Observation:
        fx, fy = d.foot
        return cls(f=d.f, x=fx, y=fy, det=d)


@dataclass(slots=True)
class Track:
    id: int
    observations: list[Observation] = field(default_factory=list)
    color: np.ndarray | None = None
    velocity: tuple[float, float] = (0.0, 0.0)

    @property
    def last(self) -> Observation:
        return self.observations[-1]

    def predict(self, dt: float, motion: np.ndarray | None) -> tuple[float, float]:
        """Where they should be after `dt` seconds, in the NEXT frame's pixels.

        Two parts, and they are different things. `motion` accounts for the camera
        having moved, which happens to every pixel whether or not anybody ran. The
        velocity term is the player's own movement, held in stabilised units so it
        survives a pan.
        """
        px, py = warp(self.last.x, self.last.y, motion)
        return (px + self.velocity[0] * dt, py + self.velocity[1] * dt)


def kit(bgr: Any, d: Detection) -> np.ndarray | None:
    """A coarse hue/value signature of the shirt."""
    from .detect import torso

    crop = torso(bgr, d)
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 2], None, [12, 4], [0, 180, 0, 256])
    total = float(hist.sum())
    if total <= 0:
        return None
    return np.asarray(hist.flatten() / total, dtype=np.float64)


def color_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """0 when the kits agree, 1 when they could not disagree more. 0.5 when unknown -
    an absent signature must neither attract nor repel."""
    if a is None or b is None:
        return 0.5
    return float(np.clip(1.0 - np.minimum(a, b).sum(), 0.0, 1.0))


def _cost(
    track: Track,
    obs: Observation,
    color: np.ndarray | None,
    fps: float,
    motion: np.ndarray | None,
) -> float | None:
    dt = max(1, obs.f - track.last.f) / fps
    px_per_m = max(track.last.det.height, obs.det.height) / PLAYER_HEIGHT_M
    gate = max(MIN_GATE_BOXES * obs.det.height, MAX_SPEED * dt * px_per_m)
    px, py = track.predict(dt, motion)
    dist = math.dist((px, py), (obs.x, obs.y))
    if dist > gate:
        return None
    return dist / gate + COLOR_WEIGHT * color_distance(track.color, color)


def run(
    observations: dict[int, list[Observation]],
    frames: list[int],
    read_frame: Any,
    *,
    fps: float,
    motions: dict[int, np.ndarray] | None = None,
) -> list[Track]:
    """Associate frame by frame, in stabilised pixels.

    `read_frame(f)` returns the image or None. `motions[f]` maps image f-1 onto image
    f; without it the camera is assumed still, which is only true of a fixed camera.
    """
    live: list[Track] = []
    done: list[Track] = []
    next_id = 1
    max_age = max(1, round(MAX_AGE_S * fps))

    for f in frames:
        # Retire BEFORE associating. The other order lets a track that is already too
        # old match anyway, because the gate grows with the gap - so whether a stale
        # track is reachable depends on whether the caller's frame list happens to be
        # contiguous, which is not a property this should rest on.
        fresh: list[Track] = []
        for track in live:
            (fresh if f - track.last.f <= max_age else done).append(track)
        live = fresh

        obs = observations.get(f, [])
        img = read_frame(f) if obs else None
        colors = [kit(img, o.det) if img is not None else None for o in obs]

        motion = motions.get(f) if motions is not None else None

        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(live):
            for oi, o in enumerate(obs):
                c = _cost(track, o, colors[oi], fps, motion)
                if c is not None:
                    pairs.append((c, ti, oi))
        pairs.sort()

        used_t: set[int] = set()
        used_o: set[int] = set()
        for _, ti, oi in pairs:
            if ti in used_t or oi in used_o:
                continue
            used_t.add(ti)
            used_o.add(oi)
            track, o = live[ti], obs[oi]
            dt = max(1, o.f - track.last.f) / fps
            # Velocity is what is left after the camera is accounted for, so it stays
            # meaningful across a pan instead of encoding the pan itself.
            sx, sy = (
                track.predict(dt, motion) if motion is not None else (track.last.x, track.last.y)
            )
            base = (
                (sx - track.velocity[0] * dt, sy - track.velocity[1] * dt)
                if motion is not None
                else (track.last.x, track.last.y)
            )
            track.velocity = ((o.x - base[0]) / dt, (o.y - base[1]) / dt)
            track.observations.append(o)
            seen = colors[oi]
            if seen is not None:
                # Rolling average: a single frame of shadow should not redefine a kit.
                prior = track.color
                track.color = seen if prior is None else 0.8 * prior + 0.2 * seen

        for oi, o in enumerate(obs):
            if oi not in used_o:
                live.append(Track(id=next_id, observations=[o], color=colors[oi]))
                next_id += 1

    done.extend(live)
    return [t for t in done if len(t.observations) >= MIN_TRACK_LENGTH]
