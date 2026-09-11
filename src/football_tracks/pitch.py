"""Top-down pitch drawing, in metres.

Only ever used to LOOK at a tracks.json - it is the picture invariant 3 asks every
stage for. Nothing here feeds a measurement, so it is allowed to be approximate about
things a viewer cannot see, and it must never become a second opinion about geometry.

Markings follow IFAB on the 105 x 68 pitch `config` declares.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

from .config import DEFAULT_PITCH, Pitch

Poly = npt.NDArray[np.float64]
Point2 = tuple[float, float]

GRASS = (58, 122, 48)
LINE = (235, 235, 235)

# IFAB, metres.
CENTRE_R = 9.15
PENALTY_DEPTH = 16.5
PENALTY_HALF_WIDTH = 20.16
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_HALF_WIDTH = 9.16
PENALTY_SPOT = 11.0
CORNER_R = 1.0


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 40) -> Poly:
    a = np.linspace(np.radians(a0), np.radians(a1), n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def model(pitch: Pitch = DEFAULT_PITCH) -> list[Poly]:
    """The markings as polylines in METRES.

    The one description of pitch geometry in this repo. `draw` renders it top-down and
    `overlay` reprojects it onto a video frame; neither restates it.
    """
    mid = pitch.middle
    polys: list[Poly] = [
        np.array(
            [[0, 0], [pitch.length, 0], [pitch.length, pitch.width], [0, pitch.width], [0, 0]],
            dtype=np.float64,
        ),
        np.array([[pitch.halfway, 0], [pitch.halfway, pitch.width]], dtype=np.float64),
        _arc(pitch.halfway, mid, CENTRE_R, 0, 360),
    ]

    for near, sign in ((0.0, 1.0), (pitch.length, -1.0)):
        for depth, half in (
            (PENALTY_DEPTH, PENALTY_HALF_WIDTH),
            (GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH),
        ):
            far = near + sign * depth
            polys.append(
                np.array(
                    [[near, mid - half], [far, mid - half], [far, mid + half], [near, mid + half]],
                    dtype=np.float64,
                )
            )

        # The penalty arc is the part of a 9.15 m circle centred on the SPOT that falls
        # outside the box - not an arc on the box edge.
        spot_x = near + sign * PENALTY_SPOT
        half_deg = np.degrees(np.arccos((PENALTY_DEPTH - PENALTY_SPOT) / CENTRE_R))
        start = 0.0 if sign > 0 else 180.0
        polys.append(_arc(spot_x, mid, CENTRE_R, start - half_deg, start + half_deg))

    for cx, cy, a0 in (
        (0.0, 0.0, 0.0),
        (pitch.length, 0.0, 90.0),
        (pitch.length, pitch.width, 180.0),
        (0.0, pitch.width, 270.0),
    ):
        polys.append(_arc(cx, cy, CORNER_R, a0, a0 + 90.0))

    return polys


def spots(pitch: Pitch = DEFAULT_PITCH) -> list[Point2]:
    """The centre spot and both penalty spots, in metres."""
    mid = pitch.middle
    return [
        (pitch.halfway, mid),
        (PENALTY_SPOT, mid),
        (pitch.length - PENALTY_SPOT, mid),
    ]


def to_px(x: float, y: float, scale: float, margin: float) -> tuple[int, int]:
    """Pitch metres -> image pixels. The single conversion this module has."""
    return (round((x + margin) * scale), round((y + margin) * scale))


def canvas_size(scale: float, margin: float, pitch: Pitch = DEFAULT_PITCH) -> tuple[int, int]:
    return (
        round((pitch.length + 2 * margin) * scale),
        round((pitch.width + 2 * margin) * scale),
    )


def draw(scale: float = 10.0, margin: float = 3.0, pitch: Pitch = DEFAULT_PITCH) -> MatLike:
    """A fresh pitch, ready to have dots put on it. Renders `model()` and nothing else."""
    w, h = canvas_size(scale, margin, pitch)
    img = np.full((h, w, 3), GRASS, dtype=np.uint8)
    t = max(1, round(scale / 8))

    for poly in model(pitch):
        pts = np.array([to_px(x, y, scale, margin) for x, y in poly], dtype=np.int32)
        cv2.polylines(img, [pts], False, LINE, t, cv2.LINE_AA)

    for sx, sy in spots(pitch):
        cv2.circle(
            img, to_px(sx, sy, scale, margin), max(2, round(0.3 * scale)), LINE, -1, cv2.LINE_AA
        )

    return img
