"""Top-down pitch drawing, in metres.

Only ever used to LOOK at a tracks.json - it is the picture invariant 3 asks every
stage for. Nothing here feeds a measurement, so it is allowed to be approximate about
things a viewer cannot see, and it must never become a second opinion about geometry.

Markings follow IFAB on the 105 x 68 pitch `config` declares.
"""

from __future__ import annotations

import cv2
import numpy as np
from cv2.typing import MatLike

from .config import PITCH_LENGTH, PITCH_WIDTH

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


def to_px(x: float, y: float, scale: float, margin: float) -> tuple[int, int]:
    """Pitch metres -> image pixels. The single conversion this module has."""
    return (round((x + margin) * scale), round((y + margin) * scale))


def canvas_size(scale: float, margin: float) -> tuple[int, int]:
    return (
        round((PITCH_LENGTH + 2 * margin) * scale),
        round((PITCH_WIDTH + 2 * margin) * scale),
    )


def draw(scale: float = 10.0, margin: float = 3.0) -> MatLike:
    """A fresh pitch, ready to have dots put on it."""
    w, h = canvas_size(scale, margin)
    img = np.full((h, w, 3), GRASS, dtype=np.uint8)
    t = max(1, round(scale / 8))

    def line(x1: float, y1: float, x2: float, y2: float) -> None:
        cv2.line(
            img, to_px(x1, y1, scale, margin), to_px(x2, y2, scale, margin), LINE, t, cv2.LINE_AA
        )

    def circle(cx: float, cy: float, r: float, thickness: int = -1) -> None:
        cv2.circle(
            img, to_px(cx, cy, scale, margin), round(r * scale), LINE, thickness, cv2.LINE_AA
        )

    def box(x1: float, y1: float, x2: float, y2: float) -> None:
        cv2.rectangle(
            img, to_px(x1, y1, scale, margin), to_px(x2, y2, scale, margin), LINE, t, cv2.LINE_AA
        )

    box(0, 0, PITCH_LENGTH, PITCH_WIDTH)
    line(PITCH_LENGTH / 2, 0, PITCH_LENGTH / 2, PITCH_WIDTH)
    circle(PITCH_LENGTH / 2, PITCH_WIDTH / 2, CENTRE_R, t)
    circle(PITCH_LENGTH / 2, PITCH_WIDTH / 2, 0.3)

    mid = PITCH_WIDTH / 2
    for near, sign in ((0.0, 1.0), (PITCH_LENGTH, -1.0)):
        box(
            near,
            mid - PENALTY_HALF_WIDTH,
            near + sign * PENALTY_DEPTH,
            mid + PENALTY_HALF_WIDTH,
        )
        box(
            near,
            mid - GOAL_AREA_HALF_WIDTH,
            near + sign * GOAL_AREA_DEPTH,
            mid + GOAL_AREA_HALF_WIDTH,
        )
        spot_x = near + sign * PENALTY_SPOT
        circle(spot_x, mid, 0.3)

        # The penalty arc is the part of a 9.15 m circle centred on the SPOT that falls
        # outside the box - not an arc on the box edge.
        half = np.degrees(np.arccos((PENALTY_DEPTH - PENALTY_SPOT) / CENTRE_R))
        start = 0.0 if sign > 0 else 180.0
        cv2.ellipse(
            img,
            to_px(spot_x, mid, scale, margin),
            (round(CENTRE_R * scale), round(CENTRE_R * scale)),
            0.0,
            start - half,
            start + half,
            LINE,
            t,
            cv2.LINE_AA,
        )

    for cx, cy, a0 in (
        (0.0, 0.0, 0.0),
        (PITCH_LENGTH, 0.0, 90.0),
        (PITCH_LENGTH, PITCH_WIDTH, 180.0),
        (0.0, PITCH_WIDTH, 270.0),
    ):
        cv2.ellipse(
            img,
            to_px(cx, cy, scale, margin),
            (round(CORNER_R * scale), round(CORNER_R * scale)),
            0.0,
            a0,
            a0 + 90.0,
            LINE,
            t,
            cv2.LINE_AA,
        )

    return img
