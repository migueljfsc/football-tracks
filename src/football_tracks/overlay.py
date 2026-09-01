"""Stage 1's picture - the pitch model reprojected onto a video frame.

A homography cannot be unit-tested. You invert it, push the markings back through it,
and look at whether the lines land on the lines. Everything numerical about stage 1 can
be right in aggregate while the frame is quietly wrong, and this is what shows it.

Kept apart from `render.py` because they work in different spaces: that one draws a
pitch top-down, this one draws a pitch onto a photograph.
"""

from __future__ import annotations

from itertools import pairwise

import cv2
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

from . import calibration, pitch

FOUND = (80, 240, 90)
MISSING = (60, 60, 240)

# A point whose homogeneous w approaches zero is on the horizon, and dividing by it
# throws the projection to infinity. Segments reaching past this are dropped rather
# than drawn, because cv2 clips a line by walking it and a coordinate of 1e9 hangs.
MAX_SPAN = 4.0


def _visible(p: npt.NDArray[np.float64], w: int, h: int) -> bool:
    return bool(
        np.all(np.isfinite(p))
        and -MAX_SPAN * w <= p[0] <= (1 + MAX_SPAN) * w
        and -MAX_SPAN * h <= p[1] <= (1 + MAX_SPAN) * h
    )


def draw(
    frame: MatLike,
    h_to_pitch: npt.NDArray[np.float64] | None,
    *,
    thickness: int = 2,
) -> MatLike:
    """The markings, drawn onto a copy of `frame`.

    `h_to_pitch` maps image pixels to metres, so it is inverted here - the drawing goes
    the other way. A frame with no homography is returned marked rather than blank, so
    a failure is visible as a failure instead of as an ordinary frame.
    """
    img = frame.copy()
    height, width = img.shape[:2]

    if h_to_pitch is None:
        cv2.putText(
            img, "no homography", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, MISSING, 3, cv2.LINE_AA
        )
        return img

    to_image = np.linalg.inv(h_to_pitch)

    for poly in pitch.model():
        pts = calibration.apply(to_image, poly)
        for a, b in pairwise(pts):
            if _visible(a, width, height) and _visible(b, width, height):
                cv2.line(
                    img,
                    (round(a[0]), round(a[1])),
                    (round(b[0]), round(b[1])),
                    FOUND,
                    thickness,
                    cv2.LINE_AA,
                )

    for spot in calibration.apply(to_image, np.array(pitch.spots(), dtype=np.float64)):
        if _visible(spot, width, height):
            cv2.circle(img, (round(spot[0]), round(spot[1])), thickness + 2, FOUND, -1, cv2.LINE_AA)

    return img


def annotate(
    img: MatLike, text: str, *, ok: bool = True, at: tuple[int, int] = (16, 40)
) -> MatLike:
    cv2.putText(
        img, text, at, cv2.FONT_HERSHEY_SIMPLEX, 1.0, FOUND if ok else MISSING, 2, cv2.LINE_AA
    )
    return img
