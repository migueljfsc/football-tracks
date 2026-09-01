"""Stage 1 - image pixels to pitch metres.

A homography is fitted per frame, because a broadcast camera pans and zooms and one
matrix for the clip is one matrix for none of it.

The correspondences come from LINE INTERSECTIONS rather than a hand-built keypoint
table. Every named pitch line has an exact analytic definition in metres, and a
homography maps straight lines to straight lines - so fitting a line to each annotated
polyline and crossing two of them gives the image projection of a pitch point whose
coordinates are known by construction. That works even where the two markings do not
physically cross, since both are infinite lines, and it means adding a new named line
adds correspondences with every other line for free.

The lines are supplied by the caller. Ground truth supplies them today; a keypoint
model supplies them later, and nothing else in this module changes.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from .config import PITCH_LENGTH, PITCH_WIDTH

Line = tuple[float, float, float]  # ax + by + c = 0
Point = tuple[float, float]

_HALF_W = PITCH_WIDTH / 2
PENALTY_HALF = 20.16
GOAL_AREA_HALF = 9.16

# Every straight pitch line SoccerNet names, in Pitchboard's metres. "top" is the y = 0
# touchline and "bottom" is y = 68: SoccerNet's naming follows a diagram whose top is
# the low-y side, which is the same way round as ours.
#
# Circles and goal frames are deliberately absent. A circle is not a line, and a goal
# post is not on the ground plane at all - projecting it with a ground homography puts
# it metres from where it is.
PITCH_LINES: dict[str, Line] = {
    "Side line left": (1.0, 0.0, 0.0),
    "Side line right": (1.0, 0.0, -PITCH_LENGTH),
    "Side line top": (0.0, 1.0, 0.0),
    "Side line bottom": (0.0, 1.0, -PITCH_WIDTH),
    "Middle line": (1.0, 0.0, -PITCH_LENGTH / 2),
    "Big rect. left main": (1.0, 0.0, -16.5),
    "Big rect. left top": (0.0, 1.0, -(_HALF_W - PENALTY_HALF)),
    "Big rect. left bottom": (0.0, 1.0, -(_HALF_W + PENALTY_HALF)),
    "Big rect. right main": (1.0, 0.0, -(PITCH_LENGTH - 16.5)),
    "Big rect. right top": (0.0, 1.0, -(_HALF_W - PENALTY_HALF)),
    "Big rect. right bottom": (0.0, 1.0, -(_HALF_W + PENALTY_HALF)),
    "Small rect. left main": (1.0, 0.0, -5.5),
    "Small rect. left top": (0.0, 1.0, -(_HALF_W - GOAL_AREA_HALF)),
    "Small rect. left bottom": (0.0, 1.0, -(_HALF_W + GOAL_AREA_HALF)),
    "Small rect. right main": (1.0, 0.0, -(PITCH_LENGTH - 5.5)),
    "Small rect. right top": (0.0, 1.0, -(_HALF_W - GOAL_AREA_HALF)),
    "Small rect. right bottom": (0.0, 1.0, -(_HALF_W + GOAL_AREA_HALF)),
}

# A frame must show lines running BOTH ways to pin a homography down. Every marking
# here is axis-aligned in pitch space, so "both ways" means at least two of the x = k
# lines and two of the y = k lines: four parallel lines constrain four of the eight
# degrees of freedom and leave the rest floating.
MIN_LINES_PER_AXIS = 2

# And it must show MORE than the bare minimum. Four lines is eight constraints for
# eight degrees of freedom: exactly determined, so the fit passes through its own data
# with no residual whatever the noise, and there is nothing left over to notice that it
# is wrong with. Measured on SNGS-147, four-line frames land 21.7 m out at the median
# while every over-determined configuration is inside 3 m. Refusing them is the same
# call as D13 - a homography that cannot be checked is not a cheaper homography, it is
# a wrong answer nobody can see.
MIN_LINES = 5
MIN_POINTS = 8


def _normalise(
    points: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Hartley normalisation: zero mean, mean distance sqrt(2) from the origin.

    Without it the DLT matrix mixes pixel counts in the thousands with ones, and the
    smallest singular vector is decided by floating-point noise rather than geometry.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    scale = np.sqrt(2) / max(1e-12, float(np.mean(np.linalg.norm(centred, axis=1))))
    t = np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]]
    )
    return centred * scale, t


def homography(
    lines: dict[str, list[dict[str, float]]], width: int, height: int
) -> npt.NDArray[np.float64] | None:
    """Image pixels -> pitch metres, or None when the frame does not show enough.

    Fitted from POINT-ON-LINE constraints rather than from line intersections. Every
    annotated point is known to lie on a named pitch line, which gives one linear
    equation in H:

        l . (H p) = 0

    Stacking them and taking the smallest right singular vector is an ordinary DLT.

    Intersections were the first approach and they fail on exactly the footage that
    matters. Under an oblique camera two pitch lines that meet at a right angle project
    to nearly parallel image lines, so their crossing point flies off and a pixel of
    error becomes tens of metres. Worse, the few intersections that survive tend to
    share a line and sit nearly collinear, which is degenerate - it fits its own points
    perfectly and puts everything else on another pitch. This formulation instead uses
    every point of every visible marking, and a line seen edge-on simply contributes
    what it can.
    """
    pts: list[tuple[float, float]] = []
    used: list[Line] = []
    axes: dict[bool, set[str]] = {True: set(), False: set()}

    for name, poly in lines.items():
        pitch_line = PITCH_LINES.get(name)
        if pitch_line is None or len(poly) < 2:
            continue
        axes[abs(pitch_line[0]) > abs(pitch_line[1])].add(name)
        for q in poly:
            pts.append((q["x"] * width, q["y"] * height))
            used.append(pitch_line)

    if len(pts) < MIN_POINTS:
        return None
    if min(len(axes[True]), len(axes[False])) < MIN_LINES_PER_AXIS:
        return None
    if len(axes[True]) + len(axes[False]) < MIN_LINES:
        return None

    image = np.array(pts, dtype=np.float64)
    normed, t_img = _normalise(image)

    # The pitch is scaled to roughly unit size for the same conditioning reason. A line
    # in scaled coordinates is (a/s, b/s, c), and the fitted matrix is unscaled after.
    s = float(PITCH_LENGTH)
    rows = []
    for (u, v), (a, b, c) in zip(normed, used, strict=True):
        # Scaling the pitch DOWN by s scales a line's normal UP: x = s*x' turns
        # a*x + b*y + c into (a*s)*x' + (b*s)*y' + c. Getting this backwards still
        # solves cleanly and still reprojects onto the markings - it just lands on a
        # pitch of the wrong size, which is why it showed up as every box off-pitch
        # rather than as a bad fit.
        a_, b_, c_ = a * s, b * s, c
        norm = math.sqrt(a_ * a_ + b_ * b_ + c_ * c_) or 1.0
        a_, b_, c_ = a_ / norm, b_ / norm, c_ / norm
        rows.append([a_ * u, a_ * v, a_, b_ * u, b_ * v, b_, c_ * u, c_ * v, c_])

    _, _, vt = np.linalg.svd(np.array(rows, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    if not np.all(np.isfinite(h)) or abs(np.linalg.det(h)) < 1e-12:
        return None

    # Undo both normalisations: image points were pre-multiplied, pitch was scaled down.
    h = np.diag([s, s, 1.0]) @ h @ t_img
    if abs(h[2, 2]) < 1e-12:
        return None
    return np.asarray(h / h[2, 2], dtype=np.float64)


def apply(h: npt.NDArray[np.float64], points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Push points through a homography. Shape (n, 2) in, (n, 2) out."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, h)
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def to_pitch(h: npt.NDArray[np.float64], x: float, y: float) -> Point:
    p = apply(h, np.array([[x, y]]))[0]
    return (float(p[0]), float(p[1]))


def lines_of(annotation: dict[str, Any]) -> dict[str, list[dict[str, float]]]:
    """The `lines` payload of a SoccerNet pitch annotation."""
    payload: dict[str, list[dict[str, float]]] = annotation.get("lines", {})
    return payload
