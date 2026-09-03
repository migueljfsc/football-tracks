"""Snap a homography back onto the markings it should be sitting on.

Propagation is open-loop: `stage1_propagate` composes a flow estimate per frame, so every
error made is kept and multiplied by every error after it. On broadcast footage that costs
1.74 m in fifty frames and 44 m in four hundred and fifty (D33, D34), which is the single
largest source of error in the pipeline -- large enough that nothing else is worth tuning
underneath it.

The pitch is painted with the answer. This closes the loop: project the model's lines into
the frame, find the white pixels they should be lying on, and refit. The carried
homography is then only ever a STARTING GUESS for the current frame rather than the
accumulated sum of every frame before it, so error stops compounding and becomes whatever
one fit is worth.

Every correspondence is a point-on-line and not a point: a white pixel says "somewhere
along this marking", never "this exact spot". `calibration.fit` already takes that kind of
evidence, because it is what a human clicking along a line gives it too.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from . import calibration
from .calibration import PITCH_LINES
from .config import PITCH_LENGTH, PITCH_WIDTH

H = npt.NDArray[np.float64]

_HALF_W = PITCH_WIDTH / 2
PENALTY_HALF = 20.16
GOAL_AREA_HALF = 9.16

# Where each marking starts and stops. The line equation alone is infinite, and sampling an
# infinite line looks for paint across the whole pitch -- the six-yard box would go hunting
# down the touchline and snap onto whatever it found.
LINE_EXTENTS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "Side line left": ((0.0, 0.0), (0.0, PITCH_WIDTH)),
    "Side line right": ((PITCH_LENGTH, 0.0), (PITCH_LENGTH, PITCH_WIDTH)),
    "Side line top": ((0.0, 0.0), (PITCH_LENGTH, 0.0)),
    "Side line bottom": ((0.0, PITCH_WIDTH), (PITCH_LENGTH, PITCH_WIDTH)),
    "Middle line": ((PITCH_LENGTH / 2, 0.0), (PITCH_LENGTH / 2, PITCH_WIDTH)),
    "Big rect. left main": ((16.5, _HALF_W - PENALTY_HALF), (16.5, _HALF_W + PENALTY_HALF)),
    "Big rect. left top": ((0.0, _HALF_W - PENALTY_HALF), (16.5, _HALF_W - PENALTY_HALF)),
    "Big rect. left bottom": ((0.0, _HALF_W + PENALTY_HALF), (16.5, _HALF_W + PENALTY_HALF)),
    "Big rect. right main": (
        (PITCH_LENGTH - 16.5, _HALF_W - PENALTY_HALF),
        (PITCH_LENGTH - 16.5, _HALF_W + PENALTY_HALF),
    ),
    "Big rect. right top": (
        (PITCH_LENGTH - 16.5, _HALF_W - PENALTY_HALF),
        (PITCH_LENGTH, _HALF_W - PENALTY_HALF),
    ),
    "Big rect. right bottom": (
        (PITCH_LENGTH - 16.5, _HALF_W + PENALTY_HALF),
        (PITCH_LENGTH, _HALF_W + PENALTY_HALF),
    ),
    "Small rect. left main": ((5.5, _HALF_W - GOAL_AREA_HALF), (5.5, _HALF_W + GOAL_AREA_HALF)),
    "Small rect. left top": ((0.0, _HALF_W - GOAL_AREA_HALF), (5.5, _HALF_W - GOAL_AREA_HALF)),
    "Small rect. left bottom": ((0.0, _HALF_W + GOAL_AREA_HALF), (5.5, _HALF_W + GOAL_AREA_HALF)),
    "Small rect. right main": (
        (PITCH_LENGTH - 5.5, _HALF_W - GOAL_AREA_HALF),
        (PITCH_LENGTH - 5.5, _HALF_W + GOAL_AREA_HALF),
    ),
    "Small rect. right top": (
        (PITCH_LENGTH - 5.5, _HALF_W - GOAL_AREA_HALF),
        (PITCH_LENGTH, _HALF_W - GOAL_AREA_HALF),
    ),
    "Small rect. right bottom": (
        (PITCH_LENGTH - 5.5, _HALF_W + GOAL_AREA_HALF),
        (PITCH_LENGTH, _HALF_W + GOAL_AREA_HALF),
    ),
}

SAMPLES_PER_LINE = 48
SEARCH_PX = 22
# Widest stripe of paint still believable as a marking, in pixels.
MAX_LINE_PX = 14
MIN_CORRESPONDENCES = 24
PASSES = 3
# A refit that moves the model further than a seed's own error is not a refinement, it is
# a different answer, and it means the search locked onto the wrong paint.
MAX_SHIFT_M = 6.0


def line_pixels(img: Any) -> npt.NDArray[np.uint8]:
    """White paint: bright, unsaturated, and brighter than the grass beside it.

    The top-hat is what makes it paint rather than anything pale. Shirts, socks, the ball
    and the hoardings are all bright and unsaturated too; what a line has and they do not
    is being narrow -- brighter than the turf a few pixels to either side.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    tophat = cv2.morphologyEx(hsv[:, :, 2], cv2.MORPH_TOPHAT, kernel).astype(np.int16)
    mask: npt.NDArray[np.uint8] = ((sat < 80) & (val > 110) & (tophat > 18)).astype(np.uint8)
    return mask


def on_pitch_mask(h: H, width: int, height: int, stride: int = 8) -> npt.NDArray[np.uint8]:
    """Which pixels the current model says are grass.

    Paint is not the only thing in a football frame that is bright, narrow and colourless:
    so are the goal net, the hoardings and the crowd behind them. They matter because the
    far touchline projects straight along the advertising boards, so a search that is
    allowed to look there will find their lettering and pull the model into the stands.
    Tested on a coarse grid and grown back -- the answer only changes over metres.
    """
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    pitch = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), h).reshape(-1, 2)
    ok = (
        np.isfinite(pitch).all(axis=1)
        & (pitch[:, 0] > -2.0)
        & (pitch[:, 0] < PITCH_LENGTH + 2.0)
        & (pitch[:, 1] > -2.0)
        & (pitch[:, 1] < PITCH_WIDTH + 2.0)
    )
    small = ok.reshape(xs.shape).astype(np.uint8)
    grown = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.asarray(grown, dtype=np.uint8)


def _sample_line(
    inv: H, a: tuple[float, float], b: tuple[float, float], width: int, height: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Points along a marking in image pixels, with the perpendicular to search along."""
    t = np.linspace(0.0, 1.0, SAMPLES_PER_LINE).reshape(-1, 1)
    pitch = np.array(a) + t * (np.array(b) - np.array(a))
    img = cv2.perspectiveTransform(pitch.reshape(-1, 1, 2).astype(np.float64), inv).reshape(-1, 2)
    # The tangent has to be measured in the IMAGE: perspective turns an evenly spaced line
    # into an unevenly spaced one, so the pitch-space direction points the wrong way.
    tangent = np.gradient(img, axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        tangent = tangent / norm
    perp = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    inside = (
        np.isfinite(img).all(axis=1)
        & np.isfinite(perp).all(axis=1)
        & (img[:, 0] >= 0)
        & (img[:, 0] < width)
        & (img[:, 1] >= 0)
        & (img[:, 1] < height)
    )
    return (
        img[inside].astype(np.float64),
        perp[inside].astype(np.float64),
    )


def _snap(
    px: float,
    py: float,
    dx: float,
    dy: float,
    mask: npt.NDArray[np.uint8],
    offsets: npt.NDArray[np.float64],
    width: int,
    height: int,
) -> tuple[float, float] | None:
    """The centre of the nearest stripe of paint along a perpendicular, if there is one."""
    us = px + offsets * dx
    vs = py + offsets * dy
    ok = (us >= 0) & (us < width) & (vs >= 0) & (vs < height)
    if not ok.any():
        return None
    hit = mask[vs[ok].astype(np.int32), us[ok].astype(np.int32)] > 0
    if not hit.any():
        return None
    offs = offsets[ok]
    # The CENTRE of the stripe, not the nearest painted pixel. A line is several pixels
    # wide, so "nearest pixel" is always its near edge -- and which edge that is depends
    # on which way the model is already wrong. It under-corrects by half a line width
    # every pass, and the fit converges to being wrong rather than to being right.
    edges = np.flatnonzero(np.diff(hit.astype(np.int8)))
    starts = np.r_[0, edges + 1]
    stops = np.r_[edges + 1, len(hit)]
    runs = [(a, b) for a, b in zip(starts, stops, strict=True) if hit[a]]
    if not runs:
        return None
    centres = [(offs[a] + offs[b - 1]) / 2.0 for a, b in runs]
    widths = [b - a for a, b in runs]
    best = int(np.argmin([abs(cn) for cn in centres]))
    # A stripe far wider than a line is not a line: it is a shirt, a sock, or the sun on
    # the turf. Taking its middle invents a marking that is not there.
    if widths[best] > MAX_LINE_PX:
        return None
    return float(px + centres[best] * dx), float(py + centres[best] * dy)


def correspondences(
    h: H, mask: npt.NDArray[np.uint8]
) -> list[tuple[tuple[float, float], tuple[float, float, float]]]:
    """Each model line paired with the paint nearest to where it thinks it is."""
    height, width = mask.shape[:2]
    inv = np.linalg.inv(h)
    offsets = np.arange(-SEARCH_PX, SEARCH_PX + 1, dtype=np.float64)
    found: list[tuple[tuple[float, float], tuple[float, float, float]]] = []

    for name, line in PITCH_LINES.items():
        pts, perp = _sample_line(inv, *LINE_EXTENTS[name], width, height)
        for (px, py), (dx, dy) in zip(pts, perp, strict=True):
            got = _snap(px, py, dx, dy, mask, offsets, width, height)
            if got is not None:
                found.append((got, line))
    return found


def _geometric(
    h: H, pairs: list[tuple[tuple[float, float], tuple[float, float, float]]]
) -> H | None:
    """The fit that minimises METRES, starting from the homography we already have.

    `calibration.fit` solves a DLT, which minimises an algebraic residual -- and that is
    not the distance anyone cares about. Its weight per constraint varies with the
    homogeneous scale, so near touchline pixels count for more than far ones, and a line
    carrying 48 snapped points outvotes one carrying 6. Applied to a homography that was
    already exactly right, it moved it half a metre and every perturbation converged to
    0.8 m rather than to zero.

    A DLT has to do it that way because it has no starting guess. Here there is one, so
    the residual can be the real thing: how far, in metres, each snapped pixel lands from
    the line it was snapped to. `soft_l1` keeps a handful of wrong snaps from steering it.
    """
    from scipy.optimize import least_squares

    image = np.array([p for p, _ in pairs], dtype=np.float64)
    lines = np.array([ln for _, ln in pairs], dtype=np.float64)
    norms = np.hypot(lines[:, 0], lines[:, 1])
    ones = np.ones((len(image), 1))
    homog = np.hstack([image, ones])

    def residual(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        m = np.append(params, 1.0).reshape(3, 3)
        proj = homog @ m.T
        w = proj[:, 2]
        w = np.where(np.abs(w) < 1e-9, 1e-9, w)
        x = proj[:, 0] / w
        y = proj[:, 1] / w
        r = (lines[:, 0] * x + lines[:, 1] * y + lines[:, 2]) / norms
        return np.asarray(r, dtype=np.float64)

    scaled = h / h[2, 2] if abs(h[2, 2]) > 1e-12 else h
    out = least_squares(residual, scaled.reshape(9)[:8], loss="soft_l1", f_scale=0.3, max_nfev=400)
    # `success` is False only when the iteration cap is hit; the estimate at that point is
    # still the best one found, and refusing it discards a good refinement for a slow one.
    fitted = np.asarray(np.append(out.x, 1.0).reshape(3, 3), dtype=np.float64)
    return fitted if bool(np.isfinite(fitted).all()) else None


def refine(h: H, img: Any, passes: int = PASSES) -> H | None:
    """A carried homography snapped back onto the paint, or None if it cannot be.

    None is a real answer and the caller keeps what it had. A frame with no visible
    markings -- a tight shot of two players, a replay wipe -- has nothing to snap to, and
    a fit from whatever few pixels were found would be worse than the guess it replaced.
    """
    paint = line_pixels(img)
    height, width = paint.shape[:2]
    current = h
    for _ in range(passes):
        mask = (paint & on_pitch_mask(current, width, height)).astype(np.uint8)
        pairs = correspondences(current, mask)
        if len(pairs) < MIN_CORRESPONDENCES:
            return None if current is h else current
        fitted = _geometric(current, pairs)
        if fitted is None:
            return None if current is h else current
        current = fitted

    # Measured on the FRAME, not at the pitch corners. In a tight shot three of the four
    # corners are off screen -- one of them twenty frame-widths out -- so a corner-based
    # shift reports the extrapolation and refuses every good refinement (D33).
    shift = calibration.observed_error(h, current, (height, width))
    if not np.isfinite(shift) or shift > MAX_SHIFT_M:
        return None
    return current
