"""The one thing a human supplies: which pixels are which pitch landmarks.

A homography needs four points whose PITCH coordinates are known. Players cannot
provide them - where a player stands is the unknown being solved for - so the seed is
always pitch geometry: a box corner, a penalty spot, the foot of a post.

Once one frame is seeded, `stage1_propagate` carries it, which held for about seven
seconds on SoccerNet before drift told. So this is a few clicks per clip, not per frame.

The file is deliberately plain JSON. It is written by the click tool here, but a
keypoint model writes the same thing, and so would Pitchboard's own import view.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .config import PITCH_LENGTH, PITCH_WIDTH

# The landmarks worth offering, in metres, for the goal at x=0. A clip showing the far
# goal is seeded with the same names mirrored, which `mirrored()` does, so a coach
# never has to think about which end the pitch model calls zero.
MID = PITCH_WIDTH / 2
# FAR means away from the camera and NEAR means toward it - never left and right,
# which depend on where the camera is standing and are ambiguous on a screen. The
# broadcast camera sits on one touchline, so "near" is always the bottom of the frame.
LANDMARKS: dict[str, tuple[float, float]] = {
    "goal post far": (0.0, MID - 3.66),
    "goal post near": (0.0, MID + 3.66),
    "6yd box far corner": (0.0, MID - 9.16),
    "6yd box near corner": (0.0, MID + 9.16),
    "6yd front far": (5.5, MID - 9.16),
    "6yd front near": (5.5, MID + 9.16),
    "penalty box far corner": (0.0, MID - 20.16),
    "penalty box near corner": (0.0, MID + 20.16),
    "penalty box front far": (16.5, MID - 20.16),
    "penalty box front near": (16.5, MID + 20.16),
    "penalty spot": (11.0, MID),
    "corner far": (0.0, 0.0),
    "corner near": (0.0, PITCH_WIDTH),
    "halfway far": (PITCH_LENGTH / 2, 0.0),
    "halfway near": (PITCH_LENGTH / 2, PITCH_WIDTH),
}


def mirrored(name: str) -> tuple[float, float]:
    """The same landmark at the other end of the pitch."""
    x, y = LANDMARKS[name]
    return (PITCH_LENGTH - x, y)


# How far a clicked point may sit from where the fitted camera puts it, in metres,
# before RANSAC calls it a misclick. In the DESTINATION space, which here is pitch
# metres and not pixels - five, the usual pixel default, would be a five-metre
# tolerance and would accept anything.
#
# Half a metre rather than one: with six points a homography has barely more
# constraints than degrees of freedom, so at a loose threshold RANSAC prefers a warped
# fit that accommodates the bad click over one that rejects it. Measured on a
# deliberate 8 m misclick, 0.5 rejects it and 1.0 absorbs it and moves the centre spot
# sixteen metres.
MISCLICK_METRES = 0.5

# Clicked points must span two dimensions, not lie along a line. Both goalposts and
# both corners of a goal are the four most obvious things to click and ALL FOUR SIT ON
# x = 0 - a degenerate set that fits perfectly and describes nothing. Measured as the
# ratio of the point cloud's two principal spreads.
MIN_SPREAD_RATIO = 0.08

# Distinct markings needed when there is nothing but traced lines to go on. Two always
# meet, and a fit that maps everything to where they meet satisfies both perfectly.
MIN_TRACED_LINES = 3
MIN_SPREAD_M = 3.0


# Named pitch lines a human can trace, and what they are in metres. Tracing beats
# clicking a corner: a corner is one exact pixel and is often out of shot, while a line
# you can see is easy to follow and just as informative once several points are stacked.
TRACEABLE: dict[str, tuple[float, float, float]] = {
    "goal line": (1.0, 0.0, 0.0),
    "6yd box front": (1.0, 0.0, -5.5),
    "penalty box front": (1.0, 0.0, -16.5),
    "6yd box far side": (0.0, 1.0, -(MID - 9.16)),
    "6yd box near side": (0.0, 1.0, -(MID + 9.16)),
    "penalty box far side": (0.0, 1.0, -(MID - 20.16)),
    "penalty box near side": (0.0, 1.0, -(MID + 20.16)),
    "far touchline": (0.0, 1.0, 0.0),
    "near touchline": (0.0, 1.0, -PITCH_WIDTH),
}


def mirrored_line(name: str) -> tuple[float, float, float]:
    """The same marking at the other end. x = k becomes x = 105 - k; y = k is unchanged."""
    a, b, c = TRACEABLE[name]
    return (-a, b, c + a * PITCH_LENGTH) if a else (a, b, c)


@dataclass(slots=True)
class Seed:
    frame: int
    points: list[tuple[tuple[float, float], tuple[float, float]]]  # (image, pitch)
    lines: list[tuple[tuple[float, float], tuple[float, float, float]]] = field(
        default_factory=list
    )  # (image, pitch line)

    def to_json(self) -> dict[str, object]:
        return {
            "version": 1,
            "frame": self.frame,
            "points": [
                {"image": [round(ix, 1), round(iy, 1)], "pitch": [px, py]}
                for (ix, iy), (px, py) in self.points
            ],
            "lines": [
                {"image": [round(ix, 1), round(iy, 1)], "line": list(ln)}
                for (ix, iy), ln in self.lines
            ],
        }


def write(path: Path, seed: Seed) -> Path:
    path.write_text(json.dumps(seed.to_json(), indent=2) + "\n")
    return path


def read(path: Path) -> Seed:
    d = json.loads(path.read_text())
    return Seed(
        frame=int(d["frame"]),
        points=[
            (
                (float(p["image"][0]), float(p["image"][1])),
                (float(p["pitch"][0]), float(p["pitch"][1])),
            )
            for p in d["points"]
        ],
        lines=[
            (
                (float(p["image"][0]), float(p["image"][1])),
                (float(p["line"][0]), float(p["line"][1]), float(p["line"][2])),
            )
            for p in d.get("lines", [])
        ],
    )


# Below this the pitch y axis and the image y axis are judged to disagree. Above the
# positive version, they agree. Between, the camera is looking along the pitch rather
# than across it and the test cannot tell - a seed from behind the goal, say.
ORIENTATION_CONFIDENT = 0.5


def orientation(seed: Seed) -> float:
    """How well the clicked pitch y axis agrees with the image y axis, -1 to 1.

    A camera above the ground plane puts what is nearer to it LOWER in the frame, and
    this codebase's convention is that nearer the camera is LARGER pitch y. So the two
    must rise together, and a strong negative means the far/near labels were swapped.

    Worth a dedicated check because the reprojection overlay - which catches everything
    else - is blind to exactly this. A football pitch is symmetric about the halfway
    line, so a y-mirrored model draws onto the real markings perfectly and the picture
    looks right while every position is flipped.
    """
    if len(seed.points) < 3:
        return 0.0
    image_y = np.array([p[0][1] for p in seed.points], dtype=np.float64)
    pitch_y = np.array([p[1][1] for p in seed.points], dtype=np.float64)
    if image_y.std() < 1e-9 or pitch_y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(image_y, pitch_y)[0, 1])


def flip_y(seed: Seed) -> Seed:
    """The same clicks, with the pitch y axis reflected. Fixes a swapped far/near."""
    return Seed(
        frame=seed.frame,
        points=[(img, (px, PITCH_WIDTH - py)) for img, (px, py) in seed.points],
        # A line a*x + b*y + c = 0 reflected in y = W/2 becomes a*x - b*y + (c + b*W).
        lines=[(img, (a, -b, c + b * PITCH_WIDTH)) for img, (a, b, c) in seed.lines],
    )


def degenerate(pitch: npt.NDArray[np.float64]) -> bool:
    """Whether the clicked landmarks lie too close to a straight line to fit a camera.

    A homography needs points spanning a plane. Four points along the goal line pin
    down nothing about the direction away from it, and the fit that comes back looks
    like any other matrix.
    """
    if len(pitch) < 4:
        return True
    centred = pitch - pitch.mean(axis=0)
    spread = np.linalg.svd(centred, compute_uv=False)
    if spread[0] < MIN_SPREAD_M:
        return True
    return bool(spread[1] / spread[0] < MIN_SPREAD_RATIO)


def homography(seed: Seed) -> npt.NDArray[np.float64] | None:
    """Image pixels -> pitch metres, from clicked landmarks and traced lines.

    Exact landmarks and traced lines go into one fit (see `calibration.fit`). A traced
    point says less than a landmark - somewhere along this marking, rather than exactly
    here - so more of them are needed, but they are far easier to place and they work
    where the corner itself is out of shot.

    Refused when the evidence is degenerate: everything along one line pins down nothing
    about the direction away from it, and fits perfectly anyway (D24).
    """
    import cv2

    from . import calibration

    pitch = np.array([p[1] for p in seed.points], dtype=np.float64)

    if not seed.lines:
        # Exact correspondences only: RANSAC, because a least-squares fit absorbs a
        # single bad click by warping the whole camera - every residual stays small
        # while a point off the edge of the clicked cluster lands tens of metres out.
        if len(seed.points) < 4 or degenerate(pitch):
            return None
        image = np.array([p[0] for p in seed.points], dtype=np.float64)
        method = cv2.RANSAC if len(seed.points) > 4 else 0
        solved, mask = cv2.findHomography(image, pitch, method, MISCLICK_METRES)
        if solved is None:
            return None
        # THE inliers must span two directions, not just the input. RANSAC will happily
        # keep a near-collinear subset, fit it perfectly, and throw away the very points
        # that pinned down the direction away from that line - which is what happened on
        # the first real clip seeded with this tool.
        if mask is not None and degenerate(pitch[mask.ravel() == 1]):
            fallback = calibration.fit(seed.points, [], 0, 0)
            return None if fallback is None else np.asarray(fallback, dtype=np.float64)
        return np.asarray(solved, dtype=np.float64)

    # Two distinct markings are ALWAYS degenerate, however many points are traced along
    # them: they cross somewhere, and a homography sending the whole image to that
    # crossing satisfies every point-on-line constraint exactly. This is structural, so
    # it is counted rather than measured - `_collapses` below catches it numerically and
    # a numerical guard is at the mercy of which machine ran the SVD.
    if not seed.points and len({ln for _, ln in seed.lines}) < MIN_TRACED_LINES:
        return None
    if not _spans_two_directions(seed):
        return None

    fitted = calibration.fit(seed.points, seed.lines, 0, 0)
    if fitted is None:
        return None
    h: npt.NDArray[np.float64] = np.asarray(fitted, dtype=np.float64)
    if _collapses(h, seed):
        return None

    # One trimmed refit. There is no RANSAC here because a traced point is not a
    # correspondence to sample from, and tracing gives many more constraints than
    # degrees of freedom, so a single bad point cannot warp the fit the way it can
    # with six exact clicks.
    bad_points = {i for i, r in enumerate(_point_residuals(h, seed)) if r > MISCLICK_METRES}
    bad_lines = {i for i, r in enumerate(_line_residuals(h, seed)) if r > MISCLICK_METRES}
    if not bad_points and not bad_lines:
        return np.asarray(h, dtype=np.float64)

    kept = Seed(
        frame=seed.frame,
        points=[p for i, p in enumerate(seed.points) if i not in bad_points],
        lines=[ln for i, ln in enumerate(seed.lines) if i not in bad_lines],
    )
    if not _spans_two_directions(kept) or len(kept.points) * 2 + len(kept.lines) < 8:
        return np.asarray(h, dtype=np.float64)
    refit = calibration.fit(kept.points, kept.lines, 0, 0)
    if refit is None or _collapses(np.asarray(refit, dtype=np.float64), kept):
        return np.asarray(h, dtype=np.float64)
    return np.asarray(refit, dtype=np.float64)


def _collapses(h: npt.NDArray[np.float64], seed: Seed) -> bool:
    """Whether the fit maps everything onto one spot.

    A point-on-line constraint says only "lands somewhere on this marking", and a
    homography that sends the ENTIRE image to the point where two traced lines cross
    satisfies every one of them exactly. Two lines always cross, so two lines alone are
    always degenerate however many points are traced along them - and the fit comes back
    with zero residuals, which is the most convincing way to be wrong.

    Caught by pushing the clicked pixels through and asking whether what comes out still
    covers any ground.
    """
    from . import calibration

    image = np.array([p[0] for p in seed.points] + [p[0] for p in seed.lines], dtype=np.float64)
    if len(image) < 3:
        return True
    got = calibration.apply(h, image)
    if not np.all(np.isfinite(got)):
        return True
    spread = np.linalg.svd(got - got.mean(axis=0), compute_uv=False)
    return bool(spread[0] < MIN_SPREAD_M or spread[1] / spread[0] < MIN_SPREAD_RATIO)


def _point_residuals(h: npt.NDArray[np.float64], seed: Seed) -> list[float]:
    """Metres between where each clicked landmark says it is and where the fit puts it."""
    from . import calibration

    if not seed.points:
        return []
    got = calibration.apply(h, np.array([p[0] for p in seed.points], dtype=np.float64))
    want = np.array([p[1] for p in seed.points], dtype=np.float64)
    return [float(v) for v in np.linalg.norm(got - want, axis=1)]


def _line_residuals(h: npt.NDArray[np.float64], seed: Seed) -> list[float]:
    """Metres from each traced point to the marking it was traced along."""
    from . import calibration

    if not seed.lines:
        return []
    got = calibration.apply(h, np.array([p[0] for p in seed.lines], dtype=np.float64))
    out = []
    for (x, y), (a, b, c) in zip(got, [ln[1] for ln in seed.lines], strict=True):
        out.append(abs(a * x + b * y + c) / max(1e-9, math.hypot(a, b)))
    return out


def _spans_two_directions(seed: Seed) -> bool:
    """Whether the evidence constrains both across the pitch AND along it.

    Markings here are axis-aligned, so this asks whether anything pins down x as well
    as y. Traced lines all running the same way - three lines parallel to the goal line,
    say - leave the camera free to slide along the pitch, and the fit that comes back
    looks like any other matrix.
    """
    xs = any(abs(a) > abs(b) for _, (a, b, _c) in seed.lines)
    ys = any(abs(b) >= abs(a) for _, (a, b, _c) in seed.lines)
    if seed.points:
        pitch = np.array([p[1] for p in seed.points], dtype=np.float64)
        if len(pitch) >= 2:
            xs = xs or bool(pitch[:, 0].std() > 1e-6)
            ys = ys or bool(pitch[:, 1].std() > 1e-6)
        # A pair of exact landmarks pins both axes at once wherever they differ.
        if len(pitch) >= 3:
            centred = pitch - pitch.mean(axis=0)
            sv = np.linalg.svd(centred, compute_uv=False)
            if sv[0] > MIN_SPREAD_M and sv[1] / sv[0] >= MIN_SPREAD_RATIO:
                return True
    return xs and ys
