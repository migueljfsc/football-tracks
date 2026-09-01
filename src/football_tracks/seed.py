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
from dataclasses import dataclass
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
MIN_SPREAD_M = 3.0


@dataclass(slots=True)
class Seed:
    frame: int
    points: list[tuple[tuple[float, float], tuple[float, float]]]  # (image, pitch)

    def to_json(self) -> dict[str, object]:
        return {
            "version": 1,
            "frame": self.frame,
            "points": [
                {"image": [round(ix, 1), round(iy, 1)], "pitch": [px, py]}
                for (ix, iy), (px, py) in self.points
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
    """Image pixels -> pitch metres, from the clicked correspondences.

    Four points is the minimum and is exactly determined, so it fits perfectly whatever
    was misclicked and cannot be checked - the same trap as D17. More points are
    strongly preferred, they must not all lie along one line, and the overlay is the
    only real verification.
    """
    import cv2

    if len(seed.points) < 4:
        return None
    image = np.array([p[0] for p in seed.points], dtype=np.float64)
    pitch = np.array([p[1] for p in seed.points], dtype=np.float64)
    if degenerate(pitch):
        return None
    # The RANSAC threshold is in the DESTINATION space, which here is PITCH METRES,
    # not pixels. Five - the usual pixel default - would be a five-metre tolerance and
    # would accept almost any misclick as an inlier.
    method = cv2.RANSAC if len(seed.points) > 4 else 0
    h, _ = cv2.findHomography(image, pitch, method, MISCLICK_METRES)
    return None if h is None else np.asarray(h, dtype=np.float64)
