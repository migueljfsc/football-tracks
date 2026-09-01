"""Carry a homography across frames that cannot be solved on their own.

Stage 1's solver needs enough pitch markings in shot. Real footage often has fewer -
the camera tightens on a challenge, or swings to a corner where only one line shows -
and on arbitrary broadcast there are no annotations at all, only whatever a detector
finds. Both cases want the same thing: a homography known at ONE frame, carried to its
neighbours.

The carrier is the ground plane itself. Features on the grass move between consecutive
frames by exactly the homography the camera's motion induces, so tracking them gives a
frame-to-frame transform D, and composing it with a known H gives the next one.

Two things this cannot do, and both are measured rather than assumed:

* It DRIFTS. Every composition multiplies in the last one's error, so a chain is only
  as long as its tolerance allows. `drift` measures how fast.
* It cannot start itself. Something must supply the first homography - the solver on a
  frame that has enough lines, a keypoint model, or a human clicking four corners (D7).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

from .config import GREEN_HI, GREEN_LO

H = npt.NDArray[np.float64]

# Features are taken only from the grass. Players, crowd and hoardings move
# independently of the ground plane, and while RANSAC would mostly reject them, not
# feeding them in is cheaper and leaves the inlier count meaning what it says.
MIN_INLIERS = 25
MAX_CORNERS = 800
QUALITY = 0.01
MIN_DISTANCE = 8

# Erosion pulls the mask off the boundary between grass and everything else, where a
# corner is half player and tracks like neither.
MASK_ERODE = 9

# Lucas-Kanade window and pyramid. Passed positionally rather than as **kwargs so the
# overload stays resolvable.
LK_WINDOW = (21, 21)
LK_LEVELS = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


def grass_mask(bgr: MatLike) -> MatLike:
    """Where the pitch is, eroded away from its own edges."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    kernel = np.ones((MASK_ERODE, MASK_ERODE), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


def between(prev_bgr: MatLike, next_bgr: MatLike) -> H | None:
    """The transform taking points in `prev` to where they land in `next`.

    None when the ground plane could not be tracked - a cut, a whip pan, or a frame
    that is mostly players. A refusal here is a gap in the chain, which is correct:
    guessing would put every later frame on a different pitch.
    """
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY,
        minDistance=MIN_DISTANCE,
        mask=grass_mask(prev_bgr),
    )
    if corners is None or len(corners) < MIN_INLIERS:
        return None

    # The stub types `nextPts` as required, but OpenCV takes None there and allocates
    # the output itself, which is the documented way to call it.
    moved, status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
        prev_gray,
        next_gray,
        corners,
        None,
        winSize=LK_WINDOW,
        maxLevel=LK_LEVELS,
        criteria=LK_CRITERIA,
    )
    if moved is None:
        return None
    ok = status.ravel() == 1
    if int(ok.sum()) < MIN_INLIERS:
        return None

    src = corners[ok].reshape(-1, 2).astype(np.float64)
    dst = moved[ok].reshape(-1, 2).astype(np.float64)
    d, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if d is None or inliers is None or int(inliers.sum()) < MIN_INLIERS:
        return None
    return np.asarray(d, dtype=np.float64)


def motions(frames_dir: Path, frames: list[int], *, cache: Path | None = None) -> dict[int, H]:
    """Frame-to-frame ground-plane transforms, keyed by the LATER frame.

    `motions[f]` maps image f-1 onto image f. Measured per pair and never accumulated,
    so unlike a carried homography this does not drift - which is what makes it safe
    for the tracker to lean on (D19).

    Computed once and cached: both propagation and stage 2 want the same transforms,
    and optical flow over a whole clip is the slowest thing in the pipeline.
    """
    if cache is not None and cache.exists():
        stored = json.loads(cache.read_text())
        return {int(k): np.array(v, dtype=np.float64) for k, v in stored["motions"].items()}

    out: dict[int, H] = {}
    prev_img: MatLike | None = None
    prev_f: int | None = None
    for f in frames:
        img = _read(frames_dir, f)
        if img is not None and prev_img is not None and prev_f == f - 1:
            d = between(prev_img, img)
            if d is not None:
                out[f] = d
        prev_img, prev_f = img, f

    if cache is not None:
        cache.write_text(
            json.dumps({"version": 1, "motions": {str(k): v.tolist() for k, v in out.items()}})
            + "\n"
        )
    return out


def carry(h_prev: H, d: H) -> H | None:
    """Move a homography onto the next frame.

    `h_prev` maps image n to pitch; `d` maps image n to image n+1. A point in image
    n+1 is therefore pushed back through `d` and then through `h_prev`.

    None when the composition has gone degenerate. Every carry multiplies in the last
    one's error, and a chain that collapses does not raise - it starts returning a
    matrix that maps the whole frame to a point, which downstream reads as every player
    standing in the same place.
    """
    if abs(np.linalg.det(d)) < 1e-12:
        return None
    out = h_prev @ np.linalg.inv(d)
    if not np.all(np.isfinite(out)) or abs(out[2, 2]) < 1e-12:
        return None
    if abs(np.linalg.det(out)) < 1e-12:
        return None
    return np.asarray(out / out[2, 2], dtype=np.float64)


@dataclass(slots=True)
class Chain:
    homographies: dict[int, H | None]
    solved_directly: int
    carried: int
    gaps: int

    @property
    def coverage(self) -> float:
        total = len(self.homographies)
        return (self.solved_directly + self.carried) / total if total else 0.0


def _read(frames_dir: Path, f: int) -> MatLike | None:
    img = cv2.imread(str(frames_dir / f"{f:06d}.jpg"))
    return None if img is None else img


# How far a homography may be carried before it is given up on, in frames. Drift is
# unbounded and a badly drifted matrix produces confident wrong positions, which is
# worse than a gap (D13). Measured on SNGS-147, a carry holds inside ~1.6 m out to 120
# frames and degrades sharply after; 50 frames is two seconds, well inside that, and it
# costs little because most gaps are short.
DEFAULT_MAX_CARRY = 50


def fill(
    frames_dir: Path,
    direct: dict[int, H | None],
    *,
    max_carry: int | None = DEFAULT_MAX_CARRY,
    motion: dict[int, H] | None = None,
) -> Chain:
    """Walk the clip forwards, using the direct fit where there is one and carrying otherwise.

    `max_carry` caps how many frames a homography may be carried before it is given up
    on. Drift is unbounded, so an uncapped chain will eventually be confidently wrong;
    the cap turns that into an honest gap.
    """
    frames = sorted(direct)
    out: dict[int, H | None] = {}
    prev_img: MatLike | None = None
    prev_f: int | None = None
    since_direct = 0
    carried = 0

    for f in frames:
        img = _read(frames_dir, f)
        current = direct.get(f)

        if current is not None:
            since_direct = 0
        elif (
            img is not None
            and prev_img is not None
            and prev_f == f - 1
            and out.get(prev_f) is not None
            and (max_carry is None or since_direct < max_carry)
        ):
            d = motion.get(f) if motion is not None else between(prev_img, img)
            previous = out[prev_f]
            if d is not None and previous is not None:
                current = carry(previous, d)
                if current is not None:
                    since_direct += 1
                    carried += 1

        out[f] = current
        prev_img, prev_f = img, f

    return Chain(
        homographies=out,
        solved_directly=sum(1 for f in frames if direct.get(f) is not None),
        carried=carried,
        gaps=sum(1 for v in out.values() if v is None),
    )


def drift(
    frames_dir: Path, direct: dict[int, H | None], seed: int, *, length: int
) -> list[tuple[int, float]]:
    """Carry from ONE seed and report the error against each frame's own direct fit.

    The measurement that says how long a chain may be. Returns `(frames carried, metres
    of disagreement)` pairs - the offset travels WITH the error because not every frame
    has a direct fit to score against, so a bare list would be indexed by scored frames
    while reading like it was indexed by carried ones.
    """
    corners = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
    seeded = direct.get(seed)
    if seeded is None:
        return []

    errors: list[tuple[int, float]] = []
    current = seeded
    prev_img = _read(frames_dir, seed)
    for f in range(seed + 1, seed + length + 1):
        img = _read(frames_dir, f)
        if img is None or prev_img is None:
            break
        d = between(prev_img, img)
        if d is None:
            break
        carried = carry(current, d)
        if carried is None:
            break
        current = carried
        prev_img = img

        truth = direct.get(f)
        if truth is None:
            continue
        # Compared at the pitch corners rather than at the matrix: a homography is only
        # ever wrong somewhere, and the corners are where a small angular error shows.
        got = cv2.perspectiveTransform(
            cv2.perspectiveTransform(corners.reshape(-1, 1, 2), np.linalg.inv(truth)), current
        ).reshape(-1, 2)
        errors.append((f - seed, float(np.max(np.linalg.norm(got - corners, axis=1)))))
    return errors
