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
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

from .calibration import observed_error
from .config import GREEN_HI, GREEN_LO

H = npt.NDArray[np.float64]

# Features are taken only from the grass. Players, crowd and hoardings move
# independently of the ground plane, and while RANSAC would mostly reject them, not
# feeding them in is cheaper and leaves the inlier count meaning what it says.
MIN_INLIERS = 25
MAX_CORNERS = 800
# Relative to the strongest corner inside the mask, so it sets how many features a frame
# offers by whatever bright thing happens to be in shot rather than by how much grass
# there is. RANSAC decides and MIN_INLIERS guards, so the bar for a CANDIDATE is low
# (D86) -- and the cap above, not this, is what limits the count on an ordinary frame.
QUALITY = 0.003
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

    It is also the most expensive refusal in the pipeline. `fill` cannot step over a
    missing link, so ONE refused pair ends the chain for every frame after it, however
    well the next pair tracks (D86).
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


def _project(h: H, x: float, y: float) -> tuple[float, float]:
    p = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), h)
    return (float(p[0][0][0]), float(p[0][0][1]))


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


def carry_back(h_next: H, d: H) -> H | None:
    """Move a homography onto the PREVIOUS frame.

    `d` maps image f-1 onto image f, so a point in f-1 goes through `d` and then
    through `h_next`. Needed because a clip is rarely best seeded at its first frame -
    the camera is often still finding the play - and a seed halfway through is useless
    if it can only travel forwards.
    """
    out = h_next @ d
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
    # How far each frame is from the nearest anchor, in frames. Zero at an anchor. This is
    # what says WHERE another seed would help, which is the one question a coach can act on.
    carried_from: dict[int, int] = field(default_factory=dict)
    # Where two anchors reach the same frame from opposite directions, how far apart the
    # two answers are, in metres. Drift measured rather than assumed -- the two chains have
    # accumulated it independently, so their disagreement is the error they have built up
    # between them (D80).
    disagreement: dict[int, float] = field(default_factory=dict)

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


# Where a frame is probed to compare two camera models, in image fractions. The centre
# and the four quarter-points: near the middle the models always agree, and at the corners
# the horizon sends the answer to infinity for both.
AGREEMENT_PROBES = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.5))

# How far two camera models may disagree, in metres, before the newer one is refused.
#
# The segmenter's failure is not coverage and not accuracy -- it solves as many frames as
# the ground-truth annotations do (615 against 613 on SNGS-116) and its fits are as good
# on the frames both solve. It is a tail: 84 of SNGS-147's 645 fitted frames put players
# more than 3 m out, and those wreck recall and poison anything carried through them.
#
# The fit's own residual cannot find them, and says so in its docstring: it is in-sample,
# so a frame with few constraints is confidently wrong and still scores well. Measured,
# good frames sit at 0.47 and bad at 0.76 -- no separation. The previous fit, walked
# forward by MEASURED motion, is independent of this frame's fit and separates them by
# twenty times: 0.17-0.34 m against 3.2-4.8 m.
AGREEMENT_M = 2.0


def disagreement(a: H, b: H, *, width: int = 1920, height: int = 1080) -> float:
    """How far two camera models put the same picture apart, in metres.

    The median over `AGREEMENT_PROBES` rather than the worst of them, because one probe
    near the horizon disagrees by hundreds of metres for models that agree everywhere a
    player stands.
    """
    gaps = [
        math.dist(_project(a, fx * width, fy * height), _project(b, fx * width, fy * height))
        for fx, fy in AGREEMENT_PROBES
    ]
    gaps.sort()
    return gaps[len(gaps) // 2]


def winnow(
    direct: dict[int, H | None],
    motion: dict[int, H],
    *,
    gate: float = AGREEMENT_M,
    max_age: int = DEFAULT_MAX_CARRY,
    width: int = 1920,
    height: int = 1080,
) -> dict[int, H | None]:
    """Direct fits, with those that contradict the frames before them dropped.

    Chained rather than pairwise: the reference is the last fit ACCEPTED, so a wrong fit
    never becomes the standard its neighbours are judged against. The reference is given
    up on after `max_age` frames, because a carry drifts and a stale one starts refusing
    good fits for disagreeing with it.
    """
    out: dict[int, H | None] = {}
    ref: tuple[int, H] | None = None
    for f in sorted(direct):
        h = direct[f]
        if h is None:
            out[f] = None
            continue
        if ref is not None and f - ref[0] <= max_age:
            carried: H | None = ref[1]
            for g in range(ref[0] + 1, f + 1):
                d = motion.get(g)
                carried = None if carried is None or d is None else carry(carried, d)
                if carried is None:
                    break
            if carried is not None and disagreement(h, carried, width=width, height=height) > gate:
                out[f] = None  # contradicts the frames before it
                continue
        out[f] = h
        ref = (f, h)
    return out


# Where a homography is resampled when two of them are mixed, in image fractions. The
# quarter-points and nothing near the edges: a corner of the frame is often above the
# horizon, where a model maps to infinity and a mix of two of them means nothing.
BLEND_CORNERS = ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75))


def blend(a: H, b: H, t: float, *, width: int = 1920, height: int = 1080) -> H | None:
    """A camera model `t` of the way from `a` to `b`.

    Mixed where it means something -- in METRES, at four points of the picture -- and
    refitted. Averaging the matrices themselves mixes nine numbers that do not vary
    independently and gives a camera that is neither.
    """
    src = np.array([[fx * width, fy * height] for fx, fy in BLEND_CORNERS], dtype=np.float32)
    pa = cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float64), a).reshape(-1, 2)
    pb = cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float64), b).reshape(-1, 2)
    if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
        return None
    mixed = ((1.0 - t) * pa + t * pb).astype(np.float32)
    out = cv2.getPerspectiveTransform(src, mixed)
    if out is None or not np.all(np.isfinite(out)) or abs(out[2, 2]) < 1e-12:
        return None
    return np.asarray(out / out[2, 2], dtype=np.float64)


# How much of the way to an anchor the chain moves in one frame.
#
# Anchoring a carried chain by REPLACEMENT is accurate per frame and worse per track: at
# every anchor the whole picture moves at once, so a player standing still takes a step
# of half a metre and the fidelity a board is built from goes with it. Measured on
# SNGS-147, hard anchors improved registration inside two metres from 62% of frames to
# 85% and took the tracks from 73.7% precision to 58.9% and the team split from 79% to
# 60%. Bleeding the same correction in over a few frames keeps the accuracy and gives the
# tracker back a camera that moves the way a camera moves.
#
# A twentieth, measured across five clips. What it buys is read on the step a STANDING
# player takes between adjacent frames, which is what the tracker and the board see:
#
#     rate    jitter p90    registration within 2 m (SNGS-147 / 116 / 060)
#     1.00     0.27-0.85 m       85%   69%   91%
#     0.30     0.07-0.22 m       86%   67%   85%
#     0.15     0.04-0.12 m       88%   64%   85%
#     0.05     0.02-0.05 m       86%   59%   89%
#
# Slower is not free -- a correction that arrives over eighty frames has not arrived --
# but every rate above a twentieth pays for its accuracy in jitter, and jitter is what
# stage 2 and the reduction are made of.
ANCHOR_RATE = 0.05


def anchor_chain(
    anchors: dict[int, H | None],
    *,
    motion: dict[int, H],
    rate: float = ANCHOR_RATE,
    hard: set[int] | None = None,
    max_carry: int | None = None,
    width: int = 1920,
    height: int = 1080,
) -> Chain:
    """Carry a chain across the clip, correcting it towards each anchor it passes.

    A complementary filter, and the two halves are chosen for what each is good at. The
    carry is exact frame to frame and wrong in the long run -- it drifts (D18). An anchor
    is unbiased in the long run and noisy frame to frame, because it is fitted from
    however many pixels of marking happened to be in shot. So the carry supplies the
    MOTION and the anchors supply the POSITION, at `rate` of the difference per frame.

    `fill` remains the honest version of "solve it or say nothing": it never mixes two
    models. This one is for the case D67 describes, where one source has the coverage and
    the other has the accuracy and neither alone makes a board.
    """
    frames = sorted(anchors)
    out: dict[int, H | None] = dict.fromkeys(frames)
    first = next((f for f in frames if anchors[f] is not None), None)
    if first is None:
        return Chain(homographies=out, solved_directly=0, carried=0, gaps=len(frames))

    corrected = 0
    current: H | None = anchors[first]
    since_anchor = 0
    for f in frames:
        if f < first:
            continue
        if f > first:
            d = motion.get(f)
            current = None if current is None or d is None else carry(current, d)
            since_anchor += 1
        here = anchors[f]
        if here is not None:
            # Nothing to correct towards yet: a chain that has broken restarts on the
            # next anchor rather than staying dark to the end of the clip.
            # A seed is taken whole rather than mixed into: it is the one fit somebody
            # looked at, and there is nothing about the chain worth keeping against it.
            share = 1.0 if hard is not None and f in hard else rate
            mixed = (
                here
                if current is None or share >= 1.0
                else blend(current, here, share, width=width, height=height)
            )
            if mixed is not None:
                current = mixed
                since_anchor = 0
                corrected += 1
        if max_carry is not None and since_anchor > max_carry:
            current = None
        out[f] = current

    # Backwards, for the frames before the first anchor. There are no anchors there by
    # construction, so this is a plain carry and drifts exactly as `fill`'s does.
    current = out[first]
    for behind, f in enumerate(reversed([f for f in frames if f < first]), start=1):
        d = motion.get(f + 1)
        current = None if current is None or d is None else carry_back(current, d)
        if max_carry is not None and behind > max_carry:
            current = None
        out[f] = current

    return Chain(
        homographies=out,
        solved_directly=sum(1 for f in frames if anchors[f] is not None),
        carried=sum(1 for f in frames if out[f] is not None) - corrected,
        gaps=sum(1 for v in out.values() if v is None),
    )


def fill(
    frames_dir: Path,
    direct: dict[int, H | None],
    *,
    max_carry: int | None = DEFAULT_MAX_CARRY,
    motion: dict[int, H] | None = None,
    snap: Callable[[H, MatLike], H | None] | None = None,
) -> Chain:
    """Direct fits where there are any, carried both ways to cover the rest.

    Two chains are built, one running with the clip and one against it, and every frame
    takes whichever carried it FEWER frames. That is what makes a second anchor worth
    clicking: a forward-only walk hands every frame between two anchors to the earlier
    one, so a clip seeded at both ends is still carried its whole length from the first.

    `max_carry` caps how many frames a homography may be carried before it is given up
    on. Drift is unbounded, so an uncapped chain will eventually be confidently wrong;
    the cap turns that into an honest gap.

    `snap` re-anchors each carried homography on its own frame before it becomes the basis
    for the next one, which is what stops drift compounding rather than merely measuring
    it. It has to happen HERE and not as a pass afterwards: snapping is a local
    correction with a capture radius of about two metres, so it can hold a chain that is
    nearly right and can do nothing at all for one that has already wandered forty. Run
    over a finished chain it refused half the frames on the clip that needed it most.
    """
    frames = sorted(direct)
    first = next((img for f in frames if (img := _read(frames_dir, f)) is not None), None)
    height, width = (first.shape[0], first.shape[1]) if first is not None else (1080, 1920)
    forward, forward_age = _carry_forward(frames_dir, frames, direct, max_carry, motion, snap)
    backward, backward_age = _carry_backward(frames_dir, frames, direct, max_carry, motion, snap)

    # Between two anchors, the two chains are MIXED in proportion to how far each has been
    # carried. Walking forward alone means a second anchor only ever helps the frames after
    # it -- everything between two of them chains off the earlier one, however far away it
    # is, and a clip seeded at both ends is carried its whole length from the first
    # (measured on a coach's clip seeded at 56 and 382: frame 380 was 324 frames from its
    # basis and metres out, with an exact fit one frame away).
    #
    # Mixed rather than SWITCHED at the midpoint, for D68's reason: at the frame where the
    # nearer anchor changes, the two chains disagree by whatever they have drifted, and
    # taking the better one moves every player at once. `splitImpossible` then cuts every
    # track in the clip at that frame -- measured on the same clip, board density fell from
    # 72% to 50% and the importer had no passage left that spanned the join.
    out: dict[int, H | None] = {}
    apart: dict[int, float] = {}
    reach: dict[int, int] = {}
    carried = 0
    for f in frames:
        if direct.get(f) is not None:
            out[f] = direct[f]
            reach[f] = 0
            continue
        ahead, behind = forward.get(f), backward.get(f)
        if ahead is None or behind is None:
            out[f] = ahead if behind is None else behind
            if out[f] is not None:
                reach[f] = forward_age[f] if behind is None else backward_age[f]
        else:
            span = forward_age[f] + backward_age[f]
            share = forward_age[f] / span if span else 0.0
            mixed = blend(ahead, behind, share, width=width, height=height)
            out[f] = mixed if mixed is not None else (ahead if share <= 0.5 else behind)
            reach[f] = min(forward_age[f], backward_age[f])
            apart[f] = disagreement(ahead, behind, width=width, height=height)
        if out[f] is not None:
            carried += 1

    return Chain(
        homographies=out,
        solved_directly=sum(1 for f in frames if direct.get(f) is not None),
        carried=carried,
        gaps=sum(1 for v in out.values() if v is None),
        carried_from=reach,
        disagreement=apart,
    )


def _carry_forward(
    frames_dir: Path,
    frames: list[int],
    direct: dict[int, H | None],
    max_carry: int | None,
    motion: dict[int, H] | None,
    snap: Callable[[H, MatLike], H | None] | None,
) -> tuple[dict[int, H | None], dict[int, int]]:
    """The chain running with the clip: each frame from the last anchor before it."""
    out: dict[int, H | None] = {}
    age: dict[int, int] = {}
    # What the NEXT frame carries from, which is deliberately not what this frame
    # reports. A snap that goes wrong is plausible enough to pass its own guard, and fed
    # back it becomes the basis for every frame after it -- one bad fit poisons the rest
    # of the chain instead of costing one frame.
    basis: dict[int, H | None] = {}
    prev_img: MatLike | None = None
    prev_f: int | None = None
    since_direct = 0

    for f in frames:
        img = _read(frames_dir, f)
        current = direct.get(f)

        if current is not None:
            since_direct = 0
        elif (
            img is not None
            and prev_img is not None
            and prev_f == f - 1
            and basis.get(prev_f) is not None
            and (max_carry is None or since_direct < max_carry)
        ):
            d = motion.get(f) if motion is not None else between(prev_img, img)
            previous = basis[prev_f]
            if d is not None and previous is not None:
                current = carry(previous, d)
                if current is not None:
                    since_direct += 1

        basis[f] = current
        if current is not None and snap is not None and img is not None:
            snapped = snap(current, img)
            if snapped is not None:
                current = snapped
        out[f] = current
        age[f] = since_direct if current is not None else 0
        prev_img, prev_f = img, f
    return out, age


def _carry_backward(
    frames_dir: Path,
    frames: list[int],
    direct: dict[int, H | None],
    max_carry: int | None,
    motion: dict[int, H] | None,
    snap: Callable[[H, MatLike], H | None] | None,
) -> tuple[dict[int, H | None], dict[int, int]]:
    """The same chain running against the clip: each frame from the next anchor after it.

    A seed is rarely placed at frame one -- the camera is usually still finding the play
    there -- so without this everything before the first anchor is blank.
    """
    out: dict[int, H | None] = {}
    basis: dict[int, H | None] = {}
    age: dict[int, int] = {}
    since_direct = 0

    for i in range(len(frames) - 1, -1, -1):
        f = frames[i]
        current = direct.get(f)
        later = frames[i + 1] if i + 1 < len(frames) else None

        if current is not None:
            since_direct = 0
        elif (
            later == f + 1
            and basis.get(later) is not None
            and (max_carry is None or since_direct < max_carry)
        ):
            d = motion.get(later) if motion is not None else None
            if d is None and later is not None:
                a, b = _read(frames_dir, f), _read(frames_dir, later)
                d = between(a, b) if a is not None and b is not None else None
            previous = basis[later] if later is not None else None
            if d is not None and previous is not None:
                current = carry_back(previous, d)
                if current is not None:
                    since_direct += 1

        basis[f] = current
        if current is not None and snap is not None:
            img = _read(frames_dir, f)
            snapped = snap(current, img) if img is not None else None
            if snapped is not None:
                current = snapped
        out[f] = current
        age[f] = since_direct if current is not None else 0
    return out, age


def drift(
    frames_dir: Path, direct: dict[int, H | None], seed: int, *, length: int
) -> list[tuple[int, float]]:
    """Carry from ONE seed and report the error against each frame's own direct fit.

    The measurement that says how long a chain may be. Returns `(frames carried, metres
    of disagreement)` pairs - the offset travels WITH the error because not every frame
    has a direct fit to score against, so a bare list would be indexed by scored frames
    while reading like it was indexed by carried ones.

    `direct` must hold only what was FITTED. Passing the carried chain scores it against
    itself and reports 0.00 m however far the camera has wandered (D33).
    """
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
        errors.append((f - seed, observed_error(truth, current, img.shape)))
    return errors
