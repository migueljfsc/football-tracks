"""One camera for a whole match: where it stands, and where each frame aims it.

A broadcast camera does not move. It sits on its gantry for ninety minutes and pans, tilts
and zooms, so every frame of every clip of that match is the same camera pointed somewhere
else -- three numbers a frame, against a homography's eight.

Both halves of that are worth having. Three numbers cannot fold the pitch over (D34) or
drift out of shape (D18), because no rotation of a camera above the ground can do either;
and the position carries BETWEEN clips, so a camera fitted from one clip's clicks registers
the next clip of the same match with no clicks at all (D96).

The position is in pitch metres and the aim is per frame, which keeps the one thing a
person supplied -- the clicks -- separate from the thing every frame has to be told.

World axes are the pitch's: x along the length, y across the width, z into the ground.
`height` is metres ABOVE it, because that is the number a person can sanity-check.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from .config import DEFAULT_PITCH, Pitch

# The camera's axes when it is aimed along the pitch's far side with no pan or tilt:
# x to the right, y down the frame, z away from the lens. A broadcast camera sits on a
# touchline, so this is the view it starts from and pan and tilt turn it from there.
LEVEL = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

# What a fit starts from when nothing better is known: a coarse sweep of where the camera
# could be aimed. A broadcast camera looks along the pitch, down at it, and the zoom range
# is wide -- so the sweep is over pan above all, which is what a pan actually changes.
COARSE: tuple[tuple[float, float, float], ...] = tuple(
    (math.radians(pan), math.radians(tilt), focal)
    for pan in range(-60, 61, 10)
    for tilt in (8.0, 13.0, 18.0)
    for focal in (3500.0, 5500.0, 8000.0)
)

# How far a solve may be from the aim it started at before the coarse sweep is tried as
# well. A pan between neighbouring frames is small; a cut is not.
NEAR_ENOUGH_M = 1.0

# Zoom is solved as its logarithm, because a lens is geometric: the step from 5000 to 6000
# pixels is the step from 500 to 600, and a solver taking linear steps crawls at one end and
# overshoots through zero at the other. The bounds are what a camera can be, and they are
# here because a solver walking uphill reaches `exp(800)` in a few steps otherwise.
LOG_FOCAL = (math.log(200.0), math.log(200_000.0))

# What a camera on a gantry can be aimed at: across the pitch rather than behind itself, and
# DOWN at the ground rather than up at the roof.
#
# Not tidiness. A pitch is symmetric end to end, and the mirror image of a camera looking at
# one goal is the same camera upside down looking at the other -- so a seed clicked on the
# wrong goal (D88) fits to 1e-14 with the tilt at 191 degrees. Bounded, there is no such
# solution and the clicks simply miss, which is what `aimed_seeds` reports.
PAN_LIMIT = (math.radians(-89.0), math.radians(89.0))
TILT_LIMIT = (math.radians(0.5), math.radians(89.0))


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


def _focal(log_focal: float) -> float:
    return float(math.exp(_clamp(log_focal, LOG_FOCAL)))


def _view(p: Any) -> View:
    """A solver's three numbers as an aim a camera could actually be pointed at."""
    return View(
        pan=_clamp(float(p[0]), PAN_LIMIT),
        tilt=_clamp(float(p[1]), TILT_LIMIT),
        focal=_focal(float(p[2])),
    )


@dataclass(frozen=True)
class View:
    """Where the camera is aimed for one frame: pan and tilt in radians, zoom in pixels."""

    pan: float
    tilt: float
    focal: float


@dataclass(frozen=True)
class Camera:
    """Where the camera stands, for a whole match. Metres, on this game's pitch."""

    x: float
    y: float
    height: float
    pitch: Pitch = DEFAULT_PITCH
    game: str = ""

    @property
    def centre(self) -> npt.NDArray[np.float64]:
        """The position in world axes, where z runs INTO the ground."""
        return np.array([self.x, self.y, -self.height], dtype=np.float64)


def lens(meta: dict[str, Any]) -> tuple[float, float]:
    """Where the lens axis sits in a clip's pixels, taken as the middle of the kept picture.

    Two clips of one match are cropped by different amounts -- 9 px and 1 px on the pair
    this was built for -- so a camera shared between them is told each clip's own crop
    rather than assuming the frames line up.

    The axis is really the middle of what the camera shot, and bars trimmed unevenly move
    the middle of the kept picture a few pixels off it. `ft frames` does not record the
    size it cropped FROM, so those few pixels are not recoverable here; at a broadcast
    zoom they are centimetres on the grass, which is under the segmenter's own noise.
    """
    x0, y0, x1, y1 = meta["crop"]
    return ((x0 + x1) / 2 - x0, (y0 + y1) / 2 - y0)


def rotation(pan: float, tilt: float) -> npt.NDArray[np.float64]:
    """Pan about the vertical, then tilt down -- the two axes a gantry head actually has."""
    cp, sp = math.cos(pan), math.sin(pan)
    ct, st = math.cos(tilt), math.sin(tilt)
    about_z = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]])
    return np.asarray(about_x @ LEVEL @ about_z, dtype=np.float64)


def matrix(
    camera: Camera, lens_at: tuple[float, float], view: View
) -> npt.NDArray[np.float64] | None:
    """Image pixels -> pitch metres for one frame, or None where the aim describes nothing.

    The same contract as `seed.homography`, so everything downstream reads this as it
    reads any other camera: nothing but stage 1 knows a camera model was involved.
    """
    if not (view.focal > 0 and camera.height > 0):
        return None
    r = rotation(view.pan, view.tilt)
    k = np.array([[view.focal, 0.0, lens_at[0]], [0.0, view.focal, lens_at[1]], [0.0, 0.0, 1.0]])
    g = k @ np.column_stack([r[:, 0], r[:, 1], -r @ camera.centre])
    if not np.all(np.isfinite(g)) or abs(float(np.linalg.det(g))) < 1e-9:
        return None
    return np.asarray(np.linalg.inv(g), dtype=np.float64)


Pairs = list[tuple[tuple[float, float], tuple[float, float, float]]]
Arcs = list[tuple[tuple[float, float], tuple[float, float, float]]]

# Markings a frame must show before its aim is believed, counting a circle as one.
#
# Three numbers need more than three constraints or they fit their evidence exactly and
# describe nothing elsewhere -- D17's rule, at a camera model's size. Measured on the clip
# this was built for: a midfield frame showing the halfway line and one touchline fitted
# both to 0.01 m and put the markings off the picture, and the residual could not see it
# because there was nothing left for it to disagree with.
MIN_MARKINGS = 3

# Constraints a seed must carry before its aim is believed. A clicked landmark is two (x and
# y), a point traced along a marking or a circle is one -- so this is four landmarks, or two
# and some traced points: what the click tool has always asked for.
#
# Two landmarks DO aim the camera, and that is not enough. A pitch is symmetric end to end,
# so the question is not whether an aim fits but whether the MIRROR of it also does (D88) --
# and a camera at the halfway line is its own mirror image, so the clicks are the only thing
# that can tell the two apart. How close a mirrored seed can get, worst case over views:
#
#     landmarks   the mirror gets within
#         2            0.02 m       -- indistinguishable
#         3            0.61 m       -- inside a plausible gate
#         4            1.63 m
#         5            4.08 m
#
# So four, which is what the click tool already asks for. The saving this model brings is
# not a cheaper seed, it is a clip that needs almost none (D96).
MIN_SEED_CONSTRAINTS = 8


def arc_misses(
    h: npt.NDArray[np.float64] | None, arcs: Arcs, cap: float = 1e3
) -> npt.NDArray[np.float64]:
    """Metres from each pixel on a painted circle to that circle, signed."""
    if h is None or not arcs:
        return np.full(len(arcs), cap)
    xy = np.array([p[0] for p in arcs], dtype=np.float64)
    q = h @ np.vstack([xy.T, np.ones(len(xy))])
    with np.errstate(invalid="ignore", divide="ignore"):
        q = q[:2] / q[2]
    circle = np.array([p[1] for p in arcs], dtype=np.float64)
    d = np.hypot(q[0] - circle[:, 0], q[1] - circle[:, 1]) - circle[:, 2]
    return np.where(np.isfinite(d), d, cap)


def line_misses(
    h: npt.NDArray[np.float64] | None, pairs: Pairs, cap: float = 1e3
) -> npt.NDArray[np.float64]:
    """Metres from each named pixel to the marking it was named for, signed.

    Signed for the solver's sake, as `seed._signed` is: a distance has a kink at zero,
    which is exactly where the answer is.
    """
    if h is None or not pairs:
        return np.full(len(pairs), cap)
    xy = np.array([p[0] for p in pairs], dtype=np.float64)
    q = h @ np.vstack([xy.T, np.ones(len(xy))])
    with np.errstate(invalid="ignore", divide="ignore"):
        q = q[:2] / q[2]
    abc = np.array([p[1] for p in pairs], dtype=np.float64)
    d = (abc[:, 0] * q[0] + abc[:, 1] * q[1] + abc[:, 2]) / np.hypot(abc[:, 0], abc[:, 1])
    return np.where(np.isfinite(d), d, cap)


# How many of the coarse sweep's aims are refined, and how much of the evidence ranks them.
# Scoring an aim is one matrix and a multiply; refining one is a solve, so the sweep sorts
# first on a sample of the pixels and only the closest few are solved from.
REFINE_BEST = 3
RANK_EVERY = 5


def _solve(
    residual: Any, starts: tuple[tuple[float, float, float], ...]
) -> tuple[View, float] | None:
    """The best aim over several starting guesses, with its median miss in metres."""
    best: tuple[View, float] | None = None
    for pan, tilt, focal in starts:
        if focal <= 0:
            continue
        got = least_squares(residual, [pan, tilt, math.log(focal)], loss="soft_l1", f_scale=0.5)
        view = _view(got.x)
        miss = float(np.median(np.abs(residual(got.x))))
        if best is None or miss < best[1]:
            best = (view, miss)
    return best


def aim(
    camera: Camera,
    lens_at: tuple[float, float],
    pairs: Pairs,
    arcs: Arcs | None = None,
    *,
    start: View | None = None,
    starts: tuple[tuple[float, float, float], ...] = COARSE,
) -> tuple[View, float] | None:
    """Pan, tilt and zoom for one frame from named pixels, with the position already known.

    Straight markings and circles together: the circle is what a midfield view has when its
    straight markings all lie in one band across the picture (D34).

    `start` is the frame before's aim, which is where a pan of a few degrees is found in
    one step. The coarse sweep is tried as well unless that lands well, because the frame
    before a cut says nothing about the frame after it.

    None where the frame shows too little to be checked -- which is most of a broadcast, and
    is a real answer rather than a failure (D67).
    """
    on = arcs or []
    if len({line for _, line in pairs} | {circle for _, circle in on}) < MIN_MARKINGS:
        return None

    def residual(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        view = _view(p)
        h = matrix(camera, lens_at, view)
        return np.concatenate([line_misses(h, pairs), arc_misses(h, on)])

    if start is not None:
        near = _solve(residual, ((start.pan, start.tilt, start.focal),))
        if near is not None and near[1] <= NEAR_ENOUGH_M:
            return near

    sample, few = pairs[::RANK_EVERY] or pairs, on[::RANK_EVERY] or on
    ranked = sorted(
        starts,
        key=lambda s: float(
            np.median(
                np.abs(
                    np.concatenate(
                        [
                            line_misses(matrix(camera, lens_at, View(*s)), sample),
                            arc_misses(matrix(camera, lens_at, View(*s)), few),
                        ]
                    )
                )
            )
        ),
    )
    return _solve(residual, tuple(ranked[:REFINE_BEST]))


def aim_from_seed(
    camera: Camera,
    lens_at: tuple[float, float],
    clicked: Any,
    *,
    start: View | None = None,
    starts: tuple[tuple[float, float, float], ...] = COARSE,
) -> tuple[View, float] | None:
    """Pan, tilt and zoom from a person's clicks, with the position already known.

    The same three numbers the segmenter's lines give, from better evidence: a landmark is
    an exact spot rather than a pixel somewhere along a marking.

    Three numbers need two landmarks and the wrong GOAL needs four, which is what is asked
    for -- see `MIN_SEED_CONSTRAINTS`, where the difference is measured.
    """
    from . import seed as seed_mod

    weight = len(clicked.points) * 2 + len(clicked.lines) + len(clicked.arcs)
    if weight < MIN_SEED_CONSTRAINTS:
        return None

    def residual(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        view = _view(p)
        h = matrix(camera, lens_at, view)
        if h is None:
            return np.full(weight, 1e3)
        miss = seed_mod._signed(h, clicked)
        return np.where(np.isfinite(miss), miss, 1e3)

    if start is not None:
        near = _solve(residual, ((start.pan, start.tilt, start.focal),))
        if near is not None and near[1] <= NEAR_ENOUGH_M:
            return near
    return _solve(residual, starts)


def between(a: View, b: View, t: float) -> View:
    """The aim `t` of the way from one to another. Zoom moves geometrically, as a lens does."""
    return View(
        pan=a.pan + (b.pan - a.pan) * t,
        tilt=a.tilt + (b.tilt - a.tilt) * t,
        focal=float(math.exp(math.log(a.focal) + (math.log(b.focal) - math.log(a.focal)) * t)),
    )


def spanned(views: dict[int, View], frames: list[int], max_gap: int) -> dict[int, View]:
    """Every frame's aim, with the ones nothing solved read off its neighbours.

    A pan is continuous, so a frame between two solved ones was aimed between them. This is
    what a carry does elsewhere (`stage1_propagate`), without a carry's drift: an
    interpolated aim is still a camera in the right place, off by however far the pan moved
    rather than by however much the chain has accumulated.
    """
    out = dict(views)
    solved = sorted(views)
    if not solved:
        return out
    for before, after in itertools.pairwise(solved):
        if after - before > max_gap:
            continue
        for f in frames:
            if before < f < after:
                out[f] = between(views[before], views[after], (f - before) / (after - before))
    return out


def fit(
    clicked: list[tuple[Any, tuple[float, float]]],
    *,
    pitch: Pitch = DEFAULT_PITCH,
    game: str = "",
    start: tuple[float, float, float] | None = None,
    aims: list[View] | None = None,
) -> tuple[Camera, list[View]] | None:
    """Where the camera stands, from several seeds of it -- each with its own lens centre.

    One seed cannot say: a frame is eight numbers of evidence and the camera's position is
    three of them, so a single view trades height against zoom and lands anywhere along
    that trade. Several views of DIFFERENT parts of the pitch cannot, because the position
    has to explain all of them at once.

    `clicked` is (seed, lens centre) so seeds from clips cropped differently can be fitted
    together. `start` and `aims` are where to begin -- a fit already made, when the question
    is how far leaving one view out moves it.

    Returns the camera and each seed's aim, in the order given -- a list and not a map by
    frame, because frame 56 exists in every clip and seeds from several are one fit. None if
    the seeds describe nothing.
    """
    from . import seed as seed_mod

    if len(clicked) < 2:
        return None
    guess = start if start is not None else (pitch.halfway, pitch.width + 35.0, 15.0)
    # Each view aimed on its own first, off the guessed position. Started from a sweep
    # instead, a fit of forty views walks until the solver gives up: one badly started view
    # pulls the shared position, which spoils every other view's aim.
    rough_rig = Camera(*guess, pitch=pitch)
    begun: list[tuple[float, float, float]] = []
    for i, (s, at) in enumerate(clicked):
        if aims is not None:
            begun.append((aims[i].pan, aims[i].tilt, aims[i].focal))
            continue
        own = aim_from_seed(rough_rig, at, s)
        if own is not None:
            begun.append((own[0].pan, own[0].tilt, own[0].focal))
        else:
            begun.append(_coarse_for(seed_mod.homography(s), at, rough_rig))

    def residual(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        camera = Camera(float(x[0]), float(x[1]), float(x[2]), pitch=pitch, game=game)
        out = []
        for (s, at), p in zip(clicked, x[3:].reshape(len(clicked), 3), strict=True):
            view = _view(p)
            h = matrix(camera, at, view)
            n = len(s.points) * 2 + len(s.lines) + len(s.arcs)
            miss = seed_mod._signed(h, s) if h is not None else np.full(n, 1e3)
            out.append(np.where(np.isfinite(miss), miss, 1e3))
        return np.concatenate(out) if out else np.zeros(0)

    x0 = [*guess]
    for pan, tilt, focal in begun:
        x0 += [pan, tilt, math.log(focal)]
    # Each seed's misses depend on the position and on its OWN aim, never on another seed's.
    # Said to the solver, a fit of forty frames costs what a fit of forty separate frames
    # does rather than forty times that.
    sizes = [len(s.points) * 2 + len(s.lines) + len(s.arcs) for s, _at in clicked]
    sparsity = np.zeros((sum(sizes), len(x0)), dtype=np.int8)
    row = 0
    for i, n in enumerate(sizes):
        sparsity[row : row + n, :3] = 1
        sparsity[row : row + n, 3 + 3 * i : 6 + 3 * i] = 1
        row += n
    # Metres, radians and a log-zoom are orders of magnitude apart; scaled by the Jacobian
    # the solver steps each by what it moves the misses, not by its units.
    got = least_squares(
        residual,
        x0,
        loss="soft_l1",
        f_scale=0.5,
        jac_sparsity=sparsity,
        method="trf",
        x_scale="jac",
    )
    camera = Camera(float(got.x[0]), float(got.x[1]), float(got.x[2]), pitch=pitch, game=game)
    if camera.height <= 0:
        return None
    return camera, [_view(p) for p in got.x[3:].reshape(len(clicked), 3)]


def _coarse_for(
    h: npt.NDArray[np.float64] | None, lens_at: tuple[float, float], camera: Camera
) -> tuple[float, float, float]:
    """The sweep's best match to a homography already fitted, as a start for the real fit."""
    from . import calibration

    if h is None:
        return (0.0, math.radians(12.0), 5500.0)
    grid = np.array(
        [[lens_at[0] * u, lens_at[1] * v] for u in (0.4, 1.0, 1.6) for v in (1.2, 1.6)],
        dtype=np.float64,
    )
    want = calibration.apply(h, grid)
    best, cost = (0.0, math.radians(12.0), 5500.0), math.inf
    for pan, tilt, focal in COARSE:
        got = matrix(camera, lens_at, View(pan, tilt, focal))
        if got is None:
            continue
        d = float(np.nanmedian(np.linalg.norm(calibration.apply(got, grid) - want, axis=1)))
        if math.isfinite(d) and d < cost:
            best, cost = (pan, tilt, focal), d
    return best


FILE = "camera.json"


def write(path: Path, camera: Camera) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "game": camera.game,
                "x": round(camera.x, 3),
                "y": round(camera.y, 3),
                "height": round(camera.height, 3),
                "pitch": {"length": camera.pitch.length, "width": camera.pitch.width},
            },
            indent=2,
        )
        + "\n"
    )
    return path


def read(path: Path) -> Camera:
    d = json.loads(path.read_text())
    p = d.get("pitch") or {}
    return Camera(
        x=float(d["x"]),
        y=float(d["y"]),
        height=float(d["height"]),
        pitch=Pitch(
            length=float(p.get("length", DEFAULT_PITCH.length)),
            width=float(p.get("width", DEFAULT_PITCH.width)),
        ),
        game=str(d.get("game", "")),
    )
