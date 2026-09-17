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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import numpy.typing as npt

from .config import DEFAULT_PITCH, Pitch

# The landmarks worth offering, in metres, for the goal at x=0. A clip showing the far
# goal is seeded with the same names mirrored, which `mirrored()` does, so a coach
# never has to think about which end the pitch model calls zero.
# FAR means away from the camera and NEAR means toward it - never left and right,
# which depend on where the camera is standing and are ambiguous on a screen. The
# broadcast camera sits on one touchline, so "near" is always the bottom of the frame.
# The centre circle's radius, for the two points where the halfway line crosses it.
CENTRE_R = 9.15


def landmarks(pitch: Pitch = DEFAULT_PITCH) -> dict[str, tuple[float, float]]:
    """The landmarks worth offering, in metres, for the goal at x=0.

    A clip showing the far goal is seeded with the same names mirrored, which `mirrored`
    does, so a coach never has to think about which end the pitch model calls zero.

    Everything here except the corners, the touchlines and the halfway line is fixed by
    the Laws and identical on every ground; what `pitch` moves is where the middle of the
    goal is, and how far away the far end and the far touchline are (D89).
    """
    mid = pitch.middle
    return {
        "goal post far": (0.0, mid - 3.66),
        "goal post near": (0.0, mid + 3.66),
        "6yd box far corner": (0.0, mid - 9.16),
        "6yd box near corner": (0.0, mid + 9.16),
        "6yd front far": (5.5, mid - 9.16),
        "6yd front near": (5.5, mid + 9.16),
        "penalty box far corner": (0.0, mid - 20.16),
        "penalty box near corner": (0.0, mid + 20.16),
        "penalty box front far": (16.5, mid - 20.16),
        "penalty box front near": (16.5, mid + 20.16),
        "penalty spot": (11.0, mid),
        "corner far": (0.0, 0.0),
        "corner near": (0.0, pitch.width),
        "halfway far": (pitch.halfway, 0.0),
        "halfway near": (pitch.halfway, pitch.width),
        # Where the halfway line crosses the centre circle. The two exact points a MIDFIELD
        # view offers and nothing else does: the circle itself is not traceable (a curve is
        # not a line) and its crossings are, which is the same trick the learned fitter uses
        # on predicted markings. Without them a camera parked on the halfway line has
        # evidence only where the paint is -- along one line and around the edges -- and
        # nothing fixes the scale through the middle of the picture, where the players are.
        "circle far": (pitch.halfway, mid - CENTRE_R),
        "circle near": (pitch.halfway, mid + CENTRE_R),
    }


def mirrored(name: str, pitch: Pitch = DEFAULT_PITCH) -> tuple[float, float]:
    """The same landmark at the other end of the pitch."""
    x, y = landmarks(pitch)[name]
    return (pitch.length - x, y)


def mirror_line(
    a: float, b: float, c: float, pitch: Pitch = DEFAULT_PITCH
) -> tuple[float, float, float]:
    """A pitch line reflected end to end. x = k becomes x = L - k; y = k is unchanged."""
    return (-a, b, c + a * pitch.length) if a else (a, b, c)


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
def traceable(pitch: Pitch = DEFAULT_PITCH) -> dict[str, tuple[float, float, float]]:
    """Named pitch lines a human can trace, and what they are in metres.

    Tracing beats clicking a corner: a corner is one exact pixel and is often out of
    shot, while a line you can see is easy to follow and just as informative once several
    points are stacked.
    """
    mid = pitch.middle
    return {
        "goal line": (1.0, 0.0, 0.0),
        "6yd box front": (1.0, 0.0, -5.5),
        "penalty box front": (1.0, 0.0, -16.5),
        "6yd box far side": (0.0, 1.0, -(mid - 9.16)),
        "6yd box near side": (0.0, 1.0, -(mid + 9.16)),
        "penalty box far side": (0.0, 1.0, -(mid - 20.16)),
        "penalty box near side": (0.0, 1.0, -(mid + 20.16)),
        "far touchline": (0.0, 1.0, 0.0),
        "near touchline": (0.0, 1.0, -pitch.width),
        # The one marking a MIDFIELD view always has, and the only line here that
        # mirroring leaves alone. Without it such a frame can only offer box lines and a
        # touchline -- every one of them in the same band across the picture, which is
        # unconstrained in depth and folds the fit over just below it (D34). A real clip
        # was refused for exactly that: `usable_seeds` saw 34% of the frame behind the
        # camera.
        "halfway line": (1.0, 0.0, -pitch.halfway),
    }


# Where each marking actually STOPS, in metres. A traceable line is stored as an
# infinite one because that is what the solver wants, but a diagram drawn from that
# claims the six-yard box runs the length of the pitch - which is what it was doing,
# pointing the coach at grass instead of at a line.
def extents(
    pitch: Pitch = DEFAULT_PITCH,
) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Where each marking actually STOPS, in metres.

    A traceable line is stored as an infinite one because that is what the solver wants,
    but a diagram drawn from that claims the six-yard box runs the length of the pitch --
    which is what it was doing, pointing the coach at grass instead of at a line.
    """
    mid = pitch.middle
    return {
        "goal line": ((0.0, 0.0), (0.0, pitch.width)),
        "6yd box front": ((5.5, mid - 9.16), (5.5, mid + 9.16)),
        "penalty box front": ((16.5, mid - 20.16), (16.5, mid + 20.16)),
        "6yd box far side": ((0.0, mid - 9.16), (5.5, mid - 9.16)),
        "6yd box near side": ((0.0, mid + 9.16), (5.5, mid + 9.16)),
        "penalty box far side": ((0.0, mid - 20.16), (16.5, mid - 20.16)),
        "penalty box near side": ((0.0, mid + 20.16), (16.5, mid + 20.16)),
        "far touchline": ((0.0, 0.0), (pitch.length, 0.0)),
        "near touchline": ((0.0, pitch.width), (pitch.length, pitch.width)),
        "halfway line": ((pitch.halfway, 0.0), (pitch.halfway, pitch.width)),
    }


def mirrored_extent(
    name: str, pitch: Pitch = DEFAULT_PITCH
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The same marking at the other end of the pitch."""
    (ax, ay), (bx, by) = extents(pitch)[name]
    return ((pitch.length - ax, ay), (pitch.length - bx, by))


def mirrored_line(name: str, pitch: Pitch = DEFAULT_PITCH) -> tuple[float, float, float]:
    """The same marking at the other end, by name."""
    return mirror_line(*traceable(pitch)[name], pitch=pitch)


def curves(pitch: Pitch = DEFAULT_PITCH) -> dict[str, tuple[float, float, float]]:
    """Painted circles a human can trace, as centre x, centre y and radius in metres.

    What a camera aimed at the far half from midfield has when the straight markings do
    not reach down the picture: the far touchline and a box corner sit in one band, the
    fit folds just below them (D34), and nothing lower is straight. The centre circle and
    the penalty arcs are lower. A point traced on one says "somewhere on this circle",
    one equation like a traced line -- but the circle bends through DEPTH, which is the
    thing a band of straight markings cannot pin down.

    The penalty arc is named by the circle it lies on, centred on the spot; only the part
    outside the box is painted, which matters for the diagram and not for the fit.
    """
    return {
        "centre circle": (pitch.halfway, pitch.middle, CENTRE_R),
        "penalty arc": (11.0, pitch.middle, CENTRE_R),
    }


def mirrored_curve(name: str, pitch: Pitch = DEFAULT_PITCH) -> tuple[float, float, float]:
    """The same curve at the other end. The centre circle is its own mirror image."""
    cx, cy, r = curves(pitch)[name]
    return (pitch.length - cx, cy, r)


def curve_extent(
    name: str, far_goal: bool, pitch: Pitch = DEFAULT_PITCH
) -> npt.NDArray[np.float64]:
    """The painted part of a curve, as a polyline in metres, for the diagram.

    The penalty arc's circle runs on into the box, where nothing is painted -- the same
    reason `extents` exists for the straight markings.
    """
    cx, cy, r = mirrored_curve(name, pitch) if far_goal else curves(pitch)[name]
    if name == "penalty arc":
        half = math.degrees(math.acos((16.5 - 11.0) / r))
        centre = 180.0 if far_goal else 0.0
        angles = np.radians(np.linspace(centre - half, centre + half, 40))
    else:
        angles = np.radians(np.linspace(0.0, 360.0, 80))
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])


Region = Literal["both", "goal end", "midfield"]

# The order `r` cycles them in. "both" first: it is the order the click tool always had,
# and the only honest one when nothing says where the camera is looking.
REGIONS: tuple[Region, ...] = ("both", "goal end", "midfield")

# What only a view of the middle of the pitch offers. Everything else belongs to a goal
# end, except the touchlines, which run through both.
MIDFIELD = frozenset(
    {"halfway far", "halfway near", "circle far", "circle near", "halfway line", "centre circle"}
)
EITHER = frozenset({"far touchline", "near touchline"})


def ordered(names: list[str], region: Region) -> list[str]:
    """`names` with what `region` shows first, then the touchlines, then everything else.

    Reordered, never filtered. The region is a guess -- a key press, or a carried camera
    that may have drifted a whole region away (D34) -- so a wrong one costs a few presses
    of `n` and cannot hide the marking that is really in shot.
    """
    if region == "both":
        return list(names)
    own = [n for n in names if n not in EITHER and (n in MIDFIELD) == (region == "midfield")]
    shared = [n for n in names if n in EITHER]
    return own + shared + [n for n in names if n not in own and n not in shared]


# Points tried along each straight marking.
REGION_SAMPLES = 40


def region_in_view(
    h: npt.NDArray[np.float64], width: int, height: int, pitch: Pitch = DEFAULT_PITCH
) -> Region:
    """Which region's markings a camera puts inside the frame; "both" for both or neither.

    `h` maps image to pitch, as a carried camera does. Either goal end counts, because
    which one is the `e` key's question and is not answered here.

    A marking behind the lens projects to a pixel as well, so a sample only counts where
    its homogeneous scale has the sign of the bottom of the frame -- ground in any
    broadcast view. `h @ inv(h)` is the identity, so the pitch scale at a sample's pixel
    is `1 / w`, and its sign is the sign of `w`.
    """
    try:
        inv = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return "both"
    ground = float(h[2] @ np.array([width / 2.0, float(height), 1.0]))
    if not np.isfinite(ground) or abs(ground) < 1e-12:
        return "both"

    def line(a: tuple[float, float], b: tuple[float, float]) -> npt.NDArray[np.float64]:
        t = np.linspace(0.0, 1.0, REGION_SAMPLES)[:, None]
        return np.array(a, dtype=np.float64) * (1.0 - t) + np.array(b, dtype=np.float64) * t

    def shown(polylines: list[npt.NDArray[np.float64]]) -> bool:
        pts = np.vstack(polylines)
        u, v, w = (np.column_stack([pts, np.ones(len(pts))]) @ inv.T).T
        front = np.sign(w) == np.sign(ground)
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = u / w, v / w
        inside = front & (x >= 0) & (x <= width) & (y >= 0) & (y <= height)
        return bool(inside.any())

    straight = extents(pitch)
    goal = [n for n in straight if n not in MIDFIELD and n not in EITHER]
    goal_end = [line(*straight[n]) for n in goal]
    goal_end += [line(*mirrored_extent(n, pitch)) for n in goal]
    goal_end += [curve_extent("penalty arc", far, pitch) for far in (False, True)]
    midfield = [line(*straight["halfway line"]), curve_extent("centre circle", False, pitch)]

    at_goal, at_middle = shown(goal_end), shown(midfield)
    if at_goal == at_middle:
        return "both"
    return "goal end" if at_goal else "midfield"


@dataclass(slots=True)
class Seed:
    frame: int
    points: list[tuple[tuple[float, float], tuple[float, float]]]  # (image, pitch)
    lines: list[tuple[tuple[float, float], tuple[float, float, float]]] = field(
        default_factory=list
    )  # (image, pitch line)
    # A fingerprint of the picture these clicks were made on. See `fingerprint`.
    image: str | None = None
    arcs: list[tuple[tuple[float, float], tuple[float, float, float]]] = field(
        default_factory=list
    )  # (image, pitch circle as centre x, centre y, radius)
    # Image -> pitch. The camera a fit through curves is refined FROM, which the other seeds
    # carried to this frame when it was clicked. Kept in the file because `usable_seeds`
    # refits every seed from its file, and a refinement is only reproducible from the
    # same start. Flips leave it alone: it came from seeds already settled, so when a
    # flip corrects the clicks it is the clicks that were wrong, not the start.
    start: npt.NDArray[np.float64] | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "version": 1,
            "frame": self.frame,
            **({"image": self.image} if self.image else {}),
            "points": [
                {"image": [round(ix, 1), round(iy, 1)], "pitch": [px, py]}
                for (ix, iy), (px, py) in self.points
            ],
            "lines": [
                {"image": [round(ix, 1), round(iy, 1)], "line": list(ln)}
                for (ix, iy), ln in self.lines
            ],
            **(
                {
                    "arcs": [
                        {"image": [round(ix, 1), round(iy, 1)], "circle": list(circle)}
                        for (ix, iy), circle in self.arcs
                    ]
                }
                if self.arcs
                else {}
            ),
            **({"start": self.start.tolist()} if self.start is not None else {}),
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
        image=d.get("image"),
        arcs=[
            (
                (float(p["image"][0]), float(p["image"][1])),
                (float(p["circle"][0]), float(p["circle"][1]), float(p["circle"][2])),
            )
            for p in d.get("arcs", [])
        ],
        start=np.array(d["start"], dtype=np.float64) if "start" in d else None,
    )


# Below this the pitch y axis and the image y axis are judged to disagree. Above the
# positive version, they agree. Between, the camera is looking along the pitch rather
# than across it and the test cannot tell - a seed from behind the goal, say.
ORIENTATION_CONFIDENT = 0.5

# How much of the frame may map behind the camera before the fit is refused.
#
# Not a tuning knob so much as a statement that a picture cannot be mostly behind the
# lens. Measured on the three seeds clicked for the Nottingham clip: the two good ones
# put 0% of the frame there and the bad one 67%, so anything in between is a wide
# margin rather than a boundary anyone has to defend.
MAX_BEHIND_CAMERA = 0.25

# How many of the fingerprint's 64 bits may differ before two pictures are called
# different. JPEG noise moves one or two; a different frame of the same shot moves a
# handful; a different match moves thirty.
MAX_UNLIKE_BITS = 12


def fingerprint(img: Any) -> str:
    """A short hash of what a frame LOOKS like, so a seed can be checked against it.

    A seed states a frame NUMBER, and frame 56 exists in every clip -- so a seed clicked
    on one match anchors the next one silently, in a coordinate frame that has nothing to
    do with it. Nothing downstream can tell: the tracks and the board share the wrong
    space, so every fidelity number stays good while the football happens somewhere else
    (D34, and it happened to a coach on his second clip).

    A difference hash: each bit says whether one cell of a coarse grey grid is brighter
    than the one to its right. Robust to compression and exposure, which is what makes it
    a test of "the same picture" rather than "the same bytes".
    """
    import cv2

    grid = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (grid[:, 1:] > grid[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def unlike(a: str, b: str) -> int:
    """How many bits two fingerprints differ by, or 64 if either is unreadable."""
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 64


BEHIND_GRID = 20


def behind_camera(h: npt.NDArray[np.float64], width: int, height: int) -> float:
    """Fraction of the frame this homography maps behind the camera.

    The check the residuals cannot make. A seed clicked entirely within a thin band of
    the frame is unconstrained in depth: it fits its own evidence to a few centimetres
    and folds over immediately below it, putting the horizon inside the picture and
    players ninety metres off the end of the pitch. Three clicked points across the top
    of a broadcast frame did exactly that, and every residual it reported was under half
    a metre (D34).

    `h` maps image to pitch, so `h[2] . p` is the homogeneous scale: where it changes
    sign the ground plane has passed through infinity, and everything past it is behind
    the lens.
    """
    a, b, c = h[2]
    xs = np.linspace(0.0, float(width), BEHIND_GRID)
    ys = np.linspace(0.0, float(height), BEHIND_GRID)
    pts = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)
    scale = pts @ np.array([a, b], dtype=np.float64) + c
    # Sign is arbitrary; what matters is that the frame does not straddle the horizon.
    behind = float((scale <= 0).mean())
    # Sign is arbitrary, so the question is how far the frame STRADDLES the horizon:
    # all-one-side is a fit that describes the whole picture, either way round.
    return min(behind, 1.0 - behind)


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
    # Traced points count, and leaving them out was a hole: a seed made of traced lines
    # with a click or two never reached the three landmarks this asked for, so the check
    # that exists to catch a swap silently returned "no opinion" on exactly the seeds most
    # likely to hold one. A point traced along a constant-y marking knows its pitch y as
    # well as a landmark does -- that is what makes the marking traceable.
    samples = [(p[0][1], p[1][1]) for p in seed.points]
    samples += [
        (img[1], -c / b) for img, (a, b, c) in seed.lines if abs(b) > abs(a) and abs(b) > 1e-9
    ]
    if len(samples) < 3:
        return 0.0
    image_y = np.array([s[0] for s in samples], dtype=np.float64)
    pitch_y = np.array([s[1] for s in samples], dtype=np.float64)
    if image_y.std() < 1e-9 or pitch_y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(image_y, pitch_y)[0, 1])


# The smallest projected triangle worth trusting a sign from, in square metres. Below
# this the fit is degenerate and the handedness is noise rather than an answer.
MIN_HANDED_AREA = 1.0


def handedness(h: npt.NDArray[np.float64], width: int, height: int) -> float:
    """Which way round the ground plane is, as seen through this camera. -1, 0 or +1.

    A camera cannot get underneath a football pitch, so every frame of a clip sees the
    plane from the same side and the image-to-pitch map has the same handedness
    throughout -- panning, zooming and tilting cannot change it. Labelling the clicks
    with the WRONG END does change it, because reflecting pitch x while far and near
    stay pinned to the image is a reflection, and a reflection reverses handedness.

    So this is the end check, and it is the exact sibling of `orientation`: between them
    they cover every way a symmetric pitch can be mislabelled. A pitch has three
    non-identity symmetries -- flip x, flip y, and both -- and `orientation` catches the
    two that move y while this catches the two that move x.

    Unlike `orientation` it cannot judge one seed alone: a clip's handedness depends on
    which touchline the camera sits on, so the answer is only meaningful against the
    other seeds of the SAME clip.

    Zero where the fit is too degenerate to have an opinion.
    """
    quarter = np.array(
        [[[width / 2.0, height / 2.0], [width * 0.75, height / 2.0], [width / 2.0, height * 0.75]]],
        dtype=np.float64,
    )
    (ax, ay), (bx, by), (cx, cy) = cv2.perspectiveTransform(quarter, h)[0]
    area = ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2.0
    if not np.isfinite(area) or abs(area) < MIN_HANDED_AREA:
        return 0.0
    return 1.0 if area > 0 else -1.0


def flip_x(seed: Seed, pitch: Pitch = DEFAULT_PITCH) -> Seed:
    """The same clicks, at the other end of the pitch. Fixes a seed clicked on the goal
    the tool was not offering -- the `e` key in the click tool, applied after the fact."""
    return Seed(
        frame=seed.frame,
        points=[(img, (pitch.length - px, py)) for img, (px, py) in seed.points],
        lines=[(img, mirror_line(a, b, c, pitch)) for img, (a, b, c) in seed.lines],
        image=seed.image,
        arcs=[(img, (pitch.length - cx, cy, r)) for img, (cx, cy, r) in seed.arcs],
        start=seed.start,
    )


def flip_y(seed: Seed, pitch: Pitch = DEFAULT_PITCH) -> Seed:
    """The same clicks, with the pitch y axis reflected. Fixes a swapped far/near."""
    return Seed(
        frame=seed.frame,
        points=[(img, (px, pitch.width - py)) for img, (px, py) in seed.points],
        # A line a*x + b*y + c = 0 reflected in y = W/2 becomes a*x - b*y + (c + b*W).
        lines=[(img, (a, -b, c + b * pitch.width)) for img, (a, b, c) in seed.lines],
        # Kept: `settle` stamps a seed before it flips one, and dropping the stamp here left
        # every far/near-corrected seed unguarded against anchoring another clip (D34).
        image=seed.image,
        arcs=[(img, (cx, pitch.width - cy, r)) for img, (cx, cy, r) in seed.arcs],
        start=seed.start,
    )


def traced_name(line: tuple[float, float, float], pitch: Pitch = DEFAULT_PITCH) -> str:
    """What a traced line is called, so a complaint about it can name it."""
    named = traceable(pitch)
    for name, here in named.items():
        if line == here:
            return name
        if line == mirrored_line(name, pitch):
            return f"{name} (far end)"
    return f"the line {line}"


def contradictions(seed: Seed) -> list[tuple[str, str]]:
    """Traced markings whose order in the PICTURE contradicts their order on the pitch.

    Nearer the camera is larger pitch y and lower in the frame, so two constant-y markings
    must run the same way in both. When they do not, one of them is labelled with the wrong
    side -- and that is a mistake the fit cannot absorb and the reprojection cannot show,
    because the fit that comes back is a compromise between two contradictory claims.

    Named rather than measured: "the near touchline is above the far touchline" is a
    sentence somebody can act on, where "no usable seed" is not.
    """
    rows: dict[tuple[float, float, float], list[float]] = {}
    for img, line in seed.lines:
        a, b, _c = line
        if abs(b) <= abs(a) or abs(b) < 1e-9:
            continue
        rows.setdefault(line, []).append(img[1])

    marks = [
        (float(np.median(ys)), -c / b, traced_name((a, b, c))) for (a, b, c), ys in rows.items()
    ]
    out: list[tuple[str, str]] = []
    for i, (image_y, pitch_y, name) in enumerate(marks):
        for other_image_y, other_pitch_y, other in marks[i + 1 :]:
            if (pitch_y - other_pitch_y) * (image_y - other_image_y) < 0:
                out.append((name, other))
    return out


def settled_handedness(others: list[Path], frames_dir: Path) -> float:
    """Which way round this clip's existing seeds see the pitch, if they agree.

    Zero when there are none, when none has an opinion, or when they disagree among
    themselves -- in the last case there is already a mislabelled seed in the folder and
    guessing which one from a new arrival would be picking a side.
    """
    seen: set[float] = set()
    for path in others:
        seeded = read(path)
        img = cv2.imread(str(frames_dir / f"{seeded.frame:06d}.jpg"))
        h = homography(seeded)
        if img is None or h is None:
            continue
        hand = handedness(h, img.shape[1], img.shape[0])
        if hand:
            seen.add(hand)
    return seen.pop() if len(seen) == 1 else 0.0


def settle(
    got: Seed,
    img: Any,
    others: list[Path],
    frames_dir: Path,
    *,
    pitch: Pitch = DEFAULT_PITCH,
) -> tuple[Seed, list[str]]:
    """A fresh set of clicks, checked against everything except the clicks themselves.

    Every check here is one the reprojection overlay CANNOT make, which is why they run
    before anything is written. A pitch is symmetric about the halfway line and again end
    to end, so a mirrored model or a seed clicked on the wrong goal draws onto the real
    markings perfectly, reports small residuals, and anchors the play 105 m away (D88).
    Between them `orientation` and `handedness` cover all three symmetries; neither alone
    does.

    `others` is every OTHER seed of this clip -- the file being written is excluded by
    the caller, because a seed compared with itself always agrees.

    Returns the settled seed and what to tell whoever clicked it. Messages rather than
    echoes so the same checks serve `ft seed` and `ft run`.
    """
    notes: list[str] = []
    # Stamped with the picture it was clicked on, so it can never anchor another clip.
    got = replace(got, image=fingerprint(img))

    agreement = orientation(got)
    if agreement < -ORIENTATION_CONFIDENT:
        notes.append("far and near look swapped - flipping the pitch y axis to match the camera")
        got = flip_y(got, pitch)
    elif agreement < ORIENTATION_CONFIDENT:
        notes.append(
            "WARNING: cannot tell which side the camera is on from these points."
            " If the board comes out mirrored, far and near are swapped."
        )

    # The END check, which needs the clip rather than the clicks. The camera cannot get
    # under the pitch, so every seed of one clip must see the ground plane the same way.
    #
    # A seed fitted through curves carries its own reference: the start was carried here
    # from seeds already settled, so it sees the ground the right way round. A MISFIT
    # cannot judge the end -- labelled with the wrong goal, the fit finds the mirror-image
    # camera and matches every click as well as the right one does, because the pitch is
    # symmetric. Only handedness tells the two apart.
    fitted = homography(got)
    if got.arcs and got.start is not None:
        mine = handedness(fitted, img.shape[1], img.shape[0]) if fitted is not None else 0.0
        theirs = handedness(got.start, img.shape[1], img.shape[0])
        if mine and theirs and mine != theirs:
            notes.append(
                "the goal in shot is the other one - flipping the pitch end to match the"
                " seeds already clicked for this clip"
            )
            got = flip_x(got, pitch)
    elif fitted is not None:
        mine = handedness(fitted, img.shape[1], img.shape[0])
        theirs = settled_handedness(others, frames_dir)
        if mine and theirs and mine != theirs:
            notes.append(
                "the goal in shot is the other one - flipping the pitch end to match the"
                " seeds already clicked for this clip"
            )
            got = flip_x(got, pitch)

    for a, b in contradictions(got):
        notes.append(
            f"WARNING: {a} and {b} are labelled with the wrong side of the pitch -"
            " nearer the camera is LOWER in the frame, and these two run the other way."
        )

    directions = {abs(a) > abs(b) for _, (a, b, _c) in got.lines}
    if got.lines and len(directions) == 1:
        notes.append(
            "NOTE: every traced line runs the same way. Lines parallel to each other pin"
            " down nothing across them - the exact points are carrying the fit."
        )
    return got, notes


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

    Traced curves go through `_through_curves` instead, which cannot be one linear fit.
    """
    import cv2

    from . import calibration

    if seed.arcs:
        return _through_curves(seed)

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
    if not seed.points and not _two_lines_each_way(seed):
        return None

    fitted = calibration.fit(seed.points, seed.lines, 0, 0)
    if fitted is None:
        return None
    h: npt.NDArray[np.float64] = np.asarray(fitted, dtype=np.float64)
    if _collapses(h, seed):
        return None

    # Trim ONE AT A TIME, worst first, rather than dropping everything that looks bad
    # in a single pass. A least-squares fit spreads a bad click's error over every other
    # point, so with two landmarks swapped EVERY residual exceeded the threshold - the
    # single-pass version dropped all eleven, found what was left degenerate, and handed
    # back the bad fit it was trying to repair. Measured on a real seed: 1.82 m from the
    # markings before, 0.2 m after.
    #
    # A drop is only kept while what remains still says something about both directions
    # and still has more constraints than degrees of freedom (D17).
    #
    # This repairs a seed whose GEOMETRY is sound and whose clicks are not. It cannot
    # rescue one where every traced marking runs the same way: there the lines pin down
    # depth and nothing pins down across, so a pair of landmarks swapped across the pitch
    # fits as well as the truth does, and no amount of trimming can prefer one. Trace a
    # line that crosses the others.
    for _ in range(len(seed.points)):
        point_res = _point_residuals(h, seed)
        line_res = _line_residuals(h, seed)
        worst_point = max(range(len(point_res)), default=-1, key=lambda i: point_res[i])
        worst_line = max(range(len(line_res)), default=-1, key=lambda i: line_res[i])
        pw = point_res[worst_point] if point_res else 0.0
        lw = line_res[worst_line] if line_res else 0.0
        if max(pw, lw) <= MISCLICK_METRES:
            break

        if pw >= lw:
            kept = Seed(
                frame=seed.frame,
                points=[p for i, p in enumerate(seed.points) if i != worst_point],
                lines=seed.lines,
            )
        else:
            kept = Seed(
                frame=seed.frame,
                points=seed.points,
                lines=[ln for i, ln in enumerate(seed.lines) if i != worst_line],
            )
        if not _spans_two_directions(kept) or len(kept.points) * 2 + len(kept.lines) < 8:
            break
        refit = calibration.fit(kept.points, kept.lines, 0, 0)
        if refit is None:
            break
        candidate = np.asarray(refit, dtype=np.float64)
        if _collapses(candidate, kept):
            break
        h, seed = candidate, kept

    return np.asarray(h, dtype=np.float64)


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

    image = np.array(
        [p[0] for p in seed.points] + [p[0] for p in seed.lines] + [p[0] for p in seed.arcs],
        dtype=np.float64,
    )
    if len(image) < 3:
        return True
    got = calibration.apply(h, image)
    if not np.all(np.isfinite(got)):
        return True
    spread = np.linalg.svd(got - got.mean(axis=0), compute_uv=False)
    return bool(spread[0] < MIN_SPREAD_M or spread[1] / spread[0] < MIN_SPREAD_RATIO)


def _signed(h: npt.NDArray[np.float64], seed: Seed) -> npt.NDArray[np.float64]:
    """Every clicked constraint's miss, in metres and with its sign, in one vector.

    Two per landmark (x and y), one per traced point on a line, one per traced point on a
    curve. Signed rather than a distance so a least-squares solver has a smooth surface to
    walk down -- a distance has a kink at zero, right where the answer is.
    """
    from . import calibration

    out: list[npt.NDArray[np.float64]] = []
    if seed.points:
        got = calibration.apply(h, np.array([p[0] for p in seed.points], dtype=np.float64))
        out.append((got - np.array([p[1] for p in seed.points], dtype=np.float64)).ravel())
    if seed.lines:
        got = calibration.apply(h, np.array([p[0] for p in seed.lines], dtype=np.float64))
        abc = np.array([p[1] for p in seed.lines], dtype=np.float64)
        dist = abc[:, 0] * got[:, 0] + abc[:, 1] * got[:, 1] + abc[:, 2]
        out.append(dist / np.maximum(1e-9, np.hypot(abc[:, 0], abc[:, 1])))
    if seed.arcs:
        got = calibration.apply(h, np.array([p[0] for p in seed.arcs], dtype=np.float64))
        circle = np.array([p[1] for p in seed.arcs], dtype=np.float64)
        out.append(np.hypot(got[:, 0] - circle[:, 0], got[:, 1] - circle[:, 1]) - circle[:, 2])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


def misfit(h: npt.NDArray[np.float64], seed: Seed) -> float:
    """The median miss of a seed's clicks under a camera, in metres."""
    miss = np.abs(_signed(h, seed))
    return float(np.median(miss)) if miss.size and np.all(np.isfinite(miss)) else math.inf


def _through_curves(seed: Seed) -> npt.NDArray[np.float64] | None:
    """A camera fitted to clicks that include traced curves, refined from a start.

    "On this circle" is quadratic in the camera, so it cannot join the one linear fit the
    straight markings use; the fit starts from a camera and improves it. Two starts are
    tried and the better result kept. `seed.start` is what the frame this exists for
    needs: its straight markings all sit in one band, so their own fit is the fold
    (D34) and refining from it goes nowhere. The straight clicks' own fit is what a first
    seed has, when there is nothing to carry from.

    The loss is robust at a misclick's width: a curve takes many clicks along a line
    players stand on, and one on a boot should move nothing.
    """
    from scipy.optimize import least_squares

    # A circle looks the same from every angle round its centre, so on its own it fixes no
    # rotation, and beside a single straight marking it can still reflect across it. Two
    # pieces of straight evidence break both.
    straight = len(seed.points) + len({ln for _, ln in seed.lines})
    if straight < 2 or len(seed.points) * 2 + len(seed.lines) + len(seed.arcs) < 8:
        return None

    plain = homography(Seed(frame=seed.frame, points=seed.points, lines=seed.lines))
    starts = [h for h in (seed.start, plain) if h is not None and abs(h[2, 2]) > 1e-12]

    def residuals(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        miss = _signed(np.append(v, 1.0).reshape(3, 3), seed)
        return np.nan_to_num(miss, nan=1e6, posinf=1e6, neginf=-1e6)

    best: npt.NDArray[np.float64] | None = None
    best_cost = math.inf
    for h0 in starts:
        try:
            sol = least_squares(
                residuals,
                (h0 / h0[2, 2]).ravel()[:8],
                loss="soft_l1",
                f_scale=MISCLICK_METRES,
                x_scale="jac",
            )
        except ValueError:
            continue
        if np.all(np.isfinite(sol.x)) and float(sol.cost) < best_cost:
            best, best_cost = np.asarray(sol.x, dtype=np.float64), float(sol.cost)
    if best is None:
        return None
    h: npt.NDArray[np.float64] = np.append(best, 1.0).reshape(3, 3)
    return None if _collapses(h, seed) else h


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


def _two_lines_each_way(seed: Seed) -> bool:
    """Whether traced lines alone can fix the SCALE in both directions.

    Spanning two directions is not enough when the evidence is lines only. Two parallel
    markings give the vanishing point and the scale between them; a single crossing line
    gives an origin along the other axis and nothing about its scale, so the camera is
    free to stretch along it and the fit that comes back is confidently wrong rather than
    refused. The halfway line and the two touchlines -- what a midfield camera shows --
    do exactly that: the fit lands the centre spot at 15 m across a 68 m pitch.

    The same rule the learned fitter uses on predicted markings, for the same reason
    (`calibration.MIN_LINES_PER_AXIS`). A landmark escapes it, because an exact point
    fixes both scales where a line fixes one.
    """
    seen = {ln for _, ln in seed.lines}
    across = sum(1 for a, b, _c in seen if abs(a) > abs(b))
    return min(across, len(seen) - across) >= 2


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
