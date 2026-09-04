"""A learned pitch-line segmenter, and the camera model fitted from it.

Why this exists, when `refine.py` already snaps a homography onto the paint: finding the
paint was never the problem. `refine.line_pixels` locates it to a median of 0.00 m under a
correct homography. What it cannot do is say WHICH marking a white pixel belongs to -- it
infers that from the homography it is trying to correct, which is why it converges only
from inside a two-metre capture radius and why it made the pipeline worse (D35).

A segmenter answers exactly that question and nothing else. Given a pixel it names the
marking, so correspondences stop being a guess and a homography can be fitted per frame
from nothing at all: no seed, no carry, and therefore no drift. The three blockers on
broadcast footage are one problem wearing three hats.

Training data is SN-GSR-2025, which is broadcast footage with per-frame line annotations
already in the format `calibration.lines_of` reads. It is used ONLY to train: at inference
the model sees the user's own clips and SoccerNet is never involved again.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from . import calibration

# The 26 markings SoccerNet labels, in a FIXED order: the index is baked into trained
# weights, so appending is safe and reordering silently relabels every prediction.
CLASSES: tuple[str, ...] = (
    "Big rect. left bottom",
    "Big rect. left main",
    "Big rect. left top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Big rect. right top",
    "Circle central",
    "Circle left",
    "Circle right",
    "Goal left crossbar",
    "Goal left post left",
    "Goal left post right",
    "Goal right crossbar",
    "Goal right post left",
    "Goal right post right",
    "Middle line",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Side line top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Small rect. left top",
    "Small rect. right bottom",
    "Small rect. right main",
    "Small rect. right top",
)
# 0 is background, so a class's channel is its index here plus one. Names are looked up
# STRIPPED: SN-Calibration-2023 writes "Goal left post left " with a trailing space, and an
# exact lookup drops it as an unknown marking -- 1,101 instances in the train split, and a
# class that then never appears in a single label. It fails as silence rather than as an
# error, which is why the strip lives at the one lookup site instead of at each caller.
INDEX: dict[str, int] = {name: i + 1 for i, name in enumerate(CLASSES)}
N_CLASSES = len(CLASSES) + 1

# Training resolution, and the hyperparameter that actually decides the answer. A line is
# thin, so the question is how many metres one predicted pixel is worth: at 640x360 a
# 3 px line upscales to a 9 px band on a 1920x1080 frame, and at the far end of the pitch
# that band is over a metre wide. Measured on held-out matches, the segmenter's pixels sit
# 0.26 m from their line where they are near the camera and 1.45 m where they are far --
# and only 3-9% of them are mislabelled, so precision and not naming is the limit (D36).
# GSR is 1920x1080 on disk, so 960x540 threw away half the linear resolution of the only
# footage the benchmark is scored on. 1280x720 buys that back. It is NOT free to raise
# further: SN-Calibration-2023 is natively 960x540, so anything above that upsamples 82%
# of the combined set and teaches the model to expect blur it will not meet at inference.
# Train above 960x540 on GSR alone (`--no-extra`), or not at all.
WIDTH, HEIGHT = 1280, 720
# Scales WITH the resolution rather than being a constant. The label is a deliberate
# dilation of a line that is 2-3 px wide at 960x540, and the same real line covers 1.33x
# more pixels at 1280x720 -- so holding this at 4 would quietly make the target thinner
# relative to the mark it describes, which is a second change dressed up as no change.
LINE_PX = 5


def rasterise(
    lines: dict[str, list[dict[str, float]]], width: int = WIDTH, height: int = HEIGHT
) -> npt.NDArray[np.uint8]:
    """A class index per pixel, from SoccerNet's normalised polylines.

    Drawn longest-marking-last so that where two markings cross, the one with more
    evidence wins the pixel. Which one wins matters less than that it is deterministic:
    an arbitrary tie makes the same crossing a different label from frame to frame, and
    the network learns the noise.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    ordered = sorted(lines.items(), key=lambda kv: len(kv[1]))
    for name, points in ordered:
        index = INDEX.get(name.strip())
        if index is None or len(points) < 2:
            continue
        pts = np.array([[p["x"] * width, p["y"] * height] for p in points], dtype=np.int32).reshape(
            -1, 1, 2
        )
        cv2.polylines(mask, [pts], False, index, LINE_PX, cv2.LINE_8)
    return mask


@dataclass(slots=True)
class Frame:
    """One training example: where the picture is, and the lines drawn on it."""

    image: Path
    lines: dict[str, list[dict[str, float]]]
    game: str


def index_clips(clips_dir: Path, names: list[str] | None = None) -> list[Frame]:
    """Every annotated frame on disk, tagged with the match it came from.

    The game tag is load-bearing. Two clips of the same match share a stadium, a camera
    and a kit, so splitting by CLIP puts near-duplicates on both sides of the split and
    reports a generalisation that was never tested. SNGS-116 and SNGS-121 are both game 7.
    """
    out: list[Frame] = []
    for clip_dir in sorted(p for p in clips_dir.iterdir() if p.is_dir()):
        if names is not None and clip_dir.name not in names:
            continue
        labels = clip_dir / "Labels-GameState.json"
        if not labels.exists():
            continue
        doc = json.loads(labels.read_text())
        game = str(doc.get("info", {}).get("game_id", doc.get("game_id", clip_dir.name)))
        by_id = {img["image_id"]: img["file_name"] for img in doc["images"]}
        for a in doc["annotations"]:
            if a.get("category_id") != 5:
                continue
            name = by_id.get(a["image_id"])
            lines = calibration.lines_of(a)
            if name is None or not lines:
                continue
            path = clip_dir / "img1" / Path(name).name
            if path.exists():
                out.append(Frame(image=path, lines=lines, game=game))
    return out


def _match_key(meta: dict[str, Any]) -> str:
    """The match an image came from, in `per_match_info.json`'s own key format."""
    return "-".join(str(meta.get(k, "")) for k in ("league", "season", "match", "date"))


def index_calibration(root: Path, split: str = "train") -> list[Frame]:
    """Every annotated frame of SN-Calibration-2023, tagged with the match it came from.

    A second reader rather than a widening of `index_clips`, because the two datasets state
    the same fact in different shapes: GSR carries its lines inside one document per clip,
    and this one writes a sidecar of normalised polylines beside each image. Both arrive as
    a `Frame`, so nothing downstream learns there are two sources.

    `match_info.json` is what makes the set usable rather than merely large. It names the
    fixture behind every image -- 290 matches over six leagues and three seasons, against
    the five anonymous ones GSR supplied. Diversity was the binding constraint on the first
    run (D36), and the same tag is what keeps `split_by_game` able to hold a match out.

    Images are natively 960x540. That matched WIDTH x HEIGHT exactly while training ran
    at that size; above it they are UPSAMPLED, which adds no detail and costs domain
    fidelity, so this source belongs only in a run at 960x540 or below.
    """
    d = root / split
    info_path = d / "match_info.json"
    info: dict[str, Any] = json.loads(info_path.read_text()) if info_path.exists() else {}

    out: list[Frame] = []
    for image in sorted(d.glob("*.jpg")):
        sidecar = image.with_suffix(".json")
        if not sidecar.exists():
            continue
        lines = json.loads(sidecar.read_text())
        if not lines:
            continue
        meta = info.get(image.name)
        out.append(
            Frame(image=image, lines=lines, game=_match_key(meta) if meta else f"{split}-unknown")
        )
    return out


def split_by_game(
    frames: list[Frame], holdout: int = 1, games: set[str] | None = None
) -> tuple[list[Frame], list[Frame]]:
    """Train and validation sets that share no match.

    `games` names the matches to hold out. It is worth naming them rather than taking the
    last N, because the clips this project BENCHMARKS on are particular ones: train on the
    match a benchmark clip came from and every number it reports afterwards is measured on
    footage the model has already seen.
    """
    if games is not None:
        return (
            [f for f in frames if f.game not in games],
            [f for f in frames if f.game in games],
        )
    names = sorted({f.game for f in frames})
    held = set(names[-holdout:]) if len(names) > holdout else set()
    return [f for f in frames if f.game not in held], [f for f in frames if f.game in held]


def batches(
    frames: list[Frame], size: int, *, shuffle: bool = True, seed: int = 0
) -> Iterator[Any]:
    """Images and masks, as torch tensors, without holding the set in memory."""
    import torch

    order = np.arange(len(frames))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), size):
        chunk = order[start : start + size]
        imgs = []
        masks = []
        for i in chunk:
            f = frames[int(i)]
            bgr = cv2.imread(str(f.image))
            if bgr is None:
                continue
            bgr = cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            imgs.append(np.transpose(rgb, (2, 0, 1)))
            masks.append(rasterise(f.lines))
        if not imgs:
            continue
        # Contiguous explicitly. Transposing HWC to CHW leaves a strided view, and MPS
        # throws "view size is not compatible" from inside the BACKWARD pass -- far from
        # the transpose, and only on this backend, because CPU and CUDA tolerate it.
        yield (
            torch.from_numpy(np.ascontiguousarray(np.stack(imgs))),
            torch.from_numpy(np.ascontiguousarray(np.stack(masks))).long(),
        )


WEIGHTS = Path("work/calib/segmenter.pt")

# A class needs this many predicted pixels before it is believed to be in shot. Below it
# the marking is a few stray pixels of another one, and a wrong correspondence is worse
# than a missing one -- it is confidently in the wrong place.
MIN_PIXELS = 60
# How many pixels of one marking are actually used. A near touchline can be 20,000 pixels
# and a far one 200, and the fit would then be a vote about which is closer to the camera.
PER_CLASS = 120


def model(weights: Path | None = None) -> Any:
    """DeepLabv3 on a MobileNetV3 backbone, 27 classes.

    A mobile-sized backbone because the whole point is that this runs on a laptop, and a
    pretrained one because 43,000 frames of one league is not enough to learn edges from
    scratch. The head is replaced rather than fine-tuned: its 21 COCO classes have nothing
    to say about a six-yard box.
    """
    import torch
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    from .detect import trust_certifi

    trust_certifi()
    net = deeplabv3_mobilenet_v3_large(
        weights=None,
        weights_backbone=None if weights else "DEFAULT",
        num_classes=N_CLASSES,
        aux_loss=True,
    )
    if weights is not None:
        net.load_state_dict(torch.load(weights, map_location="cpu"))
    return net


def device() -> Any:
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(net: Any, image: Any) -> npt.NDArray[np.uint8]:
    """A class index per pixel, at the model's own resolution."""
    import torch

    bgr = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))[None]).to(next(net.parameters()).device)
    with torch.no_grad():
        logits = net(tensor)["out"]
    return np.asarray(logits.argmax(1)[0].cpu().numpy(), dtype=np.uint8)


# A curve or its cutting line needs at least this many predicted pixels before the crossing
# is believed. Lower than MIN_PIXELS would let a handful of stray pixels place an EXACT
# correspondence, which is worth two equations and would drag the whole fit with it.
CROSSING_PIXELS = 400
# How near the line a curve pixel must be to count as touching it. A rasterised marking is
# LINE_PX wide and the two labels compete for the pixels where they overlap, so the curve's
# own pixels stop a line width short of the crossing rather than reaching it.
TOUCH_PX = 2.0 * LINE_PX
# Pixels needed across both clusters, and the gap along the line that separates them. The
# two crossings are 18.3 m apart on the centre circle and 14.6 m on a penalty arc, so any
# honest pair is far apart; anything closer is one cluster with a hole in it.
TOUCH_PIXELS = 24
TOUCH_GAP_PX = 40.0


def _touching(
    curve: tuple[Any, Any], line: tuple[float, float, float, float]
) -> list[tuple[float, float]]:
    """Where a curve meets the line that cuts it: the curve's OWN pixels, not a fitted conic.

    Fitting an ellipse and intersecting it analytically is the obvious way and it is wrong
    here. A penalty arc clipped by the frame edge is a short fragment, and `cv2.fitEllipse`
    on 3,174 px of one returns a 46x153 sliver that has nothing to do with the arc -- an
    unconstrained 5-parameter conic through a stub. Angular coverage does not tell the good
    fits from the bad either; the sliver scores 69% where a healthy circle scores 50%.

    The curve's endpoints ARE the crossing. A penalty arc is by definition the part of a
    circle outside the box, so it ENDS on the box line; the halfway line runs through the
    centre spot, so it cuts the centre circle radially. Taking the curve pixels that lie
    within a line width of the line and splitting them into two clusters needs no fit at
    all, and it degrades the right way: a clipped arc shows ONE cluster and is refused,
    where the conic silently invented a second crossing off in the grass.
    """
    xs, ys = curve
    vx, vy, x0, y0 = line
    # Perpendicular distance to the line, and position along it.
    across = (xs - x0) * -vy + (ys - y0) * vx
    along = (xs - x0) * vx + (ys - y0) * vy
    on = np.abs(across) <= TOUCH_PX
    if int(on.sum()) < TOUCH_PIXELS:
        return []

    order = np.argsort(along[on])
    t, tx, ty = along[on][order], xs[on][order], ys[on][order]
    # Two crossings, so one gap: the widest one, and it must be a real separation rather
    # than the ordinary spacing between neighbouring pixels of a single cluster.
    gaps = np.diff(t)
    if len(gaps) == 0:
        return []
    cut = int(np.argmax(gaps))
    if float(gaps[cut]) < TOUCH_GAP_PX:
        return []  # one cluster: the curve meets this line once, so it is clipped

    lo, hi = slice(0, cut + 1), slice(cut + 1, len(t))
    if min(hi.stop - hi.start, lo.stop - lo.start) < TOUCH_PIXELS // 2:
        return []
    return [(float(tx[part].mean()), float(ty[part].mean())) for part in (lo, hi)]


def _y_axis_in_image(pixels: dict[str, Any]) -> tuple[float, float] | None:
    """Which way pitch y increases on screen, read off the constant-y lines in shot.

    The two crossings of a curve are the same distance apart whichever way round they are,
    so nothing in the geometry says which is which -- and getting it backwards mirrors the
    fit. The frame answers it: two markings of KNOWN, different pitch y give the direction
    directly, and the axis gate has already guaranteed two of them exist.
    """
    seen: list[tuple[float, float, float]] = []
    for name, (px, py) in pixels.items():
        line = calibration.PITCH_LINES.get(name)
        if line is None or abs(line[1]) <= abs(line[0]):
            continue  # constant-y markings only; a constant-x one says nothing about y
        seen.append((-line[2] / line[1], float(px.mean()), float(py.mean())))
    if len(seen) < 2:
        return None
    seen.sort()
    (_, lx, ly), (_, hx, hy) = seen[0], seen[-1]
    dx, dy = hx - lx, hy - ly
    length = math.hypot(dx, dy)
    # Two markings whose pixels sit on top of each other give a direction that is noise.
    return None if length < 1.0 else (dx / length, dy / length)


# How far a computed crossing may sit from the nearest predicted pixel of the markings it
# claims to join. A few pixels of slack for the fit itself; beyond that it is extrapolation.
def _crossings(
    mask: npt.NDArray[np.uint8], sx: float, sy: float
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Exact correspondences where a curve crosses the straight marking that cuts it.

    This is what lets a MIDFIELD frame be solved at all. Such a frame has almost no straight
    paint -- SNGS-121 spends 368 consecutive frames with four usable lines against a
    MIN_LINES of five -- while the centre circle sits in the mask at 11,765 px, discarded
    for not being a line. The crossing is not the circle: it is two exact spots (D36).
    """
    pixels: dict[str, Any] = {}
    for name, index in INDEX.items():
        ys, xs = np.nonzero(mask == index)
        if len(xs) >= CROSSING_PIXELS:
            pixels[name] = (xs, ys)

    axis = _y_axis_in_image(pixels)
    if axis is None:
        return []

    out: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for curve, cutter, low, high in calibration.CURVE_CROSSINGS:
        if curve not in pixels or cutter not in pixels:
            continue
        lxs, lys = pixels[cutter]
        # `.ravel()`: fitLine hands back a column of 1-element arrays, not four scalars.
        vx, vy, x0, y0 = (
            float(v)
            for v in cv2.fitLine(
                np.stack([lxs, lys], 1).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
            ).ravel()
        )
        hits = _touching(pixels[curve], (vx, vy, x0, y0))
        if len(hits) != 2:
            continue
        # Order them along increasing pitch y, then they name themselves.
        hits.sort(key=lambda h: h[0] * axis[0] + h[1] * axis[1])
        for (hx, hy), pitch in zip(hits, (low, high), strict=True):
            out.append((((hx + 0.5) * sx, (hy + 0.5) * sy), pitch))
    return out


def _residual_m(h: Any, pairs: list[tuple[tuple[float, float], Any]]) -> float:
    """How far the segmenter's own pixels land from the lines they claim, in metres.

    A confidence estimate, and an in-sample one: the fit was chosen to minimise roughly
    this, so a frame with few constraints can be confidently wrong and still score well.
    That is why it GATES rather than selects — D35's lesson was that choosing between two
    fitters on the quantity one of them minimises is rigged, and this chooses between
    nothing. It only asks whether a fit disagrees with the evidence it was given.

    Measured worth: rank correlation 0.618 against the true error, and keeping the best
    half of frames by this takes the median error from 0.75 m to 0.56 m.
    """
    xy = np.array([p[0] for p in pairs], dtype=np.float64)
    q = h @ np.vstack([xy.T, np.ones(len(xy))])
    q = q[:2] / q[2]
    lines = np.array([p[1] for p in pairs], dtype=np.float64)
    d = np.abs(lines[:, 0] * q[0] + lines[:, 1] * q[1] + lines[:, 2]) / np.hypot(
        lines[:, 0], lines[:, 1]
    )
    return float(np.median(d))


def fit_from_mask(
    mask: npt.NDArray[np.uint8],
    width: int,
    height: int,
    max_residual_m: float | None = None,
) -> Any:
    """A homography from a predicted mask, fitted from nothing else.

    This is the payoff. Every pixel arrives already named, so a correspondence is a fact
    rather than an inference from the homography being solved -- which is what `refine`
    could never have, and why it needed a nearly-correct answer before it could improve
    one (D35).

    DLT first because it needs no starting guess, then the geometric fit because a DLT
    minimises an algebraic residual and is biased by how many pixels each marking happens
    to contribute.
    """
    from . import refine

    sx, sy = width / WIDTH, height / HEIGHT
    pairs: list[tuple[tuple[float, float], tuple[float, float, float]]] = []
    rng = np.random.default_rng(0)
    for name, index in INDEX.items():
        line = calibration.PITCH_LINES.get(name)
        if line is None:
            continue  # circles and goalposts are labelled, but the fitter speaks lines
        ys, xs = np.nonzero(mask == index)
        if len(xs) < MIN_PIXELS:
            continue
        if len(xs) > PER_CLASS:
            keep = rng.choice(len(xs), PER_CLASS, replace=False)
            xs, ys = xs[keep], ys[keep]
        pairs.extend(
            (((float(x) + 0.5) * sx, (float(y) + 0.5) * sy), line)
            for x, y in zip(xs, ys, strict=True)
        )

    seen = {ln for _, ln in pairs}
    across = sum(1 for a, b, _c in seen if abs(a) > abs(b))
    # Both axes are still required outright: no number of crossings rescues a frame that
    # can see only one direction of paint, and the gate below would happily let them try.
    if min(across, len(seen) - across) < calibration.MIN_LINES_PER_AXIS:
        return None

    crossings = _crossings(mask, sx, sy)
    # A crossing is a pair of EXACT spots, so it carries two equations where a marking
    # carries one, and a midfield frame short of MIN_LINES is short by less than it looks.
    if len(seen) + 2 * len(crossings) < calibration.MIN_LINES:
        return None

    first = calibration.fit(crossings, pairs, width, height)
    if first is None:
        return None
    # Explicit None: `a or b` on an ndarray asks for its truth value and raises.
    better = refine._geometric(first, pairs)
    fitted = first if better is None else better
    if max_residual_m is not None and _residual_m(fitted, pairs) > max_residual_m:
        return None  # the fit disagrees with its own evidence; a gap is the honest answer
    return fitted


def train(
    frames: list[Frame],
    val: list[Frame],
    *,
    epochs: int,
    batch: int,
    out: Path = WEIGHTS,
    log: Any = print,
    resume: bool = False,
) -> Path:
    """Fit the segmenter, keeping the epoch that does best on a match it never saw.

    Class weighting matters more than the architecture here. Lines are 5% of the pixels,
    so a model that predicts "background" everywhere is 95% accurate and useless, and
    plain cross-entropy walks straight into it.
    """
    import torch
    from torch.nn import functional as F

    dev = device()
    # Resuming matters more than it looks: a run long enough to matter is longer than any
    # one session reliably survives, and losing six epochs to a killed shell is how a
    # measurement never gets made.
    net = model(out if resume and out.exists() else None).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    weights = torch.ones(N_CLASSES, device=dev)
    weights[0] = 0.05
    best = float("inf")
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        net.train()
        total = seen = 0.0
        for images, masks in batches(frames, batch, seed=epoch):
            images, masks = images.to(dev), masks.to(dev)
            result = net(images)
            loss = F.cross_entropy(result["out"], masks, weight=weights)
            loss = loss + 0.4 * F.cross_entropy(result["aux"], masks, weight=weights)
            opt.zero_grad()
            # torch is an optional extra, so CI type-checks this file without it and the
            # call is Any there. unused-ignore keeps the same line honest in both.
            loss.backward()  # type: ignore[no-untyped-call, unused-ignore]
            opt.step()
            total += float(loss.detach()) * len(images)
            seen += len(images)
        train_loss = total / max(seen, 1)

        net.eval()
        vtotal = vseen = 0.0
        with torch.no_grad():
            for images, masks in batches(val, batch, shuffle=False):
                images, masks = images.to(dev), masks.to(dev)
                vloss = F.cross_entropy(net(images)["out"], masks, weight=weights)
                vtotal += float(vloss.detach()) * len(images)
                vseen += len(images)
        val_loss = vtotal / max(vseen, 1)
        log(f"epoch {epoch:>3}  train {train_loss:.4f}  val {val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save(net.state_dict(), out)
            log(f"           saved {out}")
    return out
