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
# 0 is background, so a class's channel is its index here plus one.
INDEX: dict[str, int] = {name: i + 1 for i, name in enumerate(CLASSES)}
N_CLASSES = len(CLASSES) + 1

# Training resolution. Lines are thin, so this is the one hyperparameter that is really a
# question about whether the label survives the downsample at all.
WIDTH, HEIGHT = 640, 360
LINE_PX = 3


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
        index = INDEX.get(name)
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


def fit_from_mask(mask: npt.NDArray[np.uint8], width: int, height: int) -> Any:
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
    if len(seen) < calibration.MIN_LINES or min(across, len(seen) - across) < (
        calibration.MIN_LINES_PER_AXIS
    ):
        return None
    first = calibration.fit([], pairs, width, height)
    if first is None:
        return None
    return refine._geometric(first, pairs) or first


def train(
    frames: list[Frame],
    val: list[Frame],
    *,
    epochs: int,
    batch: int,
    out: Path = WEIGHTS,
    log: Any = print,
) -> Path:
    """Fit the segmenter, keeping the epoch that does best on a match it never saw.

    Class weighting matters more than the architecture here. Lines are 5% of the pixels,
    so a model that predicts "background" everywhere is 95% accurate and useless, and
    plain cross-entropy walks straight into it.
    """
    import torch
    from torch.nn import functional as F

    dev = device()
    net = model().to(dev)
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
            loss.backward()  # type: ignore[no-untyped-call]
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
