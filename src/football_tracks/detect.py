"""Stage 2a - find people in a frame.

torch and transformers are imported INSIDE the functions that need them. They are a
two-gigabyte optional extra, and the dataclasses here are the contract every other stage
reads - `Detection` and `Sighting` have to be importable without a model, or the tests
and the type checker need a GPU stack to look at a bounding box.

RT-DETR, COCO-pretrained, Apache-2.0. It replaced torchvision's Faster R-CNN once the
pipeline was measurable enough to compare them properly, and it wins on every axis:

    Faster R-CNN  conf 0.50   83.6% recall   1.6 spurious per frame   0.26 s/frame
    RT-DETR       conf 0.50   86.2% recall   0.5 spurious per frame   0.14 s/frame

That comparison is the whole reason the first detector was a deliberate floor. The
false-positive column matters as much as recall here: everything the detector invents
competes for associations and spawns tracks, which is what fragments them (D27).

Two things that did NOT lift recall, so they are not worth retrying: feeding the model
a larger image (800px against 1333px changed nothing, because the misses are occlusions
rather than small players), and lowering the confidence floor, which buys recall at
roughly three false detections per real one.

Detections are cached to work/<clip>/detections.json. Detecting is the slow part and
tracking is the part that gets tuned, so they are separate stages on purpose.
"""

from __future__ import annotations

import json
import os
import ssl
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# COCO-pretrained, Apache-2.0. The o365 variant is trained on Objects365 as well and is
# the stronger of the two on crowded scenes, which is all of football.
MODEL = "PekingU/rtdetr_r50vd_coco_o365"

PERSON = "person"

# The ball. D4 deferred it on the grounds that it is the hardest object in the frame,
# and on this detector that turned out to be wrong: RT-DETR finds it in every sampled
# frame of all three clips. What is still hard is where it IS - a ground homography
# assumes z = 0, so an airborne ball projects metres from the truth. That is why only
# the CARRIER is derived from it and never a position (D29 in this repo's plan).
BALL = "sports ball"

# Lower than the person floor. A ball is small and often blurred, and the cost of a
# spurious one is bounded: it has to land within a couple of metres of a player before
# anything downstream believes it.
BALL_CONF = 0.15

# Held at 0.5 rather than lowered. RT-DETR at 0.4 does have the same false-positive rate
# the old detector had at 0.5, but "false" there was measured against players only, and
# what it actually finds are referees and touchline staff - real people the pipeline then
# tracks and has to be told to ignore. At 0.5 it beats the old detector on BOTH counts:
# 86.2% recall against 83.6%, and a third of the spurious boxes.
DEFAULT_CONF = 0.5


@dataclass(slots=True)
class Sighting:
    """A ball, seen. Not a track - the ball is never followed, only looked for."""

    f: int
    x: float
    y: float
    score: float


@dataclass(slots=True)
class Detection:
    f: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom middle - where the player meets the grass, and the only point on a
        box a ground homography can say anything about."""
        return ((self.x1 + self.x2) / 2, self.y2)

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def _trust_certifi() -> None:
    """Point urllib at certifi's bundle before torch downloads weights.

    A stock python.org install on macOS has no system CA bundle wired up, so the
    download dies with CERTIFICATE_VERIFY_FAILED - a failure that looks like a network
    problem and is not.
    """
    try:
        import certifi
    except ImportError:
        return
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    ssl._create_default_https_context = ssl.create_default_context


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(dev: str | None = None) -> tuple[Any, Any, str]:
    """The model, its image processor, and the device. Weights download on first use."""
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

    _trust_certifi()
    dev = dev or device()
    processor = RTDetrImageProcessor.from_pretrained(MODEL)
    # transformers ships no py.typed, so the model is Any from here on.
    model: Any = RTDetrForObjectDetection.from_pretrained(MODEL)
    model = model.eval().to(dev)
    return model, processor, dev


def on_frame(
    model: Any,
    processor: Any,
    dev: str,
    bgr: Any,
    conf: float = DEFAULT_CONF,
    ball_conf: float = BALL_CONF,
) -> tuple[list[tuple[float, ...]], list[tuple[float, float, float]]]:
    """People and balls in one pass. Two lists, because they are not the same thing:
    a person is tracked and a ball is only ever asked about."""
    import torch

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inputs)
    sizes = torch.tensor([rgb.shape[:2]])
    floor = min(conf, ball_conf)
    result = processor.post_process_object_detection(out, target_sizes=sizes, threshold=floor)[0]
    people = {i for i, name in model.config.id2label.items() if name == PERSON}
    balls = {i for i, name in model.config.id2label.items() if name == BALL}

    found: list[tuple[float, ...]] = []
    seen: list[tuple[float, float, float]] = []
    for box, label, score in zip(
        result["boxes"], result["labels"].tolist(), result["scores"].tolist(), strict=True
    ):
        x1, y1, x2, y2 = (float(v) for v in box.cpu().numpy())
        if label in people and score >= conf:
            found.append((x1, y1, x2, y2, float(score)))
        elif label in balls and score >= ball_conf:
            # The centre, not the foot. A ball has no feet, and where it meets the grass
            # is exactly what a ground homography cannot tell you when it is in the air.
            seen.append(((x1 + x2) / 2, (y1 + y2) / 2, float(score)))
    return found, seen


def run(
    frames_dir: Path, frames: list[int], *, conf: float = DEFAULT_CONF, progress: Any = None
) -> tuple[list[Detection], list[Sighting]]:
    model, processor, dev = load_model()
    out: list[Detection] = []
    balls: list[Sighting] = []
    for f in frames:
        img = cv2.imread(str(frames_dir / f"{f:06d}.jpg"))
        if img is None:
            continue
        people, seen = on_frame(model, processor, dev, img, conf)
        for x1, y1, x2, y2, s in people:
            out.append(Detection(f=f, x1=x1, y1=y1, x2=x2, y2=y2, score=s))
        for x, y, s in seen:
            balls.append(Sighting(f=f, x=x, y=y, score=s))
        if progress is not None:
            progress(f)
    return out, balls


def write(path: Path, detections: list[Detection], balls: list[Sighting], *, conf: float) -> Path:
    def rounded(d: Any) -> dict[str, Any]:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(d).items()}

    path.write_text(
        json.dumps(
            {
                "version": 1,
                "conf": conf,
                "detections": [rounded(d) for d in detections],
                "balls": [rounded(b) for b in balls],
            },
            indent=1,
        )
        + "\n"
    )
    return path


def read(path: Path) -> tuple[list[Detection], list[Sighting]]:
    data = json.loads(path.read_text())
    return (
        [Detection(**d) for d in data["detections"]],
        [Sighting(**b) for b in data.get("balls", [])],
    )


def torso(bgr: Any, d: Detection) -> Any:
    """The shirt, roughly - the middle of the upper half of the box.

    Insetting matters: the edges of a box are grass and the legs are shorts, and both
    drag a kit colour towards something it is not.
    """
    h, w = bgr.shape[:2]
    bw, bh = d.x2 - d.x1, d.y2 - d.y1
    x0 = int(np.clip(d.x1 + 0.25 * bw, 0, w - 1))
    x1 = int(np.clip(d.x2 - 0.25 * bw, 0, w))
    y0 = int(np.clip(d.y1 + 0.15 * bh, 0, h - 1))
    y1 = int(np.clip(d.y1 + 0.45 * bh, 0, h))
    if x1 <= x0 or y1 <= y0:
        return None
    return bgr[y0:y1, x0:x1]
