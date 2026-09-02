"""Stage 2a - find people in a frame.

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
import torch

# COCO-pretrained, Apache-2.0. The o365 variant is trained on Objects365 as well and is
# the stronger of the two on crowded scenes, which is all of football.
MODEL = "PekingU/rtdetr_r50vd_coco_o365"

# The only class worth asking for - a football is in COCO too, but at this resolution
# the detector finds it about as often as it invents one (D4).
PERSON = "person"

# Held at 0.5 rather than lowered. RT-DETR at 0.4 does have the same false-positive rate
# the old detector had at 0.5, but "false" there was measured against players only, and
# what it actually finds are referees and touchline staff - real people the pipeline then
# tracks and has to be told to ignore. At 0.5 it beats the old detector on BOTH counts:
# 86.2% recall against 83.6%, and a third of the spurious boxes.
DEFAULT_CONF = 0.5


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
    model: Any, processor: Any, dev: str, bgr: Any, conf: float = DEFAULT_CONF
) -> list[tuple[float, ...]]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inputs)
    sizes = torch.tensor([rgb.shape[:2]])
    result = processor.post_process_object_detection(out, target_sizes=sizes, threshold=conf)[0]
    people = {i for i, name in model.config.id2label.items() if name == PERSON}

    found: list[tuple[float, ...]] = []
    for box, label, score in zip(
        result["boxes"], result["labels"].tolist(), result["scores"].tolist(), strict=True
    ):
        if label in people:
            x1, y1, x2, y2 = (float(v) for v in box.cpu().numpy())
            found.append((x1, y1, x2, y2, float(score)))
    return found


def run(
    frames_dir: Path, frames: list[int], *, conf: float = DEFAULT_CONF, progress: Any = None
) -> list[Detection]:
    model, processor, dev = load_model()
    out: list[Detection] = []
    for f in frames:
        img = cv2.imread(str(frames_dir / f"{f:06d}.jpg"))
        if img is None:
            continue
        for x1, y1, x2, y2, s in on_frame(model, processor, dev, img, conf):
            out.append(Detection(f=f, x1=x1, y1=y1, x2=x2, y2=y2, score=s))
        if progress is not None:
            progress(f)
    return out


def write(path: Path, detections: list[Detection], *, conf: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "conf": conf,
                "detections": [
                    {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(d).items()}
                    for d in detections
                ],
            },
            indent=1,
        )
        + "\n"
    )
    return path


def read(path: Path) -> list[Detection]:
    data = json.loads(path.read_text())
    return [Detection(**d) for d in data["detections"]]


def by_frame(detections: list[Detection]) -> dict[int, list[Detection]]:
    out: dict[int, list[Detection]] = {}
    for d in detections:
        out.setdefault(d.f, []).append(d)
    return out


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
