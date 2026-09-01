"""Stage 2a - find people in a frame.

torchvision's COCO Faster R-CNN. Not state of the art, which is the point: it is BSD
licensed, it is already a dependency, and it is a FLOOR. Anything it finds a better
detector also finds, so a measurement taken with it is a lower bound on the pipeline
rather than a best case.

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
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

# COCO class 1. The only one worth asking for - a football is class 37 but at this
# resolution the detector finds it about as often as it invents one (D4).
PERSON = 1

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


def load_model(dev: str | None = None) -> tuple[Any, str]:
    _trust_certifi()
    dev = dev or device()
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).eval().to(dev)
    return model, dev


def on_frame(model: Any, dev: str, bgr: Any, conf: float = DEFAULT_CONF) -> list[tuple[float, ...]]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = model(t)[0]
    keep = (out["labels"] == PERSON) & (out["scores"] >= conf)
    boxes = out["boxes"][keep].cpu().numpy()
    scores = out["scores"][keep].cpu().numpy()
    return [
        (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s))
        for b, s in zip(boxes, scores, strict=True)
    ]


def run(
    frames_dir: Path, frames: list[int], *, conf: float = DEFAULT_CONF, progress: Any = None
) -> list[Detection]:
    model, dev = load_model()
    out: list[Detection] = []
    for f in frames:
        img = cv2.imread(str(frames_dir / f"{f:06d}.jpg"))
        if img is None:
            continue
        for x1, y1, x2, y2, s in on_frame(model, dev, img, conf):
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
