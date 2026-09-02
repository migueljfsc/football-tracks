"""A broadcast video -> the frame layout the rest of the pipeline already speaks.

SoccerNet hands out clips as numbered JPEGs under img1/ with a metadata file beside
them. Everything downstream reads that shape, so rather than teach each stage about
video files, a real clip is turned INTO that shape once, here.

Two things a SoccerNet clip does not need and a recording does:

* **The crop.** A screen recording is pillarboxed, and often carries the recorder's own
  chrome - a player UI, a subscribe button. Those bars are not black enough to ignore:
  compression noise lifts them over the grass mask's value floor, so they register as
  pitch and the optical flow tries to track them.
* **The real frame rate.** A container can claim 120fps while holding 214 frames across
  6.6 seconds. Timings downstream become scene durations, so the honest number is the
  one derived from duration and count, not the one advertised.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

# A bar is dark for its whole height, so a column is judged by how bright it gets - but
# by its 99th percentile rather than its maximum, because one hot pixel or a compression
# artefact should not make a black bar look like content.
BAR_LEVEL = 60
BAR_PERCENTILE = 99

SAMPLES = 12


@dataclass(slots=True)
class Clip:
    name: str
    fps: float
    width: int
    height: int
    frames: int
    crop: tuple[int, int, int, int]  # x0, y0, x1, y1 in the source

    def to_json(self) -> dict[str, object]:
        return {**asdict(self), "crop": list(self.crop)}


def probe(path: Path) -> tuple[float, int, int, int]:
    """Real fps, frame count, width, height."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    declared = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # A variable-rate recording lies in the header. Duration times the declared rate
    # rarely equals the frame count; the count over the duration is what actually
    # played, and that is what a scene duration has to be built from.
    # ffprobe is the only thing that knows the real duration, and it may not be
    # installed. Missing it costs accuracy, not the run: the declared rate is used, and
    # a variable-rate recording then gets slightly wrong scene timings.
    duration = 0.0
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        duration = float(out.stdout.strip())
    except (FileNotFoundError, ValueError):
        duration = 0.0

    fps = count / duration if duration > 0 and count > 0 else declared
    return fps, count, w, h


def _longest_run(bright: np.ndarray) -> tuple[int, int]:
    """The longest contiguous stretch of content, as (start, end_exclusive).

    Not the first-to-last bright column. A recorder's own furniture sits OUTSIDE the
    video — the Rio Ave clip had a subscribe button 200px into the right-hand bar — and
    taking the outermost bright columns swallows the bar between it and the picture. The
    video is the one unbroken stretch; anything else is somebody else's chrome.
    """
    best = (0, 0)
    start: int | None = None
    for i, on in enumerate([*bright, False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def content_box(path: Path, *, samples: int = SAMPLES) -> tuple[int, int, int, int]:
    """The rectangle inside the letterbox and pillarbox bars.

    Sampled across the clip and then taken as a MEDIAN of what each frame says. Not the
    union: one frame with a flash, an overlay or a fade at the edge would otherwise widen
    the crop for the whole clip, which is how a 3354px frame came back uncropped when 293
    columns of it were black in every frame but one.
    """
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * count / samples))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cols = np.percentile(gray, BAR_PERCENTILE, axis=0) > BAR_LEVEL
        rows = np.percentile(gray, BAR_PERCENTILE, axis=1) > BAR_LEVEL
        x0, x1 = _longest_run(cols)
        y0, y1 = _longest_run(rows)
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    cap.release()
    if not boxes:
        raise RuntimeError(f"{path} appears to be entirely dark")

    arr = np.array(boxes)
    return (
        int(np.median(arr[:, 0])),
        int(np.median(arr[:, 1])),
        int(np.median(arr[:, 2])),
        int(np.median(arr[:, 3])),
    )


def extract(path: Path, dest: Path, *, crop: tuple[int, int, int, int] | None = None) -> Clip:
    """Write every frame as img1/NNNNNN.jpg, and clip.json beside it."""
    fps, _count, _w, _h = probe(path)
    box = crop or content_box(path)
    x0, y0, x1, y1 = box

    frames_dir = dest / "img1"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        cv2.imwrite(
            str(frames_dir / f"{n:06d}.jpg"), frame[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
    cap.release()

    clip = Clip(name=dest.name, fps=fps, width=x1 - x0, height=y1 - y0, frames=n, crop=box)
    (dest / "clip.json").write_text(json.dumps(clip.to_json(), indent=2) + "\n")
    return clip


def load(dest: Path) -> Clip:
    d = json.loads((dest / "clip.json").read_text())
    return Clip(
        name=d["name"],
        fps=d["fps"],
        width=d["width"],
        height=d["height"],
        frames=d["frames"],
        crop=tuple(d["crop"]),
    )


def read_frame(frames_dir: Path, f: int) -> MatLike | None:
    img = cv2.imread(str(frames_dir / f"{f:06d}.jpg"))
    return None if img is None else img
