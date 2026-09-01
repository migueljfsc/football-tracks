"""Stage 0 - find the main tactical camera segment in a broadcast clip.

Broadcast football cuts constantly: replays, close-ups, crowd, dugout. Every stage
after this one assumes ONE continuous view of the pitch from ONE camera, so the first
job is to find that view and throw the rest away.

The main camera is recognisable without understanding anything about football: it is
long, it is overwhelmingly green, and it moves smoothly because it is panning rather
than cutting. Close-ups are green too but short; crowd shots are neither.

Writes work/<clip>/segments.json.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from cv2.typing import MatLike
from scenedetect import ContentDetector, detect

# OpenCV's hue channel is 0..179, so pitch green sits around 60. The saturation and
# value floors exist to reject grey stands and floodlit white, not to be precise -
# a shaded half of the pitch is much less saturated than a sunlit one.
GREEN_LO = np.array([35, 40, 40], dtype=np.uint8)
GREEN_HI = np.array([85, 255, 255], dtype=np.uint8)

# Frames are downscaled to this width before differencing. Full resolution measures
# compression noise as much as camera movement, and is slower for no gain.
MOTION_WIDTH = 320


@dataclass
class Segment:
    index: int
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    duration_s: float
    green: float
    motion: float
    main: bool


def _green_fraction(bgr: MatLike) -> float:
    """Share of the frame that is pitch, 0..1."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(cv2.inRange(hsv, GREEN_LO, GREEN_HI).mean() / 255.0)


def _motion(a: MatLike, b: MatLike) -> float:
    """Mean absolute difference between consecutive frames, 0..255.

    A proxy for camera movement rather than a measurement of it: players moving in a
    static frame register too. It separates a panning wide shot from a locked-off
    replay well enough to rank segments, which is all it is asked to do.
    """
    scale = MOTION_WIDTH / a.shape[1]
    size = (MOTION_WIDTH, max(1, round(a.shape[0] * scale)))
    ga = cv2.cvtColor(cv2.resize(a, size), cv2.COLOR_BGR2GRAY).astype(np.int16)
    gb = cv2.cvtColor(cv2.resize(b, size), cv2.COLOR_BGR2GRAY).astype(np.int16)
    return float(np.abs(ga - gb).mean())


def _sample_frames(start: int, end: int, n: int) -> list[int]:
    """Evenly spaced frame indices inside [start, end), avoiding the boundaries.

    A cut's own frames are a blend of two shots, so sampling them measures neither.
    """
    span = max(1, end - start)
    n = max(1, min(n, span))
    return [start + int((i + 0.5) * span / n) for i in range(n)]


def _score(cap: cv2.VideoCapture, start: int, end: int, samples: int) -> tuple[float, float]:
    greens: list[float] = []
    motions: list[float] = []
    for f in _sample_frames(start, end, samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        greens.append(_green_fraction(frame))
        # The very next frame, so the difference is one frame of camera movement and
        # not a second of it. Seeking is the expensive part; this read is nearly free.
        ok, nxt = cap.read()
        if ok:
            motions.append(_motion(frame, nxt))
    return (
        float(np.mean(greens)) if greens else 0.0,
        float(np.mean(motions)) if motions else 0.0,
    )


def find_segments(
    clip: Path,
    *,
    threshold: float = 27.0,
    min_seconds: float = 4.0,
    green_min: float = 0.35,
    samples: int = 6,
) -> tuple[list[Segment], dict[str, Any]]:
    """Split a clip at its cuts and score each piece for being the tactical camera.

    Returns the segments and the source metadata every later stage needs.
    """
    cuts = detect(str(clip), ContentDetector(threshold=threshold))

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source: dict[str, Any] = {
        "clip": clip.name,
        "fps": fps,
        "frames": total,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    # No cuts found means one continuous shot, which is the normal case for an already
    # trimmed clip - not an error, and not an empty result.
    spans = [(s.get_frames(), e.get_frames()) for s, e in cuts] if cuts else [(0, total)]

    segments: list[Segment] = []
    try:
        for i, (start, end) in enumerate(spans):
            green, motion = _score(cap, start, end, samples)
            duration = (end - start) / fps
            segments.append(
                Segment(
                    index=i,
                    start_frame=start,
                    end_frame=end,
                    start_s=start / fps,
                    end_s=end / fps,
                    duration_s=duration,
                    green=green,
                    motion=motion,
                    main=duration >= min_seconds and green >= green_min,
                )
            )
    finally:
        cap.release()

    return segments, source


def best(segments: list[Segment]) -> Segment | None:
    """The longest qualifying segment, or None when nothing qualifies.

    Length rather than greenness: a tight shot of the goalmouth can be greener than a
    wide one, and duration is what makes a passage of play worth reducing.
    """
    qualifying = [s for s in segments if s.main]
    return max(qualifying, key=lambda s: s.duration_s) if qualifying else None


def write(segments: list[Segment], source: dict[str, Any], out: Path) -> Path:
    # Resolved once: mypy cannot narrow `best(...) if best(...)` across two calls,
    # and calling it twice is a second answer to the same question anyway.
    pick = best(segments)
    path = out / "segments.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "source": source,
                "segments": [asdict(s) for s in segments],
                "best": pick.index if pick else None,
            },
            indent=2,
        )
        + "\n"
    )
    return path


def extract(clip: Path, segment: Segment, out: Path) -> Path:
    """Cut one segment out to its own file with ffmpeg.

    Re-encoded rather than stream-copied: a copy can only cut on keyframes, which puts
    the boundary somewhere other than the cut and leaves frames of the wrong shot at
    the front. Later stages index frames, so an off-by-a-few start is a silent error.
    """
    dest = out / f"segment_{segment.index:02d}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(clip),
            "-ss",
            f"{segment.start_s:.3f}",
            "-to",
            f"{segment.end_s:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-an",
            str(dest),
        ],
        check=True,
    )
    return dest
