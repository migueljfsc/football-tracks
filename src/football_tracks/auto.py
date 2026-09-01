"""The automatic path, end to end: frames in, tracks.json out.

Stitches detection, tracking, team assignment and registration together. Registration
is pluggable on purpose, because the whole question this pipeline exists to answer is
how much of the error belongs to which stage:

* `truth`  - a homography per frame from the ground-truth pitch lines. Holds stage 1
             fixed, so what comes out is stage 2 and 3's error alone.
* `seed`   - ONLY frame one's lines, then carried by tracking the grass. This is the
             real question: what a human clicking four corners once actually buys.

Both write the same tracks.json, scored by the same `ft score`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from . import calibration, detect, stage1_propagate, stage1_register, stage2_track
from .stage3_teams import assign
from .tracks import Sample, Track, on_pitch

Mode = Literal["truth", "seed"]


@dataclass(slots=True)
class Result:
    tracks: list[Track]
    frames: int
    detections: int
    raw_tracks: int
    dropped_off_pitch: int
    unsolved_frames: int


def homographies(
    labels: dict[str, Any], frames_dir: Path, mode: Mode, *, max_carry: int | None
) -> dict[int, Any]:
    """Per-frame homographies, either from every frame's lines or from frame one's."""
    direct = stage1_register.fit_all(labels)
    if mode == "truth":
        return stage1_propagate.fill(frames_dir, direct, max_carry=max_carry).homographies

    # Everything except the first solvable frame is thrown away, which is what a
    # human clicking once actually leaves you with.
    first = next((f for f in sorted(direct) if direct[f] is not None), None)
    seeded: dict[int, Any] = dict.fromkeys(direct)
    if first is not None:
        seeded[first] = direct[first]
    return stage1_propagate.fill(frames_dir, seeded, max_carry=max_carry).homographies


def build(
    frames_dir: Path,
    frames: list[int],
    detections: list[detect.Detection],
    homs: dict[int, Any],
    *,
    fps: float,
) -> Result:
    """Detections plus a camera model -> tracks in pitch metres.

    Projection happens BEFORE association, not after. The tracker gates on how far a
    player can run, which is a claim about metres and seconds; in image pixels the same
    gate is really a claim about how fast the camera pans.
    """
    cache: dict[int, Any] = {}

    def read_frame(f: int) -> Any:
        if f not in cache:
            cache.clear()  # one frame at a time; the tracker only ever looks back one
            cache[f] = cv2.imread(str(frames_dir / f"{f:06d}.jpg"))
        return cache[f]

    observations: dict[int, list[stage2_track.Observation]] = {}
    dropped = 0
    for d in detections:
        h = homs.get(d.f)
        if h is None:
            continue
        x, y = calibration.to_pitch(h, *d.foot)
        if not on_pitch(x, y):
            dropped += 1  # crowd, dugout, ballboys behind the hoardings
            continue
        observations.setdefault(d.f, []).append(stage2_track.Observation(f=d.f, x=x, y=y, det=d))

    raw = stage2_track.run(observations, frames, read_frame, fps=fps)

    positions = {
        t.id: [Sample(f=o.f, x=o.x, y=o.y, conf=o.det.score) for o in t.observations] for t in raw
    }
    mean_x = {tid: float(np.mean([s.x for s in ss])) for tid, ss in positions.items()}
    teams = assign(raw, mean_x)

    return Result(
        tracks=[
            Track(id=tid, team=teams.get(tid, "unknown"), number=None, samples=ss)
            for tid, ss in sorted(positions.items())
        ],
        frames=len(frames),
        detections=len(detections),
        raw_tracks=len(raw),
        dropped_off_pitch=dropped,
        unsolved_frames=sum(1 for f in frames if homs.get(f) is None),
    )
