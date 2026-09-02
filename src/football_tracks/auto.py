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
from . import seed as seed_mod
from .stage3_teams import assign
from .tracks import PLAYER_MARGIN, Sample, Track, on_pitch

Mode = Literal["truth", "seed"]


# How many frames either side the ball's position is taken a median over.
#
# The detector reports about five "sports balls" a frame - a head, a boot, a patch of
# hoarding - and the real one is only usually the most confident. What separates it from
# the impostors is that it MOVES SMOOTHLY, so a median over neighbouring frames throws
# them out without having to know which is which.
BALL_SMOOTH_FRAMES = 5


@dataclass(slots=True)
class Result:
    tracks: list[Track]
    ball: list[Sample]
    frames: int
    detections: int
    raw_tracks: int
    dropped_off_pitch: int
    unsolved_frames: int


def from_seed(
    seeded: seed_mod.Seed,
    frames: list[int],
    frames_dir: Path,
    *,
    max_carry: int | None,
    motions: dict[int, Any] | None = None,
) -> dict[int, Any]:
    """One clicked frame, carried across the clip in both directions.

    The whole automatic path for a clip nobody has annotated: a human marks landmarks
    once and `stage1_propagate` does the rest. Both directions matter - a clip is
    rarely best seeded at its first frame, because the camera is often still finding
    the play there.
    """
    h = seed_mod.homography(seeded)
    direct: dict[int, Any] = dict.fromkeys(frames)
    if h is not None and seeded.frame in direct:
        direct[seeded.frame] = h
    return stage1_propagate.fill(
        frames_dir, direct, max_carry=max_carry, motion=motions
    ).homographies


def homographies(
    labels: dict[str, Any],
    frames_dir: Path,
    mode: Mode,
    *,
    max_carry: int | None,
    motions: dict[int, Any] | None = None,
) -> dict[int, Any]:
    """Per-frame homographies, either from every frame's lines or from frame one's."""
    direct = stage1_register.fit_all(labels)
    if mode == "truth":
        return stage1_propagate.fill(
            frames_dir, direct, max_carry=max_carry, motion=motions
        ).homographies

    # Everything except the first solvable frame is thrown away, which is what a
    # human clicking once actually leaves you with.
    first = next((f for f in sorted(direct) if direct[f] is not None), None)
    seeded: dict[int, Any] = dict.fromkeys(direct)
    if first is not None:
        seeded[first] = direct[first]
    return stage1_propagate.fill(
        frames_dir, seeded, max_carry=max_carry, motion=motions
    ).homographies


def ball_path(
    balls: list[detect.Sighting],
    homs: dict[int, Any],
    frames: list[int],
    smooth: int = BALL_SMOOTH_FRAMES,
) -> list[Sample]:
    """Where the ball is, per frame, in pitch metres.

    The most confident sighting per frame, projected, then median-filtered over its
    neighbours. Picking the sighting NEAREST a player scores worse than this and it is
    worth saying why: there are several candidates a frame, so "nearest a player"
    reliably selects whichever false positive happens to stand beside somebody.

    The positions are NOT to be trusted as positions. A ground homography assumes z = 0,
    so a ball in flight lands metres from where it is. They are good for one question -
    who is nearest - and that question is all a board needs (D29).
    """
    best: dict[int, tuple[float, float]] = {}
    per_frame: dict[int, list[detect.Sighting]] = {}
    for b in balls:
        per_frame.setdefault(b.f, []).append(b)

    for f, seen in per_frame.items():
        h = homs.get(f)
        if h is None:
            continue
        top = max(seen, key=lambda b: b.score)
        best[f] = calibration.to_pitch(h, top.x, top.y)

    out: list[Sample] = []
    for f in frames:
        near = [best[g] for g in range(f - smooth, f + smooth + 1) if g in best]
        if not near:
            continue
        x = float(np.median([p[0] for p in near]))
        y = float(np.median([p[1] for p in near]))
        if on_pitch(x, y):
            out.append(Sample(f=f, x=x, y=y))
    return out


def build(
    frames_dir: Path,
    frames: list[int],
    detections: list[detect.Detection],
    homs: dict[int, Any],
    *,
    fps: float,
    motions: dict[int, Any] | None = None,
    balls: list[detect.Sighting] | None = None,
) -> Result:
    """Detections plus a camera model -> tracks in pitch metres.

    Tracking runs FIRST, in stabilised image pixels, and projection happens after. The
    other order made stage 2 inherit every wobble in stage 1's homography, which is
    what breaks tracks all at once (D19). This way a drifting homography moves the
    positions and leaves the identities intact.
    """
    cache: dict[int, Any] = {}

    def read_frame(f: int) -> Any:
        if f not in cache:
            cache.clear()  # one frame at a time; the tracker only ever looks back one
            cache[f] = cv2.imread(str(frames_dir / f"{f:06d}.jpg"))
        return cache[f]

    # Drop anyone standing off the pitch BEFORE tracking, not after. Two fifths of what
    # the detector finds is crowd, dugout staff and ballboys behind the hoardings, and
    # while they were always discarded at the end, until then they were competing for
    # associations and spawning tracks. Filtering first nearly halves the track count.
    #
    # This is the one place stage 2 consults stage 1 (D22), and only as a filter: the
    # association itself still never sees a homography, so a drifting camera moves which
    # detections are considered and cannot move the identities.
    observations: dict[int, list[stage2_track.Observation]] = {}
    dropped = 0
    for d in detections:
        h = homs.get(d.f)
        if h is None:
            continue
        if not on_pitch(*calibration.to_pitch(h, *d.foot), PLAYER_MARGIN):
            dropped += 1
            continue
        observations.setdefault(d.f, []).append(stage2_track.Observation.of(d))

    raw = stage2_track.run(observations, frames, read_frame, fps=fps, motions=motions)

    positions: dict[int, list[Sample]] = {}
    for t in raw:
        samples: list[Sample] = []
        for o in t.observations:
            h = homs.get(o.f)
            if h is None:
                continue
            x, y = calibration.to_pitch(h, o.x, o.y)
            if not on_pitch(x, y):
                dropped += 1
                continue
            samples.append(Sample(f=o.f, x=x, y=y, conf=o.det.score))
        if samples:
            positions[t.id] = samples

    mean_x = {tid: float(np.mean([s.x for s in ss])) for tid, ss in positions.items()}
    teams = assign([t for t in raw if t.id in positions], mean_x)

    return Result(
        ball=ball_path(balls or [], homs, frames),
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
