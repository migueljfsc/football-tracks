"""Stage 0's pure helpers.

Only the numerical parts are tested, which is the same bargain Pitchboard's engine
makes: cheap where the maths is, and a picture where the maths meets a video. Whether
the cut detector finds the right cut is not a unit test, it is stage 0's overlay.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from cv2.typing import MatLike

from football_tracks.stage0_segment import (
    Segment,
    _green_fraction,
    _motion,
    _sample_frames,
    best,
    write,
)


def hsv_block(h: int, s: int, v: int, w: int = 64, ht: int = 64) -> MatLike:
    """A flat BGR image of one HSV colour, so tests state the hue they mean."""
    img = np.zeros((ht, w, 3), dtype=np.uint8)
    img[:, :] = (h, s, v)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


PITCH = hsv_block(60, 200, 150)  # grass
STANDS = hsv_block(0, 0, 128)  # unsaturated grey, whatever its hue


def test_green_fraction_reads_grass_as_pitch() -> None:
    assert _green_fraction(PITCH) == pytest.approx(1.0, abs=0.02)


def test_green_fraction_rejects_unsaturated_grey() -> None:
    # The saturation floor is what separates stands from pitch. Hue alone would call
    # grey green, since grey has no meaningful hue at all.
    assert _green_fraction(STANDS) == pytest.approx(0.0, abs=0.02)


def test_green_fraction_is_a_share_not_a_verdict() -> None:
    half = np.vstack([PITCH, STANDS])
    assert _green_fraction(half) == pytest.approx(0.5, abs=0.02)


def test_motion_is_zero_for_a_still_camera() -> None:
    assert _motion(PITCH, PITCH.copy()) == pytest.approx(0.0, abs=0.5)


def test_motion_saturates_on_a_cut() -> None:
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    white = np.full((64, 64, 3), 255, dtype=np.uint8)
    assert _motion(black, white) == pytest.approx(255.0, abs=0.5)


def test_sample_frames_stays_inside_the_segment() -> None:
    frames = _sample_frames(0, 100, 5)
    assert len(frames) == 5
    assert all(0 <= f < 100 for f in frames)
    assert frames == sorted(frames)


def test_sample_frames_avoids_the_boundaries() -> None:
    # A cut's own frames blend two shots, so scoring them measures neither.
    frames = _sample_frames(0, 100, 5)
    assert frames[0] > 0
    assert frames[-1] < 99


def test_sample_frames_never_asks_for_more_than_the_segment_has() -> None:
    assert _sample_frames(10, 11, 6) == [10]


def seg(index: int, duration: float, *, main: bool) -> Segment:
    return Segment(
        index=index,
        start_frame=0,
        end_frame=int(duration * 25),
        start_s=0.0,
        end_s=duration,
        duration_s=duration,
        green=0.6,
        motion=3.0,
        main=main,
    )


def test_best_prefers_the_longest_qualifying_segment() -> None:
    # Length rather than greenness: a tight shot of the goalmouth can be greener than
    # a wide one, and duration is what makes a passage worth reducing.
    segments = [seg(0, 5.0, main=True), seg(1, 12.0, main=True), seg(2, 30.0, main=False)]
    picked = best(segments)
    assert picked is not None
    assert picked.index == 1


def test_best_is_none_when_nothing_qualifies() -> None:
    assert best([seg(0, 2.0, main=False)]) is None


def test_write_records_the_pick(tmp_path: object) -> None:
    from pathlib import Path

    out = Path(str(tmp_path))
    segments = [seg(0, 3.0, main=False), seg(1, 9.0, main=True)]
    path = write(segments, {"clip": "x.mp4", "fps": 25.0}, out)
    data = json.loads(path.read_text())
    assert data["best"] == 1
    assert len(data["segments"]) == 2


def test_module_boundary_is_stage_zero_only() -> None:
    """Stage 0 must not learn about SoccerNet, or the seam stops being a seam."""
    import football_tracks.stage0_segment as s0

    assert not hasattr(s0, "soccernet")
