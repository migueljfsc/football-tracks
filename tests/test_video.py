"""Ingesting a recording: the crop, and the frame rate it claims versus the one it has."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from football_tracks import video

W, H = 320, 240
BAR = 40


def make(path: Path, *, frames: int = 20, fps: float = 30.0, pillarbox: int = BAR) -> None:
    """A green frame inside dark bars, written as a real file."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (W, H))
    rng = np.random.default_rng(0)
    for _ in range(frames):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        # Bars are not pure black in a real recording; compression noise lifts them,
        # which is exactly why the detector uses a level rather than testing for zero.
        img[:, :] = rng.integers(0, 12, (H, W, 3), dtype=np.uint8)
        img[:, pillarbox : W - pillarbox] = (55, 130, 50)
        writer.write(img)
    writer.release()


def test_pillarbox_is_found_and_removed(tmp_path: Path) -> None:
    src = tmp_path / "clip.mp4"
    make(src)
    x0, y0, x1, y1 = video.content_box(src)
    assert x0 == pytest.approx(BAR, abs=3)
    assert x1 == pytest.approx(W - BAR, abs=3)
    assert (y0, y1) == (0, H)


def test_extract_writes_the_frame_layout_the_pipeline_expects(tmp_path: Path) -> None:
    src = tmp_path / "clip.mp4"
    make(src, frames=8)
    clip = video.extract(src, tmp_path / "out")

    assert clip.frames == 8
    assert clip.width == pytest.approx(W - 2 * BAR, abs=6)
    assert sorted(p.name for p in (tmp_path / "out" / "img1").glob("*.jpg"))[:2] == [
        "000001.jpg",
        "000002.jpg",
    ]
    # Frames are 1-based and zero-padded, because every stage indexes them by name.
    assert video.read_frame(tmp_path / "out" / "img1", 1) is not None
    assert video.read_frame(tmp_path / "out" / "img1", 99) is None


def test_the_clip_metadata_round_trips(tmp_path: Path) -> None:
    src = tmp_path / "clip.mp4"
    make(src, frames=5)
    written = video.extract(src, tmp_path / "out")
    assert video.load(tmp_path / "out") == written
