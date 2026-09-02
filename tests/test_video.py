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


def make_with_chrome(path: Path, *, frames: int = 20, pillarbox: int = BAR) -> None:
    """A pillarboxed clip with a recorder's own button sitting OUT in the right bar."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (W, H))
    rng = np.random.default_rng(0)
    for _ in range(frames):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :] = rng.integers(0, 12, (H, W, 3), dtype=np.uint8)
        img[:, pillarbox : W - pillarbox] = (55, 130, 50)
        img[H - 30 : H - 10, W - 25 : W - 5] = (240, 240, 240)
        writer.write(img)
    writer.release()


def test_a_button_out_in_the_bar_does_not_widen_the_crop(tmp_path: Path) -> None:
    # A recorder's furniture sits OUTSIDE the video. Taking the outermost bright columns
    # swallows the whole bar between it and the picture, which is how a 3354px frame came
    # back uncropped. The video is the one unbroken stretch of content.
    src = tmp_path / "chrome.mp4"
    make_with_chrome(src)
    x0, _y0, x1, _y1 = video.content_box(src)
    assert x1 <= W - BAR + 4
    assert x0 == pytest.approx(BAR, abs=4)


def test_one_odd_frame_does_not_widen_the_crop(tmp_path: Path) -> None:
    # A flash, a fade or an overlay on a single frame would widen the crop for the whole
    # clip if the samples were unioned. They are taken as a median instead.
    src = tmp_path / "flash.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (W, H))
    rng = np.random.default_rng(0)
    for i in range(24):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :] = rng.integers(0, 12, (H, W, 3), dtype=np.uint8)
        img[:, BAR : W - BAR] = (55, 130, 50)
        if i == 12:
            img[:, :] = (200, 200, 200)  # one frame of white, edge to edge
        writer.write(img)
    writer.release()
    x0, _y0, x1, _y1 = video.content_box(src)
    assert x0 == pytest.approx(BAR, abs=4)
    assert x1 == pytest.approx(W - BAR, abs=4)


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


def test_a_missing_ffprobe_costs_accuracy_not_the_run(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Without ffprobe the container's declared rate is used.

    It is the only thing that knows the real duration, and a recording routinely lies
    about its frame rate — but a machine without it should still be able to open a clip,
    and get slightly wrong scene timings rather than a stack trace.
    """
    import subprocess

    src = tmp_path / "clip.mp4"
    make(src, frames=6, fps=30.0)

    def gone(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError(2, "No such file or directory", "ffprobe")

    monkeypatch.setattr(subprocess, "run", gone)
    fps, count, _w, _h = video.probe(src)
    assert count == 6
    assert fps == pytest.approx(30.0, abs=0.1)
