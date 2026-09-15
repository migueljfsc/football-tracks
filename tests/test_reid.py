"""Appearance - the pure parts.

The network needs torch and a checkpoint, neither of which CI has, so what is tested here is
everything done with its output: how crops are made, how looks are averaged and compared, and
that embeddings of other detections are never read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from football_tracks import reid
from football_tracks.detect import Detection


def test_a_look_is_the_unit_mean_of_its_crops() -> None:
    look = reid.look(
        [np.array([2.0, 0.0], dtype=np.float32), np.array([0.0, 5.0], dtype=np.float32)]
    )
    assert look is not None
    assert np.allclose(look, [np.sqrt(0.5), np.sqrt(0.5)])
    assert reid.look([]) is None


def test_distance_is_zero_for_one_look_and_one_for_unrelated_looks() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    assert reid.distance(a, a) == pytest.approx(0.0)
    assert reid.distance(a, np.array([0.0, 3.0], dtype=np.float32)) == pytest.approx(1.0)


def test_embeddings_of_other_detections_are_not_read(tmp_path: Path) -> None:
    dets = [
        Detection(f=1, x1=0.0, y1=0.0, x2=10.0, y2=20.0, score=0.9),
        Detection(f=2, x1=5.0, y1=5.0, x2=15.0, y2=25.0, score=0.9),
    ]
    features = np.eye(2, reid.FEATURES, dtype=np.float32)
    path = reid.write(tmp_path / "appearance.npz", dets, features, np.array([True, False]))
    got = reid.read(path, dets)
    assert got is not None
    assert list(got) == [reid.key(dets[0])]
    moved = [dets[0], Detection(f=2, x1=6.0, y1=5.0, x2=15.0, y2=25.0, score=0.9)]
    assert reid.read(path, moved) is None
    assert reid.read(tmp_path / "missing.npz", dets) is None


def test_a_crop_is_what_the_network_was_trained_on() -> None:
    bgr = np.full((100, 100, 3), 128, dtype=np.uint8)
    c = reid.crop(bgr, Detection(f=1, x1=10.0, y1=10.0, x2=40.0, y2=90.0, score=0.9))
    assert c is not None
    assert c.shape == (3, reid.CROP_H, reid.CROP_W)
    assert c.dtype == np.float32
    assert reid.crop(bgr, Detection(f=1, x1=10.0, y1=10.0, x2=12.0, y2=90.0, score=0.9)) is None
