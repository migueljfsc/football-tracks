"""The pitch-line segmenter's data layer.

Torch is not installed in CI, so nothing here may import it. `calib` keeps every torch
import inside the function that needs it for exactly this reason -- the rasteriser and the
split are ordinary numerical code and are where the load-bearing mistakes live.
"""

from __future__ import annotations

import numpy as np

from football_tracks import calib
from football_tracks.calibration import PITCH_LINES


def test_every_class_has_a_distinct_index() -> None:
    assert len(calib.INDEX) == len(calib.CLASSES)
    assert sorted(calib.INDEX.values()) == list(range(1, len(calib.CLASSES) + 1))
    assert 0 not in calib.INDEX.values(), "0 is background"


def test_the_fitter_knows_every_straight_marking_the_model_predicts() -> None:
    """A predicted class the fitter cannot name contributes nothing.

    Circles and goalposts are labelled and predicted on purpose -- they are evidence the
    network learns from -- but only the straight markings become correspondences.
    """
    straight = {n for n in calib.CLASSES if not n.startswith(("Circle", "Goal"))}
    assert straight == set(PITCH_LINES), straight.symmetric_difference(set(PITCH_LINES))


def test_rasterise_draws_the_named_class() -> None:
    lines = {"Middle line": [{"x": 0.5, "y": 0.0}, {"x": 0.5, "y": 1.0}]}
    mask = calib.rasterise(lines, width=64, height=32)
    assert set(np.unique(mask).tolist()) == {0, calib.INDEX["Middle line"]}
    # down the middle, not along an edge
    assert mask[:, 30:34].any()
    assert not mask[:, :4].any()


def test_rasterise_ignores_a_class_it_does_not_know() -> None:
    mask = calib.rasterise({"Not a marking": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]}, 32, 32)
    assert not mask.any()


def test_rasterise_needs_two_points_to_draw_a_line() -> None:
    assert not calib.rasterise({"Middle line": [{"x": 0.5, "y": 0.5}]}, 32, 32).any()


def test_the_split_never_puts_one_match_on_both_sides() -> None:
    """Two clips of a game share a stadium, a camera and a kit.

    Splitting by clip or by frame reports a generalisation that was never tested, and the
    two clips already on disk are both game 7 -- so this is the default outcome, not an
    edge case.
    """
    frames = [
        calib.Frame(image=__import__("pathlib").Path(f"{g}-{i}.jpg"), lines={}, game=g)
        for g in ("7", "8", "9")
        for i in range(3)
    ]
    train, val = calib.split_by_game(frames, holdout=1)
    assert {f.game for f in train}.isdisjoint({f.game for f in val})
    assert len(val) == 3
    assert len(train) == 6


def test_the_split_is_empty_rather_than_leaky_when_there_is_one_match() -> None:
    frames = [
        calib.Frame(image=__import__("pathlib").Path(f"{i}.jpg"), lines={}, game="7")
        for i in range(4)
    ]
    train, val = calib.split_by_game(frames, holdout=1)
    assert val == []
    assert len(train) == 4
