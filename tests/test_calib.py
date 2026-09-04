"""The pitch-line segmenter's data layer.

Torch is not installed in CI, so nothing here may import it. `calib` keeps every torch
import inside the function that needs it for exactly this reason -- the rasteriser and the
split are ordinary numerical code and are where the load-bearing mistakes live.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from football_tracks import calib, calibration
from football_tracks.calibration import PITCH_LINES
from football_tracks.config import PITCH_LENGTH, PITCH_WIDTH


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


def test_rasterise_tolerates_the_trailing_space_in_the_calibration_labels() -> None:
    """SN-Calibration-2023 writes one marking with a trailing space.

    An exact lookup drops it as an unknown class -- silently, and for 1,101 instances of
    the train split, leaving a class that never appears in a single label.
    """
    points = [{"x": 0.4, "y": 0.4}, {"x": 0.6, "y": 0.6}]
    padded = calib.rasterise({"Goal left post left ": points}, width=64, height=32)
    exact = calib.rasterise({"Goal left post left": points}, width=64, height=32)
    assert np.array_equal(padded, exact)
    assert padded.any()


def test_index_calibration_tags_each_frame_with_its_match(tmp_path: Path) -> None:
    """The match tag is what `split_by_game` holds out, so it has to survive the read."""
    split = tmp_path / "train"
    split.mkdir()
    (split / "match_info.json").write_text(
        json.dumps(
            {
                "00000.jpg": {
                    "league": "england_epl",
                    "season": "2014-2015",
                    "match": " Chelsea - Burnley",
                    "date": "15-02-21",
                },
                "00001.jpg": {
                    "league": "spain_laliga",
                    "season": "2016-2017",
                    "match": " Barcelona - Betis",
                    "date": "17-08-20",
                },
            }
        )
    )
    lines = {"Middle line": [{"x": 0.5, "y": 0.0}, {"x": 0.5, "y": 1.0}]}
    for name in ("00000", "00001"):
        (split / f"{name}.jpg").write_bytes(b"")
        (split / f"{name}.json").write_text(json.dumps(lines))

    frames = calib.index_calibration(tmp_path, "train")
    assert len(frames) == 2
    assert len({f.game for f in frames}) == 2, "two fixtures must not collapse to one tag"
    assert frames[0].game.startswith("england_epl-2014-2015")


def test_index_calibration_is_empty_rather_than_loud_when_nothing_is_fetched(
    tmp_path: Path,
) -> None:
    """`--extra` defaults on, so an unfetched dataset must degrade to the GSR-only run."""
    assert calib.index_calibration(tmp_path, "train") == []


def test_curve_crossings_are_where_the_geometry_says_they_are() -> None:
    """The whole value of a crossing is that it is EXACT, so the table must be exact."""
    centre = (PITCH_LENGTH / 2, PITCH_WIDTH / 2)
    for curve, _cutter, low, high in calibration.CURVE_CROSSINGS:
        spot = centre if curve == "Circle central" else None
        if spot is None:
            # A penalty arc is 9.15 m from its penalty spot, which is 11 m from the goal.
            spot = (11.0, centre[1]) if curve == "Circle left" else (94.0, centre[1])
        for point in (low, high):
            got = math.dist(point, spot)
            assert got == pytest.approx(calibration.CENTRE_RADIUS), (
                f"{curve} crossing is {got:.3f} m from its centre, not 9.15"
            )
        assert low[1] < high[1], "the pair must be ordered by pitch y for the caller"


def test_a_curve_touching_a_line_twice_names_both_crossings() -> None:
    """The crossing is the curve's own pixels, so the test hands it pixels."""
    # A vertical line at x = 100, and a curve that meets it at y = 40 and y = 200.
    ys = np.concatenate([np.arange(30, 51), np.arange(190, 211)])
    xs = np.full(len(ys), 100.0)
    hits = calib._touching((xs, ys.astype(float)), (0.0, 1.0, 100.0, 0.0))
    assert len(hits) == 2
    lo, hi = sorted(hits, key=lambda h: h[1])
    assert lo[1] == pytest.approx(40.0)
    assert hi[1] == pytest.approx(200.0)


def test_a_curve_clipped_to_one_touch_names_no_crossing() -> None:
    """The clipped-arc case: one cluster is not a pair, and inventing the second is the bug.

    SNGS-121 frame 200 has 3,174 px of penalty arc cut off by the frame edge. Fitting a
    conic to it produced a 46x153 sliver; this must refuse instead.
    """
    ys = np.arange(30, 51).astype(float)
    xs = np.full(len(ys), 100.0)
    assert calib._touching((xs, ys), (0.0, 1.0, 100.0, 0.0)) == []


def test_a_curve_that_never_reaches_the_line_names_no_crossing() -> None:
    """Pixels a long way off the line are not a crossing, however many there are."""
    ys = np.concatenate([np.arange(30, 51), np.arange(190, 211)]).astype(float)
    xs = np.full(len(ys), 400.0)  # 300 px from the line
    assert calib._touching((xs, ys), (0.0, 1.0, 100.0, 0.0)) == []


def test_the_y_axis_needs_two_markings_of_different_known_y() -> None:
    """One constant-y line cannot say which way y increases, and guessing mirrors the fit."""
    one = {"Side line top": (np.array([10, 20]), np.array([5, 5]))}
    assert calib._y_axis_in_image(one) is None
    # A constant-X marking says nothing about y either, however many there are.
    with_x = one | {"Middle line": (np.array([1, 2]), np.array([9, 9]))}
    assert calib._y_axis_in_image(with_x) is None
    both = one | {"Side line bottom": (np.array([10, 20]), np.array([95, 95]))}
    axis = calib._y_axis_in_image(both)
    assert axis is not None and axis[1] > 0.9, "pitch y increases DOWN this image"
