"""Scoring a produced tracks.json against ground truth.

Both sides are the same format written by the same writer (D12), so these tests build
documents by hand and never touch the dataset.
"""

from __future__ import annotations

from typing import Any

from football_tracks.score import match_frame, score


def trk(
    tid: int, team: str, pts: list[tuple[int, float, float]], number: int | None = None
) -> dict[str, Any]:
    return {
        "id": tid,
        "team": team,
        "number": number,
        "samples": [{"f": f, "x": x, "y": y} for f, x, y in pts],
    }


def doc(*tracks: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "source": {"clip": "t", "fps": 25.0, "startFrame": 1, "endFrame": 10},
        "pitch": {"length": 105.0, "width": 68.0},
        "tracks": list(tracks),
        "ball": None,
    }


def test_a_perfect_prediction_scores_perfectly() -> None:
    d = doc(trk(1, "home", [(1, 10.0, 10.0), (2, 11.0, 10.0)]))
    s = score(d, d)
    assert s.recall == 1.0
    assert s.precision == 1.0
    assert s.median_error_m == 0.0
    assert s.team_accuracy == 1.0
    assert s.identity_purity == 1.0
    assert s.id_switches == 0


def test_a_missed_player_costs_recall_not_precision() -> None:
    truth = doc(trk(1, "home", [(1, 10.0, 10.0)]), trk(2, "away", [(1, 30.0, 30.0)]))
    pred = doc(trk(1, "home", [(1, 10.0, 10.0)]))
    s = score(truth, pred)
    assert s.recall == 0.5
    assert s.precision == 1.0


def test_a_hallucinated_player_costs_precision_not_recall() -> None:
    truth = doc(trk(1, "home", [(1, 10.0, 10.0)]))
    pred = doc(trk(1, "home", [(1, 10.0, 10.0)]), trk(2, "home", [(1, 30.0, 30.0)]))
    s = score(truth, pred)
    assert s.recall == 1.0
    assert s.precision == 0.5


def test_a_position_beyond_the_radius_is_not_a_match() -> None:
    truth = doc(trk(1, "home", [(1, 10.0, 10.0)]))
    assert score(truth, doc(trk(1, "home", [(1, 11.5, 10.0)]))).matched == 1
    assert score(truth, doc(trk(1, "home", [(1, 14.0, 10.0)]))).matched == 0


def test_an_id_switch_is_visible_where_recall_is_not() -> None:
    # The failure this whole pipeline is most likely to die of. Recall stays perfect
    # because every player is still found; only purity shows the swap.
    truth = doc(
        trk(1, "home", [(1, 10.0, 10.0), (2, 10.0, 10.0)]),
        trk(2, "home", [(1, 40.0, 40.0), (2, 40.0, 40.0)]),
    )
    pred = doc(
        trk(7, "home", [(1, 10.0, 10.0), (2, 40.0, 40.0)]),
        trk(8, "home", [(1, 40.0, 40.0), (2, 10.0, 10.0)]),
    )
    s = score(truth, pred)
    assert s.recall == 1.0
    assert s.identity_purity == 0.5
    assert s.id_switches == 2


def test_the_wrong_team_is_counted_separately_from_the_wrong_place() -> None:
    truth = doc(trk(1, "home", [(1, 10.0, 10.0)]))
    s = score(truth, doc(trk(1, "away", [(1, 10.0, 10.0)])))
    assert s.recall == 1.0
    assert s.team_accuracy == 0.0


def test_a_wrong_shirt_number_is_reported_apart_from_an_unread_one() -> None:
    # D5. Unread is free; wrong attaches a run to the wrong player invisibly. A scorer
    # that lumps them together hides the only jersey error that actually costs anything.
    truth = doc(trk(1, "home", [(1, 10.0, 10.0)], number=9))
    assert score(truth, doc(trk(1, "home", [(1, 10.0, 10.0)], number=9))).jersey_correct == 1
    assert score(truth, doc(trk(1, "home", [(1, 10.0, 10.0)], number=6))).jersey_wrong == 1
    assert score(truth, doc(trk(1, "home", [(1, 10.0, 10.0)], number=None))).jersey_missing == 1


def test_a_number_on_a_track_that_was_never_found_is_unread_not_wrong() -> None:
    truth = doc(trk(1, "home", [(1, 10.0, 10.0)], number=9))
    s = score(truth, doc(trk(1, "home", [(1, 90.0, 60.0)], number=9)))
    assert (s.jersey_correct, s.jersey_wrong, s.jersey_missing) == (0, 0, 1)


def test_matching_is_nearest_first_and_never_double_assigns() -> None:
    gt = [(1, 10.0, 10.0), (2, 10.5, 10.0)]
    pred = [(7, 10.4, 10.0)]
    pairs = match_frame(gt, pred)
    assert len(pairs) == 1
    assert pairs[0][0] == 2  # 0.1 m beats 0.4 m
