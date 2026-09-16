"""Where a chain is weakest, and how that is said to whoever clicks."""

from __future__ import annotations

import numpy as np

from football_tracks import guide, seed
from football_tracks.stage1_propagate import Chain

H = np.eye(3)


def chain(
    reach: dict[int, int], apart: dict[int, float] | None = None, lost: tuple[int, ...] = ()
) -> Chain:
    homs = {f: H for f in reach} | dict.fromkeys(lost)
    return Chain(
        homographies=homs,
        solved_directly=sum(1 for r in reach.values() if r == 0),
        carried=sum(1 for r in reach.values() if r > 0),
        gaps=len(lost),
        carried_from=reach,
        disagreement=apart or {},
    )


def test_a_clicked_frame_is_its_own_anchor_and_a_missing_one_is_lost() -> None:
    c = chain({1: 0, 2: 1}, lost=(3,))
    assert guide.standing(c, 1) == "clicked"
    assert guide.standing(c, 3) == "lost"
    # And a frame the chain never heard of is lost too, rather than a KeyError.
    assert guide.standing(c, 99) == "lost"


def test_distance_grades_a_frame_only_one_click_reaches() -> None:
    near, far, farther = guide.GOOD_REACH, guide.FAIR_REACH, guide.FAIR_REACH + 1
    c = chain({1: 0, 2: near, 3: far, 4: farther})
    assert [guide.standing(c, f) for f in (2, 3, 4)] == ["good", "fair", "poor"]


def test_measured_disagreement_outranks_distance() -> None:
    # Two hundred frames from either click would be poor by distance, and the two chains
    # agreeing to a metre is better evidence than that.
    c = chain({1: 0, 2: 200, 3: 200, 4: 200, 5: 0}, apart={2: 1.0, 3: 3.0, 4: 9.0})
    assert [guide.standing(c, f) for f in (2, 3, 4)] == ["good", "fair", "poor"]


def test_the_next_seed_is_where_the_clicks_disagree_most_when_they_can() -> None:
    got = guide.next_seed(chain({1: 0, 2: 300, 3: 10, 4: 0}, apart={3: 6.0}))
    assert got is not None and got[0] == 3


def test_the_next_seed_is_the_furthest_frame_with_one_click() -> None:
    got = guide.next_seed(chain({1: 0, 2: 40, 3: 90}))
    assert got is not None and got[0] == 3


def test_a_long_carry_from_one_click_outranks_two_clicks_that_nearly_agree() -> None:
    # nottingham: 732 frames hung off one anchor, and the suggestion went to 2.1 m.
    got = guide.next_seed(chain({1: 0, 2: 800, 3: 5, 4: 0}, apart={3: 2.1}))
    assert got is not None and got[0] == 2


def test_a_stretch_with_no_camera_is_clicked_in_its_middle() -> None:
    got = guide.next_seed(chain({1: 0, 2: 300}, apart={}, lost=(3, 4, 5, 6, 7)))
    assert got is not None and got[0] == 5


def test_nothing_carried_means_nowhere_to_click() -> None:
    assert guide.next_seed(chain({1: 0, 2: 0})) is None
    assert guide.next_seed(None) is None


def test_the_overview_counts_clicked_frames_and_coverage() -> None:
    c = chain({1: 0, 2: 5, 3: 0}, apart={2: 1.25}, lost=(4,))
    assert guide.overview(c) == (
        "2 frames clicked, camera followed on 75% of the clip, clicks agree within 1.2 m"
    )


def test_a_seed_no_camera_fits_is_said_plainly() -> None:
    assert "don't fit a camera" in guide.refusal(seed.Seed(frame=1, points=[]), 1920, 1080)


def test_a_stretch_somebody_gave_up_on_is_not_suggested_again() -> None:
    # Frames 2-3 are one poor stretch. Skipping frame 3 passes over frame 2 as well: the
    # angle that defeated one frame defeats its neighbours.
    c = chain({1: 0, 2: 300, 3: 299, 4: 0, 5: 130})
    first = guide.next_seed(c)
    assert first is not None and first[0] == 2
    after = guide.next_seed(c, avoid={3})
    assert after is not None and after[0] == 5


def test_stretches_are_consecutive_runs_of_one_standing() -> None:
    c = chain({1: 0, 2: 300, 3: 299, 4: 0, 5: 10}, lost=(6, 7))
    assert guide.stretches(c) == [
        (1, 1, "clicked"),
        (2, 3, "poor"),
        (4, 4, "clicked"),
        (5, 5, "good"),
        (6, 7, "lost"),
    ]


def test_every_refusal_says_how_to_skip() -> None:
    no_camera = guide.refusal(seed.Seed(frame=1, points=[]), 1920, 1080)
    no_start = guide.refusal(
        seed.Seed(frame=1, points=[], arcs=[((1.0, 1.0), (52.5, 34.0, 9.15))]), 1920, 1080
    )
    assert "Q: skip this frame" in no_camera and "Q: skip this frame" in no_start
    assert "first camera" in no_start
