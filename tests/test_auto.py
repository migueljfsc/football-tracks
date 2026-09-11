"""The automatic path's own helpers — the pure ones."""

from __future__ import annotations

from football_tracks.tracks import Sample


def test_the_ball_goes_through_whoever_is_nearest_and_stays_there() -> None:
    # Nearest player, within reach, for long enough that his side stops being a detail.
    # Not a claim about control: only that the move went through him (D91).
    from football_tracks.auto import CARRIER_RADIUS_M, on_the_ball

    fps = 25.0
    ball = [Sample(f=f, x=10.0 + f * 0.2, y=30.0, conf=0.9) for f in range(40)]
    with_it = [Sample(f=f, x=10.0 + f * 0.2, y=30.5, conf=0.9) for f in range(40)]
    away = [Sample(f=f, x=60.0, y=10.0, conf=0.9) for f in range(40)]
    brushed = [
        Sample(f=f, x=10.0 + f * 0.2, y=30.5, conf=0.9)
        if f < 5
        else Sample(f=f, x=60.0, y=10.0, conf=0.9)
        for f in range(40)
    ]
    got = on_the_ball(ball, {1: with_it, 2: away, 3: brushed}, fps)
    assert got == {1}
    assert CARRIER_RADIUS_M > 0


def test_no_ball_means_nobody_was_on_it() -> None:
    from football_tracks.auto import on_the_ball

    assert on_the_ball([], {1: [Sample(f=1, x=0.0, y=0.0, conf=0.9)]}, 25.0) == set()
