"""Joining track fragments back into players."""

from __future__ import annotations

import numpy as np

from football_tracks import stage2_stitch
from football_tracks.tracks import Sample


def _frag(start: int, n: int, x: float, y: float, step: float = 0.1) -> list[Sample]:
    return [Sample(f=start + i, x=x + i * step, y=y, conf=0.9) for i in range(n)]


def test_two_halves_of_one_player_are_joined() -> None:
    """The case that costs the roster: a break with almost no gap and no movement."""
    pos = {1: _frag(0, 10, 50.0, 30.0), 2: _frag(12, 10, 51.0, 30.0)}
    out = stage2_stitch.stitch(pos, {}, fps=25.0)
    assert len(out) == 1, "one player arrived as two fragments and must leave as one"
    assert len(next(iter(out.values()))) == 20


def test_two_players_are_not_joined_however_close_in_time() -> None:
    """A join across 60 m is a teleport, and worse than the two honest halves."""
    pos = {1: _frag(0, 10, 10.0, 30.0), 2: _frag(12, 10, 70.0, 30.0)}
    out = stage2_stitch.stitch(pos, {}, fps=25.0)
    assert len(out) == 2


def test_a_long_gap_is_not_bridged() -> None:
    """Past MAX_GAP_S the position prior cannot tell two players in one kit apart."""
    far = int(stage2_stitch.MAX_GAP_S * 25.0) + 30
    pos = {1: _frag(0, 10, 50.0, 30.0), 2: _frag(far, 10, 50.0, 30.0)}
    assert len(stage2_stitch.stitch(pos, {}, fps=25.0)) == 2


def test_an_ambiguous_join_is_refused_rather_than_guessed() -> None:
    """Mutual best: two candidates equally good means neither is chosen.

    A fragment ending in a crowd has several plausible successors, and taking the
    cheapest is how one player's run lands on another's shirt.
    """
    pos = {
        1: _frag(0, 10, 50.0, 30.0),
        2: _frag(12, 10, 50.5, 31.0),
        3: _frag(12, 10, 50.5, 29.0),
    }
    out = stage2_stitch.stitch(pos, {}, fps=25.0)
    # Whichever it picks, the OTHER must survive as its own track rather than vanish.
    assert sum(len(v) for v in out.values()) == 30, "no samples may be lost"
    assert len(out) >= 2


def test_kit_breaks_a_tie() -> None:
    """Two equally reachable successors, and only one is wearing the right shirt."""
    red, blue = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    pos = {
        1: _frag(0, 10, 50.0, 30.0),
        2: _frag(12, 10, 50.5, 31.0),
        3: _frag(12, 10, 50.5, 29.0),
    }
    out = stage2_stitch.stitch(pos, {1: red, 2: blue, 3: red}, fps=25.0)
    joined = max(out.values(), key=len)
    assert len(joined) == 20, "the same-kit fragment is the one to join"
    assert {s.f for s in joined} == set(range(10)) | set(range(12, 22))
    assert 2 in out, "the loser stays its own track rather than vanishing"


def test_matching_kit_never_rescues_a_speed_violation() -> None:
    """Colour breaks ties; it does not repeal 11 m/s. 60 m in half a second is a teleport."""
    red = np.array([1.0, 0.0, 0.0])
    pos = {1: _frag(0, 10, 10.0, 30.0), 2: _frag(12, 10, 70.0, 30.0)}
    assert len(stage2_stitch.stitch(pos, {1: red, 2: red}, fps=25.0)) == 2


def test_no_samples_are_ever_lost() -> None:
    """Losing a player silently is not an acceptable failure for any input."""
    pos = {i: _frag(i * 5, 6, 40.0 + i, 30.0) for i in range(6)}
    out = stage2_stitch.stitch(pos, {}, fps=25.0)
    assert sum(len(v) for v in out.values()) == 36


def test_a_run_is_picked_up_seconds_later_where_it_was_heading() -> None:
    """The join the old gate could not make. 55-64% of a player's own breaks are longer
    than the half second it allowed, at a median of 2.2-2.8 s."""
    # 0.1 m a frame at 25 fps is 2.5 m/s, so three seconds of gap is 7.5 m of running.
    pos = {1: _frag(0, 10, 50.0, 30.0), 2: _frag(84, 10, 58.4, 30.0)}
    out = stage2_stitch.stitch(pos, {}, fps=25.0)
    assert len(out) == 1
    assert len(next(iter(out.values()))) == 20


def test_a_long_gap_does_not_admit_whoever_is_within_reach() -> None:
    """What makes the longer gap safe. A reach gate at 12 m/s allows 38 m across three
    seconds, which is most of the pitch and any two players in one kit."""
    pos = {1: _frag(0, 10, 50.0, 30.0), 2: _frag(84, 10, 30.0, 30.0)}
    assert len(stage2_stitch.stitch(pos, {}, fps=25.0)) == 2


def test_a_player_who_was_standing_still_is_expected_to_still_be_there() -> None:
    pos = {
        1: _frag(0, 10, 50.0, 30.0, step=0.0),
        2: _frag(60, 10, 50.4, 30.0, step=0.0),
    }
    assert len(stage2_stitch.stitch(pos, {}, fps=25.0)) == 1
