"""Joining track fragments back into players."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_two_kits_are_not_one_player_however_well_the_prediction_lands() -> None:
    # The same guard the tracker makes (D78), one gap later. A fragment ending where the
    # next begins is the whole of the geometric case for a join, and it is exactly what an
    # opponent standing in the runner's path also looks like -- so the kit decides, and a
    # weight cannot: the join is refused outright.
    yellow, white = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    pos = {1: _frag(0, 10, 50.0, 30.0), 2: _frag(12, 10, 51.0, 30.0)}
    assert len(stage2_stitch.stitch(pos, {1: yellow, 2: white}, fps=25.0)) == 2
    assert len(stage2_stitch.stitch(pos, {1: yellow, 2: yellow}, fps=25.0)) == 1
    # An unread kit is not a disagreement, and must still join.
    assert len(stage2_stitch.stitch(pos, {1: yellow}, fps=25.0)) == 1


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


def _jittery(start: int, n: int, x: float, y: float, step: float, fps: float) -> list[Sample]:
    """A player running in a straight line, seen through a camera model that wobbles."""
    wobble = [0.0, 0.35, -0.3, 0.25, -0.35, 0.3, -0.25, 0.2]
    return [
        Sample(f=start + i, x=x + i * step, y=y + wobble[i % len(wobble)], conf=0.9)
        for i in range(n)
    ]


def test_velocity_reads_the_same_run_at_any_frame_rate() -> None:
    """It was read over five SAMPLES, which is half a second at 25 fps and an eighth of one
    at 33 -- and an eighth of a second turns position noise into a sprint sideways."""
    slow = _jittery(0, 30, 50.0, 30.0, 5.0 / 25, 25.0)
    fast = _jittery(0, 60, 50.0, 30.0, 5.0 / 50, 50.0)
    vx_slow, vy_slow = stage2_stitch._velocity(slow, 25.0, at_end=True)
    vx_fast, vy_fast = stage2_stitch._velocity(fast, 50.0, at_end=True)
    assert vx_slow == pytest.approx(vx_fast, abs=0.6)
    assert vy_slow == pytest.approx(vy_fast, abs=0.6)
    assert vx_slow == pytest.approx(5.0, abs=1.0), "the player is running at 5 m/s"


def test_a_run_and_a_shot_are_one_player_at_a_high_frame_rate() -> None:
    """The real failure: a 33 fps clip, a break of a third of a second, two fragment ends
    two metres apart -- and the board drew the shot as a pass to somebody standing there."""
    fps = 33.4
    step = -6.0 / fps  # a 6 m/s run towards the goal
    runner = _jittery(44, 70, 20.0, 23.0, step, fps)
    # Where that run has reached by the time the tracker picks him up again, eleven frames
    # after it lost him -- two metres on, which is what the real pair looked like.
    resumes = runner[-1].x + step * 11
    shooter = _jittery(125, 69, resumes, 22.8, step, fps)
    out = stage2_stitch.stitch({17: runner, 21: shooter}, {}, fps=fps)
    assert len(out) == 1, "one player who ran and then shot, not two"
    assert len(next(iter(out.values()))) == len(runner) + len(shooter)


def _every_other(start: int, n: int, x: float, y: float, step: float, odd: bool) -> list[Sample]:
    """Half a player's frames — what one of two tracks taking turns on him looks like."""
    return [
        Sample(f=start + i, x=x + i * step, y=y, conf=0.9) for i in range(n) if (i % 2 == 1) == odd
    ]


def test_two_tracks_taking_turns_on_one_player_are_one_player() -> None:
    # The tracker renumbers a player without ever losing him, so both ids stay live and
    # alternate frames. `stitch` cannot see it: it asks whether one fragment CONTINUES
    # another and refuses anything overlapping (D90).
    positions = {
        1: _every_other(100, 60, 40.0, 30.0, 0.1, odd=False),
        2: _every_other(100, 60, 40.3, 30.2, 0.1, odd=True),
    }
    assert stage2_stitch.duplicates(positions, 25.0) == {2: 1}


def test_a_striker_and_the_man_marking_him_are_two_players() -> None:
    # The case distance alone cannot separate, and the reason the frame test exists: these
    # two run a metre apart for the whole move, and merging them destroys two players.
    # The detector finds each of them every frame, which is what says there are two.
    both = {
        1: _frag(100, 60, 40.0, 30.0, 0.1),
        2: _frag(100, 60, 40.0, 31.2, 0.1),
    }
    assert stage2_stitch.duplicates(both, 25.0) == {}


def test_tracks_far_apart_are_left_alone_however_they_interleave() -> None:
    positions = {
        1: _every_other(100, 60, 40.0, 30.0, 0.1, odd=False),
        2: _every_other(100, 60, 40.0, 50.0, 0.1, odd=True),
    }
    assert stage2_stitch.duplicates(positions, 25.0) == {}


def test_merging_keeps_one_sample_a_frame_and_fills_the_gaps() -> None:
    positions = {
        1: _every_other(100, 60, 40.0, 30.0, 0.1, odd=False),
        2: _every_other(100, 60, 40.3, 30.2, 0.1, odd=True),
    }
    merged = stage2_stitch.merge(positions, {2: 1})
    assert 2 not in merged
    frames = [s.f for s in merged[1]]
    assert frames == sorted(frames)
    assert len(frames) == len(set(frames)) == 60
