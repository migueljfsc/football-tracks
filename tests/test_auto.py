"""The automatic path's own helpers — the pure ones."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_tracks import auto
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


def test_every_stage_has_its_command() -> None:
    """One command per stage, and the CLI is the only way in.

    Worth asserting because nothing else does: a command is registered by a decorator, and
    a decorator separated from its function by an edit binds to whatever follows instead.
    `ft seed` vanished that way -- the pipeline still ran, every test still passed, and the
    one step that needs a human was simply not there.
    """
    from football_tracks.cli import app

    registered = {c.callback.__name__ for c in app.registered_commands if c.callback}
    for name in (
        "auto",
        "bench",
        "calibrate",
        "camera",
        "detect",
        "frames",
        "learn",
        "pitch",
        "reg-eval".replace("-", "_"),
        "reid",
        "render",
        "run",
        "score",
        "seed",
        "segment",
        "truth",
    ):
        assert name in registered, f"`ft {name.replace('_', '-')}` is not registered"
    # And no helper: a function under a stray decorator is registered as a command nobody meant.
    assert "_pipeline" not in registered, "`_pipeline` is a helper, not a command"


def test_a_clip_named_for_a_match_stays_named_for_it(tmp_path: Path, monkeypatch: Any) -> None:
    """`--game` on `ft run` or `ft auto` is recorded beside the clip, so the next command on it
    finds the match's camera and kits without being told again -- and a clip nobody named is
    simply not in a match, rather than an error for the commands that do not need one."""
    from football_tracks import cli

    monkeypatch.setattr(cli, "CLIPS", tmp_path)
    (tmp_path / "Untitled_1").mkdir()
    (tmp_path / "Untitled_1" / "clip.json").write_text(json.dumps({"name": "Untitled_1"}))

    assert cli._maybe_game("Untitled_1", None) is None
    cli._name_game("Untitled_1", "milan-benfica")
    assert cli._game_of("Untitled_1", None) == "milan-benfica"
    # Said outright, the flag wins over what was recorded.
    assert cli._game_of("Untitled_1", "another-match") == "another-match"
    # A clip with no clip.json at all -- a SoccerNet one -- is left alone.
    cli._name_game("nowhere", "milan-benfica")
    assert not (tmp_path / "nowhere").exists()


def test_a_still_player_stops_twitching() -> None:
    """Two wobbles land on every position -- the camera's aim and the detector's box -- and
    a coach reads them off the dot video as *"the player dots are very twitchy"*."""
    jitter = [0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3]
    samples = [Sample(f=f, x=50.0 + j, y=34.0) for f, j in enumerate(jitter, start=1)]
    out = auto.settle(samples, window=2)
    assert [s.f for s in out] == [s.f for s in samples]
    assert max(abs(s.x - 50.0) for s in out[2:-2]) < 0.15


def test_a_run_arrives_where_it_was_going() -> None:
    """The window is centred, so a constant speed is not delayed and not shortened."""
    samples = [Sample(f=f, x=float(f), y=34.0) for f in range(1, 12)]
    out = auto.settle(samples, window=2)
    assert [round(s.x, 6) for s in out[2:-2]] == [float(f) for f in range(3, 10)]


def test_a_sample_with_no_neighbours_is_left_alone() -> None:
    """A gap is not filled and its edges are not dragged across it (D8)."""
    samples = [
        Sample(f=1, x=10.0, y=34.0),
        Sample(f=2, x=10.0, y=34.0),
        Sample(f=99, x=60.0, y=34.0),
    ]
    out = auto.settle(samples, window=2)
    assert out[-1].x == 60.0
    assert [s.f for s in out] == [1, 2, 99]
