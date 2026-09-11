"""Naming the side a track is on, and declining to when the kit does not say.

`assign` is the one stage that can put a player in the wrong colour, and a wrong colour
is not a cosmetic error: the board draws the pass between two shirts, so a mislabelled
player turns a pass into a turnover that never happened.
"""

from __future__ import annotations

import numpy as np

from football_tracks import stage3_teams
from football_tracks.stage2_track import Track
from football_tracks.stage3_teams import assign, kit_colours
from football_tracks.tracks import TeamLabel


def kitted(track_id: int, kit: tuple[float, float, float]) -> Track:
    t = Track(id=track_id)
    t.kit_sum = np.array(kit, dtype=np.float64)
    t.kit_seen = 1
    t.side_sum = np.array(kit, dtype=np.float64)
    t.side_seen = 1
    t.side_weight = 1.0
    return t


RED = (1.0, 0.0, 0.0)
BLUE = (0.0, 0.0, 1.0)


def test_two_clear_kits_become_two_sides() -> None:
    tracks = [kitted(i, RED) for i in range(4)] + [kitted(10 + i, BLUE) for i in range(4)]
    mean_x = {t.id: (20.0 if t.id < 10 else 80.0) for t in tracks}
    out = assign(tracks, mean_x)
    assert {out[i] for i in range(4)} == {"home"}
    assert {out[10 + i] for i in range(4)} == {"away"}


def test_a_kit_on_the_boundary_is_declined_rather_than_guessed() -> None:
    """A coin flip reaches the board as a player in the wrong colour, and the pass drawn
    to them is a turnover that never happened.

    Two kits that shade into each other, as they do under floodlights and a low camera:
    the ends of the range are still clear and the middle of it is not.
    """
    reds = [(1.0, 0.0, 0.0), (0.85, 0.0, 0.15), (0.7, 0.0, 0.3), (0.55, 0.0, 0.45)]
    blues = [(0.0, 0.0, 1.0), (0.15, 0.0, 0.85), (0.3, 0.0, 0.7), (0.45, 0.0, 0.55)]
    tracks = [kitted(i, k) for i, k in enumerate(reds)]
    tracks += [kitted(10 + i, k) for i, k in enumerate(blues)]
    mean_x = {t.id: (20.0 if t.id < 10 else 80.0) for t in tracks}

    out = assign(tracks, mean_x)
    assert out[0] == "home", "the reddest kit is not in doubt"
    assert out[10] in {"away", "referee"}, "nor is the bluest"
    boundary = {out[3], out[13]}
    assert "unknown" in boundary, f"a kit between the two must not be guessed at: {boundary}"


def test_a_track_with_no_kit_at_all_is_still_unknown() -> None:
    tracks = [kitted(i, RED) for i in range(3)] + [kitted(10 + i, BLUE) for i in range(3)]
    blind = Track(id=50)
    tracks.append(blind)
    mean_x = {t.id: (20.0 if t.id < 10 else 80.0) for t in tracks}
    mean_x[50] = 50.0
    assert assign(tracks, mean_x)[50] == "unknown"


def worn(track_id: int, hue: int | None, share: float, bgr: tuple[float, float, float]) -> Track:
    """A track whose signature puts `share` of itself in one hue, and the rest colourless.

    `hue` is a bin of the twelve the signature keeps, each thirty real degrees wide: 0 is
    red, 2 is green, 7 is blue. None puts everything in the colourless bin, which is what
    a white, grey or black kit looks like. Bins 2 to 5 are the PITCH's, struck out before
    anything is painted, so a kit cannot be written in them.
    """
    t = Track(id=track_id)
    sig = np.zeros(49, dtype=np.float64)
    if hue is None:
        sig[48] = 1.0
    else:
        sig[hue * 4 + 2] = share
        sig[48] = 1.0 - share
    t.side_sum, t.side_seen, t.side_weight = sig, 1, 1.0
    t.kit_sum, t.kit_seen = sig, 1
    t.tone_sum = np.array(bgr, dtype=np.float64)
    return t


def test_a_kit_is_painted_from_its_biggest_hue_and_not_from_an_average() -> None:
    # The average of a torso crop is blunt: floodlights, blur, grass and a white sleeve all
    # pull it towards grey, so a red shirt and a green one average to much the same olive.
    # The histogram never mixed them -- which is why it can name the sides at all (D92).
    teams: dict[int, TeamLabel] = {1: "home", 2: "away"}
    olive = (73.0, 101.0, 109.0)
    got = kit_colours([worn(1, 0, 0.6, olive), worn(2, 7, 0.6, olive)], teams)

    assert got is not None
    # Both sides measure the SAME mean colour, and are still painted apart.
    assert got["home"] != got["away"]


def test_a_trim_does_not_decide_the_kit() -> None:
    # A white shirt with a coloured collar puts a few percent in that hue. Painting the
    # side by it is worse than painting it white.
    teams: dict[int, TeamLabel] = {1: "home", 2: "away"}
    got = kit_colours(
        [worn(1, 2, 0.05, (215.0, 215.0, 215.0)), worn(2, 0, 0.6, (40.0, 40.0, 190.0))], teams
    )
    assert got is not None
    assert got["home"] == "#e6e6e6"


def test_two_sides_painted_the_same_are_not_offered_at_all() -> None:
    # A board painting both sides one colour is worse than one painting them its own two.
    teams: dict[int, TeamLabel] = {1: "home", 2: "away"}
    red = (40.0, 40.0, 200.0)
    assert kit_colours([worn(1, 0, 0.6, red), worn(2, 0, 0.6, red)], teams) is None


def test_a_side_with_no_shirt_read_offers_no_colour() -> None:
    teams: dict[int, TeamLabel] = {1: "home", 2: "away"}
    assert kit_colours([worn(1, 0, 0.6, (40.0, 40.0, 200.0)), kitted(2, BLUE)], teams) is None


def test_a_shirt_with_no_colour_is_not_given_one() -> None:
    # White, grey and black kits have no hue to read -- the signature gathers every one of
    # their pixels into the colourless bin -- so brightness decides, which an average is
    # perfectly good at.
    teams: dict[int, TeamLabel] = {1: "home", 2: "away"}
    got = kit_colours(
        [worn(1, None, 0.0, (215.0, 215.0, 215.0)), worn(2, None, 0.0, (35.0, 35.0, 35.0))], teams
    )
    assert got is not None
    assert got["home"] == "#e6e6e6"
    assert got["away"] == "#2b2b2b"


def logged(track_id: int, kits: list[tuple[int, tuple[float, float]]]) -> Track:
    t = Track(id=track_id)
    for f, kit in kits:
        t.saw_kit(np.array(kit, dtype=np.float64), None, f)
    return t


CENTRES = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))


def test_a_declined_track_is_cut_where_the_shirt_changes() -> None:
    # Only ever asked of a track already declined (D72), which is what makes it safe: a
    # declined track is thrown away, so a cut that explains it costs nothing when it fails
    # and returns two players when it works.
    two = logged(
        1, [(f, (1.0, 0.0)) for f in range(1, 13)] + [(f, (0.0, 1.0)) for f in range(13, 25)]
    )
    got = stage3_teams.two_shirts(two, CENTRES)
    assert got is not None
    at, first, second = got
    assert at == 13
    assert (first, second) == (0, 1)


def test_one_player_in_changing_light_is_not_two_players() -> None:
    # Both halves land on the same side: the kit moved, the shirt did not.
    same = logged(
        2, [(f, (1.0, 0.0)) for f in range(1, 13)] + [(f, (0.8, 0.2)) for f in range(13, 25)]
    )
    assert stage3_teams.two_shirts(same, CENTRES) is None


def test_a_cut_that_leaves_a_half_still_ambiguous_is_refused() -> None:
    muddy = logged(
        3, [(f, (1.0, 0.0)) for f in range(1, 13)] + [(f, (0.5, 0.5)) for f in range(13, 25)]
    )
    assert stage3_teams.two_shirts(muddy, CENTRES) is None


def test_a_track_with_too_few_readings_is_left_alone() -> None:
    short = logged(
        4, [(f, (1.0, 0.0)) for f in range(1, 6)] + [(f, (0.0, 1.0)) for f in range(6, 11)]
    )
    assert stage3_teams.two_shirts(short, CENTRES) is None


def _a_clip(rng: np.random.Generator) -> tuple[list[Track], dict[int, float]]:
    """Six a side with a real spread, and three officials.

    The officials matter: `MAX_REFEREES` is spent on the oddest kits, and without anything
    genuinely odd on the pitch a merely AMBIGUOUS one is the oddest thing there and gets
    named an official instead of being declined. Real clips have both.
    """
    tracks, mean_x = [], {}
    for i in range(12):
        base = np.array(RED if i < 6 else BLUE, dtype=np.float64)
        worn = np.abs(base + rng.normal(0, 0.08, 3))
        tracks.append(kitted(i, (float(worn[0]), float(worn[1]), float(worn[2]))))
        mean_x[i] = 25.0 if i < 6 else 80.0
    for j, odd in enumerate([(0.0, 1.0, 0.0), (0.05, 0.95, 0.0), (0.0, 0.95, 0.05)]):
        tracks.append(kitted(90 + j, odd))
        mean_x[90 + j] = 50.0 + j
    return tracks, mean_x


# Leaning red, and short of the margin KIT_MARGIN asks for.
BETWEEN = (0.52, 0.0, 0.48)


def test_a_track_the_ball_went_through_is_named_on_the_plain_comparison() -> None:
    # KIT_MARGIN buys silence, and silence is the right price for the twenty-one players
    # who are not on the ball. For the one who IS, declining does not leave the move
    # uncoloured -- Pitchboard fields nobody it cannot name -- it leaves the move undrawn,
    # and a coach watched his striker receive, run and win a penalty while the board stood
    # somebody else offside in his place (D91).
    tracks, mean_x = _a_clip(np.random.default_rng(0))
    tracks.append(kitted(99, BETWEEN))
    mean_x[99] = 40.0

    assert assign(tracks, mean_x)[99] == "unknown"
    assert assign(tracks, mean_x, carried={99})[99] == "home"


def test_the_exemption_is_one_track_and_not_a_looser_threshold() -> None:
    # The ball going through somebody says nothing about anybody else's shirt. A carrier
    # who was already named changes nothing, and the ambiguous track stays declined.
    tracks, mean_x = _a_clip(np.random.default_rng(0))
    tracks.append(kitted(99, BETWEEN))
    mean_x[99] = 40.0

    assert assign(tracks, mean_x, carried={0})[99] == "unknown"
    assert assign(tracks, mean_x, carried={99})[99] == "home"


def test_a_kit_the_pitch_is_wearing_is_not_a_kit() -> None:
    # Every torso crop is part shirt and part grass, and for a dark kit the grass is the
    # larger share: read without striking the pitch out, SNGS-147's red-and-black team came
    # out green (D92). A side whose only colour is the pitch's has none.
    teams: dict[int, TeamLabel] = {1: "home", 2: "away"}
    assert kit_colours([worn(1, 3, 0.8, (60.0, 120.0, 40.0)), worn(2, 0, 0.6, RED)], teams) is None


def test_a_side_its_own_signature_cannot_settle_is_not_painted() -> None:
    # 41% colourless against 25% of everything else is Sporting's green-and-white hoops,
    # whose green IS the pitch's. Painted anyway they came out yellow; the board's palette
    # is the better answer (D92).
    teams: dict[int, TeamLabel] = {1: "home", 2: "away"}
    undecided = Track(id=1)
    sig = np.zeros(49, dtype=np.float64)
    sig[0 * 4 + 2] = 0.06
    sig[1 * 4 + 2] = 0.19
    sig[2 * 4 + 2] = 0.34
    sig[48] = 0.41
    undecided.side_sum, undecided.side_seen, undecided.side_weight = sig, 1, 1.0
    undecided.tone_sum = np.array((73.0, 101.0, 109.0), dtype=np.float64)

    assert kit_colours([undecided, worn(2, 0, 0.6, RED)], teams) is None
