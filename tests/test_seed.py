"""The human seed: clicked landmarks -> a camera model."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from football_tracks import calibration, seed
from football_tracks.config import DEFAULT_PITCH, PITCH_LENGTH, PITCH_WIDTH, Pitch

PITCH_TO_IMAGE = np.asarray(
    cv2.getPerspectiveTransform(
        np.array(
            [[0, 0], [PITCH_LENGTH, 0], [PITCH_LENGTH, PITCH_WIDTH], [0, PITCH_WIDTH]], np.float32
        ),
        np.array([[620, 300], [1480, 315], [1880, 1010], [80, 960]], np.float32),
    ),
    dtype=np.float64,
)


def clicked(names: list[str], *, jitter: float = 0.0) -> seed.Seed:
    rng = np.random.default_rng(0)
    pts = []
    for n in names:
        pitch = seed.landmarks()[n]
        img = calibration.apply(PITCH_TO_IMAGE, np.array([pitch]))[0]
        if jitter:
            img = img + rng.normal(0, jitter, 2)
        pts.append(((float(img[0]), float(img[1])), pitch))
    return seed.Seed(frame=1, points=pts)


# Spread across the pitch on purpose. The obvious four - both posts and both corners -
# all sit on x = 0 and are degenerate; `test_landmarks_along_one_line_are_refused`
# pins that.
SIX = [
    "goal post far",
    "goal post near",
    "penalty box front far",
    "penalty box front near",
    "6yd front far",
    "penalty spot",
]


def test_clicked_landmarks_recover_the_camera() -> None:
    h = seed.homography(clicked(SIX))
    assert h is not None
    got = calibration.apply(h, calibration.apply(PITCH_TO_IMAGE, np.array([[52.5, 34.0]])))
    assert got[0] == pytest.approx([52.5, 34.0], abs=0.01)


def test_four_points_is_the_minimum() -> None:
    assert seed.homography(clicked(SIX[:3])) is None
    assert seed.homography(clicked(SIX[:4])) is not None


def test_a_misclick_is_absorbed_when_there_are_spare_points() -> None:
    # Four points fit whatever was misclicked and cannot be checked - the same trap as
    # D17 one level up. With six, RANSAC has something to disagree with.
    pts = clicked(SIX).points
    (ix, iy), pitch = pts[2]
    pts[2] = ((ix + 60.0, iy - 40.0), pitch)
    h = seed.homography(seed.Seed(frame=1, points=pts))
    assert h is not None
    got = calibration.apply(h, calibration.apply(PITCH_TO_IMAGE, np.array([[52.5, 34.0]])))
    assert got[0] == pytest.approx([52.5, 34.0], abs=1.5)


def test_landmarks_along_one_line_are_refused() -> None:
    # Both posts and both corners of a goal are the four most natural things to click,
    # and all four sit on x = 0. That fits perfectly and describes nothing, so it is
    # refused rather than returned - the same call as D17.
    on_the_goal_line = ["goal post far", "goal post near", "corner far", "corner near"]
    assert seed.homography(clicked(on_the_goal_line)) is None


def test_the_far_goal_is_the_same_landmark_mirrored() -> None:
    # So a coach never has to think about which end the pitch model calls zero.
    assert seed.landmarks()["goal post far"][0] == 0.0
    assert seed.mirrored("goal post far")[0] == PITCH_LENGTH
    assert seed.mirrored("goal post far")[1] == seed.landmarks()["goal post far"][1]


def test_a_seed_survives_a_round_trip(tmp_path: Path) -> None:
    original = clicked(SIX)
    seed.write(tmp_path / "seed.json", original)
    back = seed.read(tmp_path / "seed.json")
    assert back.frame == original.frame
    assert len(back.points) == len(original.points)
    assert back.points[0][1] == original.points[0][1]


def flipped(s: seed.Seed) -> seed.Seed:
    return seed.Seed(
        frame=s.frame, points=[(i, (px, PITCH_WIDTH - py)) for i, (px, py) in s.points]
    )


def test_a_correctly_seeded_clip_agrees_with_the_camera() -> None:
    # The synthetic camera has the near touchline at the bottom of the frame, which is
    # what a broadcast camera on a touchline always gives.
    assert seed.orientation(clicked(SIX)) > seed.ORIENTATION_CONFIDENT


def test_swapped_far_and_near_is_detected() -> None:
    # The failure the reprojection overlay is blind to: a pitch is symmetric about the
    # halfway line, so a y-mirrored model lands on the real markings perfectly and only
    # the arithmetic can tell.
    assert seed.orientation(flipped(clicked(SIX))) < -seed.ORIENTATION_CONFIDENT


def test_flipping_restores_the_orientation() -> None:
    assert seed.orientation(seed.flip_y(flipped(clicked(SIX)))) > seed.ORIENTATION_CONFIDENT


def test_flipping_keeps_the_clicks_and_moves_only_the_pitch_side() -> None:
    original = clicked(SIX)
    turned = seed.flip_y(original)
    assert [p[0] for p in turned.points] == [p[0] for p in original.points]
    assert [p[1][0] for p in turned.points] == [p[1][0] for p in original.points]


def traced(name: str, n: int = 6) -> list[tuple[tuple[float, float], tuple[float, float, float]]]:
    """Points along a named marking, projected through the synthetic camera."""
    a, b, c = seed.traceable()[name]
    out = []
    for t in np.linspace(0.15, 0.85, n):
        p = (-c / a, t * PITCH_WIDTH) if abs(a) > abs(b) else (t * PITCH_LENGTH, -c / b)
        img = calibration.apply(PITCH_TO_IMAGE, np.array([p]))[0]
        out.append(((float(img[0]), float(img[1])), (a, b, c)))
    return out


def test_tracing_two_crossing_markings_recovers_the_camera() -> None:
    # What a tight goalmouth shot actually offers: long clear lines whose corners are
    # off screen. Clicking anywhere along them is enough.
    s = seed.Seed(
        frame=1,
        points=[],
        lines=traced("goal line")
        + traced("penalty box front")
        + traced("far touchline")
        + traced("near touchline"),
    )
    h = seed.homography(s)
    assert h is not None
    got = calibration.apply(h, calibration.apply(PITCH_TO_IMAGE, np.array([[52.5, 34.0]])))
    assert got[0] == pytest.approx([52.5, 34.0], abs=0.05)


def test_a_midfield_view_can_be_traced_from_what_it_shows() -> None:
    """What a camera parked near the centre circle offers: the halfway line, a box line
    at the far end of the shot, and the touchlines. A real clip was refused because the
    halfway line could not be traced at all, leaving everything in one band."""
    s = seed.Seed(
        frame=1,
        points=[],
        lines=traced("halfway line")
        + traced("penalty box front")
        + traced("far touchline")
        + traced("near touchline"),
    )
    h = seed.homography(s)
    assert h is not None
    got = calibration.apply(h, calibration.apply(PITCH_TO_IMAGE, np.array([[52.5, 34.0]])))
    assert got[0] == pytest.approx([52.5, 34.0], abs=0.05)


def test_one_line_across_and_two_along_is_refused_rather_than_guessed() -> None:
    """The halfway line and both touchlines look like plenty and are not: two parallel
    markings fix the scale between them, and the single crossing line fixes an origin and
    no scale at all. Fitted anyway it puts the centre spot 19 m from where it belongs."""
    s = seed.Seed(
        frame=1,
        points=[],
        lines=traced("halfway line") + traced("far touchline") + traced("near touchline"),
    )
    assert seed.homography(s) is None


def test_the_halfway_line_is_the_same_line_from_either_end() -> None:
    a, b, c = seed.traceable()["halfway line"]
    ma, mb, mc = seed.mirrored_line("halfway line")
    # Mirroring negates it, which is the same line: -x + 52.5 = 0 is x = 52.5.
    assert (ma, mb, mc) == pytest.approx((-a, b, -c))


def test_the_circle_crossings_are_the_same_from_either_end() -> None:
    # They sit ON the halfway line, so which goal is far cannot move them -- and they are
    # the only exact points a midfield view offers.
    for name in ("circle far", "circle near"):
        assert seed.mirrored(name) == seed.landmarks()[name]


def test_a_marking_labelled_with_the_wrong_side_is_named() -> None:
    """The failure a real clip hit: the top of the picture traced as the NEAR touchline,
    which is where the far one is. The fit that comes back is a compromise between two
    contradictory claims, and "no usable seed" says nothing about which."""
    top = [((x, 300.0), seed.traceable()["near touchline"]) for x in (200.0, 900.0, 1600.0)]
    lower = [((x, 700.0), seed.traceable()["penalty box near side"]) for x in (300.0, 1000.0)]
    got = seed.contradictions(seed.Seed(frame=1, points=[], lines=top + lower))
    assert got == [("near touchline", "penalty box near side")] or got == [
        ("penalty box near side", "near touchline")
    ]


def test_markings_the_right_way_round_are_not_complained_about() -> None:
    # Nearer the camera is lower in the frame: the near touchline BELOW the box's near side.
    near = [((x, 900.0), seed.traceable()["near touchline"]) for x in (200.0, 900.0)]
    box = [((x, 500.0), seed.traceable()["penalty box near side"]) for x in (300.0, 1000.0)]
    assert seed.contradictions(seed.Seed(frame=1, points=[], lines=near + box)) == []


def test_the_orientation_check_reads_traced_lines_too() -> None:
    """It asked for three clicked landmarks, so a seed made of traced lines and a click
    or two -- the ones most likely to hold a swap -- never got an opinion at all."""
    swapped = [((x, 300.0), seed.traceable()["near touchline"]) for x in (200.0, 900.0)]
    swapped += [((x, 800.0), seed.traceable()["far touchline"]) for x in (300.0, 1000.0)]
    assert seed.orientation(seed.Seed(frame=1, points=[], lines=swapped)) < 0


def test_tracing_only_parallel_markings_is_refused() -> None:
    # Three lines all parallel to the goal line leave the camera free to slide along
    # the pitch. The fit would come back looking like any other matrix.
    s = seed.Seed(
        frame=1,
        points=[],
        lines=traced("goal line") + traced("6yd box front") + traced("penalty box front"),
    )
    assert seed.homography(s) is None


def test_landmarks_and_traces_combine() -> None:
    s = seed.Seed(
        frame=1,
        points=[clicked(["goal post far"]).points[0], clicked(["goal post near"]).points[0]],
        lines=traced("penalty box front")
        + traced("penalty box near side")
        + traced("far touchline"),
    )
    h = seed.homography(s)
    assert h is not None
    got = calibration.apply(h, calibration.apply(PITCH_TO_IMAGE, np.array([[30.0, 40.0]])))
    assert got[0] == pytest.approx([30.0, 40.0], abs=0.5)


def test_two_traced_lines_alone_are_refused() -> None:
    # Two lines always cross, and a homography sending the whole image to that crossing
    # satisfies every point-on-line constraint exactly. It fits with zero residuals,
    # which is the most convincing way to be wrong.
    s = seed.Seed(
        frame=1,
        points=[],
        lines=traced("penalty box front") + traced("penalty box near side"),
    )
    assert seed.homography(s) is None


def test_a_traced_seed_can_be_flipped_and_round_tripped(tmp_path: Path) -> None:
    s = seed.Seed(frame=1, points=[], lines=traced("goal line") + traced("far touchline"))
    seed.write(tmp_path / "s.json", s)
    back = seed.read(tmp_path / "s.json")
    assert len(back.lines) == len(s.lines)
    # Flipping y must move the marking to the mirrored side, not leave it be.
    flipped_line = seed.flip_y(s).lines[-1][1]
    assert flipped_line != s.lines[-1][1]


def test_one_bad_click_does_not_drag_the_others_with_it() -> None:
    """A least-squares fit spreads a bad click's error over every other point.

    On a real seed that meant EVERY residual exceeded the misclick threshold, so the
    single-pass trim dropped all eleven landmarks, found what remained degenerate, and
    returned the very fit it had been trying to repair. Dropping the worst one at a
    time and refitting recovers instead.
    """
    names = ["goal post far", "goal post near", "6yd front far", "6yd front near", "penalty spot"]
    points = list(clicked(names).points)
    (image, pitch) = points[2]
    points[2] = ((image[0] + 220.0, image[1] - 160.0), pitch)  # one badly placed click

    s = seed.Seed(
        frame=1,
        points=points,
        # Crossing markings, so the geometry itself is sound and only the click is wrong.
        lines=traced("goal line") + traced("far touchline") + traced("near touchline"),
    )
    h = seed.homography(s)
    assert h is not None
    got = calibration.apply(h, calibration.apply(PITCH_TO_IMAGE, np.array([[30.0, 40.0]])))
    assert got[0] == pytest.approx([30.0, 40.0], abs=1.0)


def test_trimming_stops_before_the_evidence_runs_out() -> None:
    # Dropping outliers must not eat the constraints. Four exact points are already
    # exactly determined, so nothing can be removed and the fit is returned as it is.
    s = clicked(SIX[:4])
    assert seed.homography(s) is not None


def test_extents_lie_on_their_lines() -> None:
    """Each drawn segment must sit on the line the solver constrains against."""
    for name, ((ax, ay), (bx, by)) in seed.extents().items():
        a, b, c = seed.traceable()[name]
        assert abs(a * ax + b * ay + c) < 1e-9, name
        assert abs(a * bx + b * by + c) < 1e-9, name
        assert (ax, ay) != (bx, by), name


def test_extents_cover_every_traceable_marking() -> None:
    assert set(seed.extents()) == set(seed.traceable())


def test_mirrored_extent_is_on_the_mirrored_line() -> None:
    for name in seed.traceable():
        (ax, ay), (bx, by) = seed.mirrored_extent(name)
        a, b, c = seed.mirrored_line(name)
        assert abs(a * ax + b * ay + c) < 1e-9, name
        assert abs(a * bx + b * by + c) < 1e-9, name


def test_text_panel_darkens_a_tall_frame() -> None:
    """A negative slice origin selects nothing, so the panel silently vanished."""
    import numpy as np

    from football_tracks import seedui

    for width, height in ((1280, 720), (2774, 1508), (3840, 2160)):
        base = np.full((height, width, 3), 255, dtype=np.uint8)
        out = seedui._draw(base, [], [], "penalty spot", False, False)
        # A band under the first line of text, left of where any glyph reaches.
        assert out[8:20, 4:12].mean() < 200, (width, height)


def test_behind_camera_accepts_a_fit_that_describes_the_whole_frame() -> None:
    # Horizon far above the picture: every pixel is in front of the lens.
    h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 3000.0]])
    assert seed.behind_camera(h, 1000, 1000) == 0.0


def test_behind_camera_catches_a_fit_folded_through_the_horizon() -> None:
    """A seed clicked in a band of the frame fits its clicks and folds below them.

    Its residuals stay under half a metre while it puts players ninety metres off the
    end of the pitch, so nothing the fit reports about itself can catch this (D34).
    """
    # Horizon through the middle: half the frame maps behind the camera.
    h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, -500.0]])
    assert seed.behind_camera(h, 1000, 1000) > seed.MAX_BEHIND_CAMERA


def test_behind_camera_does_not_care_which_way_the_sign_runs() -> None:
    a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 3000.0]])
    b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 3000.0]])
    assert seed.behind_camera(a, 1000, 1000) == seed.behind_camera(b, 1000, 1000)


def test_refine_pulls_a_nudged_model_back_onto_the_paint(tmp_path: Path) -> None:
    """A synthetic pitch: two lines each way, and a model nudged off them.

    The refinement is local. It corrects what the carry has just done to a homography
    that was nearly right, which is why it belongs inside the carry loop and not after
    it: run over a finished chain that has already wandered, it is out of range and
    refuses (D35).
    """
    from football_tracks import refine as refine_mod

    h, w = 720, 1280
    img = np.full((h, w, 3), (60, 140, 60), dtype=np.uint8)
    # Model: pitch metres map to pixels by a factor of 12, so lines land where the
    # markings would be. Only a scaling, but the fit does not know that.
    scale = 12.0
    truth = np.array([[1 / scale, 0.0, 0.0], [0.0, 1 / scale, 0.0], [0.0, 0.0, 1.0]])
    for x in (5.5, 16.5):
        cv2.line(img, (int(x * scale), 0), (int(x * scale), h), (250, 250, 250), 5)
    for y in (24.84, 43.16):
        cv2.line(img, (0, int(y * scale)), (w, int(y * scale)), (250, 250, 250), 5)

    nudged = truth @ np.array([[1.0, 0.0, 6.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]])
    before = calibration.observed_error(truth, nudged, img.shape)
    got = refine_mod.refine(nudged, img)
    assert got is not None
    assert calibration.observed_error(truth, got, img.shape) < before


def test_the_diagram_can_be_moved_out_of_the_way_and_hidden() -> None:
    """A landmark can be anywhere, including under whichever corner the diagram is in --
    and a click on the diagram is not a click on the pitch, so the rectangle it occupies
    is what the click filter uses."""
    from football_tracks import seedui

    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    panel = np.zeros((200, 300, 3), dtype=np.uint8)

    corners = {c: seedui._inset_rect(frame, panel, c) for c in seedui.CORNERS if c}
    assert corners["bottom left"] == (16, 684, 316, 884)
    assert corners["bottom right"] == (1284, 684, 1584, 884)
    assert corners["top right"] == (1284, 16, 1584, 216)
    # Every corner is reachable, and the last stop is no diagram at all.
    assert None in seedui.CORNERS
    assert seedui._inset_rect(frame, panel, None) is None


def test_a_diagram_too_big_for_the_frame_is_not_drawn() -> None:
    from football_tracks import seedui

    tiny = np.zeros((100, 100, 3), dtype=np.uint8)
    panel = np.zeros((200, 300, 3), dtype=np.uint8)
    assert seedui._inset_rect(tiny, panel, "bottom left") is None


def test_every_diagram_is_the_same_size_whatever_it_draws() -> None:
    """What lets the click filter measure the inset once. It used to measure it per frame
    by drawing the CURRENT selection in point mode, which raises the moment you switch to
    tracing: a line name is not in the landmark table."""
    from football_tracks import pitch as pitch_mod
    from football_tracks import seedui

    width = 1600
    sizer = pitch_mod.draw(max(seedui.DIAGRAM_MIN_SCALE, width / 420), 2.0)
    for name in seed.landmarks():
        assert seedui._diagram(name, False, width).shape == sizer.shape
    for name in seed.traceable():
        assert seedui._diagram(name, True, width, trace=True).shape == sizer.shape


def test_a_seed_is_stamped_with_the_picture_it_was_clicked_on() -> None:
    # A frame NUMBER is not an identity -- frame 56 exists in every clip -- so a seed left
    # behind by the last clip anchors the next one silently, in a coordinate frame that has
    # nothing to do with it. Measured on a coach's second clip: the board was unrecognisable
    # and every fidelity number stayed good (D34).
    rng = np.random.default_rng(0)
    one = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    other = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    same_shot = cv2.convertScaleAbs(one, alpha=1.05, beta=4)

    print_ = seed.fingerprint(one)
    assert seed.unlike(print_, seed.fingerprint(one)) == 0
    assert seed.unlike(print_, seed.fingerprint(same_shot)) <= seed.MAX_UNLIKE_BITS
    assert seed.unlike(print_, seed.fingerprint(other)) > seed.MAX_UNLIKE_BITS


def test_an_unreadable_fingerprint_is_treated_as_a_different_picture() -> None:
    assert seed.unlike("not a hash", "0123456789abcdef") == 64


def test_a_stamp_survives_a_round_trip(tmp_path: Path) -> None:
    written = seed.Seed(frame=7, points=[((1.0, 2.0), (3.0, 4.0))], image="0123456789abcdef")
    assert seed.read(seed.write(tmp_path / "seed.json", written)).image == "0123456789abcdef"
    # And a seed written before this existed still reads, with nothing to check against.
    assert seed.read(seed.write(tmp_path / "old.json", seed.Seed(frame=7, points=[]))).image is None


def _view(frame: int, flip_x: bool = False, flip_y: bool = False) -> seed.Seed:
    """A seed of a camera on one touchline, optionally mislabelled by a pitch symmetry."""
    corners = [
        ((200.0, 900.0), (0.0, 68.0)),
        ((1700.0, 900.0), (52.5, 68.0)),
        ((700.0, 300.0), (0.0, 0.0)),
        ((1500.0, 300.0), (52.5, 0.0)),
        ((1100.0, 500.0), (26.25, 34.0)),
        ((400.0, 700.0), (11.0, 54.16)),
    ]
    points = [
        (img, (105.0 - x if flip_x else x, 68.0 - y if flip_y else y)) for img, (x, y) in corners
    ]
    return seed.Seed(frame=frame, points=points)


def test_every_seed_of_a_clip_sees_the_pitch_the_same_way_round() -> None:
    # A camera cannot get underneath a pitch, so panning from one goal to the other cannot
    # change the handedness of the image-to-pitch map. Labelling the clicks with the wrong
    # END can, because that is a reflection (D88).
    right = seed.homography(_view(1))
    mirrored = seed.homography(_view(600, flip_x=True))
    assert right is not None and mirrored is not None

    assert seed.handedness(right, 1920, 1080) == -seed.handedness(mirrored, 1920, 1080)


def test_the_end_check_and_the_far_near_check_cover_every_way_a_pitch_is_symmetric() -> None:
    # A pitch has three non-identity symmetries and the two guards divide them: reflecting
    # y flips `orientation`, reflecting x flips `handedness`, and reflecting both flips
    # `orientation` while leaving `handedness` alone, because two reflections are a
    # rotation. Neither check alone is enough and together nothing gets through.
    base = _view(1)
    straight = seed.handedness(_check(base), 1920, 1080)
    for flip_x, flip_y in ((True, False), (False, True), (True, True)):
        got = _view(1, flip_x=flip_x, flip_y=flip_y)
        turned = seed.handedness(_check(got), 1920, 1080) != straight
        swapped = seed.orientation(got) < 0
        assert turned or swapped, (flip_x, flip_y)


def _check(got: seed.Seed) -> npt.NDArray[np.float64]:
    h = seed.homography(got)
    assert h is not None
    return h


def test_flipping_the_end_twice_is_where_it_started() -> None:
    there = seed.flip_x(_view(9))
    back = seed.flip_x(there)
    for (_, a), (_, b) in zip(_view(9).points, back.points, strict=True):
        assert a == b
    assert seed.flip_x(_view(9)).points != _view(9).points


def test_a_traced_line_flips_end_to_end_and_a_sideways_one_does_not() -> None:
    # x = 5.5 is the 6yd line at the near goal and x = 99.5 at the far one; y = 13.84 is
    # the same marking whichever end is in shot.
    assert seed.mirror_line(1.0, 0.0, -5.5) == (-1.0, 0.0, 99.5)
    assert seed.mirror_line(0.0, 1.0, -13.84) == (0.0, 1.0, -13.84)


def test_a_degenerate_fit_has_no_opinion_about_which_way_round_it_is() -> None:
    # Zero rather than a coin flip, so a fit that cannot answer never refuses another seed.
    collapsed = np.array([[1e-9, 0.0, 0.0], [0.0, 1e-9, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    assert seed.handedness(collapsed, 1920, 1080) == 0.0


def test_a_narrower_pitch_moves_the_goal_and_leaves_the_markings_alone() -> None:
    # The Laws fix a goal at 7.32 m and a penalty spot 11 m out on every ground; what a
    # smaller pitch moves is where the middle is, and how far away the far side is (D89).
    small = Pitch(length=100.0, width=64.0)
    wide, narrow = seed.landmarks(), seed.landmarks(small)

    assert wide["penalty spot"] == (11.0, 34.0)
    assert narrow["penalty spot"] == (11.0, 32.0)
    for marks in (wide, narrow):
        posts = marks["goal post near"][1] - marks["goal post far"][1]
        assert posts == pytest.approx(7.32)
    assert narrow["corner near"] == (0.0, 64.0)
    assert narrow["halfway far"] == (50.0, 0.0)


def test_the_other_end_is_the_other_end_of_THIS_pitch() -> None:
    small = Pitch(length=100.0, width=64.0)
    assert seed.mirrored("goal post far", small)[0] == 100.0
    assert seed.mirror_line(1.0, 0.0, -16.5, small) == (-1.0, 0.0, 83.5)
    assert seed.mirror_line(1.0, 0.0, -16.5) == (-1.0, 0.0, 88.5)


def test_a_seed_written_for_one_pitch_is_not_silently_read_as_another() -> None:
    # The trap this exists for: the same clicks, labelled against two pitch sizes, fit two
    # different cameras and NEITHER complains. The residuals are fine both ways, because
    # the clicks agree with whatever they were told they meant. Nothing downstream can
    # catch it, which is why `ft pitch` warns when a clip is already seeded (D89).
    camera = np.array([[12.0, 3.0, 300.0], [0.0, 9.0, 150.0], [0.0, 0.006, 1.0]], dtype=np.float64)
    small = Pitch(length=100.0, width=64.0)
    names = [
        "corner far",
        "penalty box front far",
        "penalty spot",
        "penalty box front near",
        "corner near",
        "halfway far",
    ]

    def clicked(pitch: Pitch) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        marks = seed.landmarks(pitch)
        spots = np.array([[marks[n] for n in names]], dtype=np.float64)
        shot = cv2.perspectiveTransform(spots, camera)[0]
        return [((float(u), float(v)), marks[n]) for (u, v), n in zip(shot, names, strict=True)]

    # One coach, one set of pixels. Only the label on the pitch differs.
    pixels = [img for img, _ in clicked(DEFAULT_PITCH)]

    def labelled(pitch: Pitch) -> seed.Seed:
        marks = [p for _, p in clicked(pitch)]
        return seed.Seed(frame=1, points=list(zip(pixels, marks, strict=True)))

    as_wide = seed.homography(labelled(DEFAULT_PITCH))
    as_narrow = seed.homography(labelled(small))
    assert as_wide is not None and as_narrow is not None
    assert not np.allclose(as_wide / as_wide[2, 2], as_narrow / as_narrow[2, 2], atol=1e-3)
