"""The human seed: clicked landmarks -> a camera model."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from football_tracks import calibration, seed
from football_tracks.config import PITCH_LENGTH, PITCH_WIDTH

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
        pitch = seed.LANDMARKS[n]
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
    assert seed.LANDMARKS["goal post far"][0] == 0.0
    assert seed.mirrored("goal post far")[0] == PITCH_LENGTH
    assert seed.mirrored("goal post far")[1] == seed.LANDMARKS["goal post far"][1]


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
    a, b, c = seed.TRACEABLE[name]
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
    for name, ((ax, ay), (bx, by)) in seed.EXTENTS.items():
        a, b, c = seed.TRACEABLE[name]
        assert abs(a * ax + b * ay + c) < 1e-9, name
        assert abs(a * bx + b * by + c) < 1e-9, name
        assert (ax, ay) != (bx, by), name


def test_extents_cover_every_traceable_marking() -> None:
    assert set(seed.EXTENTS) == set(seed.TRACEABLE)


def test_mirrored_extent_is_on_the_mirrored_line() -> None:
    for name in seed.TRACEABLE:
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
