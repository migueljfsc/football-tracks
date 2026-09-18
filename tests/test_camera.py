"""One camera for a match: the position, the aim, and what each is recovered from."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from football_tracks import calibration, camera, seed
from football_tracks.config import DEFAULT_PITCH

# A gantry on the near touchline, level with the halfway line: where a broadcast camera
# actually is, and where the Milan-Benfica clips put it (D96).
RIG = camera.Camera(x=52.5, y=104.0, height=15.0, game="test")
LENS = (1379.0, 775.0)


def project(view: camera.View, points: list[tuple[float, float]]) -> npt.NDArray[np.float64]:
    """Pitch metres -> image pixels, which is the camera run backwards."""
    h = camera.matrix(RIG, LENS, view)
    assert h is not None
    return calibration.apply(np.linalg.inv(h), np.array(points, dtype=np.float64))


def drawn(view: camera.View, names: list[str], per_line: int = 12) -> camera.Pairs:
    """What the segmenter would hand over for this view: named pixels along named markings."""
    pairs: camera.Pairs = []
    for name in names:
        a, b, c = calibration.PITCH_LINES[name]
        # Walk the marking in metres, then look at where this camera draws it.
        if abs(a) > abs(b):
            along = np.linspace(2.0, DEFAULT_PITCH.width - 2.0, per_line)
            pts = [(-c / a, float(t)) for t in along]
        else:
            along = np.linspace(2.0, DEFAULT_PITCH.length - 2.0, per_line)
            pts = [(float(t), -c / b) for t in along]
        pairs.extend(((float(x), float(y)), (a, b, c)) for x, y in project(view, pts))
    return pairs


def test_a_camera_and_an_aim_agree_with_themselves() -> None:
    view = camera.View(pan=math.radians(-10), tilt=math.radians(12), focal=5500.0)
    h = camera.matrix(RIG, LENS, view)
    assert h is not None
    spots = [(52.5, 34.0), (16.5, 13.84), (88.5, 54.16)]
    back = calibration.apply(h, project(view, spots))
    assert np.allclose(back, np.array(spots), atol=1e-6)


def test_a_camera_underground_describes_nothing() -> None:
    # Not a tuning guard: the solver walks, and a camera below the pitch fits the markings
    # upside down and reports nothing wrong with itself.
    below = camera.Camera(x=52.5, y=104.0, height=-15.0)
    assert camera.matrix(below, LENS, camera.View(0.0, 0.2, 5000.0)) is None
    assert camera.matrix(RIG, LENS, camera.View(0.0, 0.2, 0.0)) is None


def test_the_aim_is_recovered_from_the_markings_it_draws() -> None:
    """Three numbers, from the same named pixels the segmenter produces (D96)."""
    view = camera.View(pan=math.radians(-18), tilt=math.radians(11), focal=6200.0)
    pairs = drawn(view, ["Side line top", "Side line bottom", "Big rect. left main", "Middle line"])
    got = camera.aim(RIG, LENS, pairs)
    assert got is not None
    found, miss = got
    assert miss < 0.05
    assert abs(found.pan - view.pan) < math.radians(0.5)
    assert abs(found.tilt - view.tilt) < math.radians(0.5)
    assert abs(found.focal - view.focal) < 50


def test_a_frame_showing_too_little_is_refused_however_well_it_fits() -> None:
    """D17 at a camera model's size: three numbers need more than three constraints.

    The midfield frames of the clip this was built for show the halfway line and one
    touchline. Fitted, they agree with themselves to 0.01 m and put the markings off the
    picture -- there is nothing left over for a residual to disagree with.
    """
    view = camera.View(pan=0.0, tilt=math.radians(12), focal=5500.0)
    two = drawn(view, ["Side line top", "Middle line"])
    assert camera.aim(RIG, LENS, two) is None
    assert camera.aim(RIG, LENS, []) is None


def test_the_centre_circle_is_what_rescues_a_midfield_frame() -> None:
    """The one marking such a view always has that bends through DEPTH (D34, D96)."""
    view = camera.View(pan=math.radians(2), tilt=math.radians(12), focal=5500.0)
    two = drawn(view, ["Side line top", "Middle line"])
    spots = [
        (52.5 + 9.15 * math.cos(a), 34.0 + 9.15 * math.sin(a))
        for a in np.linspace(0, 2 * math.pi, 24)
    ]
    circle = [((float(u), float(v)), (52.5, 34.0, 9.15)) for u, v in project(view, spots)]
    got = camera.aim(RIG, LENS, two, circle)
    assert got is not None
    found, miss = got
    assert miss < 0.05
    assert abs(found.focal - view.focal) < 50


def test_the_lens_axis_moves_with_the_crop() -> None:
    """One camera, two clips, two crops: the axis is in each clip's own pixels (D96).

    The Milan-Benfica pair, which is cropped by 9 px and by 1 px -- a camera shared between
    them without this lands four pixels out on one of them.
    """
    assert camera.lens({"crop": [9, 1, 2768, 1552]}) == (1379.5, 775.5)
    assert camera.lens({"crop": [1, 1, 2765, 1552]}) == (1382.0, 775.5)


def test_the_position_is_recovered_from_seeds_of_different_views() -> None:
    """What `ft camera` does: several clicked frames, one position that explains them all."""
    views = [
        camera.View(pan=math.radians(-25), tilt=math.radians(11), focal=6000.0),
        camera.View(pan=math.radians(0), tilt=math.radians(13), focal=5200.0),
        camera.View(pan=math.radians(22), tilt=math.radians(10), focal=6800.0),
    ]
    names = list(seed.landmarks())
    clicked = []
    for i, view in enumerate(views):
        spots = [seed.landmarks()[n] for n in names]
        shot = project(view, spots)
        points = [((float(u), float(v)), spots[j]) for j, (u, v) in enumerate(shot)]
        clicked.append((seed.Seed(frame=i + 1, points=points), LENS))

    got = camera.fit(clicked, pitch=DEFAULT_PITCH, game="test")
    assert got is not None
    found, aims = got
    assert math.dist((found.x, found.y, found.height), (RIG.x, RIG.y, RIG.height)) < 0.5
    assert abs(aims[2].focal - views[1].focal) < 100


def test_one_seed_cannot_place_the_camera() -> None:
    view = camera.View(pan=0.0, tilt=math.radians(12), focal=5500.0)
    spots = [seed.landmarks()[n] for n in list(seed.landmarks())[:6]]
    shot = project(view, spots)
    points = [((float(u), float(v)), spots[j]) for j, (u, v) in enumerate(shot)]
    assert camera.fit([(seed.Seed(frame=1, points=points), LENS)]) is None


def test_an_aim_between_two_is_between_them() -> None:
    a = camera.View(pan=0.0, tilt=math.radians(10), focal=4000.0)
    b = camera.View(pan=math.radians(20), tilt=math.radians(14), focal=8000.0)
    half = camera.between(a, b, 0.5)
    assert abs(half.pan - math.radians(10)) < 1e-9
    assert abs(half.tilt - math.radians(12)) < 1e-9
    # Zoom is geometric, so halfway between 4000 and 8000 is their geometric mean.
    assert abs(half.focal - math.sqrt(4000.0 * 8000.0)) < 1e-6


def test_a_gap_is_spanned_but_a_cut_is_not() -> None:
    views = {
        1: camera.View(0.0, 0.2, 5000.0),
        10: camera.View(math.radians(9), 0.2, 5000.0),
        200: camera.View(math.radians(-40), 0.3, 9000.0),
    }
    frames = [1, 5, 10, 100, 200]
    out = camera.spanned(views, frames, max_gap=25)
    assert abs(out[5].pan - math.radians(4)) < 1e-9
    # 100 sits in a gap of 190 frames: whatever happened there, it was not one pan.
    assert 100 not in out
    assert set(out) == {1, 5, 10, 200}
