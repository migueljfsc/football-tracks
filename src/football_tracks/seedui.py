"""A click tool for seeding one frame.

An OpenCV window, because it needs no web stack and runs where the frames already are.
It is deliberately thin: it writes seed.json and nothing else, so the same file can
later come from a keypoint model or from Pitchboard's own import view (D23).

Controls are printed on the window rather than in a manual, since the whole thing is
used once per clip and then forgotten.
"""

from __future__ import annotations

from typing import Any

import cv2

from . import pitch as pitch_mod
from .seed import LANDMARKS, Seed, mirrored

WINDOW = "seed - click a landmark, then pick its name"
MARK = (60, 240, 90)
TEXT = (255, 255, 255)
TARGET = (40, 90, 250)

# The diagram is the whole usability of this tool. A landmark name means nothing on its
# own - "6yd front far" is only obvious once you have seen it marked on a pitch - so a
# small top-down pitch sits in the corner with the wanted point on it.
# Pixels per metre in the inset. Scaled to the frame so it stays readable whether the
# clip is 720p or an upscaled recording.
DIAGRAM_MIN_SCALE = 4.0


def _diagram(name: str, far_goal: bool, width: int) -> Any:
    scale = max(DIAGRAM_MIN_SCALE, width / 420)
    img = pitch_mod.draw(scale, 2.0)
    x, y = mirrored(name) if far_goal else LANDMARKS[name]
    px, py = pitch_mod.to_px(x, y, scale, 2.0)
    r = max(10, round(scale * 2.2))
    cv2.circle(img, (px, py), r, TARGET, max(2, r // 4), cv2.LINE_AA)
    cv2.drawMarker(img, (px, py), TARGET, cv2.MARKER_CROSS, r * 2, max(2, r // 5), cv2.LINE_AA)
    cv2.putText(
        img,
        name,
        (10, img.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.7, scale / 9),
        TARGET,
        2,
        cv2.LINE_AA,
    )
    return img


def _inset(base: Any, panel: Any) -> None:
    """Drop the diagram into the bottom-left corner, over the frame."""
    ph, pw = panel.shape[:2]
    h = base.shape[0]
    y0, x0 = h - ph - 16, 16
    if y0 < 0 or x0 + pw > base.shape[1]:
        return
    cv2.rectangle(base, (x0 - 4, y0 - 4), (x0 + pw + 4, y0 + ph + 4), (0, 0, 0), -1)
    base[y0 : y0 + ph, x0 : x0 + pw] = panel


def _draw(base: Any, seed_points: list[Any], cursor: str, far_goal: bool) -> Any:
    img = base.copy()
    for (ix, iy), (px, py) in seed_points:
        cv2.circle(img, (int(ix), int(iy)), 7, MARK, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"{px:.0f},{py:.0f}",
            (int(ix) + 10, int(iy) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            MARK,
            2,
            cv2.LINE_AA,
        )
    end = "FAR goal" if far_goal else "NEAR goal"
    enough = len(seed_points) >= 4
    lines = [
        f"CLICK: {cursor}      (marked on the diagram, bottom left)",
        f"{len(seed_points)} placed   |   {end} end"
        " - press 'e' if the goal in shot is the other one",
        "n = SKIP this landmark if it is not in shot     p = back     u = undo",
        f"s = save{'' if enough else '  (needs 4 or more)'}     q = quit",
    ]
    scale = max(0.9, base.shape[1] / 2200)
    for i, line in enumerate(lines):
        y = int(44 + i * 46 * scale)
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(
            img,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            MARK if i == 0 else TEXT,
            2,
            cv2.LINE_AA,
        )
    _inset(img, _diagram(cursor, far_goal, base.shape[1]))
    return img


def collect(frame: Any, frame_index: int) -> Seed | None:
    """Run the window until saved or abandoned. Returns None if abandoned."""
    names = list(LANDMARKS)
    state = {"i": 0, "far": False}
    points: list[Any] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            name = names[state["i"]]
            pitch = mirrored(name) if state["far"] else LANDMARKS[name]
            points.append(((float(x), float(y)), pitch))
            state["i"] = min(state["i"] + 1, len(names) - 1)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1600, 900)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        cv2.imshow(WINDOW, _draw(frame, points, names[state["i"]], bool(state["far"])))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(WINDOW)
            return None
        if key == ord("s") and len(points) >= 4:
            cv2.destroyWindow(WINDOW)
            return Seed(frame=frame_index, points=points)
        if key == ord("n"):
            state["i"] = (state["i"] + 1) % len(names)
        if key == ord("p"):
            state["i"] = (state["i"] - 1) % len(names)
        if key == ord("e"):
            state["far"] = not state["far"]
        if key == ord("u") and points:
            points.pop()
