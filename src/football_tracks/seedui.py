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
from .seed import LANDMARKS, TRACEABLE, Seed, mirrored, mirrored_line

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


def _diagram(name: str, far_goal: bool, width: int, trace: bool = False) -> Any:
    scale = max(DIAGRAM_MIN_SCALE, width / 420)
    img = pitch_mod.draw(scale, 2.0)
    if trace:
        a, b, c = mirrored_line(name) if far_goal else TRACEABLE[name]
        # Draw the whole marking, since tracing means "anywhere along this".
        if abs(a) > abs(b):
            p0, p1 = (-c / a, 0.0), (-c / a, 68.0)
        else:
            p0, p1 = (0.0, -c / b), (105.0, -c / b)
        cv2.line(
            img,
            pitch_mod.to_px(*p0, scale, 2.0),
            pitch_mod.to_px(*p1, scale, 2.0),
            TARGET,
            max(3, round(scale / 2)),
            cv2.LINE_AA,
        )
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


def _draw(
    base: Any,
    seed_points: list[Any],
    traced: list[Any],
    cursor: str,
    far_goal: bool,
    trace: bool,
) -> Any:
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
    for (ix, iy), _ in traced:
        cv2.circle(img, (int(ix), int(iy)), 6, TARGET, -1, cv2.LINE_AA)

    end = "FAR goal" if far_goal else "NEAR goal"
    enough = len(seed_points) * 2 + len(traced) >= 8
    mode = "TRACE ALONG" if trace else "CLICK"
    lines = [
        f"{mode}: {cursor}      (marked on the diagram, bottom left)",
        f"{len(seed_points)} points + {len(traced)} traced   |   {end} end"
        " - press 'e' if the goal in shot is the other one",
        "t = switch point/trace mode    n = next    p = back    u = undo",
        f"s = save{'' if enough else '  (needs more evidence)'}     q = quit",
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
    _inset(img, _diagram(cursor, far_goal, base.shape[1], trace))
    return img


def collect(frame: Any, frame_index: int) -> Seed | None:
    """Run the window until saved or abandoned. Returns None if abandoned.

    Two modes, because they suit different footage. POINT mode wants an exact landmark,
    which is precise when the corner is in shot. TRACE mode wants several clicks
    anywhere along a named line, which is what a tight goalmouth shot actually offers -
    long clear markings whose corners are off screen.
    """
    names = list(LANDMARKS)
    line_names = list(TRACEABLE)
    state = {"i": 0, "far": False, "trace": False}
    points: list[Any] = []
    traced: list[Any] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["trace"]:
            name = line_names[state["i"]]
            line = mirrored_line(name) if state["far"] else TRACEABLE[name]
            traced.append(((float(x), float(y)), line))
            return  # stay on the same line - tracing wants several clicks
        name = names[state["i"]]
        pitch = mirrored(name) if state["far"] else LANDMARKS[name]
        points.append(((float(x), float(y)), pitch))
        state["i"] = min(state["i"] + 1, len(names) - 1)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1600, 900)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        active = line_names if state["trace"] else names
        state["i"] = min(int(state["i"]), len(active) - 1)
        cv2.imshow(
            WINDOW,
            _draw(
                frame, points, traced, active[state["i"]], bool(state["far"]), bool(state["trace"])
            ),
        )
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(WINDOW)
            return None
        if key == ord("s") and (len(points) * 2 + len(traced)) >= 8:
            cv2.destroyWindow(WINDOW)
            return Seed(frame=frame_index, points=points, lines=traced)
        if key == ord("n"):
            state["i"] = (int(state["i"]) + 1) % len(active)
        if key == ord("p"):
            state["i"] = (int(state["i"]) - 1) % len(active)
        if key == ord("e"):
            state["far"] = not state["far"]
        if key == ord("t"):
            state["trace"] = not state["trace"]
            state["i"] = 0
        if key == ord("u"):
            if state["trace"] and traced:
                traced.pop()
            elif not state["trace"] and points:
                points.pop()
