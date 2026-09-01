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

from .seed import LANDMARKS, Seed, mirrored

WINDOW = "seed - click a landmark, then pick its name"
MARK = (60, 240, 90)
TEXT = (255, 255, 255)


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
    end = "FAR goal (x=105)" if far_goal else "NEAR goal (x=0)"
    lines = [
        f"{len(seed_points)} points  |  end: {end}  (press 'e' to switch)",
        f"next: {cursor}",
        "click = place   n/p = change landmark   u = undo   s = save   q = quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            img, line, (14, 34 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, TEXT, 2, cv2.LINE_AA
        )
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
