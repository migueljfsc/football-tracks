"""A click tool for seeding one frame.

An OpenCV window, because it needs no web stack and runs where the frames already are.
It is deliberately thin: it writes seed.json and nothing else, so the same file can
later come from a keypoint model or from Pitchboard's own import view (D23).

Controls are printed on the window rather than in a manual, since the whole thing is
used once per clip and then forgotten.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import guide as guide_mod
from . import overlay as overlay_mod
from . import pitch as pitch_mod
from .config import DEFAULT_PITCH, Pitch
from .seed import (
    REGIONS,
    Seed,
    curve_extent,
    curves,
    extents,
    landmarks,
    mirrored,
    mirrored_curve,
    mirrored_extent,
    mirrored_line,
    ordered,
    region_in_view,
    traceable,
)
from .stage1_propagate import Chain

WINDOW = "seed - click a landmark, then pick its name"
MARK = (60, 240, 90)
TEXT = (255, 255, 255)
TARGET = (40, 90, 250)

# The instructions sit over the picture, and a crowd or a floodlit stand behind them is
# unreadable however the text is outlined. A dark panel under the block fixes it; 60%
# lets enough of the frame through that nothing underneath is hidden, which matters
# because a landmark can be up there.
TEXT_PANEL_OPACITY = 0.6

# The diagram is the whole usability of this tool. A landmark name means nothing on its
# own - "6yd front far" is only obvious once you have seen it marked on a pitch - so a
# small top-down pitch sits in the corner with the wanted point on it.
# Pixels per metre in the inset. Scaled to the frame so it stays readable whether the
# clip is 720p or an upscaled recording.
DIAGRAM_MIN_SCALE = 4.0


def _diagram(
    name: str, far_goal: bool, width: int, trace: bool = False, pitch: Pitch = DEFAULT_PITCH
) -> Any:
    scale = max(DIAGRAM_MIN_SCALE, width / 420)
    img = pitch_mod.draw(scale, 2.0, pitch)
    if trace:
        thickness = max(4, round(scale / 1.6))
        if name in curves(pitch):
            # Only the painted part: the penalty arc's circle runs on inside the box.
            poly = curve_extent(name, far_goal, pitch)
            drawn = np.array([pitch_mod.to_px(x, y, scale, 2.0) for x, y in poly], dtype=np.int32)
            cv2.polylines(img, [drawn], False, TARGET, thickness, cv2.LINE_AA)
        else:
            # The marking's real extent, not the infinite line the solver stores. Drawing the
            # infinite one claims the six-yard box runs the length of the pitch, which points
            # the coach at grass rather than at a line.
            p0, p1 = mirrored_extent(name, pitch) if far_goal else extents(pitch)[name]
            cv2.line(
                img,
                pitch_mod.to_px(*p0, scale, 2.0),
                pitch_mod.to_px(*p1, scale, 2.0),
                TARGET,
                thickness,
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

    x, y = mirrored(name, pitch) if far_goal else landmarks(pitch)[name]
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


# Where the diagram may sit, and the order `d` cycles them in. The top left is not on the
# list: the controls live there, and a diagram under them is unreadable. `None` is the
# fifth stop -- a landmark can be anywhere, including under the corner the diagram is in,
# and on a tight shot every corner is somewhere a coach needs to click.
CORNERS: tuple[str | None, ...] = ("bottom left", "bottom right", "top right", None)


def _inset_rect(base: Any, panel: Any, corner: str | None) -> tuple[int, int, int, int] | None:
    """Where the diagram goes, or None when it is hidden or would not fit."""
    if corner is None:
        return None
    ph, pw = panel.shape[:2]
    h, w = base.shape[:2]
    if ph + 32 > h or pw + 32 > w:
        return None
    x0 = 16 if "left" in corner else w - pw - 16
    y0 = 16 if "top" in corner else h - ph - 16
    return (x0, y0, x0 + pw, y0 + ph)


def _inset(base: Any, panel: Any, corner: str | None) -> None:
    """Drop the diagram into a corner, over the frame."""
    rect = _inset_rect(base, panel, corner)
    if rect is None:
        return
    x0, y0, x1, y1 = rect
    cv2.rectangle(base, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 0, 0), -1)
    base[y0:y1, x0:x1] = panel


def _draw(
    base: Any,
    seed_points: list[Any],
    traced: list[Any],
    cursor: str,
    far_goal: bool,
    trace: bool,
    corner: str | None = "bottom left",
    message: str | None = None,
    curved: list[Any] | None = None,
    region: str = "both",
) -> Any:
    img = base.copy()
    curved = curved or []
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
    for (ix, iy), _ in curved:
        cv2.circle(img, (int(ix), int(iy)), 6, TARGET, -1, cv2.LINE_AA)
        cv2.circle(img, (int(ix), int(iy)), 9, TEXT, 2, cv2.LINE_AA)

    end = "FAR goal" if far_goal else "NEAR goal"
    enough = len(seed_points) * 2 + len(traced) + len(curved) >= 8
    mode = "TRACE ALONG" if trace else "CLICK"
    where = f"marked on the diagram, {corner}" if corner else "diagram hidden - press d"
    lines = [
        f"{mode}: {cursor}      ({where})",
        f"{len(seed_points)} points + {len(traced)} traced"
        f"{f' + {len(curved)} on curves' if curved else ''}   |   {end} end"
        " - press 'e' if the goal in shot is the other one",
        f"t = points / trace lines and circles    r = region: {region.upper()}"
        "    n = next    p = back    u = undo",
        f"d = move the diagram / hide it     s = save"
        f"{'' if enough else ' (needs more evidence)'}"
        # After a refusal this frame may offer nothing more, and the way out must say so.
        f"     q = {'skip this frame' if message else 'quit'}",
    ]
    colours = [MARK] + [TEXT] * (len(lines) - 1)
    # Why the last attempt on this frame was not kept, above everything else: it is the
    # first thing to read on coming back to a frame that was already clicked once.
    if message:
        parts = message.split("\n")
        lines[:0] = parts
        colours[:0] = [TARGET] * len(parts)
    scale = max(0.9, base.shape[1] / 2200)
    step = int(46 * scale)
    widest = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] for line in lines)
    pad = int(14 * scale)
    # Clamped at zero: a negative index is a slice from the far edge, so an oversized
    # font on a tall frame silently selects nothing and the panel never appears.
    x0, y0 = max(0, 16 - pad), max(0, 44 - step + pad // 2)
    x1 = min(img.shape[1], 16 + widest + pad)
    y1 = min(img.shape[0], 44 + step * (len(lines) - 1) + pad)

    panel = img[y0:y1, x0:x1]
    if panel.size:
        img[y0:y1, x0:x1] = cv2.addWeighted(
            panel, 1 - TEXT_PANEL_OPACITY, np.zeros_like(panel), TEXT_PANEL_OPACITY, 0
        )

    for i, line in enumerate(lines):
        y = 44 + i * step
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(
            img,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            colours[i],
            2,
            cv2.LINE_AA,
        )
    _inset(img, _diagram(cursor, far_goal, base.shape[1], trace), corner)
    return img


def collect(
    frame: Any,
    frame_index: int,
    pitch: Pitch = DEFAULT_PITCH,
    message: str | None = None,
    window: str | None = None,
    camera: Any = None,
) -> Seed | None:
    """Run the window until saved or abandoned. Returns None if abandoned.

    `message` is shown above the controls - why the last clicks on this frame were refused.
    `window` is a window already on screen to draw in and leave open, which is how `ft run`
    keeps one window for the whole session (see `_open`); without it the tool opens its own
    and closes it on the way out, as `ft seed` always has.

    Two modes, because they suit different footage. POINT mode wants an exact landmark,
    which is precise when the corner is in shot. TRACE mode wants several clicks
    anywhere along a named line, which is what a tight goalmouth shot actually offers -
    long clear markings whose corners are off screen. The circles are traced the same way,
    after the lines, for the shot whose straight markings all sit in one band.

    `r` puts one region's markings first -- a goal end, midfield, or both, which is the
    order they are listed in. `camera` is where the other seeds carry this frame, image to
    pitch, and picks the region to start on; without one it starts on both.
    """
    curve_table = curves(pitch)
    height, width = frame.shape[:2]
    region = "both" if camera is None else region_in_view(camera, width, height, pitch)
    state: dict[str, Any] = {
        "i": 0,
        "far": False,
        "trace": False,
        "corner": 0,
        "rect": None,
        "region": REGIONS.index(region),
    }

    def offered(trace: bool) -> list[str]:
        names = list(traceable(pitch)) + list(curve_table) if trace else list(landmarks(pitch))
        return ordered(names, REGIONS[int(state["region"])])

    points: list[Any] = []
    traced: list[Any] = []
    curved: list[Any] = []
    # Which name each click was made against, so undo can put the diagram back on it. The
    # previous index is not enough: `n` and `p` skip names, a traced line takes many clicks
    # before `n` moves on to the next one, and `r` reorders the list between clicks. Trace
    # mode keeps one history for lines and curves together, so undo takes back the last
    # click whichever list it went into.
    points_at: list[str] = []
    trace_history: list[tuple[bool, str]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # A click on the diagram is not a click on the pitch. Recording it puts a landmark
        # wherever the diagram happens to be, which is a mistake nothing downstream can
        # see -- and the fix a coach reaches for is `d`, not undo.
        rect = state["rect"]
        if rect is not None and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
            return
        name = offered(bool(state["trace"]))[state["i"]]
        if state["trace"]:
            if name in curve_table:
                circle = mirrored_curve(name, pitch) if state["far"] else curve_table[name]
                curved.append(((float(x), float(y)), circle))
            else:
                line = mirrored_line(name, pitch) if state["far"] else traceable(pitch)[name]
                traced.append(((float(x), float(y)), line))
            trace_history.append((name in curve_table, name))
            return  # stay on the same marking - tracing wants several clicks
        spot = mirrored(name, pitch) if state["far"] else landmarks(pitch)[name]
        points.append(((float(x), float(y)), spot))
        points_at.append(name)
        state["i"] = min(state["i"] + 1, len(landmarks(pitch)) - 1)

    # One panel, measured once: every diagram is the same size whatever it draws, so the
    # rectangle the click filter needs does not depend on the mode -- and asking for it in
    # the wrong mode looks a line name up in the landmark table and raises.
    sizer = pitch_mod.draw(max(DIAGRAM_MIN_SCALE, frame.shape[1] / 420), 2.0)

    name = window or WINDOW
    _open(name)
    cv2.setMouseCallback(name, on_mouse)

    while True:
        active = offered(bool(state["trace"]))
        state["i"] = min(int(state["i"]), len(active) - 1)
        corner = CORNERS[int(state["corner"]) % len(CORNERS)]
        state["rect"] = _inset_rect(frame, sizer, corner)
        cv2.imshow(
            name,
            _draw(
                frame,
                points,
                traced,
                active[state["i"]],
                bool(state["far"]),
                bool(state["trace"]),
                corner,
                message,
                curved,
                REGIONS[int(state["region"])],
            ),
        )
        key = cv2.waitKey(20) & 0xFF
        # A window closed with its own button never sends `q`, and in `ft run` the loop
        # around this would wait on it forever. Only once seen open, for `scrub`'s reason.
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
            state["open"] = True
        elif state.get("open"):
            return None
        if key == ord("q"):
            if window is None:
                cv2.destroyWindow(name)
            return None
        if key == ord("s") and (len(points) * 2 + len(traced) + len(curved)) >= 8:
            if window is None:
                cv2.destroyWindow(name)
            return Seed(frame=frame_index, points=points, lines=traced, arcs=curved)
        if key == ord("n"):
            state["i"] = (int(state["i"]) + 1) % len(active)
        if key == ord("p"):
            state["i"] = (int(state["i"]) - 1) % len(active)
        if key == ord("e"):
            state["far"] = not state["far"]
        if key == ord("d"):
            state["corner"] = (int(state["corner"]) + 1) % len(CORNERS)
        if key == ord("t"):
            state["trace"] = not state["trace"]
            state["i"] = 0
        if key == ord("r"):
            state["region"] = (int(state["region"]) + 1) % len(REGIONS)
            state["i"] = 0
        if key == ord("u"):
            if state["trace"] and trace_history:
                was_curve, undone = trace_history.pop()
                (curved if was_curve else traced).pop()
                state["i"] = offered(True).index(undone)
            elif not state["trace"] and points:
                points.pop()
                state["i"] = offered(False).index(points_at.pop())


SCRUB = "football-tracks - place the pitch"

# How far the bracket keys jump. A second of footage, near enough, which is the unit a
# camera move happens in - stepping a frame at a time to cross a pan is unusable.
JUMP = 25

# Arrow keys as `waitKeyEx` reports them, which depends on the window backend: Windows, GTK
# and Cocoa each have their own. Checked BEFORE the low byte is read as a letter, because
# GTK's left arrow is 0xFF51 and its low byte is 'Q'.
LEFT_KEYS = frozenset({2424832, 65361, 63234})
RIGHT_KEYS = frozenset({2555904, 65363, 63235})
ENTER_KEYS = frozenset({10, 13})

# What each standing looks like on the timeline, BGR. A clicked frame is drawn as good and
# marked above the bar, because one frame is too thin to read as a colour of its own.
STANDING_COLOURS: dict[str, tuple[int, int, int]] = {
    "clicked": (90, 200, 90),
    "good": (90, 200, 90),
    "fair": (40, 190, 230),
    "poor": (40, 90, 250),
    "lost": (85, 85, 85),
}
LEGEND = (("good", "good"), ("fair", "check"), ("poor", "poor"), ("lost", "lost"))
# The bar before there is a chain to judge it by: the first pick has nothing to colour.
UNJUDGED = (110, 110, 110)
SUGGEST = (255, 200, 60)
# Darker than the click tool's panel, which keeps 60% because a landmark can sit under it.
# Nothing is clicked on the scrubber's frame, and its warnings are coloured text over
# whatever the broadcast put in that corner -- a red scorebug swallowed the orange ones.
SCRUB_PANEL_OPACITY = 0.8
CLICKED = (255, 255, 255)
GROUND = 24


def _banner(img: Any, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Lines of text over a dark panel, so a crowd or a floodlit stand behind stays readable."""
    scale = max(0.9, img.shape[1] / 2200)
    step = int(46 * scale)
    widest = max(
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] for text, _ in lines
    )
    pad = int(14 * scale)
    x0, y0 = max(0, 16 - pad), max(0, 44 - step + pad // 2)
    x1 = min(img.shape[1], 16 + widest + pad)
    y1 = min(img.shape[0], 44 + step * (len(lines) - 1) + pad)

    panel = img[y0:y1, x0:x1]
    if panel.size:
        img[y0:y1, x0:x1] = cv2.addWeighted(
            panel, 1 - SCRUB_PANEL_OPACITY, np.zeros_like(panel), SCRUB_PANEL_OPACITY, 0
        )
    for i, (text, colour) in enumerate(lines):
        y = 44 + i * step
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def _x(i: int, n: int, width: int) -> int:
    """The middle of frame i's column on a timeline `width` pixels wide."""
    return (i * width // n + (i + 1) * width // n) // 2


def _inside(x: int, size: int, width: int) -> int:
    """A marker's centre, moved in far enough that the first and last frame's are whole."""
    return max(size // 2 + 2, min(x, width - size // 2 - 2))


def _strip(width: int, frames: list[int], chain: Chain | None) -> tuple[Any, int, int]:
    """The timeline without its cursor: the bar, the clicked frames, the legend.

    Built once per window rather than per redraw, since it changes only when the chain
    does. Returns the image and the bar's top and bottom rows.
    """
    height = max(72, round(width * 0.05))
    img = np.full((height, width, 3), GROUND, dtype=np.uint8)
    top, bottom = round(height * 0.28), round(height * 0.58)
    n = len(frames)
    for i, f in enumerate(frames):
        x0 = i * width // n
        x1 = max(x0 + 1, (i + 1) * width // n)
        img[top:bottom, x0:x1] = (
            UNJUDGED if chain is None else STANDING_COLOURS[guide_mod.standing(chain, f)]
        )
    if chain is None:
        return img, top, bottom

    marker = max(10, round(height * 0.16))
    for i, f in enumerate(frames):
        if chain.carried_from.get(f) == 0:
            at = (_inside(_x(i, n, width), marker, width), top - marker // 2 - 2)
            cv2.drawMarker(img, at, CLICKED, cv2.MARKER_TRIANGLE_DOWN, marker, 2, cv2.LINE_AA)

    scale = max(0.45, height / 160)
    base = height - max(6, round(height * 0.1))
    swatch = max(8, round(height * 0.12))

    def label(text: str, x: int) -> int:
        cv2.putText(img, text, (x, base), cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT, 1, cv2.LINE_AA)
        return x + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] + swatch * 2

    x = 16
    for key, text in LEGEND:
        cv2.rectangle(img, (x, base - swatch), (x + swatch, base), STANDING_COLOURS[key], -1)
        x = label(text, x + swatch + 6)
    for colour, shape, text in (
        (CLICKED, cv2.MARKER_TRIANGLE_DOWN, "clicked"),
        (SUGGEST, cv2.MARKER_DIAMOND, "suggested"),
    ):
        cv2.drawMarker(
            img, (x + swatch // 2, base - swatch // 2), colour, shape, swatch + 4, 2, cv2.LINE_AA
        )
        x = label(text, x + swatch + 6)
    return img, top, bottom


def _timeline(strip: Any, top: int, bottom: int, i: int, n: int, suggest: int | None) -> Any:
    """The strip with where you are, and where the chain suggests clicking."""
    out = strip.copy()
    width = out.shape[1]
    if suggest is not None:
        size = max(12, round((bottom - top) * 0.9))
        at = (_inside(_x(suggest, n, width), size, width), top - size // 2 - 2)
        cv2.drawMarker(out, at, SUGGEST, cv2.MARKER_DIAMOND, size, 3, cv2.LINE_AA)
    x = _x(i, n, width)
    cv2.line(out, (x, top - 6), (x, bottom + 6), (0, 0, 0), 7)
    cv2.line(out, (x, top - 6), (x, bottom + 6), CLICKED, 3)
    return out


def _lines(
    chain: Chain | None, f: int, i: int, n: int, notice: str | None
) -> list[tuple[str, tuple[int, int, int]]]:
    """What the scrubber says, in words for whoever is clicking rather than for the pipeline."""
    if chain is None:
        lines = [
            ("Pick a frame with plenty of pitch lines in view", MARK),
            (f"frame {f}  ({i + 1} of {n})", TEXT),
        ]
        finish = "Q: quit"
    else:
        kind = guide_mod.standing(chain, f)
        lines = [
            ("Do the green lines sit on the pitch's own lines?", MARK),
            (guide_mod.overview(chain), TEXT),
            (
                f"frame {f}: {guide_mod.describe(chain, f)}",
                TEXT
                if kind in ("clicked", "good")
                else STANDING_COLOURS["poor" if kind == "lost" else kind],
            ),
        ]
        finish = "Q: they fit - finish"
    if notice:
        lines.append((notice, SUGGEST))
    lines.append(("A / D or arrows: step     [ ]: one second     click or drag the bar", TEXT))
    lines.append((f"ENTER: click the pitch on this frame     {finish}", TEXT))
    return lines


def working(text: str, done: int = 0, total: int = 0) -> None:
    """A message, and a bar when the size of the job is known, in the scrubber's window.

    For the waits between one click and the next. The longest is optical flow over the
    whole clip, the slowest step in the pipeline, and to somebody watching the window
    rather than the terminal it would otherwise look like a hang.

    Only into a window already on screen, never a new one. A wait is the caller blocking
    the event loop for seconds, and a window put up just before that never gets to finish
    coming forward (see `_open`) -- on a coach's machine every click on the scrubber after
    it was lost, for a whole session, while 187 mouse moves reached it.
    """
    if not _showing(SCRUB):
        return
    canvas = np.full((900, 1600, 3), GROUND, dtype=np.uint8)
    cv2.putText(canvas, text, (80, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.4, TEXT, 2, cv2.LINE_AA)
    if total:
        x0, y0, x1, y1 = 80, 470, 1520, 500
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 70, 70), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + round((x1 - x0) * done / total), y1), MARK, -1)
        cv2.putText(
            canvas, f"{done / total:.0%}", (x0, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT, 2
        )
    cv2.imshow(SCRUB, canvas)
    cv2.waitKey(1)


def _showing(name: str) -> bool:
    """Whether a window of this name is on screen. OpenCV raises for one it never made."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


# How long a new window's events are serviced before anything else may run, in seconds.
# Long enough for the activation macOS hands a new window to complete; measured on the
# machine that found this, it takes under 0.2 s.
SETTLE_S = 0.3


def _open(name: str) -> None:
    """Put a window on screen, creating it only if it is not already there.

    On macOS an app is only brought forward while its event loop is running, and OpenCV's
    loop runs only inside `waitKey`. A window put up and then left while the caller computed
    for three seconds never became active at all -- and OpenCV's view does not accept a
    click that has to activate its app, so from then on every click was discarded while the
    mouse moves still arrived. Measured on a coach's machine, for a whole `ft run` session:
    187 moves, no clicks, the app inactive throughout. So a new window's events are serviced
    for `SETTLE_S` before anything else is allowed to run, and `working` never makes one.

    A window closed with its own button is still known to OpenCV and cannot be shown again,
    so that one is destroyed properly and made anew.
    """
    if _showing(name):
        return
    with contextlib.suppress(cv2.error):
        cv2.destroyWindow(name)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1600, 900)
    end = time.monotonic() + SETTLE_S
    while time.monotonic() < end:
        cv2.waitKey(10)


def close() -> None:
    """Close the scrubber's window if it is open. Backends differ on closing one that is not."""
    try:
        cv2.destroyWindow(SCRUB)
        cv2.waitKey(1)
    except cv2.error:
        pass


def scrub(
    frames_dir: Path,
    frames: list[int],
    *,
    chain: Chain | None = None,
    start: int | None = None,
    notice: str | None = None,
) -> int | None:
    """Walk the clip and pick a frame. None if abandoned.

    Two jobs in one window, because they are the same act of looking. With no chain it
    shows the footage, which is how the FIRST seed is chosen - frame 1 is routinely a cut, a
    close-up, or the camera still finding the play, and nothing but a person can see that.
    With one it draws the reprojected markings over every frame, which is stage 1's picture
    (invariant 3) and the only way to catch a camera that is wrong in a way the residuals
    survive; and the timeline under it shows the whole chain at once, so the weak stretches
    are visible before anybody scrubs to them.

    `start` is where the chain suggests clicking, and is marked on the timeline as such.
    `notice` is what happened last - a click saved, or not.

    The timeline is the slider. OpenCV's trackbar was another control to click into, and a
    trackbar cannot be taken off a window again -- which would leave it on the click tool,
    now that both draw in the same window. The window stays open on the way out; whoever
    runs the session closes it (`close`).

    Disposable for the same reason the click tool is: the FILE is the interface (D23).
    """
    if not frames:
        return None
    n = len(frames)
    suggest = frames.index(start) if start is not None and start in frames else None
    at = suggest if suggest is not None else 0
    state: dict[str, int] = {"want": at, "height": 0, "width": 1}

    def on_mouse(event: int, x: int, y: int, flags: int, _param: Any) -> None:
        # Seeking by the timeline, which is where the colours say to go.
        pressed = event == cv2.EVENT_LBUTTONDOWN or (
            event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
        )
        if pressed and state["height"] and y >= state["height"]:
            state["want"] = max(0, min(x * n // state["width"], n - 1))

    _open(SCRUB)
    cv2.setMouseCallback(SCRUB, on_mouse)

    strip: Any = None
    top = bottom = 0
    shown: Any = None
    drawn = -1
    # A window closed with its own button never returns a key, so the loop has to notice.
    # Only once it has been SEEN open: some backends report nothing useful before the first
    # frame is shown, and reading that as closed would end the loop before it began.
    seen_open = False
    while True:
        i = state["want"]
        if i != drawn:
            drawn = i
            f = frames[i]
            raw = cv2.imread(str(frames_dir / f"{f:06d}.jpg"))
            if raw is not None:
                if strip is None:
                    strip, top, bottom = _strip(raw.shape[1], frames, chain)
                    state["height"], state["width"] = raw.shape[0], raw.shape[1]
                h = chain.homographies.get(f) if chain is not None else None
                picture = overlay_mod.draw(raw, h) if h is not None else raw.copy()
                _banner(picture, _lines(chain, f, i, n, notice))
                shown = np.vstack([picture, _timeline(strip, top, bottom, i, n, suggest)])
        if shown is not None:
            cv2.imshow(SCRUB, shown)

        key = cv2.waitKeyEx(20)
        if cv2.getWindowProperty(SCRUB, cv2.WND_PROP_VISIBLE) >= 1:
            seen_open = True
        elif seen_open:
            return None
        if key == -1:
            continue

        if key in LEFT_KEYS:
            step = -1
        elif key in RIGHT_KEYS:
            step = 1
        else:
            char = chr(key & 0xFF).lower()
            if char == "q":
                return None
            if key & 0xFF in ENTER_KEYS:
                return frames[state["want"]]
            step = {"a": -1, "d": 1, "[": -JUMP, "]": JUMP}.get(char, 0)
        if step:
            state["want"] = max(0, min(state["want"] + step, n - 1))
