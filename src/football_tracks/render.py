"""tracks.json -> a top-down video of coloured dots.

The picture invariant 3 asks for. It is deliberately not specific to the ground-truth
path: anything that writes a tracks.json can be looked at with this, which is what
makes it stage 4's proof as well as this stage's.

If the dots move like a football team, the positions are right. Nothing about the
numbers tells you that, which is the whole reason this module exists.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
from cv2.typing import MatLike

from . import pitch
from .tracks import TeamLabel

# BGR. Keepers are the outfield colour lightened rather than a colour of their own -
# a keeper is on a side, and a third hue reads as a third team.
COLORS: dict[str, tuple[int, int, int]] = {
    "home": (220, 130, 40),
    "away": (60, 60, 225),
    "gkHome": (240, 200, 140),
    "gkAway": (150, 150, 245),
    "referee": (40, 220, 240),
    "unknown": (180, 180, 180),
}

TRAIL_FRAMES = 12


def _by_frame(doc: dict[str, Any]) -> dict[int, list[tuple[dict[str, Any], dict[str, Any]]]]:
    """frame -> [(track, sample)]. Samples are sparse, so frames can be missing."""
    out: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for t in doc["tracks"]:
        for s in t["samples"]:
            out[s["f"]].append((t, s))
    return out


def frame(
    doc: dict[str, Any],
    f: int,
    *,
    scale: float = 10.0,
    margin: float = 3.0,
    trails: dict[int, list[tuple[float, float]]] | None = None,
    labels: bool = True,
) -> MatLike:
    """One rendered frame."""
    img = pitch.draw(scale, margin)
    per = _by_frame(doc)

    if trails:
        for tid, pts in trails.items():
            trail_team = next((t["team"] for t in doc["tracks"] if t["id"] == tid), "unknown")
            color = COLORS.get(trail_team, COLORS["unknown"])
            for i in range(1, len(pts)):
                cv2.line(
                    img,
                    pitch.to_px(*pts[i - 1], scale, margin),
                    pitch.to_px(*pts[i], scale, margin),
                    tuple(int(c * 0.55) for c in color),
                    max(1, round(scale / 10)),
                    cv2.LINE_AA,
                )

    r = max(3, round(scale * 0.55))
    for t, s in per.get(f, []):
        team: TeamLabel = t["team"]
        color = COLORS.get(team, COLORS["unknown"])
        centre = pitch.to_px(s["x"], s["y"], scale, margin)
        cv2.circle(img, centre, r, color, -1, cv2.LINE_AA)
        cv2.circle(img, centre, r, (20, 20, 20), max(1, r // 4), cv2.LINE_AA)

        if labels and t.get("number") is not None:
            text = str(t["number"])
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale / 40, 1)
            cv2.putText(
                img,
                text,
                (centre[0] - tw // 2, centre[1] + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale / 40,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        img, f"f{f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )
    return img


def video(doc: dict[str, Any], out: Path, *, scale: float = 10.0, margin: float = 3.0) -> Path:
    """Every frame the file has a sample for, at the source's frame rate."""
    per = _by_frame(doc)
    frames = sorted(per)
    if not frames:
        raise ValueError("no samples to render")

    w, h = pitch.canvas_size(scale, margin)
    fps = float(doc["source"]["fps"])
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a writer for {out}")

    trails: dict[int, list[tuple[float, float]]] = defaultdict(list)
    try:
        for f in frames:
            for t, s in per[f]:
                pts = trails[t["id"]]
                pts.append((s["x"], s["y"]))
                del pts[:-TRAIL_FRAMES]
            writer.write(frame(doc, f, scale=scale, margin=margin, trails=dict(trails)))
    finally:
        writer.release()
    return out


def still(doc: dict[str, Any], f: int, out: Path, *, scale: float = 10.0) -> Path:
    cv2.imwrite(str(out), frame(doc, f, scale=scale))
    return out


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data
