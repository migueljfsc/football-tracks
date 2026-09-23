"""A coach's corrections, read back off a board he fixed in Pitchboard (D104).

Every change a coach makes to an imported board is a label nobody else has: on his club, his
broadcaster and his camera. D98 and D100 found that identity is what such labels buy -- a model
that has seen a side's players names them far better -- and D101 that three matches of ball
annotations do not carry to a fourth. Pitchboard keeps what its importer said on the board
(`origin`, its D88); this diffs the board against that and keeps what the coach changed.

Four kinds of label, each keyed by the video frame and the track it is about:

* **carrier** -- at this frame the ball was with this track, or with nobody, or with a player
  the tracker never had. What the ball's ranking lacks (D101).
* **number** -- this track wears this shirt number, every frame of it. What the number reader
  lacks (D102): crops on this broadcaster's fonts and this club's kit.
* **position** -- at this frame this track was really here. Registration and tracking error,
  measured where the coach looked.
* **side** -- this track is on the other team. The kit split's mistakes, which on Sporting's
  hoops are the ones that matter (D92).

A player the coach removed is recorded too: a track fielded that should not have been, whatever
the reason. Nothing is inferred from what the coach LEFT alone -- an untouched scene may be
right or merely unexamined, and only a change says someone looked.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

# How far a token has to move before a drag is a correction rather than a nudge, in metres.
# Pitchboard stores positions to the centimetre and a coach's hand is not that steady; a metre is
# about the camera model's own error, so anything smaller says nothing about where he was.
MOVED_M = 1.0

LABELS_FILE = "labels.json"


@dataclass
class Labels:
    clip: str
    fps: float
    carriers: list[dict[str, Any]] = field(default_factory=list)
    numbers: list[dict[str, Any]] = field(default_factory=list)
    positions: list[dict[str, Any]] = field(default_factory=list)
    sides: list[dict[str, Any]] = field(default_factory=list)
    removed: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[int] = field(default_factory=list)

    def count(self) -> int:
        return (
            len(self.carriers)
            + len(self.numbers)
            + len(self.positions)
            + len(self.sides)
            + len(self.removed)
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def resolve(track: int, known: set[int]) -> int | None:
    """The id in the tracks file of a track Pitchboard names.

    Pitchboard splits a track it finds an impossible jump in, and numbers the n-th piece
    `id * 1000 + n` (its `splitImpossible`); the piece's own span says which frames it was.
    """
    if track in known:
        return track
    if track // 1000 in known:
        return track // 1000
    return None


def corrections(board: dict[str, Any], known: set[int]) -> Labels:
    """Everything the coach changed on `board`, against what its importer said.

    `known` is the track ids in the tracks file the board was built from; a track the file does
    not have is reported in `unresolved` rather than guessed at.
    """
    origin = board.get("origin")
    if not origin:
        raise ValueError("this board was not imported from video, or predates `origin` (D88)")
    out = Labels(clip=origin["clip"], fps=float(origin["fps"]))
    players = origin["players"]
    missing: set[int] = set()

    def track(player_id: str | None) -> dict[str, Any] | None:
        """The track behind a token, as frames and an id the tracks file knows."""
        if player_id is None or player_id not in players:
            return None
        was = players[player_id]
        tid = resolve(int(was["track"]), known)
        if tid is None:
            missing.add(int(was["track"]))
            return None
        return {"track": tid, "from": int(was["from"]), "to": int(was["to"])}

    side_of = {
        p["id"]: ("home" if i == 0 else "away")
        for i, team in enumerate(board["teams"])
        for p in team["players"]
    }
    number_of = {p["id"]: p["number"] for team in board["teams"] for p in team["players"]}

    for scene in board["scenes"]:
        said = origin["scenes"].get(scene["id"])
        if said is None:
            continue  # a scene the coach added was never a frame of the video
        frame = int(said["frame"])

        if scene.get("carrier") != said.get("carrier"):
            now = scene.get("carrier")
            where = scene["positions"].get(now) if now else None
            out.carriers.append(
                {
                    "frame": frame,
                    "was": track(said.get("carrier")),
                    "now": track(now),
                    # A carrier the coach placed himself has no track: where he stood is the label.
                    "handPlaced": now is not None and now not in players,
                    "at": [where["x"], where["y"]] if where else None,
                }
            )

        for pid, (x0, y0) in said["positions"].items():
            now_at = scene["positions"].get(pid)
            if now_at is None:
                continue
            if math.hypot(now_at["x"] - x0, now_at["y"] - y0) < MOVED_M:
                continue
            behind = track(pid)
            if behind is None:
                continue
            out.positions.append(
                {"frame": frame, **behind, "was": [x0, y0], "now": [now_at["x"], now_at["y"]]}
            )

    for pid, was in players.items():
        behind = track(pid)
        if behind is None:
            continue
        if pid not in side_of:
            out.removed.append(behind)
            continue
        if side_of[pid] != was["side"]:
            out.sides.append({**behind, "side": side_of[pid], "was": was["side"]})
        if number_of[pid] != was["number"]:
            out.numbers.append({**behind, "number": number_of[pid], "was": was["number"]})

    out.unresolved = sorted(missing)
    return out
