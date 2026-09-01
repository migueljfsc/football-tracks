"""Score a produced tracks.json against ground truth.

This is what D12 bought: both sides of the comparison are the same format written by
the same writer, so scoring is a diff rather than a translation. "70% accurate" stops
being a feeling about a video.

Everything here is pure and takes loaded documents, so the tests need no dataset.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

# How close a predicted position must be to count as the same player, in metres. Two
# metres is about a player's own width plus the slop in a good homography: tight enough
# that a neighbour cannot be claimed, loose enough that a correct detection is not lost.
MATCH_RADIUS = 2.0

# home <-> away, for scoring the team split without scoring which name it chose.
_SWAP: dict[str | None, str | None] = {
    "home": "away",
    "away": "home",
    "gkHome": "gkAway",
    "gkAway": "gkHome",
}


@dataclass(slots=True)
class Score:
    frames: int
    gt_samples: int
    pred_samples: int
    matched: int
    median_error_m: float
    p90_error_m: float
    team_accuracy: float
    identity_purity: float
    id_switches: int
    jersey_gt_total: int
    jersey_correct: int
    jersey_wrong: int
    jersey_missing: int

    @property
    def recall(self) -> float:
        return self.matched / self.gt_samples if self.gt_samples else 0.0

    @property
    def precision(self) -> float:
        return self.matched / self.pred_samples if self.pred_samples else 0.0


def _index(doc: dict[str, Any]) -> dict[int, list[tuple[int, float, float]]]:
    """frame -> [(track_id, x, y)]."""
    out: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for t in doc["tracks"]:
        for s in t["samples"]:
            out[s["f"]].append((t["id"], s["x"], s["y"]))
    return out


def _teams(doc: dict[str, Any]) -> dict[int, str]:
    return {t["id"]: t["team"] for t in doc["tracks"]}


def _numbers(doc: dict[str, Any]) -> dict[int, int | None]:
    return {t["id"]: t.get("number") for t in doc["tracks"]}


def match_frame(
    gt: list[tuple[int, float, float]],
    pred: list[tuple[int, float, float]],
    radius: float = MATCH_RADIUS,
) -> list[tuple[int, int, float]]:
    """Pair ground-truth positions with predicted ones, nearest first.

    Greedy by ascending distance rather than optimal assignment. Inside a radius this
    tight the two agree almost always, and greedy has the property that matters here:
    it is deterministic and it can be read, so a surprising score can be traced to a
    pair rather than to a solver.
    """
    pairs = []
    for gid, gx, gy in gt:
        for pid, px, py in pred:
            d = math.dist((gx, gy), (px, py))
            if d <= radius:
                pairs.append((d, gid, pid))
    pairs.sort()

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    out = []
    for d, gid, pid in pairs:
        if gid in used_gt or pid in used_pred:
            continue
        used_gt.add(gid)
        used_pred.add(pid)
        out.append((gid, pid, d))
    return out


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    i = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[i]


def score(truth: dict[str, Any], pred: dict[str, Any], *, radius: float = MATCH_RADIUS) -> Score:
    gt_by_frame, pred_by_frame = _index(truth), _index(pred)
    gt_teams, pred_teams = _teams(truth), _teams(pred)
    gt_numbers, pred_numbers = _numbers(truth), _numbers(pred)

    errors: list[float] = []
    team_hits = 0
    team_hits_swapped = 0
    matched = 0
    # gt track -> which predicted tracks it matched, and in what order
    assigned: dict[int, list[int]] = defaultdict(list)

    frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    for f in frames:
        for gid, pid, d in match_frame(gt_by_frame.get(f, []), pred_by_frame.get(f, []), radius):
            matched += 1
            errors.append(d)
            assigned[gid].append(pid)
            want, got_team = gt_teams.get(gid), pred_teams.get(pid)
            if want == got_team:
                team_hits += 1
            if want == _SWAP.get(got_team, got_team):
                team_hits_swapped += 1

    # Purity: the share of a ground-truth track's matched samples that went to its
    # single most common predicted partner. A tracker that swaps a player halfway
    # scores 0.5 here while recall stays perfect, which is exactly the failure that
    # recall alone cannot see.
    purities = []
    switches = 0
    for pids in assigned.values():
        purities.append(Counter(pids).most_common(1)[0][1] / len(pids))
        switches += sum(1 for a, b in pairwise(pids) if a != b)

    correct = wrong = missing = 0
    gt_numbered = 0
    for gid, number in gt_numbers.items():
        if number is None:
            continue
        gt_numbered += 1
        partners = assigned.get(gid)
        if not partners:
            missing += 1
            continue
        got = pred_numbers.get(Counter(partners).most_common(1)[0][0])
        if got is None:
            missing += 1
        elif got == number:
            correct += 1
        else:
            wrong += 1

    return Score(
        frames=len(frames),
        gt_samples=sum(len(v) for v in gt_by_frame.values()),
        pred_samples=sum(len(v) for v in pred_by_frame.values()),
        matched=matched,
        median_error_m=_percentile(errors, 0.5),
        p90_error_m=_percentile(errors, 0.9),
        # Permutation-invariant. Which cluster an unsupervised split calls "home" is
        # arbitrary - the pipeline settles it by which end the side plays at, and that
        # tie can go either way on a clip where both teams cover the same ground.
        # Scoring the raw labelling would measure that coin flip rather than whether
        # the two sides were told apart at all, which is the actual question.
        team_accuracy=max(team_hits, team_hits_swapped) / matched if matched else 0.0,
        identity_purity=sum(purities) / len(purities) if purities else 0.0,
        id_switches=switches,
        jersey_gt_total=gt_numbered,
        jersey_correct=correct,
        jersey_wrong=wrong,
        jersey_missing=missing,
    )


def report(s: Score) -> str:
    lines = [
        f"frames            {s.frames}",
        f"samples           {s.pred_samples} predicted / {s.gt_samples} true",
        f"recall            {s.recall:6.1%}",
        f"precision         {s.precision:6.1%}",
        f"position error    {s.median_error_m:.2f} m median, {s.p90_error_m:.2f} m p90",
        f"team split        {s.team_accuracy:6.1%}  (best of the two labellings)",
        f"identity purity   {s.identity_purity:6.1%}  ({s.id_switches} switches)",
        f"shirt numbers     {s.jersey_correct} right, {s.jersey_wrong} WRONG,"
        f" {s.jersey_missing} unread, of {s.jersey_gt_total}",
    ]
    if s.jersey_wrong:
        # D5: an unread number imports as a generic token and costs nothing. A wrong
        # one attaches a run to the wrong player and nobody downstream can see it.
        lines.append("                  a wrong number is worse than no number - see D5")
    return "\n".join(lines)
