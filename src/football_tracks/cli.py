"""Command line entry point. One command per stage."""

from __future__ import annotations

import importlib.util
import json
import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, cast

import cv2
import typer

from . import auto as auto_mod
from . import (
    calibration,
    soccernet,
    stage0_segment,
    stage1_propagate,
    stage1_register,
    stage3_teams,
    tracks,
)
from . import camera as camera_mod
from . import detect as detect_mod
from . import guide as guide_mod
from . import overlay as overlay_mod
from . import reid as reid_mod
from . import render as render_mod
from . import score as score_mod
from . import seed as seed_mod
from . import video as video_mod
from .config import (
    CALIB_DATA,
    CLIPS,
    WORK,
    Pitch,
    game_dir,
    read_pitch,
    work_dir,
    write_pitch,
)

app = typer.Typer(add_completion=False, help="Broadcast clip -> player tracks in pitch metres.")


def _size(root: Path, labels: dict[str, Any] | None) -> tuple[int, int]:
    """Frame size, from SoccerNet's labels or from clip.json."""
    if labels is not None:
        return (int(labels["images"][0]["width"]), int(labels["images"][0]["height"]))
    clip = video_mod.load(root)
    return (clip.width, clip.height)


def _clip_meta(clip: str) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads((CLIPS / clip / "clip.json").read_text()))


def _lens_of(clip: str) -> tuple[float, float]:
    """Where the lens axis falls in this clip's pixels: from its crop, or a frame's middle.

    SoccerNet's clips are uncropped broadcast frames and have no `clip.json` to say so.
    """
    if (CLIPS / clip / "clip.json").exists():
        return camera_mod.lens(_clip_meta(clip))
    width, height = _size(CLIPS / clip, soccernet.Clip(name=clip, root=CLIPS / clip).labels())
    return camera_mod.lens({"crop": [0, 0, width, height]})


def _game_of(clip: str, game: str | None) -> str:
    """Which match a clip is from: said outright, remembered, or named by SoccerNet's labels."""
    if game:
        return game
    try:
        named = str(_clip_meta(clip).get("game") or "")
    except (OSError, ValueError):
        named = ""
    labelled = soccernet.Clip(name=clip, root=CLIPS / clip)
    if not named and labelled.labels_path.exists():
        found = labelled.labels()["info"].get("game_id")
        named = f"sngs-{found}" if found else ""
    if not named:
        raise typer.BadParameter(
            f"{clip} is not in a game yet - pass --game, or run `ft camera <a seeded clip>"
            " --game <name>` on a clip of the same match first"
        )
    return named


def _maybe_game(clip: str, game: str | None) -> str | None:
    """The clip's match if anything names one, and None rather than an error if nothing does."""
    try:
        return _game_of(clip, game)
    except typer.BadParameter:
        return None


def _name_game(clip: str, game: str) -> None:
    """Remember a clip's match in its own clip.json, so it need not be said twice.

    A SoccerNet clip has no clip.json, and needs none: its labels name the game already.
    """
    path = CLIPS / clip / "clip.json"
    if not path.exists():
        return
    meta = _clip_meta(clip)
    meta["game"] = game
    path.write_text(json.dumps(meta, indent=2) + "\n")


def _game_camera(clip: str, game: str | None) -> tuple[camera_mod.Camera, tuple[float, float]]:
    """The match's camera, and where its lens axis falls in THIS clip's pixels."""
    named = _game_of(clip, game)
    path = game_dir(named, create=False) / camera_mod.FILE
    if not path.exists():
        raise typer.BadParameter(
            f"no camera for {named} - run `ft camera <a seeded clip of it> --game {named}`"
        )
    return camera_mod.read(path), _lens_of(clip)


def _carry(mode: str, carry: int) -> int | None:
    """`--carry` as the stages mean it: -1 is uncapped, 0 is none, N caps the chain."""
    return carry if carry >= 0 else None


def _fps(root: Path, labels: dict[str, Any] | None) -> float:
    """SoccerNet states it; a recording carries it in clip.json (and the container lies)."""
    if labels is not None:
        return float(labels["info"]["frame_rate"])
    return video_mod.load(root).fps


@app.command()
def segment(
    clip: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Source video.")],
    threshold: Annotated[
        float, typer.Option(help="Cut sensitivity. Lower finds more cuts.")
    ] = 27.0,
    min_seconds: Annotated[float, typer.Option(help="Shortest segment worth keeping.")] = 4.0,
    green_min: Annotated[
        float, typer.Option(help="Least pitch-green a tactical shot may be.")
    ] = 0.35,
    samples: Annotated[int, typer.Option(help="Frames scored per segment.")] = 6,
    extract: Annotated[
        int | None, typer.Option(help="Also cut this segment out to its own mp4.")
    ] = None,
) -> None:
    """Stage 0 - split a broadcast clip at its cuts and find the tactical camera."""
    out = work_dir(clip)
    segments, source = stage0_segment.find_segments(
        clip,
        threshold=threshold,
        min_seconds=min_seconds,
        green_min=green_min,
        samples=samples,
    )

    typer.echo(f"{source['clip']}  {source['width']}x{source['height']}  {source['fps']:.2f} fps")
    typer.echo(f"{'':>3} {'start':>8} {'end':>8} {'dur':>7} {'green':>6} {'motion':>7}")
    for s in segments:
        mark = "*" if s.main else " "
        typer.echo(
            f"{s.index:>2}{mark} {s.start_s:>8.2f} {s.end_s:>8.2f} {s.duration_s:>7.2f}"
            f" {s.green:>6.2f} {s.motion:>7.2f}"
        )

    path = stage0_segment.write(segments, source, out)
    typer.echo(f"\nwrote {path}")

    pick = stage0_segment.best(segments)
    if pick is None:
        typer.echo("no segment qualified - lower --green-min or --min-seconds")
    else:
        typer.echo(f"main camera: segment {pick.index} ({pick.duration_s:.1f}s)")

    if extract is not None:
        chosen = next((s for s in segments if s.index == extract), None)
        if chosen is None:
            raise typer.BadParameter(f"no segment {extract}")
        typer.echo(f"wrote {stage0_segment.extract(clip, chosen, out)}")


@app.command()
def clips(
    split: Annotated[str, typer.Option(help="train, valid, test or challenge.")] = "test",
) -> None:
    """List the clips in a SoccerNet GSR split, without downloading any of them."""
    with soccernet.open_split(split) as zf:
        names = soccernet.list_clips(zf)
    typer.echo(f"{len(names)} clips in {split}")
    typer.echo("  ".join(names))


@app.command()
def fetch(
    clip: Annotated[str, typer.Argument(help="Clip name, e.g. SNGS-147.")],
    split: Annotated[str, typer.Option(help="train, valid, test or challenge.")] = "test",
    limit: Annotated[
        int | None, typer.Option(help="Fetch only the first N frames, for a quick look.")
    ] = None,
) -> None:
    """Download one SoccerNet GSR clip - labels and frames - into data/clips/.

    Pulls a single clip out of the split's multi-gigabyte zip by range request, so
    this costs about 150 MB rather than the whole 8.85 GB.
    """
    with soccernet.open_split(split) as zf:
        typer.echo(f"fetching {clip} from {split} ...")
        out = soccernet.fetch(zf, clip, CLIPS, limit=limit)
    n = len(list((out / "img1").glob("*.jpg")))
    typer.echo(f"wrote {out}  ({n} frames)")


@app.command()
def truth(
    clip: Annotated[str, typer.Argument(help="A clip already fetched into data/clips/.")],
    referees: Annotated[
        bool, typer.Option(help="Keep referees rather than dropping them.")
    ] = False,
    interval_s: Annotated[
        float,
        typer.Option(
            "--interval-s",
            help="Seconds between stored positions. 0 keeps every frame, which is what a"
            " yardstick wants.",
        ),
    ] = 0.0,
) -> None:
    """Ground-truth labels -> tracks.json, with no CV in the loop.

    The yardstick every later stage is scored against, and a real file for
    Pitchboard's importer to be built against before any of the vision works.

    Interval defaults to 0 HERE, unlike everywhere else, for the reason `ft bench` gives:
    `ft score` counts samples, so a truth file reduced to a 0.1 s grid holds two fifths of
    the samples of the 25 fps run being scored against it, and every recall and precision
    number is then a fact about the grid. Writing this file at the default silently halved
    SNGS-116's true samples from 10,148 to 5,104 and took its precision from 74.9% to
    37.5% without a line of pipeline code changing.
    """
    c = soccernet.Clip(name=clip, root=CLIPS / clip)
    if not c.labels_path.exists():
        raise typer.BadParameter(f"no labels at {c.labels_path} - run `ft fetch {clip}` first")

    labels = c.labels()
    built = soccernet.to_tracks(labels, keep_referees=referees)
    ball = soccernet.to_ball(labels)
    info = labels["info"]
    frames = [int(img["file_name"].split(".")[0]) for img in labels["images"]]

    out = work_dir(Path(clip))
    path = tracks.write(
        out / "truth.json",
        clip=clip,
        fps=float(info["frame_rate"]),
        start_frame=min(frames),
        end_frame=max(frames),
        tracks=built,
        ball=ball,
        width=labels["images"][0]["width"],
        height=labels["images"][0]["height"],
        interval_s=interval_s,
    )

    named = sum(1 for t in built if t.number is not None)
    total_samples = sum(len(t.samples) for t in built)
    typer.echo(
        f"{len(built)} tracks, {total_samples} samples, {named} with a shirt number,"
        f" ball on {len(ball)} frames"
    )
    typer.echo(f"wrote {path}")


@app.command()
def render(
    path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="A tracks.json or truth.json.")
    ],
    scale: Annotated[float, typer.Option(help="Pixels per metre.")] = 10.0,
    still: Annotated[int | None, typer.Option(help="Render one frame as a PNG instead.")] = None,
) -> None:
    """Draw a tracks file as a top-down video of coloured dots.

    The picture invariant 3 asks for. If the dots move like a football team the
    positions are right, and nothing in the numbers can tell you that.
    """
    doc = render_mod.load(path)
    if still is not None:
        out = render_mod.still(doc, still, path.with_suffix(f".f{still}.png"), scale=scale)
    else:
        out = render_mod.video(doc, path.with_suffix(".mp4"), scale=scale)
    typer.echo(f"wrote {out}")


@app.command("calib-train")
def calib_train(
    epochs: Annotated[int, typer.Option(help="Passes over the training set.")] = 12,
    batch: Annotated[int, typer.Option(help="Frames per step.")] = 8,
    stride: Annotated[
        int, typer.Option(help="Use every Nth annotated frame; 750 consecutive ones are one shot.")
    ] = 5,
    holdout: Annotated[int, typer.Option(help="Matches held out, when none are named.")] = 1,
    resume: Annotated[
        bool, typer.Option(help="Continue from the saved weights rather than starting over.")
    ] = False,
    holdout_games: Annotated[
        str,
        typer.Option(
            help="Comma-separated game ids to hold out. Defaults to the matches the"
            " benchmark clips come from, so the benchmark stays honest."
        ),
    ] = "7,8",
    extra: Annotated[
        bool, typer.Option(help="Add SN-Calibration-2023, if it has been fetched.")
    ] = True,
    extra_stride: Annotated[
        int, typer.Option(help="Use every Nth calibration frame; they are single shots, not runs.")
    ] = 1,
) -> None:
    """Train the pitch-line segmenter (D36).

    Validation is by MATCH, never by clip or by frame. Two clips of one game share a
    stadium, a camera and a kit, so any other split reports a generalisation that was
    never tested.
    """
    from . import calib

    frames = calib.index_clips(CLIPS)[::stride]
    held = {g.strip() for g in holdout_games.split(",") if g.strip()} or None
    train_set, val_set = calib.split_by_game(frames, holdout=holdout, games=held)

    # SN-Calibration-2023 arrives with its OWN match-disjoint split, so it is added on
    # either side of ours rather than re-split: 290 matches to train on and 55 to validate
    # against, none shared. It is what makes the set diverse enough to be worth training on
    # at all, and GSR stays in because the benchmark clips are GSR footage.
    if extra:
        train_set += calib.index_calibration(CALIB_DATA, "train")[::extra_stride]
        val_set += calib.index_calibration(CALIB_DATA, "valid")[::extra_stride]

    if not train_set:
        raise typer.BadParameter(f"no annotated frames under {CLIPS} - run `ft fetch` first")
    games = sorted({f.game for f in train_set + val_set})
    typer.echo(
        f"{len(train_set) + len(val_set)} frames over {len(games)} matches"
        f" -> train {len(train_set)}, validate {len(val_set)}"
    )
    if not val_set:
        raise typer.BadParameter("no held-out match; fetch more clips or lower --holdout")
    calib.train(train_set, val_set, epochs=epochs, batch=batch, log=typer.echo, resume=resume)


@app.command("reg-eval")
def reg_eval(
    clip: Annotated[str, typer.Argument(help="A clip with ground-truth pitch lines.")],
    mode: Annotated[str, typer.Option(help="'truth', 'seed' or 'camera'.")] = "seed",
    game: Annotated[
        str | None,
        typer.Option(help="`--mode camera`: the match. Defaults to SoccerNet's game id."),
    ] = None,
    carry: Annotated[int, typer.Option(help="Frames a homography may be carried.")] = -1,
    weights: Annotated[Path | None, typer.Option(help="Segmenter weights.")] = None,
    within: Annotated[
        str, typer.Option(help="Comma-separated metre thresholds to report shares at.")
    ] = "1,2,5",
) -> None:
    """How much of a clip a registration solves, and how well, over ALL of it.

    Scoring only the frames a model already solves rewards refusing the hard ones: a model
    that answers a tenth of a clip perfectly beats one that answers all of it within a metre,
    and the second makes the better board (D67). Five training runs were killed against that
    number before this replaced it.

    This one counts every frame the ground truth can judge. A frame with no homography
    is not skipped, it is a miss -- which is the whole difference, and the reason the
    shares below are shares of the CLIP rather than of the answers.
    """
    import numpy as np

    if mode not in ("truth", "seed", "camera"):
        raise typer.BadParameter("mode must be 'truth', 'seed' or 'camera'")
    picked: auto_mod.Mode = cast("auto_mod.Mode", mode)
    # The match's camera, fitted from its OTHER clips -- `ft camera <clips> --truth` -- so
    # this clip's labels judge a camera that never saw them.
    rig, lens_at = _game_camera(clip, game) if mode == "camera" else (None, None)

    c = soccernet.Clip(name=clip, root=CLIPS / clip)
    if not c.labels_path.exists():
        raise typer.BadParameter(f"{clip} has no ground-truth lines to score against")
    labels = c.labels()
    out = work_dir(Path(clip))
    frames = sorted(int(p.stem) for p in c.frames_dir.glob("*.jpg"))
    width, height = _size(CLIPS / clip, labels)

    truth = stage1_register.fit_all(labels)
    got = auto_mod.homographies(
        labels,
        c.frames_dir,
        picked,
        max_carry=_carry(mode, carry),
        motions=stage1_propagate.motions(c.frames_dir, frames, cache=out / "motions.json"),
        weights=weights,
        rig=rig,
        lens_at=lens_at,
        aims_cache=out / "aims.json",
    )

    judged = [f for f in frames if truth.get(f) is not None]
    if not judged:
        raise typer.BadParameter(f"{clip} has no frame the ground truth can judge")
    errors: list[float] = []
    for f in judged:
        want, have = truth[f], got.get(f)
        if want is not None and have is not None:
            errors.append(calibration.observed_error(want, have, (height, width)))
    solved = len(errors)
    thresholds = [float(t) for t in within.split(",") if t.strip()]
    shares = "  ".join(
        f"within {t:g} m {sum(1 for e in errors if e <= t) / len(judged):.0%}" for t in thresholds
    )
    typer.echo(
        f"{clip} {mode}: registered {solved}/{len(judged)} ({solved / len(judged):.0%})  {shares}"
    )
    if errors:
        e = np.array(errors)
        typer.echo(
            f"  on the frames it solved: p50 {np.median(e):.2f} m"
            f"  p90 {np.percentile(e, 90):.2f} m  worst {e.max():.2f} m"
        )

    # And the same question asked where the PLAYERS are, which is not where the probes
    # are. A camera model is only ever used to place people, and they stand in a band
    # across the middle of the frame: a fit can agree better with the annotated lines and
    # put the players further from where they were (D68).
    at_players, boxes, thrown = stage1_register.player_errors(labels, got)
    if boxes:
        shares = "  ".join(
            f"within {t:g} m {sum(1 for x in at_players if x <= t) / boxes:.0%}" for t in thresholds
        )
        p = np.array(at_players) if at_players else np.array([float("nan")])
        typer.echo(
            f"  at the players: {boxes} boxes  {shares}"
            f"   p50 {np.median(p):.2f} m  p90 {np.percentile(p, 90):.2f} m"
            f"  thrown off the pitch {thrown}"
        )


@app.command()
def bench(
    clips: Annotated[
        str,
        typer.Option(help="Comma-separated clip names. Defaults to every clip with frames."),
    ] = "",
    interval_s: Annotated[
        float, typer.Option("--interval-s", help="Held at 0 so recall counts samples, not slots.")
    ] = 0.0,
    mode: Annotated[
        str,
        typer.Option(
            help="Registration to run every clip through: 'seed' simulates one click on frame"
            " one, 'camera' aims each match's own camera (D96) and is what a match ships on."
        ),
    ] = "seed",
) -> None:
    """Run every clip end to end and print one table.

    The command this project did not have, and the absence cost it: the numbers recorded
    in PLAN.md came from a bespoke sweep, so nothing since could be compared with them and
    two conclusions were nearly drawn from the difference. A benchmark that anyone can
    re-run is worth more than a better number nobody can reproduce.

    Interval defaults to 0 here and nowhere else. `ft score` counts samples, so a file
    reduced to a time grid scores a third of the recall of the same tracking at 25 fps --
    which is a fact about the grid and not about the pipeline.
    """
    import json as json_mod

    names = (
        [c.strip() for c in clips.split(",") if c.strip()]
        if clips
        else sorted(p.name for p in CLIPS.iterdir() if p.is_dir() and (p / "img1").is_dir())
    )
    if not names:
        raise typer.BadParameter(f"no clips in {CLIPS}")

    header = (
        f"{'clip':<16} {'tracks':>6} {'recall':>7} {'precis':>7} {'error':>8} "
        f"{'purity':>7} {'teams':>6}  notes"
    )
    typer.echo(header)
    typer.echo("-" * len(header))
    for name in names:
        out = work_dir(Path(name))
        pred_path = out / "tracks.json"
        try:
            _pipeline(name, mode, -1, interval_s)
        except Exception as exc:
            typer.echo(f"{name:<16} {'-':>6} {'-':>7} {'-':>7} {'-':>8} {'-':>7} {'-':>6}  {exc}")
            continue
        doc = json_mod.loads(pred_path.read_text())
        tracks_n = len(doc["tracks"])
        gt = out / "truth.json"
        if gt.exists():
            s = score_mod.score(render_mod.load(gt), render_mod.load(pred_path))
            typer.echo(
                f"{name:<16} {tracks_n:>6} {s.recall:>6.1%} {s.precision:>6.1%}"
                f" {s.median_error_m:>6.2f} m {s.identity_purity:>6.1%}"
                f" {s.team_accuracy:>5.0%}  ground truth"
            )
        else:
            # No truth to score against, so report what CAN be checked: a broadcast clip
            # with a plausible roster and a real spread of positions is not proof of a
            # good board, but a roster of three is proof of a bad one.
            samples = sum(len(t["samples"]) for t in doc["tracks"])
            frames = {s["f"] for t in doc["tracks"] for s in t["samples"]}
            teams: dict[str, int] = {}
            for t in doc["tracks"]:
                teams[t["team"]] = teams.get(t["team"], 0) + 1
            per = f"{samples / len(frames):.1f}/frame" if frames else "-"
            typer.echo(
                f"{name:<16} {tracks_n:>6} {'-':>7} {'-':>7} {'-':>8} {'-':>7} {'-':>6}"
                f"  no truth: {per}, {len(frames)} frames, "
                + " ".join(f"{k}={v}" for k, v in sorted(teams.items()))
            )


@app.command()
def score(
    prediction: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="A produced tracks.json.")
    ],
    truth_path: Annotated[
        Path | None,
        typer.Option("--truth", help="Ground truth. Defaults to truth.json beside it."),
    ] = None,
    radius: Annotated[float, typer.Option(help="Match radius in metres.")] = score_mod.MATCH_RADIUS,
) -> None:
    """Diff a produced tracks.json against ground truth."""
    gt = truth_path or prediction.parent / "truth.json"
    if not gt.exists():
        raise typer.BadParameter(f"no ground truth at {gt} - run `ft truth <clip>` first")
    typer.echo(
        score_mod.report(
            score_mod.score(render_mod.load(gt), render_mod.load(prediction), radius=radius)
        )
    )


@app.command()
def calibrate(
    clip: Annotated[str, typer.Argument(help="A clip already fetched into data/clips/.")],
    frame: Annotated[
        int | None, typer.Option(help="Draw the overlay for this frame instead of measuring.")
    ] = None,
    video: Annotated[
        bool, typer.Option(help="Draw the overlay for every frame, as an mp4.")
    ] = False,
    carry: Annotated[
        int,
        typer.Option(
            help="Carry a homography this many frames across gaps the solver cannot fill."
            " 0 disables it; a negative value means uncapped."
        ),
    ] = stage1_propagate.DEFAULT_MAX_CARRY,
    drift_from: Annotated[
        int | None,
        typer.Option(help="Measure how far a homography carried from this frame wanders."),
    ] = None,
) -> None:
    """Stage 1 - fit a homography per frame from the pitch lines, and check it.

    With no options this MEASURES: ground-truth boxes are pushed through the fitted
    homography and compared with the position SoccerNet recorded for them, which
    isolates the camera model from detection and gives the pipeline's error ceiling.

    With --frame or --video it draws the picture, which is the only way to see a
    homography that is wrong in a way the averages survive.
    """
    c = soccernet.Clip(name=clip, root=CLIPS / clip)
    out = work_dir(Path(clip))
    frames_all = sorted(int(p.stem) for p in c.frames_dir.glob("*.jpg"))
    if not frames_all:
        raise typer.BadParameter(f"no frames in {c.frames_dir}")

    motions = stage1_propagate.motions(c.frames_dir, frames_all, cache=out / "motions.json")
    seed_path = out / "seed.json"
    labels: dict[str, object] | None = None

    # What was fitted from evidence rather than carried to. A carry can only be scored
    # against this, and on a seeded clip it is the ONE clicked frame - `from_seed` has
    # already carried by the time it returns, so `homs` is the chain, not the evidence.
    direct: dict[int, Any] = {}
    chain: stage1_propagate.Chain | None = None

    if c.labels_path.exists():
        # SoccerNet: every frame carries its own pitch lines.
        labels = c.labels()
        homs = stage1_register.fit_all(labels)
        direct = {f: h for f, h in homs.items() if h is not None}
    elif seed_path.exists():
        # A clip nobody annotated: every clicked frame, carried both ways between them.
        clicked, refused = auto_mod.usable_seeds(out, c.frames_dir)
        for path, why in refused:
            typer.echo(f"IGNORING {path.name}: {why}")
        if not clicked:
            raise typer.BadParameter(f"{clip} has no usable seed")
        chain = auto_mod.chain_from_seeds(
            clicked,
            frames_all,
            c.frames_dir,
            max_carry=None,
            motions=motions,
        )
        homs = chain.homographies
        direct = {s.frame: seed_mod.homography(s) for s in clicked}
    else:
        raise typer.BadParameter(
            f"{clip} has neither SoccerNet labels nor {seed_path} - run `ft seed {clip}` first"
        )

    if carry != 0 and labels is not None:
        chain = stage1_propagate.fill(
            c.frames_dir, homs, max_carry=None if carry < 0 else carry, motion=motions
        )
        homs = chain.homographies

    if drift_from is not None:
        # What DEFAULT_MAX_CARRY is set from. Carrying is unbounded in principle, so the
        # cap is only honest while somebody can re-derive the number behind it.
        #
        # Scored against `direct` and never against `homs`: on a clip with no labels every
        # entry in `homs` IS the carry, so scoring against it compares the chain with
        # itself and reports 0.00 m however far the camera has wandered.
        truth = dict(direct)
        if len(truth) < 2:
            raise typer.BadParameter(
                f"{clip} has one seeded frame and no per-frame labels, so there is nothing"
                " independent to score a carry against. Seed a second frame further on"
                f" (`ft seed {clip} --frame N --check`) and measure again."
            )
        if drift_from not in truth:
            raise typer.BadParameter(f"frame {drift_from} has no homography to carry")
        # To the end of the clip. Capping the walk short of the next piece of evidence
        # reports "nothing to carry" for a chain that simply had not reached it yet.
        walked = stage1_propagate.drift(
            c.frames_dir, truth, drift_from, length=max(frames_all) - drift_from
        )
        if not walked:
            raise typer.BadParameter(
                f"carrying from frame {drift_from} reached no other fitted frame -"
                " the chain breaks before the next one (a cut, or grass the flow cannot hold)"
            )
        typer.echo(f"{'carried':>9} {'error on screen':>16}")
        for carried, error in walked:
            typer.echo(f"{carried:>8}f {error:>14.2f} m")
        return

    if frame is not None or video:
        import cv2

        frames = sorted(homs) if video else [frame] if frame is not None else []
        writer = None
        for f in frames:
            src = c.frames_dir / f"{f:06d}.jpg"
            if not src.exists():
                if not video:
                    raise typer.BadParameter(f"no frame at {src} - fetch the clip without --limit")
                continue
            raw = cv2.imread(str(src))
            if raw is None:
                continue
            img = overlay_mod.draw(raw, homs[f])
            overlay_mod.annotate(
                img, f"f{f}" if homs[f] is not None else f"f{f} unsolved", ok=homs[f] is not None
            )
            if not video:
                dest = out / f"calib.f{f}.png"
                cv2.imwrite(str(dest), img)
                typer.echo(f"wrote {dest}")
            else:
                if writer is None:
                    h, w = img.shape[:2]
                    dest = out / "calib.mp4"
                    writer = cv2.VideoWriter(
                        str(dest),
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        _fps(CLIPS / clip, labels),
                        (w, h),
                    )
                writer.write(img)
        if writer is not None:
            writer.release()
            typer.echo(f"wrote {out / 'calib.mp4'}")
        return

    if chain is not None:
        typer.echo(
            f"carried           {chain.carried} frames across gaps"
            f" ({chain.solved_directly} solved directly, {chain.gaps} left unsolved)"
        )
    if labels is None:
        solved = sum(1 for h in homs.values() if h is not None)
        typer.echo(f"frames solved     {solved}/{len(homs)}  ({solved / max(1, len(homs)):.1%})")
        for line in guide_mod.weakest(clip, chain, homs):
            typer.echo(line)
        typer.echo("no ground truth here - check the overlay with --frame or --video")
        return
    typer.echo(stage1_register.report(stage1_register.evaluate(labels, homs)))


def _detect(clip: str, conf: float, *, label: str = "detecting") -> tuple[str, Path]:
    """`ft detect`'s work, shared with `ft run`: what it found, and where it wrote it."""
    c = soccernet.Clip(name=clip, root=CLIPS / clip)
    frames = sorted(int(p.stem) for p in c.frames_dir.glob("*.jpg"))
    if not frames:
        raise typer.BadParameter(f"no frames in {c.frames_dir} - run `ft fetch {clip}` first")

    out = work_dir(Path(clip))
    with typer.progressbar(frames, label=label) as bar:
        found, balls = detect_mod.run(
            c.frames_dir, frames, conf=conf, progress=lambda _f: bar.update(1)
        )
    path = detect_mod.write(out / "detections.json", found, balls, conf=conf)
    said = (
        f"{len(found)} people over {len(frames)} frames"
        f" ({len(found) / max(1, len(frames)):.1f}/frame),"
        f" {len(balls)} ball sightings"
    )
    return said, path


@app.command()
def detect(
    clip: Annotated[str, typer.Argument(help="A clip already fetched into data/clips/.")],
    conf: Annotated[
        float, typer.Option(help="Detection confidence floor.")
    ] = detect_mod.DEFAULT_CONF,
) -> None:
    """Stage 2a - find people in every frame, and cache them.

    Slow, and separate from tracking on purpose: tracking is the part that gets tuned.
    """
    said, path = _detect(clip, conf)
    typer.echo(said)
    typer.echo(f"wrote {path}")


def _reid(clip: str, *, label: str = "embedding") -> tuple[str, Path]:
    """`ft reid`'s work, shared with `ft run`: what it embedded, and where it wrote it."""
    c = soccernet.Clip(name=clip, root=CLIPS / clip)
    out = work_dir(Path(clip))
    dets_path = out / "detections.json"
    if not dets_path.exists():
        raise typer.BadParameter(f"no {dets_path} - run `ft detect {clip}` first")
    detections, _balls = detect_mod.read(dets_path)
    frames = sorted({d.f for d in detections})
    with typer.progressbar(frames, label=label) as bar:
        features, valid = reid_mod.run(
            c.frames_dir, detections, WORK / "reid", progress=lambda _f: bar.update(1)
        )
    path = reid_mod.write(out / "appearance.npz", detections, features, valid)
    return f"{int(valid.sum())} of {len(detections)} detections embedded", path


@app.command()
def reid(
    clip: Annotated[str, typer.Argument(help="A clip already fetched into data/clips/.")],
) -> None:
    """Stage 2c - embed how every detected player looks, and cache it.

    Needs `ft detect` first. The tracker never reads it; the stitcher does, to spend its contact
    slack only on a join whose two ends look like one man (D95). The weights download to
    work/reid/ on first use and are checked against a pinned hash before every load.
    """
    said, path = _reid(clip)
    typer.echo(said)
    typer.echo(f"wrote {path}")


def _pipeline(
    clip: str,
    mode: str,
    carry: int,
    interval_s: float,
    stitch: bool = True,
    game: str | None = None,
) -> tuple[Path, auto_mod.Result]:
    """Frames in, tracks.json out. Shared by `ft auto` and `ft bench`.

    Extracted so the benchmark runs the SAME pipeline the user runs, rather than a
    second copy of it that can drift away from it silently.
    """
    if mode not in ("truth", "seed", "camera"):
        raise typer.BadParameter("mode must be 'truth', 'seed' or 'camera'")
    picked: auto_mod.Mode = cast("auto_mod.Mode", mode)

    c = soccernet.Clip(name=clip, root=CLIPS / clip)
    out = work_dir(Path(clip))
    pitch = read_pitch(out)
    dets_path = out / "detections.json"
    if not dets_path.exists():
        raise typer.BadParameter(f"no {dets_path} - run `ft detect {clip}` first")

    frames = sorted(int(p.stem) for p in c.frames_dir.glob("*.jpg"))
    detections, balls = detect_mod.read(dets_path)
    appearance_path = out / "appearance.npz"
    appearance = reid_mod.read(appearance_path, detections)
    if appearance is None and appearance_path.exists():
        typer.echo(
            f"IGNORING {appearance_path.name}: it was embedded from other detections"
            f" - run `ft reid {clip}`"
        )
    motions = stage1_propagate.motions(c.frames_dir, frames, cache=out / "motions.json")

    labels: dict[str, Any] | None = None
    seed_path = out / "seed.json"
    if picked == "camera":
        # The camera came from another clip of the match, so this one need not be clicked at
        # all -- and where it was, those clicks are three numbers like any other frame (D96).
        rig, lens_at = _game_camera(clip, game)
        pitch = rig.pitch
        # A SoccerNet clip's frame rate and size live in its labels; nothing here reads the
        # lines in them, which is what keeps the score honest.
        labels = c.labels() if c.labels_path.exists() else None
        aimed, refused = auto_mod.aimed_seeds(out, c.frames_dir, rig, lens_at)
        for path, why in refused:
            typer.echo(f"IGNORING {path.name}: {why}")
        homs = auto_mod.camera_homographies(
            c.frames_dir, rig, lens_at, seeds=aimed, cache=out / "aims.json"
        )
        solved = sum(1 for h in homs.values() if h is not None)
        typer.echo(f"camera {rig.game}: {solved}/{len(homs)} frames aimed")
    elif c.labels_path.exists():
        labels = c.labels()
        homs = auto_mod.homographies(
            labels,
            c.frames_dir,
            picked,
            max_carry=_carry(mode, carry),
            motions=motions,
        )
    elif seed_path.exists():
        # A real clip: one seeded frame is all the camera information there is.
        usable, refused = auto_mod.usable_seeds(out, c.frames_dir)
        for path, why in refused:
            typer.echo(f"IGNORING {path.name}: {why}")
        if not usable:
            raise typer.BadParameter(f"{clip} has no usable seed")
        homs = auto_mod.from_seeds(
            usable, frames, c.frames_dir, max_carry=_carry(mode, carry), motions=motions
        )
    else:
        raise typer.BadParameter(
            f"{clip} has neither SoccerNet labels nor {seed_path} - run `ft seed {clip}` first"
        )

    # Which kit is `home` is the MATCH's to say once its first clip has said it, so every clip
    # of it names the same team the same way whichever end they are attacking (D99).
    match = _maybe_game(clip, game)
    kits_path = game_dir(match, create=False) / stage3_teams.KITS_FILE if match else None
    stored = stage3_teams.read_kits(kits_path) if kits_path is not None else None
    result = auto_mod.build(
        c.frames_dir,
        frames,
        detections,
        homs,
        fps=_fps(CLIPS / clip, labels),
        motions=motions,
        balls=balls,
        appearance=appearance,
        stitch=stitch,
        pitch=pitch,
        kits=stored,
    )
    if kits_path is not None and stored is None and result.signatures is not None:
        stage3_teams.write_kits(kits_path, result.signatures, clip)
        colours = result.kits or {}
        typer.echo(
            f"{match}: kits recorded from {clip} - home wears"
            f" {colours.get('home', 'the kit on the left here')} in every clip of it from now on"
        )

    path = tracks.write(
        out / "tracks.json",
        clip=clip,
        fps=_fps(CLIPS / clip, labels),
        start_frame=min(frames),
        end_frame=max(frames),
        tracks=result.tracks,
        ball=result.ball,
        width=_size(CLIPS / clip, labels)[0],
        height=_size(CLIPS / clip, labels)[1],
        pitch=pitch,
        interval_s=interval_s,
        kits=result.kits,
    )
    return path, result


@app.command()
def auto(
    clip: Annotated[str, typer.Argument(help="A clip already fetched into data/clips/.")],
    mode: Annotated[
        str,
        typer.Option(
            help="'truth' uses every frame's lines; 'seed' uses only frame one's, or the"
            " clicked seeds of a real clip; 'camera' aims the match's own camera at every"
            " frame and needs no clicks."
        ),
    ] = "seed",
    game: Annotated[
        str | None,
        typer.Option(help="Which match this clip is from, for `--mode camera`."),
    ] = None,
    carry: Annotated[int, typer.Option(help="Frames a homography may be carried.")] = -1,
    interval_s: Annotated[
        float,
        typer.Option(
            "--interval-s",
            help="Seconds between stored positions. 0 writes every frame, which is finer"
            " than the pipeline is accurate.",
        ),
    ] = tracks.DEFAULT_INTERVAL_S,
    stitch: Annotated[
        bool,
        typer.Option(help="Join track fragments that are each other's best continuation."),
    ] = True,
) -> None:
    """The automatic path end to end - frames in, tracks.json out.

    `--mode truth` holds stage 1 fixed so the score is stages 2 and 3 alone.
    `--mode seed` throws away every line annotation but frame one's, which is what a
    human clicking four corners once actually leaves you with -- and on a real clip,
    carries the frames that were clicked.
    `--mode camera` reads the learned pitch lines as three numbers -- pan, tilt and zoom -- off the
    position the match was shot from, which another clip's clicks already established. No
    seed on this clip, and nothing to carry (D96).
    """
    if game:
        _name_game(clip, game)
    path, result = _pipeline(clip, mode, carry, interval_s, stitch, game)
    typer.echo(
        f"mode {mode}: {result.detections} detections -> {result.raw_tracks} raw tracks"
        f" -> {len(result.tracks)} kept"
    )
    typer.echo(
        f"dropped off pitch {result.dropped_off_pitch}, frames with no homography"
        f" {result.unsolved_frames}, ball located on {len(result.ball)} frames"
    )
    typer.echo(f"wrote {path}")


# `ft run` speaks at two levels: headings at the margin, and everything a step says under
# them, indented to line up with the heading's text.
INDENT = " " * 6

# What `ft detect` and `ft reid` import when they run, and nothing imports at the top of a
# module, so a missing extra surfaces only when they start -- by which time somebody has
# spent ten minutes clicking. The whole `vision` extra rather than torch alone: D77 is the
# account of how pillow and torchvision came to be listed at all.
VISION_MODULES = ("torch", "torchvision", "transformers", "PIL")

RUN_STEPS = (
    "extract frames",
    "pitch size",
    "place the camera",
    "detect people",
    "embed appearance",
    "track",
    "render",
)


def _say(text: str) -> None:
    typer.echo(f"{INDENT}{text}")


def _warn(text: str) -> None:
    typer.secho(f"{INDENT}{text}", fg=typer.colors.YELLOW)


def _duration(seconds: float) -> str:
    """How long, in the fewest units that still say it."""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _shown(path: Path) -> str:
    """A path as it is typed from the repo root, which is where `ft` runs."""
    try:
        return str(path.relative_to(WORK.parent))
    except ValueError:
        return str(path)


@dataclass(slots=True)
class _Outcome:
    """What a step came to, set by the step itself, for the heading's last line and the table."""

    text: str = "done"


class _Steps:
    """`ft run`'s steps as numbered headings, and a table of what each came to.

    The terminal is the operator's record -- what ran, what was skipped because it was already
    cached, how long the slow parts took -- while the window speaks to whoever is clicking. A
    step that raises is marked before the error reaches typer, so the last heading above a
    traceback says which step it came from.
    """

    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names
        self.rows: list[tuple[str, str, float | None]] = []
        self.began = time.monotonic()

    def _heading(self, name: str) -> None:
        typer.echo("")
        typer.secho(
            f"[{self.names.index(name) + 1}/{len(self.names)}] {name}",
            bold=True,
            fg=typer.colors.CYAN,
        )

    @contextmanager
    def step(self, name: str) -> Iterator[_Outcome]:
        self._heading(name)
        began = time.monotonic()
        outcome = _Outcome()
        try:
            yield outcome
        except (KeyboardInterrupt, typer.Exit, typer.Abort):
            typer.secho(f"{INDENT}stopped after {_duration(time.monotonic() - began)}", dim=True)
            raise
        except Exception:
            typer.secho(
                f"{INDENT}failed after {_duration(time.monotonic() - began)}", fg=typer.colors.RED
            )
            raise
        took = time.monotonic() - began
        typer.secho(f"{INDENT}{outcome.text}  ({_duration(took)})", fg=typer.colors.GREEN)
        self.rows.append((name, outcome.text, took))

    def skip(self, name: str, why: str, tag: str) -> None:
        """A step that did not need to run: `why` under its heading, `tag` in the table."""
        self._heading(name)
        typer.secho(f"{INDENT}skipped - {why}", dim=True)
        self.rows.append((name, f"skipped - {tag}", None))

    def summary(self, artefacts: list[tuple[str, str]]) -> None:
        width = max(len(n) for n in self.names)
        wide = 52
        rule = typer.style("-" * (width + wide + 14), dim=True)
        typer.echo(f"\n  {rule}")
        for name, text, took in self.rows:
            shown = text if len(text) <= wide else text[: wide - 3] + "..."
            timing = _duration(took) if took is not None else "-"
            typer.echo(
                f"  {name:<{width}}  {typer.style(f'{shown:<{wide}}', dim=took is None)}"
                f"  {timing:>8}"
            )
        typer.echo(f"  {rule}")
        total = _duration(time.monotonic() - self.began)
        typer.secho(f"  {'total':<{width}}  {'':<{wide}}  {total:>8}", bold=True)
        typer.echo("")
        for label, value in artefacts:
            typer.echo(f"  {typer.style(f'{label:<{width}}', bold=True)}  {value}")


def _missing_vision(work: Path, fresh: bool) -> list[str]:
    """The `vision` extra's modules this run will need and cannot import.

    None needed when both caches are here and agree with each other, because then neither
    stage that imports them runs.
    """
    dets = work / "detections.json"
    if not fresh and dets.exists():
        detections, _balls = detect_mod.read(dets)
        if reid_mod.read(work / "appearance.npz", detections) is not None:
            return []
    return [m for m in VISION_MODULES if importlib.util.find_spec(m) is None]


def _ask_pitch(work: Path) -> Pitch:
    """How big this pitch is, asked before anything is clicked.

    Before, and not after: the seed's landmarks are WRITTEN in these dimensions, so a
    pitch set afterwards describes a different pitch from the one that was clicked. The
    wrong answer here is the quietest failure in the repo - every player moves a metre or
    two across the pitch, the residuals stay small, and the overlay still lands on the
    real paint (D89).
    """
    here = read_pitch(work)
    _say("the Laws fix the markings and leave the field variable - only elite competition")
    _say("pins it at 105 x 68, and the wrong size moves every player (D89)")
    return Pitch(
        length=float(
            typer.prompt(f"{INDENT}length, goal line to goal line (m)", default=here.length)
        ),
        width=float(typer.prompt(f"{INDENT}width, touchline to touchline (m)", default=here.width)),
    )


def _motions(
    frames_dir: Path, numbers: list[int], work: Path, *, window: bool = False
) -> dict[int, Any]:
    """How the camera moves between frames, with progress the one time it is measured.

    The slowest step in the pipeline and cached after, so the bar appears once per clip.
    `window` draws it in the scrubber as well, for the wait that falls between two clicks,
    where whoever is clicking is watching the window and not the terminal.
    """
    cache = work / "motions.json"
    if cache.exists():
        return stage1_propagate.motions(frames_dir, numbers, cache=cache)
    from . import seedui

    total = len(numbers)
    done = 0
    with typer.progressbar(length=total, label=f"{INDENT}following the camera") as bar:

        def tick(_f: int) -> None:
            nonlocal done
            done += 1
            bar.update(1)
            if window and (done % 10 == 0 or done == total):
                seedui.working("Following the camera through the clip...", done, total)

        return stage1_propagate.motions(frames_dir, numbers, cache=cache, progress=tick)


def _click(
    clip: str,
    frame: int,
    *,
    check: bool,
    work: Path,
    frames_dir: Path,
    start: Any = None,
    rig: camera_mod.Camera | None = None,
    lens_at: tuple[float, float] | None = None,
) -> tuple[str, bool]:
    """Click one frame until the pipeline will use it, or whoever is clicking gives up.

    `ft seed` writes whatever was clicked and leaves the checking to `ft calibrate`. Here
    the person is still at the window, so a seed `usable_seeds` refuses is moved aside --
    named rather than deleted, because it is human work -- and the click tool reopens on the
    same frame saying why, in words that need no knowledge of the pipeline. The refusal in
    the pipeline's own words goes to the terminal.

    Some camera angles offer nothing more to click, so every refusal ends in a way out:
    `q` in the click tool skips the frame. `start` is the camera the chain carries here,
    which a fit through traced curves is refined from.

    With `rig`, the clicks are judged as an AIM of the match's camera rather than as a
    camera of their own: a frame too thin to fit a homography is still worth three numbers,
    and a seed clicked on the wrong goal misses instead of fitting perfectly (D88, D96).

    Returns what the scrubber should say next, and whether the frame was given up on
    after a refusal -- which is what stops the scrubber suggesting that stretch again.
    """
    from . import seedui

    message: str | None = None
    while True:
        written = _seed_one(
            clip,
            frame,
            check=check,
            message=message,
            say=_say,
            start=start,
            window=seedui.SCRUB,
            rig=rig,
            lens_at=lens_at,
        )
        if written is None:
            if message is not None:
                return (
                    f"Skipped frame {frame} - the camera there stays as it was followed,"
                    " and that stretch won't be suggested again.",
                    True,
                )
            return f"Nothing saved on frame {frame}.", False
        path, notes = written
        if rig is not None and lens_at is not None:
            _aimed, refused = auto_mod.aimed_seeds(work, frames_dir, rig, lens_at)
        else:
            _usable, refused = auto_mod.usable_seeds(work, frames_dir)
        why = next((reason for p, reason in refused if p == path), None)
        if why is None:
            more = f" ({len(notes)} note{'' if len(notes) == 1 else 's'} in the terminal)"
            return f"Saved your clicks on frame {frame}.{more if notes else ''}", False
        # Out of the `seed.*.json` glob, or it would go on anchoring the clip regardless.
        aside = path.with_name(f"refused.{path.name}")
        path.replace(aside)
        _warn(f"REFUSED {path.name}, moved to {aside.name}: {why}")
        img = video_mod.read_frame(frames_dir, frame)
        message = (
            f"{why}\nQ: skip this frame."
            if rig is not None or img is None
            else guide_mod.refusal(seed_mod.read(aside), img.shape[1], img.shape[0])
        )


def _seed_until_happy(clip: str, frames_dir: Path, numbers: list[int], work: Path) -> str:
    """Click, look, click again - until whoever is here says the camera is right.

    Where to click next is not a judgement a person can make from the footage. Drift is
    invisible until the markings are drawn back onto the frame, and the frame where two
    anchors disagree most is arithmetic nobody does by eye (D83). So the loop proposes the
    frame, opens the window already there, and shows the reprojection over every frame on
    the way; `q` ends it.

    One seed cannot cross a pan (D34), and anchors reach BOTH ways (D80), so a second
    click is the normal case rather than a repair.

    The motion between frames is measured only once a first seed exists: picking and
    clicking a frame needs none of it, and it is the slowest thing in the pipeline.

    Nothing here can trap anybody. A frame given up on after a refusal takes its whole
    stretch out of the suggestions, `q` in the scrubber always finishes, and whatever is
    still weak at that point is named in the terminal rather than left for the board to
    reveal.

    Returns what the camera came to, short enough for the run's summary table.
    """
    from . import seedui

    motions: dict[int, Any] | None = None
    notice: str | None = None
    warned: set[Path] = set()
    skipped: set[int] = set()
    try:
        while True:
            usable, refused = auto_mod.usable_seeds(work, frames_dir)
            for path, why in refused:
                if path not in warned:
                    _warn(f"IGNORING {path.name}: {why}")
                    warned.add(path)

            if not usable:
                picked = seedui.scrub(frames_dir, numbers, notice=notice)
                if picked is None:
                    _warn("nothing clicked, so there is no camera and nothing after this can run")
                    _warn(f"pick it up again with: ft run {clip}")
                    raise typer.Exit(1)
                _say(f"clicking frame {picked}")
                notice, _gave_up = _click(
                    clip, picked, check=False, work=work, frames_dir=frames_dir
                )
                continue

            if motions is None:
                motions = _motions(frames_dir, numbers, work, window=True)
            seedui.working("Placing the pitch on every frame...")
            chain = auto_mod.chain_from_seeds(
                usable, numbers, frames_dir, max_carry=None, motions=motions
            )
            _say(guide_mod.overview(chain))
            for line in guide_mod.weakest(clip, chain, chain.homographies):
                _say(line)

            nxt = guide_mod.next_seed(chain, avoid=skipped)
            picked = seedui.scrub(
                frames_dir,
                numbers,
                chain=chain,
                start=nxt[0] if nxt else None,
                notice=notice,
            )
            if picked is None:
                return _still_weak(chain, skipped)
            _say(f"clicking frame {picked}")
            notice, gave_up = _click(
                clip,
                picked,
                check=True,
                work=work,
                frames_dir=frames_dir,
                start=chain.homographies.get(picked),
            )
            if gave_up:
                skipped.add(picked)
    finally:
        seedui.close()


def _fit_game_camera(clip: str, game: str, work: Path, frames_dir: Path) -> str:
    """Fit the match's camera from the clicks just made, so the next clip needs none.

    Several seeds aimed at different parts of the pitch are what pin a position down (D96);
    fewer say so rather than writing a camera nobody should trust.
    """
    usable, _refused = auto_mod.usable_seeds(work, frames_dir)
    if len(usable) < 2:
        return f"{game} has no camera yet: it takes two seeds or more"
    lens_at = camera_mod.lens(_clip_meta(clip))
    got = camera_mod.fit(
        [(s, lens_at) for s in sorted(usable, key=lambda s: s.frame)],
        pitch=read_pitch(work),
        game=game,
    )
    if got is None:
        return f"{game} has no camera: these seeds describe none"
    rig = got[0]
    camera_mod.write(game_dir(game) / camera_mod.FILE, rig)
    _name_game(clip, game)
    return (
        f"{game}'s camera is at ({rig.x:.1f}, {rig.y:.1f}), {rig.height:.1f} m up"
        " - the next clip of it needs no clicks"
    )


def _aim_until_happy(
    clip: str,
    frames_dir: Path,
    numbers: list[int],
    work: Path,
    rig: camera_mod.Camera,
    lens_at: tuple[float, float],
) -> str:
    """Place the camera with the match's own, and ask for clicks only where it cannot see.

    The clip may need nothing at all: where the segmenter reads the markings, three numbers
    off a position another clip established register the frame (D96). What is left is the
    stretches it cannot read -- a close-up, a crowd shot, a replay -- and those are offered
    one at a time. They cost what a seed always cost -- four landmarks or so, because that is
    what tells the right goal from the wrong one -- and there are far fewer of them.

    The segmenter's pass is cached, so clicking and looking again costs a second rather than
    another two minutes.
    """
    from . import seedui

    notice: str | None = None
    skipped: set[int] = set()
    warned: set[Path] = set()
    with typer.progressbar(length=len(numbers), label=f"{INDENT}reading the lines") as bar:
        views, _ = auto_mod.camera_aims(
            frames_dir,
            rig,
            lens_at,
            cache=work / "aims.json",
            progress=lambda _f: bar.update(1),
        )
    read = len(views)
    try:
        while True:
            aimed, refused = auto_mod.aimed_seeds(work, frames_dir, rig, lens_at)
            for path, why in refused:
                if path not in warned:
                    _warn(f"IGNORING {path.name}: {why}")
                    warned.add(path)
            clicked = {}
            for seeded in aimed:
                got = camera_mod.aim_from_seed(rig, lens_at, seeded)
                if got is not None:
                    clicked[seeded.frame] = got[0]
            chain = auto_mod.camera_chain(
                rig, lens_at, {**views, **clicked}, numbers, set(clicked), auto_mod.MAX_SPAN_FRAMES
            )
            solved = sum(1 for h in chain.homographies.values() if h is not None)
            _say(
                f"{rig.game}'s camera: {solved}/{len(numbers)} frames"
                f" ({read} read from the lines, {len(clicked)} clicked, {chain.carried} spanned)"
            )
            blind = [
                (first, last) for first, last, kind in guide_mod.stretches(chain) if kind == "lost"
            ]
            for first, last in blind[:3]:
                _say(f"no camera      {first}-{last} ({last - first + 1} frames)")
            dark = sum(last - first + 1 for first, last in blind)
            # A stretch somebody gave up on is not offered again: nothing in this loop may
            # trap them, and a close-up has nothing to click whatever the wizard thinks.
            offer = [run for run in blind if not any(run[0] <= f <= run[1] for f in skipped)]
            if not offer:
                return f"{solved}/{len(numbers)} frames" + (
                    f", {dark} without a camera" if dark else ", no clicking needed"
                )

            first, last = max(offer, key=lambda run: run[1] - run[0])
            picked = seedui.scrub(
                frames_dir, numbers, chain=chain, start=(first + last) // 2, notice=notice
            )
            if picked is None:
                return f"{solved}/{len(numbers)} frames, {dark} without a camera"
            _say(f"clicking frame {picked}")
            notice, gave_up = _click(
                clip, picked, check=True, work=work, frames_dir=frames_dir, rig=rig, lens_at=lens_at
            )
            if gave_up:
                skipped.add(picked)
    finally:
        seedui.close()


def _still_weak(chain: stage1_propagate.Chain, skipped: set[int]) -> str:
    """Name every stretch still poor or lost when the clicking stops, and sum them up.

    Finishing with a weak stretch is allowed -- some angles offer nothing to click -- but
    it is never silent: those are the frames where the board's players may be metres out.
    """
    weak = [s for s in guide_mod.stretches(chain) if s[2] in ("poor", "lost")]
    for first, last, kind in weak:
        span = f"frames {first}-{last}" if last > first else f"frame {first}"
        gave_up = any(first <= f <= last for f in skipped)
        _warn(f"{span} still {kind}{' - skipped, nothing more to click there' if gave_up else ''}")
    clicked = sum(1 for r in chain.carried_from.values() if r == 0)
    frames = sum(last - first + 1 for first, last, _kind in weak)
    said = f"{clicked} frame{'' if clicked == 1 else 's'} clicked"
    return f"{said}, {frames} still weak" if frames else f"{said}, none weak"


@app.command()
def run(
    source: Annotated[
        str,
        typer.Argument(
            help="A video file to extract, or the name of a clip already in data/clips/."
        ),
    ],
    name: Annotated[str | None, typer.Option(help="Clip name. Defaults to the file stem.")] = None,
    mode: Annotated[
        str,
        typer.Option(
            help="'seed' carries the clicked frames; 'camera' aims the match's own camera,"
            " which is chosen automatically when the game has one."
        ),
    ] = "seed",
    game: Annotated[
        str | None,
        typer.Option(
            help="Which match this clip is from. With a camera already fitted for it the"
            " clip needs no clicks; without one, this clip's clicks fit it for the next."
        ),
    ] = None,
    carry: Annotated[int, typer.Option(help="Frames a homography may be carried.")] = -1,
    interval_s: Annotated[
        float,
        typer.Option(
            "--interval-s", help="Seconds between stored positions. 0 writes every frame."
        ),
    ] = tracks.DEFAULT_INTERVAL_S,
    length: Annotated[
        float, typer.Option(help="Pitch length in metres. Given, the question is not asked.")
    ] = 0.0,
    width: Annotated[
        float, typer.Option(help="Pitch width in metres. Given, the question is not asked.")
    ] = 0.0,
) -> None:
    """A recording in, tracks.json out - asking only for what a person has to supply.

    The stage commands remain, and this runs the same code they do. What it adds is the
    ORDER, which is not obvious and where every trap in this pipeline lives: the pitch
    dimensions must be set before a landmark is clicked (D89), a re-detect invalidates the
    appearance cache and the stitcher then silently loses its contact slack (D95), and one
    seed is rarely enough because a chain drifts without announcing it (D18, D34).

    `source` is inferred: a file is extracted, anything else is a clip already extracted,
    which is how a clip whose detections are already cached is picked up again.
    """
    src = Path(source)
    fresh = src.is_file()
    clip = name or (src.stem if fresh else source)
    if not fresh and not (CLIPS / clip / "img1").exists():
        raise typer.BadParameter(
            f"{source} is neither a video file nor a clip in {CLIPS} - pass a recording"
            " to extract, or the name of one already extracted"
        )
    # Checked here and not where `_pipeline` checks it, which is after the clicking.
    if mode not in ("truth", "seed", "camera"):
        raise typer.BadParameter("mode must be 'truth', 'seed' or 'camera'")

    work = work_dir(Path(clip))
    missing = _missing_vision(work, fresh)
    if missing:
        typer.secho(
            f"the vision extra is not installed (no {', '.join(missing)}). Detecting and"
            " embedding need it, and that is better found out now than after the clicking:",
            fg=typer.colors.RED,
            err=True,
        )
        typer.echo("    uv sync --extra vision", err=True)
        raise typer.Exit(1)

    steps = _Steps(RUN_STEPS)
    typer.secho(f"ft run {clip}", bold=True)

    if fresh:
        with steps.step("extract frames") as out:
            got = _extract(src, name, say=_warn)
            out.text = f"{got.frames} frames at {got.fps:.2f} fps, {got.width}x{got.height}"
    else:
        steps.skip("extract frames", "already extracted", "already extracted")

    frames_dir = CLIPS / clip / "img1"
    numbers = sorted(int(p.stem) for p in frames_dir.glob("*.jpg"))
    if not numbers:
        raise typer.BadParameter(f"no frames in {frames_dir}")
    annotated = soccernet.Clip(name=clip, root=CLIPS / clip).labels_path.exists()

    # An annotated clip is on SoccerNet's 105 x 68 by their convention, which is what its
    # own pitch lines are named against (`calibration.PITCH_LINES`), so there is nothing to
    # ask -- and a size written here would move every player off the lines it came with.
    if annotated:
        steps.skip(
            "pitch size", "annotated clips are 105 x 68 by SoccerNet's convention", "annotated"
        )
    else:
        with steps.step("pitch size") as out:
            here = read_pitch(work)
            want = (
                Pitch(length=length or here.length, width=width or here.width)
                if (length or width)
                else _ask_pitch(work)
            )
            if not (90.0 <= want.length <= 120.0 and 45.0 <= want.width <= 90.0):
                raise typer.BadParameter(
                    f"{want.length:g} x {want.width:g} m is not a football pitch -"
                    " the Laws allow 90-120 by 45-90"
                )
            if auto_mod.seed_paths(work) and (want.length, want.width) != (
                here.length,
                here.width,
            ):
                _warn(
                    "this clip is already seeded, and those clicks were written against"
                    f" {here.length:g} x {here.width:g} m - re-seed, or they describe a"
                    " different pitch"
                )
            write_pitch(work, want)
            out.text = f"{want.length:g} x {want.width:g} m"

    # A match already seeded once has a camera, and a clip of it needs no clicks at all --
    # the lines in the picture say where that camera was aimed (D96). Clicks are asked for
    # only where the segmenter can read nothing.
    if game and not annotated:
        _name_game(clip, game)
    named = game or str(_clip_meta(clip).get("game") or "") if not annotated else ""
    fitted = (game_dir(named, create=False) / camera_mod.FILE) if named else None
    rig = camera_mod.read(fitted) if fitted is not None and fitted.exists() else None
    if rig is not None and mode == "seed":
        mode = "camera"
    elif mode == "camera" and rig is None:
        raise typer.BadParameter(
            "--mode camera needs a camera for the game - pass --game, or run `ft camera"
            " <a seeded clip of it> --game <name>` first"
        )

    if annotated:
        steps.skip(
            "place the camera", "annotated: its own pitch lines register every frame", "annotated"
        )
    elif rig is not None:
        with steps.step("place the camera") as out:
            out.text = _aim_until_happy(
                clip, frames_dir, numbers, work, rig, camera_mod.lens(_clip_meta(clip))
            )
    else:
        with steps.step("place the camera") as out:
            out.text = _seed_until_happy(clip, frames_dir, numbers, work)
            # The clicking just done is what places the camera for every OTHER clip of this
            # match, and it is only knowable once: fit it now rather than asking again later.
            if named:
                out.text += f"; {_fit_game_camera(clip, named, work, frames_dir)}"

    dets = work / "detections.json"
    if dets.exists():
        steps.skip(
            "detect people", "detections.json is cached - delete it to detect again", "cached"
        )
    else:
        with steps.step("detect people") as out:
            out.text, _written = _detect(clip, detect_mod.DEFAULT_CONF, label=f"{INDENT}detecting")

    # Never detect without embedding after it. `appearance.npz` is keyed by detection, so a
    # re-detect invalidates it -- and the pipeline's answer to a stale one is to carry on
    # with no contact slack at all, which is a quiet regression to D94 rather than an error.
    detections, _balls = detect_mod.read(dets)
    if reid_mod.read(work / "appearance.npz", detections) is not None:
        steps.skip("embed appearance", "appearance.npz matches these detections", "cached")
    else:
        with steps.step("embed appearance") as out:
            out.text, _written = _reid(clip, label=f"{INDENT}embedding")

    with steps.step("track") as out:
        _motions(frames_dir, numbers, work)
        path, result = _pipeline(clip, mode, carry, interval_s, game=named or None)
        _say(
            f"{result.detections} detections -> {result.raw_tracks} raw tracks"
            f" -> {len(result.tracks)} kept"
        )
        _say(
            f"dropped off pitch {result.dropped_off_pitch},"
            f" frames with no homography {result.unsolved_frames}"
        )
        out.text = f"{len(result.tracks)} tracks, ball located on {len(result.ball)} frames"

    # The picture invariant 3 asks for, and the first thing to look at: if the dots do not
    # move like a football team, no number on the board will say so.
    with steps.step("render") as out:
        doc = render_mod.load(path)
        sampled = len({s["f"] for t in doc["tracks"] for s in t["samples"]})
        with typer.progressbar(length=sampled, label=f"{INDENT}drawing") as bar:
            movie = render_mod.video(
                doc, path.with_suffix(".mp4"), progress=lambda _f: bar.update(1)
            )
        out.text = f"{sampled} frames of dots"

    # The board is the only judgement that counts, and seven per-frame wins have failed to
    # reach it (D36) -- so the run ends by naming it rather than by reporting a metric.
    steps.summary(
        [
            ("tracks", _shown(path)),
            ("video", _shown(movie)),
            ("board", f"pnpm board ../football-tracks/{_shown(path)}"),
        ]
    )


# Everything cached from a clip's frames and keyed by frame NUMBER -- which a re-extraction
# keeps and a trimmed recording shifts, so each of these would be reused against footage it
# was never measured on. The camera aims are keyed by weights and camera, not by the
# picture, and a trimmed clip re-extracted under its old name reads every one of them a few
# hundred frames out.
FROM_FRAMES = (
    "motions.json",
    "detections.json",
    "appearance.npz",
    "aims.json",
)


def _extract(
    source: Path, name: str | None, *, say: Callable[[str], None] = typer.echo
) -> video_mod.Clip:
    """`ft frames`' work, shared with `ft run`. `say` hears what was thrown away to do it."""
    clip_name = name or source.stem
    dest = CLIPS / clip_name
    before = len(list((dest / "img1").glob("*.jpg")))
    # Which match this is: a person said so, extraction does not measure it, and a trimmed
    # recording re-extracted under its name is still that match. Said aloud rather than kept
    # quietly, because the same name could be reused for another match's recording.
    try:
        match = str(_clip_meta(clip_name).get("game") or "")
    except (OSError, ValueError):
        match = ""
    clip = video_mod.extract(source, dest)
    if match:
        _name_game(clip_name, match)
        say(f"kept the match it was named for, {match} - pass --game to name another")
    if before > clip.frames:
        say(
            f"replaced {before} frames already under this name - a shorter recording used to"
            " leave the tail of the longer one behind, and everything downstream read the two"
            " as one clip"
        )

    # The frames are this command's own output. So is everything CACHED from them, and a
    # cache keyed by frame number is silently reused against different footage -- a real
    # clip was re-extracted over another and carried the previous one's optical flow, which
    # nothing downstream could notice. Those go.
    #
    # The seed does not: it is the only human work in the pipeline, and a coach who
    # reframed the same match may well want it. It is named instead, because a seed from
    # another camera fits nothing and says nothing about it.
    work = work_dir(Path(clip_name))
    if before:
        for cached in FROM_FRAMES:
            path = work / cached
            if path.exists():
                path.unlink()
                say(f"dropped {cached}: it was measured from the frames just replaced")
        # Every seed, not just the primary one: the extras are anchors too, and one left
        # behind by the previous clip is the worst kind of stale cache -- it is human work,
        # it looks deliberate, and it puts the football in the wrong half. Moved rather
        # than deleted, and out of the name the pipeline reads.
        for stale in auto_mod.seed_paths(work):
            aside = stale.with_name(f"stale.{stale.name}")
            stale.rename(aside)
            say(
                f"moved {stale.name} to {aside.name}: it was clicked on the clip that was"
                " here before, and a seed only fits the picture it was clicked on"
            )
    return clip


@app.command()
def frames(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A video file.")],
    name: Annotated[str | None, typer.Option(help="Clip name. Defaults to the file stem.")] = None,
) -> None:
    """Turn a broadcast video into the numbered-JPEG layout the pipeline speaks.

    Detects and removes pillarbox and letterbox bars, and records the REAL frame rate
    rather than the one the container claims - a screen recording routinely lies.
    """
    clip = _extract(source, name)
    typer.echo(
        f"{clip.frames} frames at {clip.fps:.2f} fps, {clip.width}x{clip.height}"
        f" (cropped from {clip.crop})"
    )
    typer.echo(f"wrote {CLIPS / clip.name}")


@app.command()
def pitch(
    clip: str,
    length: float = typer.Option(0.0, help="Goal line to goal line, in metres."),
    width: float = typer.Option(0.0, help="Touchline to touchline, in metres."),
) -> None:
    """How big this clip's pitch really is.

    The Laws fix the markings and leave the field variable -- 90 to 120 m long, 45 to 90
    wide -- and only elite competition pins it, so the 105 x 68 default is right for a
    European night and wrong for most football. It matters because the seed's landmarks
    are written in it: click a goal post as `(0, W/2 - 3.66)` with the wrong W and every
    player lands a metre or two off across the pitch, with good residuals and a touchline
    that is not where the touchline is (D89).

    Set it BEFORE seeding. A seed already clicked was written against whatever was in
    force then, and this does not go back and change it.
    """
    work = work_dir(Path(clip))
    if not length and not width:
        here = read_pitch(work)
        typer.echo(f"{clip}: {here.length:g} x {here.width:g} m")
        return
    here = read_pitch(work)
    want = Pitch(length=length or here.length, width=width or here.width)
    if not (90.0 <= want.length <= 120.0 and 45.0 <= want.width <= 90.0):
        raise typer.BadParameter(
            f"{want.length:g} x {want.width:g} m is not a football pitch -"
            " the Laws allow 90-120 by 45-90"
        )
    if auto_mod.seed_paths(work):
        typer.echo(
            "WARNING: this clip is already seeded, and those clicks were written against"
            f" {here.length:g} x {here.width:g} m. Re-seed, or they describe a different pitch."
        )
    typer.echo(f"wrote {write_pitch(work, want)}")


@app.command()
def camera(
    clips: Annotated[
        list[str],
        typer.Argument(help="Clips of one match: seeded ones, or with --truth, SoccerNet's."),
    ],
    game: Annotated[
        str | None, typer.Option(help="What to call the match. Defaults to the first clip's.")
    ] = None,
    truth: Annotated[
        bool,
        typer.Option(
            help="Fit from SoccerNet's labelled lines rather than from clicks -- how a"
            " benchmark match gets a camera its other clips can be scored against."
        ),
    ] = False,
    per_clip: Annotated[
        int, typer.Option(help="With --truth: labelled frames to take from each clip.")
    ] = 4,
    overlay: Annotated[
        bool, typer.Option(help="Draw the markings back onto a frame of each.")
    ] = True,
) -> None:
    """Where the camera stood for this whole match, from what its clips were clicked with.

    A broadcast camera does not move between clips of the same game -- it pans, tilts and
    zooms from one gantry -- so the position several views agree on is the position every
    other clip of that match was also shot from. `ft auto <other clip> --mode camera` then
    registers those clips with no clicking at all (D96).

    It takes SEVERAL views, aimed at different parts of the pitch. One frame is eight
    numbers of evidence and the position is three of them, so a single view trades height
    against zoom and settles anywhere along that trade. What the fit prints is how far the
    position moves when each clip -- or, from one clip, each seed -- is left out of it,
    which is the number that says whether the views really pinned it.
    """
    import numpy as np

    views_of: list[tuple[str, seed_mod.Seed, tuple[float, float]]] = []
    pitch: Pitch | None = None
    for clip in clips:
        root = CLIPS / clip
        lens_at = _lens_of(clip)
        if truth:
            labelled = soccernet.Clip(name=clip, root=root)
            if not labelled.labels_path.exists():
                raise typer.BadParameter(f"{clip} has no labelled lines to fit from")
            labels = labelled.labels()
            seen = stage1_register.evidence(labels)
            # Frames whose lines pin a homography, spread across the clip: a clip is a few
            # seconds of one pan, and its ends are the views furthest apart.
            usable = sorted(f for f, h in stage1_register.fit_all(labels).items() if h is not None)
            step = max(1, (len(usable) - 1) // max(1, per_clip - 1))
            chosen = list(dict.fromkeys(usable[::step][:per_clip]))
            seeds = [
                seed_mod.Seed(frame=f, points=[], lines=seen[f][0], arcs=seen[f][1]) for f in chosen
            ]
            # SoccerNet's lines are named on their own 105 x 68 (`calibration.PITCH_LINES`).
            here = Pitch()
        else:
            work = work_dir(Path(clip))
            found, refused = auto_mod.usable_seeds(work, root / "img1")
            for path, why in refused:
                typer.echo(f"IGNORING {clip}/{path.name}: {why}")
            seeds = sorted(found, key=lambda s: s.frame)
            here = read_pitch(work)
        if pitch is not None and (here.length, here.width) != (pitch.length, pitch.width):
            raise typer.BadParameter(
                f"{clip} is set to a {here.length:g} x {here.width:g} m pitch and the others"
                f" to {pitch.length:g} x {pitch.width:g} - one match is one pitch (D89)"
            )
        pitch = here
        views_of += [(clip, s, lens_at) for s in seeds]

    if pitch is None or len(views_of) < 2:
        raise typer.BadParameter(
            f"{len(views_of)} usable view(s); a camera needs at least two, and several aimed at"
            " different parts of the pitch is what pins it"
        )
    try:
        named = _game_of(clips[0], game)
    except typer.BadParameter:
        named = clips[0]

    got = camera_mod.fit([(s, at) for _clip, s, at in views_of], pitch=pitch, game=named)
    if got is None:
        raise typer.BadParameter("these views describe no camera")
    rig, views = got
    typer.echo(
        f"{named}: camera at ({rig.x:.1f}, {rig.y:.1f}) m, {rig.height:.1f} m up,"
        f" on a {pitch.length:g} x {pitch.width:g} m pitch, from {len(views_of)} views"
    )
    for clip in clips:
        rows = [
            (s, view, at) for (c, s, at), view in zip(views_of, views, strict=True) if c == clip
        ]
        mine = [
            seed_mod.misfit(h, s)
            for s, v, at in rows
            if (h := camera_mod.matrix(rig, at, v)) is not None
        ]
        own = [
            seed_mod.misfit(h, s)
            for s, _v, _at in rows
            if (h := seed_mod.homography(s)) is not None
        ]
        typer.echo(
            f"{INDENT}{clip}: {len(rows)} views, {np.median(mine):.2f} m"
            f" (their own fits {np.median(own):.2f} m)"
            if mine and own
            else f"{INDENT}{clip}: {len(rows)} views"
        )

    # Each clip -- or from one clip, each view -- left out in turn. A position that moves
    # metres when one goes was never fitted from the others; it was that one's own trade
    # between height and zoom.
    groups = clips if len(clips) > 1 else [str(i) for i in range(len(views_of))]
    moved = []
    for leave in groups:
        kept = [
            i
            for i, (c, _s, _at) in enumerate(views_of)
            if (c if len(clips) > 1 else str(i)) != leave
        ]
        if len(kept) < 2:
            continue
        without = camera_mod.fit(
            [(views_of[i][1], views_of[i][2]) for i in kept],
            pitch=pitch,
            game=named,
            start=(rig.x, rig.y, rig.height),
            aims=[views[i] for i in kept],
        )
        if without is not None:
            other = without[0]
            moved.append(math.dist((rig.x, rig.y, rig.height), (other.x, other.y, other.height)))
    if moved:
        what = "clip" if len(clips) > 1 else "view"
        typer.echo(f"{INDENT}leaving any one {what} out moves it by at most {max(moved):.1f} m")

    path = camera_mod.write(game_dir(named) / camera_mod.FILE, rig)
    for clip in clips:
        _name_game(clip, named)
    typer.echo(f"wrote {path}")

    if overlay:
        drawn: set[str] = set()
        for (clip, s, at), view in zip(views_of, views, strict=True):
            if clip in drawn:
                continue
            img = video_mod.read_frame(CLIPS / clip / "img1", s.frame)
            h = camera_mod.matrix(rig, at, view)
            if img is None or h is None:
                continue
            dest = work_dir(Path(clip)) / f"camera.f{s.frame}.png"
            cv2.imwrite(str(dest), overlay_mod.draw(img, h))
            drawn.add(clip)
            typer.echo(f"{INDENT}wrote {dest}")


def name_of(frame: int, check: bool) -> str:
    """What a seed for this frame is called: the primary, or an extra anchor."""
    return f"seed.{frame}.json" if check else "seed.json"


def _carried(clip: str, frame: int) -> Any:
    """The camera this clip's other seeds carry to `frame`, or None when there are none.

    What a fit through traced curves starts from when `ft seed` runs on its own; `ft run`
    already has the chain and passes its camera straight in.
    """
    frames_dir = CLIPS / clip / "img1"
    work = work_dir(Path(clip))
    usable, _refused = auto_mod.usable_seeds(work, frames_dir)
    if not usable:
        return None
    numbers = sorted(int(p.stem) for p in frames_dir.glob("*.jpg"))
    chain = auto_mod.chain_from_seeds(
        usable, numbers, frames_dir, max_carry=None, motions=_motions(frames_dir, numbers, work)
    )
    return chain.homographies.get(frame)


def _seed_one(
    clip: str,
    frame: int,
    *,
    check: bool,
    message: str | None = None,
    say: Callable[[str], None] = typer.echo,
    start: Any = None,
    window: str | None = None,
    rig: camera_mod.Camera | None = None,
    lens_at: tuple[float, float] | None = None,
) -> tuple[Path, list[str]] | None:
    """Click one frame and write it: where it went, and what `settle` had to say about it.

    None if it was abandoned. Shared by `ft seed` and `ft run` so the checks a seed cannot
    make for itself run identically whichever way a person got here: the two mirror
    symmetries, which the reprojection overlay is blind to, and the contradictory labels,
    which no fit can absorb (D88).

    A seed with traced curves is stamped with the camera its fit starts from -- `start`
    when the caller has one, otherwise wherever the other seeds carry this frame. `window`
    is a window already open to click in, which `ft run` passes so the session keeps one.
    `start` also picks which region's markings the click tool offers first.
    """
    from . import seedui

    root = CLIPS / clip
    img = video_mod.read_frame(root / "img1", frame)
    if img is None:
        raise typer.BadParameter(f"no frame {frame} in {root / 'img1'}")

    work = work_dir(Path(clip))
    pitch = read_pitch(work)
    got = seedui.collect(img, frame, pitch, message=message, window=window, camera=start)
    if got is None:
        say("abandoned; nothing written")
        return None
    if got.arcs:
        got = replace(got, start=start if start is not None else _carried(clip, frame))

    mine = name_of(frame, check)
    notes: list[str] = []
    if rig is not None and lens_at is not None:
        # The symmetry checks `settle` makes are the camera's now. A pitch is symmetric and
        # a seed cannot see that in itself (D88), but a camera in a known place can: clicked
        # on the wrong goal, no aim of it fits, which `aimed_seeds` reports as a plain miss.
        # The stamp still matters -- a seed must never anchor another clip's frame (D34).
        got = replace(got, image=seed_mod.fingerprint(img))
    else:
        others = [p for p in auto_mod.seed_paths(work) if p.name != mine]
        got, notes = seed_mod.settle(got, img, others, root / "img1", pitch=pitch)
    for note in notes:
        say(note)

    path = seed_mod.write(work / mine, got)
    aimed = (
        camera_mod.aim_from_seed(rig, lens_at, got)
        if rig is not None and lens_at is not None
        else None
    )
    fitted = (
        f"the camera aimed to {aimed[1]:.2f} m"
        if aimed is not None
        else "NO aim of the camera"
        if rig is not None
        else "a homography"
        if seed_mod.homography(got) is not None
        else "NO homography"
    )
    say(
        f"{len(got.points)} points + {len(got.lines)} traced"
        f"{f' + {len(got.arcs)} on curves' if got.arcs else ''} -> {fitted}"
    )
    say(f"wrote {path}")
    return path, notes


@app.command()
def seed(
    clip: Annotated[str, typer.Argument(help="A clip in data/clips/.")],
    frame: Annotated[int, typer.Option(help="Which frame to seed.")] = 1,
    check: Annotated[
        bool,
        typer.Option(
            help="Write seed.<frame>.json instead of replacing seed.json. The pipeline reads"
            " those as EXTRA ANCHORS, so this is how a clip that pans gets a second one"
            " (D34); it is also what `ft calibrate` scores a carried fit against."
        ),
    ] = False,
) -> None:
    """Click pitch landmarks on one frame, to seed the camera model.

    The only human step in the automatic path. Click a landmark, and the name shown is
    the one it is recorded as - so click them in the order listed, or press n/p to
    choose. Four is the minimum and is exactly determined, which means it fits whatever
    was misclicked and cannot be checked (D17); six or more is much safer.
    """
    if _seed_one(clip, frame, check=check) is None:
        raise typer.Exit(1)
    typer.echo(f"now check it: ft calibrate {clip} --frame {frame}")


if __name__ == "__main__":
    app()
