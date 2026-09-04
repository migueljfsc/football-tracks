"""Command line entry point. One command per stage."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, cast

import cv2
import typer

from . import auto as auto_mod
from . import calibration, soccernet, stage0_segment, stage1_propagate, stage1_register, tracks
from . import detect as detect_mod
from . import overlay as overlay_mod
from . import refine as refine_mod
from . import render as render_mod
from . import score as score_mod
from . import seed as seed_mod
from . import video as video_mod
from .config import CALIB_DATA, CLIPS, work_dir

app = typer.Typer(add_completion=False, help="Broadcast clip -> player tracks in pitch metres.")


def _size(root: Path, labels: dict[str, Any] | None) -> tuple[int, int]:
    """Frame size, from SoccerNet's labels or from clip.json."""
    if labels is not None:
        return (int(labels["images"][0]["width"]), int(labels["images"][0]["height"]))
    clip = video_mod.load(root)
    return (clip.width, clip.height)


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
        width=labels["images"][0]["width"],
        height=labels["images"][0]["height"],
        interval_s=interval_s,
    )

    named = sum(1 for t in built if t.number is not None)
    total_samples = sum(len(t.samples) for t in built)
    typer.echo(f"{len(built)} tracks, {total_samples} samples, {named} with a shirt number")
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


@app.command("calib-eval")
def calib_eval(
    clip: Annotated[str, typer.Argument(help="A clip with ground-truth pitch lines.")],
    weights: Annotated[Path | None, typer.Option(help="Trained segmenter.")] = None,
    stride: Annotated[int, typer.Option(help="Score every Nth frame.")] = 10,
) -> None:
    """Score a per-frame fit from the segmenter against the annotated one.

    The question the whole idea turns on: can a homography be fitted from the picture
    alone, with no seed and nothing carried? A good human seed is 0.15-0.3 m, so anything
    worse than about half a metre is not worth replacing seeding with.
    """
    import numpy as np

    from . import calib

    c = soccernet.Clip(name=clip, root=CLIPS / clip)
    if not c.labels_path.exists():
        raise typer.BadParameter(f"{clip} has no ground-truth lines to score against")
    truth = stage1_register.fit_all(c.labels())
    net = calib.model(weights or calib.WEIGHTS).to(calib.device()).eval()

    errors: list[float] = []
    attempted = solved = 0
    for f in sorted(truth)[::stride]:
        want = truth[f]
        if want is None:
            continue
        img = cv2.imread(str(c.frames_dir / f"{f:06d}.jpg"))
        if img is None:
            continue
        attempted += 1
        got = calib.fit_from_mask(calib.predict(net, img), img.shape[1], img.shape[0])
        if got is None:
            continue
        solved += 1
        errors.append(calibration.observed_error(want, got, img.shape))

    if not errors:
        typer.echo(f"{clip}: solved 0 of {attempted} frames")
        return
    e = np.array(errors)
    typer.echo(
        f"{clip}: solved {solved}/{attempted} ({solved / attempted:.0%})"
        f"  median {np.median(e):.2f} m  p90 {np.percentile(e, 90):.2f} m"
        f"  worst {e.max():.2f} m"
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
    snap: Annotated[bool, typer.Option(help="Re-anchor on the markings (D35).")] = False,
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
            _pipeline(name, "seed", -1, interval_s, snap)
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

    if c.labels_path.exists():
        # SoccerNet: every frame carries its own pitch lines.
        labels = c.labels()
        homs = stage1_register.fit_all(labels)
        direct = {f: h for f, h in homs.items() if h is not None}
    elif seed_path.exists():
        # A clip nobody annotated: every clicked frame, carried both ways between them.
        clicked, refused = auto_mod.usable_seeds(out, c.frames_dir)
        for path, behind in refused:
            typer.echo(
                f"IGNORING {path.name}: it maps {behind:.0%} of its frame behind the"
                " camera, so it is wrong AT the anchor and not merely far from it."
                " Click evidence lower in the frame - a fit needs depth, not just points."
            )
        if not clicked:
            raise typer.BadParameter(f"{clip} has no usable seed")
        homs = auto_mod.from_seeds(
            clicked,
            frames_all,
            c.frames_dir,
            max_carry=None,
            motions=motions,
        )
        direct = {s.frame: seed_mod.homography(s) for s in clicked}
    else:
        raise typer.BadParameter(
            f"{clip} has neither SoccerNet labels nor {seed_path} - run `ft seed {clip}` first"
        )

    chain = None
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
        typer.echo("no ground truth here - check the overlay with --frame or --video")
        return
    typer.echo(stage1_register.report(stage1_register.evaluate(labels, homs)))


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
    c = soccernet.Clip(name=clip, root=CLIPS / clip)
    frames = sorted(int(p.stem) for p in c.frames_dir.glob("*.jpg"))
    if not frames:
        raise typer.BadParameter(f"no frames in {c.frames_dir} - run `ft fetch {clip}` first")

    out = work_dir(Path(clip))
    with typer.progressbar(frames, label="detecting") as bar:
        found, balls = detect_mod.run(
            c.frames_dir, frames, conf=conf, progress=lambda _f: bar.update(1)
        )
    path = detect_mod.write(out / "detections.json", found, balls, conf=conf)
    typer.echo(
        f"{len(found)} people over {len(frames)} frames"
        f" ({len(found) / max(1, len(frames)):.1f}/frame),"
        f" {len(balls)} ball sightings"
    )
    typer.echo(f"wrote {path}")


@app.command()
def _pipeline(
    clip: str,
    mode: str,
    carry: int,
    interval_s: float,
    snap: bool,
    max_residual: float = 0.0,
    stitch: bool = True,
) -> tuple[Path, auto_mod.Result]:
    """Frames in, tracks.json out. Shared by `ft auto` and `ft bench`.

    Extracted so the benchmark runs the SAME pipeline the user runs, rather than a
    second copy of it that can drift away from it silently.
    """
    if mode not in ("truth", "seed", "segmenter"):
        raise typer.BadParameter("mode must be 'truth', 'seed' or 'segmenter'")
    picked: auto_mod.Mode = cast("auto_mod.Mode", mode)

    c = soccernet.Clip(name=clip, root=CLIPS / clip)
    out = work_dir(Path(clip))
    dets_path = out / "detections.json"
    if not dets_path.exists():
        raise typer.BadParameter(f"no {dets_path} - run `ft detect {clip}` first")

    frames = sorted(int(p.stem) for p in c.frames_dir.glob("*.jpg"))
    detections, balls = detect_mod.read(dets_path)
    motions = stage1_propagate.motions(c.frames_dir, frames, cache=out / "motions.json")

    labels: dict[str, Any] | None = None
    seed_path = out / "seed.json"
    if c.labels_path.exists():
        labels = c.labels()
        homs = auto_mod.homographies(
            labels,
            c.frames_dir,
            picked,
            max_carry=None if carry < 0 else carry,
            motions=motions,
            snap=refine_mod.refine if snap else None,
            max_residual_m=max_residual or None,
        )
    elif seed_path.exists():
        # A real clip: one seeded frame is all the camera information there is.
        usable, refused = auto_mod.usable_seeds(out, c.frames_dir)
        for path, behind in refused:
            typer.echo(f"IGNORING {path.name}: maps {behind:.0%} of its frame behind the camera")
        if not usable:
            raise typer.BadParameter(f"{clip} has no usable seed")
        homs = auto_mod.from_seeds(
            usable,
            frames,
            c.frames_dir,
            max_carry=None if carry < 0 else carry,
            motions=motions,
            snap=refine_mod.refine if snap else None,
        )
    else:
        raise typer.BadParameter(
            f"{clip} has neither SoccerNet labels nor {seed_path} - run `ft seed {clip}` first"
        )

    result = auto_mod.build(
        c.frames_dir,
        frames,
        detections,
        homs,
        fps=_fps(CLIPS / clip, labels),
        motions=motions,
        balls=balls,
        stitch=stitch,
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
        interval_s=interval_s,
    )
    return path, result


@app.command()
def auto(
    clip: Annotated[str, typer.Argument(help="A clip already fetched into data/clips/.")],
    mode: Annotated[
        str,
        typer.Option(
            help="'truth' uses every frame's lines; 'seed' uses only frame one's;"
            " 'segmenter' uses the learned ones and no annotations at all."
        ),
    ] = "seed",
    carry: Annotated[int, typer.Option(help="Frames a homography may be carried.")] = -1,
    interval_s: Annotated[
        float,
        typer.Option(
            "--interval-s",
            help="Seconds between stored positions. 0 writes every frame, which is finer"
            " than the pipeline is accurate.",
        ),
    ] = tracks.DEFAULT_INTERVAL_S,
    snap: Annotated[
        bool,
        typer.Option(
            help="Re-anchor each carried homography on its own frame's markings."
            " Improves the camera model and makes the tracks WORSE (D35); off by default."
        ),
    ] = False,
    max_residual: Annotated[
        float,
        typer.Option(
            help="Segmenter mode: refuse a frame whose fit disagrees with its own predicted"
            " pixels by more than this many metres. 0 keeps every solvable frame."
        ),
    ] = 0.0,
    stitch: Annotated[
        bool,
        typer.Option(help="Join track fragments that are each other's best continuation."),
    ] = True,
) -> None:
    """The automatic path end to end - frames in, tracks.json out.

    `--mode truth` holds stage 1 fixed so the score is stages 2 and 3 alone.
    `--mode seed` throws away every line annotation but frame one's, which is what a
    human clicking four corners once actually leaves you with.
    `--mode segmenter` uses no annotations at all: every frame is registered from the
    learned pitch lines, so there is nothing to seed and nothing to carry (D36).
    """
    path, result = _pipeline(clip, mode, carry, interval_s, snap, max_residual, stitch)
    typer.echo(
        f"mode {mode}: {result.detections} detections -> {result.raw_tracks} raw tracks"
        f" -> {len(result.tracks)} kept"
    )
    typer.echo(
        f"dropped off pitch {result.dropped_off_pitch}, frames with no homography"
        f" {result.unsolved_frames}, ball located on {len(result.ball)} frames"
    )
    typer.echo(f"wrote {path}")


@app.command()
def frames(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A video file.")],
    name: Annotated[str | None, typer.Option(help="Clip name. Defaults to the file stem.")] = None,
) -> None:
    """Turn a broadcast video into the numbered-JPEG layout the pipeline speaks.

    Detects and removes pillarbox and letterbox bars, and records the REAL frame rate
    rather than the one the container claims - a screen recording routinely lies.
    """
    dest = CLIPS / (name or source.stem)
    clip = video_mod.extract(source, dest)
    typer.echo(
        f"{clip.frames} frames at {clip.fps:.2f} fps, {clip.width}x{clip.height}"
        f" (cropped from {clip.crop})"
    )
    typer.echo(f"wrote {dest}")


@app.command()
def seed(
    clip: Annotated[str, typer.Argument(help="A clip in data/clips/.")],
    frame: Annotated[int, typer.Option(help="Which frame to seed.")] = 1,
    check: Annotated[
        bool,
        typer.Option(
            help="Write seed.<frame>.json as independent truth to score a carry against,"
            " rather than replacing the seed the pipeline runs from."
        ),
    ] = False,
) -> None:
    """Click pitch landmarks on one frame, to seed the camera model.

    The only human step in the automatic path. Click a landmark, and the name shown is
    the one it is recorded as - so click them in the order listed, or press n/p to
    choose. Four is the minimum and is exactly determined, which means it fits whatever
    was misclicked and cannot be checked (D17); six or more is much safer.
    """
    from . import seedui

    root = CLIPS / clip
    img = video_mod.read_frame(root / "img1", frame)
    if img is None:
        raise typer.BadParameter(f"no frame {frame} in {root / 'img1'}")

    got = seedui.collect(img, frame)
    if got is None:
        typer.echo("abandoned; nothing written")
        raise typer.Exit(1)

    # The reprojection overlay CANNOT see a mirrored y axis: a pitch is symmetric about
    # the halfway line, so a flipped model draws onto the real markings perfectly. So it
    # is checked here, arithmetically, before anything is written.
    agreement = seed_mod.orientation(got)
    if agreement < -seed_mod.ORIENTATION_CONFIDENT:
        typer.echo("far and near look swapped - flipping the pitch y axis to match the camera")
        got = seed_mod.flip_y(got)
    elif agreement < seed_mod.ORIENTATION_CONFIDENT:
        typer.echo(
            "WARNING: cannot tell which side the camera is on from these points."
            " If the board comes out mirrored, far and near are swapped."
        )

    directions = {abs(a) > abs(b) for _, (a, b, _c) in got.lines}
    if got.lines and len(directions) == 1:
        typer.echo(
            "NOTE: every traced line runs the same way. Lines parallel to each other pin"
            " down nothing across them - the exact points are carrying the fit."
        )

    name = f"seed.{frame}.json" if check else "seed.json"
    path = seed_mod.write(work_dir(Path(clip)) / name, got)
    h = seed_mod.homography(got)
    typer.echo(
        f"{len(got.points)} points + {len(got.lines)} traced"
        f" -> {'a homography' if h is not None else 'NO homography'}"
    )
    typer.echo(f"wrote {path}")
    typer.echo(f"now check it: ft calibrate {clip} --frame {frame}")


if __name__ == "__main__":
    app()
