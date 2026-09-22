# The pipeline

How a broadcast clip becomes `tracks.json`. Each stage reads the previous stage's artefact
from `work/<clip>/` and writes its own, so any stage can be rerun without the ones before it.

The reasoning behind each choice is in [`decisions/`](./decisions); this describes what runs.

## The three invariants

1. **`tracks.json` is the only contract.** Nothing downstream of it touches video; nothing
   upstream of it knows about Pitchboard's schema. The seam is deliberate — the video half is
   the part most likely to be thrown away and rewritten, and it must be replaceable without
   the TypeScript half noticing.

2. **No pixels cross the boundary.** Positions in `tracks.json` are pitch metres, origin at
   the top-left corner of a 105 × 68 pitch — the same space and the same origin Pitchboard's
   `BoardDoc` uses. Image coordinates stay inside the stage that produced them.

3. **Every stage writes an artefact, and every stage has a picture.** A homography cannot be
   unit-tested; you look at it. Each stage drops its output in `work/<clip>/` and can render
   an overlay or a top-down video proving it did what it claims. A stage with no visual check
   is a stage you cannot debug.

## The pipeline

Each stage reads the previous stage's artefact from `work/<clip>/` and writes its own. They
run independently, so a failing stage 3 is re-run without redoing stage 1.

### Stage 0 — segment `segments.json`

Broadcast cuts constantly: replays, close-ups, crowd, dugout. Everything downstream assumes
one continuous view of the pitch from one camera, so the first job is to find that view and
throw away the rest.

Shot-boundary detection, then score each segment on duration, pitch-green fraction and frame-
to-frame motion. The main tactical camera is long, very green, and moves smoothly.

- *Check:* the reported segment boundaries match where the cuts are.
- *Difficulty:* low. This will not be what fails.

### Stage 1 — registration — **one camera per match** (D96)

Per-frame 3 × 3 matrix mapping image pixels to pitch metres. Broadcast pans and zooms, so this
is a matrix per frame -- but a broadcast camera does not MOVE: it stands on a gantry and turns.
So a whole match is registered as one camera position plus three numbers a frame, pan, tilt and
zoom, and that is what ships.

- **The first clip of a match is seeded.** `ft seed` asks for landmarks clicked on a few frames
  (four per frame at least, because two cannot tell one goal from the other, D88). Those fits
  are carried across the frames between them by tracking the grass (D18), which covers the clip
  and drifts, and they fit where the match's camera stands: `ft camera`.
- **Every later clip is read, not clicked.** A learned segmenter (`calib.py`, D36) names every
  pitch marking in the frame; `camera.aim` turns them into that frame's pan, tilt and zoom.
  Frames with nothing readable are spanned between the aims either side, and only where a
  stretch is blind does `ft run` ask for a click. Three numbers cannot fold a pitch or drift out
  of shape, which is what beat every free per-frame fit (D67, D68, both removed).

On the eleven benchmark clips, each aimed by a camera fitted from its match's OTHER clips,
97.7% of players land within 2 m of where SoccerNet puts them, against 72.7% from one seed.

- *Check:* **reproject the pitch model back onto the video** (`ft calibrate --frame N`, or
  `--video`). Lines land on lines or they do not. It is what caught D16, D34 and D88.
- *Difficulty:* the highest of any stage, and now the most finished.

### Stage 2 — detect and track `detections.json`

Person detector per frame, then a tracker to string detections into tracks with stable ids.

RT-DETR finds 94-98% of visible players (D28); the tiled pass that also finds the ball is
most of a clip's compute. Association runs in STABILISED pixels, so a wobble in stage 1 cannot
break a track (D22), and refuses kits that plainly disagree (D78).

**Identity is where the loss is now.** A track breaks where players touch and where the camera
looks away, and a stitcher joins the pieces afterwards: a fragment's best continuation that is
also chosen back (D53, D69), across a contact only where both ends look like one man (D94, D95),
and two tracks taking turns on one man are merged (D90). What is left is 26 points: a player's
tracks together hold 85% of his time on screen and the best one 59% (D97).

- *Check:* boxes and ids overlaid on the video, and `ft score`'s best-track coverage.
- *Difficulty:* the remaining risk is concentrated here.

### Stage 2c — appearance `appearance.npz`

`ft reid` embeds every detection with OSNet-AIN, a person re-identification network, into a
512-d vector where one person lands near himself and far from everybody else. The tracker never
reads it. The stitcher does, at one decision only: a join across a box that holds two men gets
`CONTACT_SLACK_M` more position gate if, and only if, the clean crops at the two ends look like
one man (`LOOK_APART`, D95). Without the file the pipeline is exactly what it was.

The network is vendored (`osnet.py`, MIT); the weights download on first use, are checked
against a pinned SHA-256 before every load and loaded `weights_only`, and are never committed.

- *Check:* the clean crops each end of a join is judged on, side by side (D95).
- *Difficulty:* the model is borrowed. The risk is team-mates, who share everything but the man.

### Stage 3 — teams `teams.json`

Split the tracks into two kits on the axis of greatest variance, not by k-means (D31), with
keepers and officials held out of the split (D64). A side the kit does not settle is `unknown`
rather than a coin flip (D72), except for a track the ball went through (D91). Within a clip
`home` is the side defending the left; across a match it is a KIT, remembered in
`work/games/<match>/kits.json`, so half time does not swap the names (D99).

- *Check:* crops grouped by assigned side, in a contact sheet, and the board's colours.
- *Difficulty:* low, except for a kit struck with the grass (D92).

### Stage 4 — project `tracks.json`

Apply stage 1's per-frame homography to stage 2's tracks. Positions become metres. A position
off the pitch is dropped, never clamped (D13), and anyone standing off it was dropped before
tracking (D27). Last, each position is averaged with the samples within 0.12 s of it, which
removes the camera's and the detector's per-frame wobble without filling any gap (D103).

**This is the proof.** Render the result as a top-down video of coloured dots. If the dots
move like a football team, the hard part is done and everything after is engineering. If they
jitter, swim or cross the touchline, the fault is upstream in stage 1 or 2 and this stage is
how you find out which.

- *Check:* the top-down dot video.
- *Difficulty:* low in itself; it is the integration test for everything before it.

### Stage 5 — numbers — **a reader exists; it is not wired in** (D32, D102)

Best-effort shirt numbers, voted per track and never read per frame (D5): a `null` number is an
answer, and a guessed one attaches a run to the wrong player where nothing downstream can see it.

A general-purpose OCR fails on this footage and fails confidently (D32). The numbers are there
-- across one track at broadcast size, ten crops in sixteen show one clearly -- and a small
reader pretrained on drawn numbers and fine-tuned on SoccerNet names 8% of benchmark tracks with
none wrong (D102). That is a board name, and too few to join fragments with. More real data is
[PLAN.md](../PLAN.md)'s path 3.

- *Check:* resolved number against the crops that voted for it.
- *Difficulty:* medium; the failure mode has to stay silence rather than a wrong answer.

### Not a stage — the ball

Found for one question, who has it (D29). The tiled detector's candidates (D57), the most
confident one a frame and none below 0.75 (D73), and drawn straight across the gaps of up to two
seconds between the sightings it keeps (D101). Its position is not a position: a ball in the air
projects metres from where it is (D66).
