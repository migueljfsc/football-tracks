# football-tracks — where this stands

Turn a broadcast football clip into player tracks in pitch metres, so Pitchboard can import a
play instead of the coach drawing it by hand.

**Target for v0 is 70%** — good enough that a coach corrects the board instead of drawing it
from nothing. A comparison, not an absolute. This is a proof, not a product.

This file is the current state and the next move, and nothing else. What the pipeline is lives
in [`docs/`](docs); why every piece of it is the way it is -- and every attempt that failed --
is in [the decisions](docs/decisions/README.md), indexed by what became of each one. The history
this file used to carry is there and in git.

## Where it stands — 22 September 2026

**The bar is met on a coach's own clips.** He read two Milan-Benfica boards scene by scene
against the footage and would correct both rather than draw them; he put one at about 90% of
the way there, and the dot videos *"read as football"*. One of the two was never clicked: the
match camera registered it from the other's seeds.

**What ships**, end to end on one command (`ft run <clip> --game <match>`):

- **Registration: one camera per match** (D96). The first clip of a match is seeded by clicking
  landmarks; that fits where the broadcast camera stands, and every later clip is read as three
  numbers a frame -- pan, tilt, zoom -- from the learned pitch lines, with no clicks at all.
- **Detection and tracking**: RT-DETR (D28), association in stabilised pixels (D22), a stitcher
  that joins fragments where appearance agrees (D94, D95), positions averaged over the samples
  around them (D103).
- **Teams**: the kit's axis of greatest variance (D31), declining a side it cannot settle (D72),
  and a match that remembers its kits so `home` is one team in every clip (D99).
- **The ball**: the most confident of the tiled detector's candidates (D57, D73), bridged across
  gaps of up to two seconds (D101) -- used for one question, who has it (D29).

On the eleven SoccerNet benchmark clips, each registered by a camera that never saw it:

    players within 2 m of the truth            97.7%     (D96)
    recall / precision                    82.9% / 95.9%  (D96)
    a player's best single track holds           59%     of his time on screen (D97)
    board shows the right side on the ball      67.2%    frame by frame (D103)

A clip takes about ten minutes on this laptop, most of it the tiled ball pass in detection.

## What limits it now

Two ceilings are measured, and both are identity rather than geometry.

**The ball's position is worth twelve points of possession** (D101). With the candidate nearest
the real ball chosen every frame, possession-right goes from 60.7% to 72.6%; the candidates
exist in 71-96% of frames and COCO's confidence ranks them badly. It is also every remaining
error on the coach's two boards: a carrier named a pass early, a heavy pass credited to a man
who never received it, the flight of a shot handed to players. A verifier trained on three
SoccerNet matches did not rank them better on a fourth.

**Fragmentation costs 26 points** (D97): a player's tracks together hold 85% of his time on
screen and the best one 59%. The biggest share is a player leaving the picture while the camera
looks elsewhere and coming back as a new track. Position cannot say which team-mate came back,
and no appearance model tried can either (D97, D98, D100). A readable shirt number can: a
reader pretrained on drawn numbers names 8% of tracks with none wrong (D102) -- right, and far
too few to join anything.

**What is not known**: whether the bar holds outside one match. Nottingham, from another match
and another broadcaster, is unreviewed, and no Sporting clip -- the team this is for -- has been
run. Sporting's green-and-white hoops on green grass are the kit the pipeline already refuses to
paint (D92).

## Paths forward

Ordered by what each would tell or buy against what it costs. None is started.

1. **Run it where it will be used.** Two or three Sporting clips, and the nottingham review
   sheet. An hour of the coach's time and some compute, and it is the only way to know whether
   90% is a property of the pipeline or of one match. Everything below is worth more or less
   depending on what it finds.

2. ~~**Draw an unseen player as unseen**~~ **-- done** (Pitchboard's D87). A player with no
   sighting near a scene is drawn faded, fades in as the play reaches him, and turns solid when
   the coach drags him.

3. **Shirt numbers from real data** (D102). Drawn numbers got a small reader from 9% to useful at
   the confident end; what it lacks is real legible crops at scale and a bigger input than 64 px.
   SoccerNet publishes a jersey-number set of labelled tracklets -- its access terms need
   checking first. A reader that names a third of tracks at D5's standard would join fragments
   with certainty where it names both, and give the board real names. The largest lever on
   fragmentation this repo has found.

4. ~~**Keep the coach's corrections.**~~ **-- the capture is done** (D104, Pitchboard's D88).
   A board from video remembers what its importer said, a player can be moved to the other side,
   and `ft learn <board.json>` turns what the coach changed into labels beside the clip:
   carriers, numbers, positions, sides. What is left is USING them -- in 3 and 5 -- once enough
   corrected boards exist to train on.

5. **The ball with more matches, or over time** (D101). The ranking a verifier learned on three
   SoccerNet matches did not transfer; a verifier trained across many, or scoring candidate
   TRACKLETS rather than single frames, is the untested half. Large and uncertain -- only after 4
   has produced data, or if SoccerNet's own larger sets turn out to be usable.

6. **A match at a time.** Several clips of one match through `ft run` in one command, and a way
   in Pitchboard to keep a match's boards together. Workflow, not accuracy; worth it once the
   coach is importing weekly.

**Not these** -- each is measured and closed: another detector (it finds 94-98% of visible
players); another segmenter run (D71); appearance models for fragmentation (D97, D98, D100);
the importer's carrier rules (the remaining invented carriers trace to the ball's position,
D101); a ball verifier on three matches (D101); snapping to painted lines (D35).

## Milestones

| # | done when | |
|---|---|---|
| M0 | scaffold and the ground-truth path: `ft truth`, `ft render`, `ft score` | **done** |
| M1 | reprojected pitch lines sit on the real lines | **done** -- the match camera, 97.7% of players within 2 m (D96) |
| M2 | tracks survive with few enough id switches to count | **partly** -- stitching ships; the best track holds 59% of a player (D97) |
| M3 | teams cluster cleanly | **done** (D63, D99) |
| M4 | the top-down dot video looks like football | **done** -- judged by a coach (D103) |
| M5 | numbers resolve for a useful share of tracks | **open** -- 8% at no errors (D102) |
| M6 | Pitchboard turns `tracks.json` into a board | **done** |
| M7 | a coach corrects rather than draws | **done on two clips of one match** -- see *What is not known* |

## Non-goals

Real-time. Multi-camera. Event detection (tackles, fouls). Ball height. Anything that needs a
GPU bigger than the laptop. A web service -- this runs locally, from a terminal, on files.

Player identity across clips stays a non-goal for opponents. For the coach's own side it is
what path 4 would buy, and D100 is why it has to come from his labels rather than from a model.

## Open questions

- Does the match camera hold on a lower broadcast camera? The coach's footage is broadcast replays
  and YouTube highlights only -- no club camera -- so pans stay, and so does fragmentation's
  largest share.
- How does Sporting's kit split? The side question reads hue, and theirs is struck with the
  grass (D87, D92).
- Can SoccerNet's jersey-number and tracking sets be used under the same terms as GSR?
