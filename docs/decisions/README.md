# Decisions — the index

Every decision in this repo, by what became of it. The full account of each is in the file
named beside it; source comments cite them by number, so `grep -rn "D96" src docs` finds
both the reasoning and the code it shaped.

**Pitchboard numbers its decisions separately.** The two logs collide from D1 up, so a comment
citing the other repo always names it -- *"Pitchboard's D52"*. D37-D52 and D65 are unused here.

## Standing rules

The contract, the licences, and how anything here is measured. These are not up for re-deciding without new evidence.

| | decision | file |
|---|---|---|
| D1 | sibling repo, not `pitchboard/tools/` | [project](project.md) |
| D2 | Python pinned to 3.12 | [project](project.md) |
| D3 | `tracks.json` is the contract, and it is frozen before the CV works | [project](project.md) |
| D5 | shirt numbers are voted per track, never read per frame | [numbers](numbers.md) |
| D6 | SoccerNet first | [project](project.md) |
| D8 | track samples are sparse | [tracking](tracking.md) |
| D9 | Ultralytics YOLO is AGPL-3.0 | [project](project.md) |
| D10 | SoccerNet enters at stage 1 | [project](project.md) |
| D14 | ground truth is `truth.json`, a prediction is `tracks.json` | [project](project.md) |
| D15 | the scorecard separates the failures the averages hide | [project](project.md) |
| D19 | stage 1's accuracy is not what stage 2 needs | [registration](registration.md) |
| D22 | stage 2 does not depend on stage 1 | [tracking](tracking.md) |
| D33 | a carry can only be scored against evidence it did not produce, and on a broadcast clip that evidence has to be clicked | [registration](registration.md) |
| D70 | a camera model is judged where the PLAYERS are, and on two clips the ground truth cannot judge it at all | [registration](registration.md) |

## What ships — registration

| | decision | file |
|---|---|---|
| D7 | automatic registration first, human-seeded as the fallback *(automatic now means the match camera (D96); the first clip of a match is still seeded)* | [registration](registration.md) |
| D11 | a clip is fetched by range request, never a split download | [project](project.md) |
| D12 | ground truth is written through the same writer as the CV path | [project](project.md) |
| D13 | an off-pitch position is dropped, never clamped | [tracking](tracking.md) |
| D16 | the homography is fitted from POINT-ON-LINE constraints, not from line intersections | [registration](registration.md) |
| D17 | a frame must show MORE lines than the fit strictly needs | [registration](registration.md) |
| D18 | a homography is carried across gaps by tracking the ground plane | [registration](registration.md) |
| D23 | the seed is a file, not a UI | [registration](registration.md) |
| D24 | four clicked landmarks are not enough, and the obvious four are degenerate | [registration](registration.md) |
| D25 | a seed propagates in BOTH directions | [registration](registration.md) |
| D26 | a human can TRACE a line as well as click a point | [registration](registration.md) |
| D34 | a seed is checked against the frame it claims to describe, not against its own clicks; and one seed does not cross a broadcast clip | [registration](registration.md) |
| D36 | the camera model is learned, because the missing thing was never the paint but the NAME of it *(the segmenter's lines now feed the match camera rather than a free fit)* | [registration](registration.md) |
| D55 | seed mode was seeding from the FIRST solvable frame, and the first frame is the worst one | [registration](registration.md) |
| D80 | a second anchor only helped the frames after it, because the chain walked one way | [registration](registration.md) |
| D82 | a seed is stamped with the picture it was clicked on, because a frame number is not an identity | [registration](registration.md) |
| D83 | the chain now says where it is weakest, because that is the only thing a coach can act on | [registration](registration.md) |
| D86 | the chain broke for want of corners, not for want of grass | [registration](registration.md) |
| D88 | a pitch is symmetric END TO END as well as side to side, and only one of those was guarded | [registration](registration.md) |
| D89 | a pitch is 105 x 68 only in elite competition, and the seed is written in metres | [registration](registration.md) |
| D96 | one camera for a whole match, three numbers a frame | [registration](registration.md) |

## What ships — detection and tracking

| | decision | file |
|---|---|---|
| D20 | the tracker is ours | [tracking](tracking.md) |
| D27 | anyone standing off the pitch is dropped BEFORE tracking, not after | [tracking](tracking.md) |
| D28 | the detector is RT-DETR, and the first one was deliberately a floor | [tracking](tracking.md) |
| D30 | a track that has lost its player gives up quickly | [tracking](tracking.md) |
| D53 | stage 2 fragments every player, and joining the pieces afterwards is safe where lengthening the tracker's memory was not | [tracking](tracking.md) |
| D56 | the tracker smoothed the kit colour and not the velocity, and velocity was doing the harder job | [tracking](tracking.md) |
| D69 | the stitcher was refusing most of what there is to join, and reach was the wrong gate | [tracking](tracking.md) |
| D77 | a velocity read over a fixed number of SAMPLES measures the frame rate | [tracking](tracking.md) |
| D78 | a colour WEIGHT is a preference, and a preference loses when the right player is missing | [tracking](tracking.md) |
| D79 | the stitcher makes the same claim across a longer gap, so it needs the same veto | [tracking](tracking.md) |
| D90 | two tracks in the same place at the same time are one player, and the detector says which | [tracking](tracking.md) |
| D94 | a track breaks where players touch, and of four ways to rejoin it three ship | [tracking](tracking.md) |
| D95 | which of two men a merged box belongs to is a question about how they look, and answering it exposed a fault in every join | [tracking](tracking.md) |
| D103 | a position is measured twice over and wobbles both times, and averaging the samples that exist settles it without moving anybody | [tracking](tracking.md) |

## What ships — teams

| | decision | file |
|---|---|---|
| D21 | which cluster is "home" is decided by which end the side plays at *(within a clip; across a match the kit decides (D99))* | [teams](teams.md) |
| D31 | the two kits are told apart on their axis of greatest variance, not by k-means | [teams](teams.md) |
| D63 | a labelling that fields nineteen players on one side is wrong whatever the kit colours say, and a pitch knows it with no ground truth at all | [teams](teams.md) |
| D64 | an official is an odd kit that is not standing in a goal, and until 6 September one reached nearly every board | [teams](teams.md) |
| D72 | a side the kit split is not sure of is declined, because a wrong colour is a pass that never happened | [teams](teams.md) |
| D81 | the board wears the kits, because a coach reads his own clip in them | [teams](teams.md) |
| D85 | a DECLINED track may be cut where its shirt changes, and only a declined one | [teams](teams.md) |
| D87 | a shirt answers three questions, and only one of them is about hue | [teams](teams.md) |
| D91 | the margin is silence, and silence costs more for the man on the ball | [teams](teams.md) |
| D92 | the kit is painted from the signature that NAMES the sides, and refused where that signature will not settle | [teams](teams.md) |
| D99 | across the clips of one match, `home` is a kit, not an end | [teams](teams.md) |

## What ships — the ball

| | decision | file |
|---|---|---|
| D29 | the ball is found for ONE question: who has it | [ball](ball.md) |
| D54 | the detector calls the penalty spot a ball, and the board believed it | [ball](ball.md) |
| D57 | the ball was three pixels wide, and everything else about the ball was downstream of that | [ball](ball.md) |
| D58 | the ball's margin was a share of the pitch and was read as metres, so the gate was 157 m wide | [ball](ball.md) |
| D59 | a set piece is the one moment the ball's position is known before it is seen, and that is worth a rule of its own | [ball](ball.md) |
| D60 | a stationary false positive is only distinguishable from a placed ball by WHERE it is standing | [ball](ball.md) |
| D73 | the ball is asserted at 0.75, and the reason is the PASSES rather than the ball | [ball](ball.md) |

## What ships — learning from the coach

| | decision | file |
|---|---|---|
| D104 | a coach's corrections come back as labels, through the board he already exports | [project](project.md) |

## Superseded

Right when written; replaced by the decision named.

| | decision | file |
|---|---|---|
| D4 | no ball in v0 (superseded by D29) *(by D29)* | [ball](ball.md) |
| D62 | the camera model is the biggest lever in the pipeline, and the segmenter's problem is neither coverage nor accuracy but a tail of confidently wrong fits *(by D96)* | [registration](registration.md) |
| D74 | more seeds do not fix the camera; only stronger ones help, and only a little *(by D96)* | [registration](registration.md) |
| D76 | two tracks alive at once are two players, not one player twice *(refined by D90, which finds the one case where it is wrong)* | [tracking](tracking.md) |

## Measured and not built

Each of these was built or prototyped, measured, and refused. They are the cheapest thing in the repo to read and the most expensive to rediscover.

| | decision | file |
|---|---|---|
| D32 | shirt-number OCR does not work on this footage, and the failures are confident *(a general OCR; D102 is the trained reader that replaced the question)* | [numbers](numbers.md) |
| D35 | snapping the camera model onto the painted lines improves the camera model and makes the tracks worse. Off by default *(removed from the code)* | [registration](registration.md) |
| D61 | the kit signature separates the two teams almost perfectly, and gating on it still does not fix the id switch. Written, measured, reverted | [teams](teams.md) |
| D66 | the ball has three separate faults, and the camera model is not one of them | [ball](ball.md) |
| D67 | the two registrations fail in opposite directions, and the board needs both halves *(the segmenter-only mode, removed from the code)* | [registration](registration.md) |
| D68 | the two registrations were joined, and the join makes no board better *(the hybrid mode, removed from the code)* | [registration](registration.md) |
| D71 | the segmenter's coverage cannot be trained, because the frames it refuses carry no paint to read | [registration](registration.md) |
| D75 | the ball's recall cannot be bought by believing weaker candidates, and the ceiling is the detector | [ball](ball.md) |
| D84 | two ways to catch a track that changes shirt, both measured, both refused | [teams](teams.md) |
| D93 | the ball cannot be recovered from this detector by any selection rule, and nearness to a player is evidence AGAINST it | [tracking](tracking.md) |
| D97 | with the camera fixed, a player's missing half is in pieces; appearance cannot say which pieces are his, and a veto built on it breaks the tackles it was meant to protect | [tracking](tracking.md) |
| D98 | a re-id network trained on football tells team-mates apart on clean crops and barely at the ends of tracks, which is where the stitcher asks | [tracking](tracking.md) |
| D100 | tagging a coach's own players names them within a match about as well as the stitcher joins them, and tags from earlier matches do not carry to the next | [tracking](tracking.md) |
| D101 | the ball costs a board twelve points of possession, the candidates to recover them are already found, and a verifier trained on three matches does not pick them *(the verifier; the gap-bridging it measured ships)* | [ball](ball.md) |
| D102 | the number IS on the shirt and readable; what D32 lacked was a reader trained to read it, and the one built here names 8% of tracks and gets none of them wrong *(8% of tracks named, all right; not integrated)* | [numbers](numbers.md) |
