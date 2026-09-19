# Decisions — numbers

Stage 5 — shirt numbers. Attempted and abandoned.

Every decision here was measured before it was made. They are kept because the code
cites them by number, and because the failures are worth as much as the successes.

**D5 — shirt numbers are voted per track, never read per frame.** See stage 5. The
consequence that matters: an unresolved number is `null` and imports as a generic token. Never
guess. A wrong number silently attaches a run to the wrong player, which is worse than no name
at all, and the coach cannot see that it happened.

**D32 — shirt-number OCR does not work on this footage, and the failures are confident.**
Measured on SNGS-147 with easyocr over the largest thirty sightings of each track, voting
by summed confidence exactly as this plan proposed:

    #20 -> 1   WRONG        #9  -> 9   RIGHT
    #23 -> 0   WRONG        #44 -> no answer
    #27 -> 1   WRONG        #4  -> no answer
    #50 -> 0   WRONG        #5  -> no answer
                            #14 -> no answer

One right, four wrong, four silent. Per crop: 3 right against 12 wrong.

The failure is not noise, which voting would survive. It reads ONE DIGIT out of a
two-digit number and is sure about it: `#20` scored `1` at 0.97 against `2` at 0.44. A
margin test does not save that, because the wrong answer wins by a mile.

So the number would be wrong four times as often as right, and D5 says plainly which way
that trade goes: an unread number imports as a generic token and costs nothing, a wrong
one attaches a run to the wrong player where nobody downstream can see it. Numbers stay
null.

Two things would have to change before this is worth revisiting: a detector-side crop
that finds the NUMBER rather than the middle of a torso, and a recogniser trained on
jersey digits rather than on printed text. Both are real projects. Trying a different
general-purpose OCR is not.

**D102 — the number IS on the shirt and readable; what D32 lacked was a reader trained to read
it, and the one built here names 8% of tracks and gets none of them wrong.** D100 ended on "what
is left is a readable number", so the footage was looked at first: sampled across one SNGS-116
track, at 142 px of player, ten of sixteen crops show a clear `93`. Numbers are legible on a
tactical camera for about half the frames of a player that size -- what they are not is legible
in a GIVEN frame, which is what D32 asked of easyocr and D5 already forbids.

**Trained on SoccerNet alone it learns the league, not the digits.** Torso crops (18-52% of the
box) from eighteen non-benchmark clips, labelled with the track's number on every frame whether
the number faces the camera or not, two heads for tens and units:

    trained on                              tracks named right (validation / benchmark)
    18 clips                                        9%  /  7%
    18 clips, tall crops, small-loss picked         3%  /  5%

Eighteen clips hold about fifty distinct numbers over a few hundred tracks, so the network learns
which numbers that league wears. The small-loss trick -- keep the batch's best-fitting 40%, the
usual answer to noisy labels -- made it worse, because the crops it keeps are the ones it has
already memorised.

**Drawn numbers fix the shortage.** Pretraining on 120,000 synthetic numbers -- a digit pair on a
shirt-coloured field, foreshortened, blurred, part-occluded, downscaled to 26-64 px and back --
costs nothing and teaches the digits themselves. Zero-shot, having never seen a real shirt, it is
45% right on the 6% of benchmark tracks it is confident about. Fine-tuned on the same eighteen
clips:

    asserted above   tracks named   of them right
    nothing                 100%         29.7%
    0.2                      22%         76.9%
    0.3                       8%        100.0%

**What that is and is not worth.** At 0.3 it names fourteen of 175 benchmark tracks and gets every
one right, which is a shirt number on a board under D5's rule. It does NOT touch fragmentation:
joining a player's pieces needs both pieces named, and 8% each way is nothing. For that the
coverage has to be several times higher, which means more real clips than three matches, a bigger
input than 64 px, and a legibility signal the annotation does not carry. Not integrated.
