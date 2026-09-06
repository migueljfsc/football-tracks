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
