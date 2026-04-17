# Project: Timer

**Student:** Phil Cote

**Camera used:** Horizon

**One-line pitch:** A 60-min timer for `whiteboard` | `group work` | `focus time` using magnetic fiducial markers (arUco) and CV. The timer's behavior depends on what Horizon sees.

## What I tried

> *I wanted to create an intuitive, casual, and tactile timer experience for both teachers and students. I started by seeing if Horizon could see handwriting. I also tried using Teachable Machine image model training — first using hand gestures, then numbers. Next, I tried Hand Detection, which worked better than the model training, but still difficult to think of how it would be sensed by the classroom consistently. Finally, I arrived at arUco tags attached to magnetically reinforced foam core, which were cheap and easy to produce.*

## What worked

• The arUco tags where the most responsive of all the interactions I tested — code is open-source so it was easy for Claude to find it when errors came up.

• The hand detection iteration technically worked, however it felt too limiting for the purpose of the timer.

• The hand written commands were successful in capturing the feel of a non-screen interaction.

## What broke

• Image model training did not work as well as I'd hoped — I learned that it's better for predicting specific objects, not interpreting commands.

• I have yet to resolve the positioning and mounting of the camera in a way which disappears for people without losing sight.

## Screenshot

<img src="../artifacts/phil-timer-screenshot.png" width="400">

In this screenshot, the on/off switch has been placed, and the timer has been set. Turn the timer off by flipping it over and hiding the arUco tag.

## If I had another week

**Next Move:** Now that I have working software, the next move would be to refine the hardware solution. Where does Horizon live?

**Add:** Another aspect of timers which I haven't explored yet, is sound. I also think if the countdown display could be projected, it could reinforce the spatial experience.

**Fix:** There's currently nothing broken at the moment, however I'm sure that there will be issues with the program once I start fleshing it out spatially.

**Rethink:** I might rethink the readability of the tags themselves. Currently, they are optimized for computer vision, but might not be fully intuitive for humans yet.

---

*Submission checklist:*
- [x] File named `<firstname>-<project>-retro.md` and placed in `docs/whitepaper/retros/`
- [x] Screenshot added to `docs/whitepaper/artifacts/` and linked above
- [x] "What broke" section is honest and specific
- [x] No code snippets — this is a story, not a tutorial
- [X] Opened as a pull request, not pushed to `main`
