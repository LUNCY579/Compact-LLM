# Autonomous session plan, 2026-09-14

Thresholds fixed here BEFORE any of these runs, per the programme's pre-registration rule.

## Standing facts these rest on

* Panel noise floor: same-config seeds give 15-24 discordant rows (NOISE_FLOOR.md). Panel
  accuracy differences under ~24 discordant rows are unreadable.
* Forge noise floor: UNMEASURED. Forge *accuracy* differences stay unread until it exists.
* Forge *termination* needs no floor: the observed effect was 15-0 and 16-0, p~5e-5, in two
  independently trained models (FORGE_TRANSFER.md).
* No B28 arm has ever run WSD. The inherited cosine:0.03 warmup ends after ~1.2 optimizer steps.
* Weight decay has been 0.0 in every run in this programme.

## Run order and pre-registered readings

**A1. B28last20 + wsd:0.2:0.2 + beta2 0.95, seed 17, SAVE=final.**
One variable changed against the banked cosine run (67/150 panel, 183/200 forge termination).
  - Termination PRESERVED (>= 195/200) and panel >= 60  -> WSD is a strictly better recipe.
    This is the win condition.
  - Termination still collapsed (<= 190/200) -> the collapse is not a schedule artifact; it is
    caused by training on short rows, and the next lever is data, not optimizer.
  - Panel < 55 -> WSD hurts this arm; do not propagate it to other B28 arms.
Termination is readable now. Panel accuracy vs 67 is NOT readable alone (noise floor 24);
only a swing beyond that counts.

**A2. B28last20 + weight_decay, seed 17.** Only if A1 leaves termination collapsed. Decay pulls
toward the source, which is the anti-forgetting form of the same question.

**B. Forge noise floor.** B28last20 at seeds 23/29/41 with SAVE=final, then probe all four on
forge. Serves three purposes at once: the forge floor, four identical-tensor-set soup
ingredients, and replacements for the checkpoints SAVE='none' discarded.

**C. Soup**, only once B shows the floor. Four ingredients of one configuration is the correct
Wortsman setup; two ingredients differing by 15 panel rows was not.

## Rules for this session

* One model job at a time. No diagnostic commands while a job runs -- host contention already
  killed one completed training run at the probe step.
* SAVE=final on everything from here. SAVE='none' cost a finished 39-step run whose probe
  aborted, because there were no weights to re-probe.
* Every claim gets the paired test, not the raw count. The raw forge 32->27 looked like a
  regression and was p=0.30 sampling.
* Report negatives. The termination finding is the most valuable result of the day and it is
  a negative.
