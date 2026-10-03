# What does a self-teacher actually learn from?

A small experiment on **on-policy self-distillation (OPSD)**: a model is trained to
match itself conditioned on extra, privileged information, so that it later solves
the problem without it.

Tufa Labs [showed](https://arxiv.org/abs/2605.30070) that the **student vs
self-teacher accuracy gap**, measured before any training, linearly predicts how
much OPSD will help. The gap is therefore worth understanding on its own: it says
what a given kind of privileged context is worth.

They measured it on code, where the privileged context is execution feedback. This
repo measures it on **grade-school math** (GSM8K), and asks a question their setup
cannot ask: **what happens when the privileged context is wrong?**

No training here. This measures the predictor, not the final improvement.

## Setup

Qwen3-1.7B, 300 GSM8K test problems, 8 rollouts each, temperature 0.7,
`enable_thinking=False`. The student solves; wherever it fails, the self-teacher
retries the same problem with extra text prepended. Following Tufa's gating,
privileged context goes **only** to rollouts the student got wrong (462 of 2400).

The teacher template is copied verbatim from their `actor.yaml`. The contexts are
built from the dataset's own reference solution, so nothing is model-generated.

**recovery** = share of the student's failed rollouts the self-teacher fixes.

## Results

| what the teacher sees | recovery | vs `none` | 95% CI | copies the context |
|---|---|---|---|---|
| full solution, answer included | 0.926 | +0.463 | [+0.361, +0.565] | 0.000 |
| **same solution, numbers shifted** | **0.751** | **+0.288** | [+0.177, +0.400] | 0.087 |
| all steps but the last | 0.713 | +0.250 | [+0.130, +0.361] | — |
| first reasoning step | 0.610 | +0.147 | [+0.037, +0.255] | — |
| another problem's solution | 0.491 | +0.028 | [−0.074, +0.130] | 0.000 |
| nothing (plain resampling) | 0.463 | — | — | — |

Student accuracy 0.807. CIs are bootstrapped over problems.
"Copies the context" = how often the model returned the number the context points at.

Three things stand out.

**1. Resampling alone recovers 46% of failures.** Any measurement of privileged
context has to be read against that, not against zero.

**2. A corrupted solution helps almost as much as a correct one.** It contains wrong
arithmetic and a wrong final answer, yet it beats having the real reasoning steps,
and the model copies its answer only 8.7% of the time. What transfers is the
*procedure*; the model re-derives the numbers itself.

**3. An irrelevant solution is ignored, not followed.** Another problem's solution
leaves recovery where it was and is never copied.

So on math the useful part of a privileged context looks like the plan, not the
values — and a plausible-but-wrong context is still useful rather than harmful. That
is encouraging for OPSD, since a self-teacher's privileged information is not
guaranteed to be correct.

## Caveats

- No OPSD training was run, so none of this shows what the final improvement would be.
- One model, one dataset, one seed.
- GSM8K is old and widely cited, so some contamination is likely.
- Corruption shifts every computed result by the same small delta, which keeps the
  text self-consistent but is a narrow notion of "wrong".
- On two-step problems `gold_step1` and `gold_steps` coincide.

## Reproduce

Open [`colab.ipynb`](colab.ipynb) on a GPU runtime; it runs end to end in about 45
minutes on a T4. Locally:

```bash
pip install -r requirements.txt
python src/inspect_prompts.py --index 3            # one example of each context
python src/run_eval.py --n-problems 300 --n-samples 8 --tag run
python src/analyze.py --tag run
```

`DESIGN.md` records how each condition maps to Tufa's `CONSTRUCTION_METHOD` and which
of their gating rules are mirrored.

## Credits

Templates and gating from
[Tufalabs/opsd-predictive-law](https://github.com/Tufalabs/opsd-predictive-law)
(Apache 2.0), built on SDPO and verl. Independent and unaffiliated.
