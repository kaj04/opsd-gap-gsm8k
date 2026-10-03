# Design notes

Source: [Tufalabs/opsd-predictive-law](https://github.com/Tufalabs/opsd-predictive-law).
Files read: `README.md`, `eval_self_teacher_gap_1step.sbatch`,
`experiments/rich_feedback/run_eval_self_teacher_gap_1step.sh`,
`verl/trainer/ppo/ray_trainer.py` (`_resolve_self_teacher_inputs`),
`verl/trainer/config/actor/actor.yaml` (templates).

## How they measure the gap

- 131 LiveCodeBench v6 problems, 8 rollouts each, sparse 0/1 score.
- One training step with `lr=0`: weights are frozen, the step only exists to walk
  the trainer path student rollout -> reprompt build -> teacher rollout.
- `gap = self-teacher score - student score`, both logged as means.
- `enable_thinking: false`; `<think>` traces stripped from demonstrations.

## What is mirrored here

- Teacher template, verbatim:
  `{prompt}{solution}{feedback}\n\nCorrectly solve the original question.`
  with `\n\nCorrect solution:\n\n{...}` for the solution block.
- `dont_reprompt_on_self_success=True`: privileged context only for failed rollouts.
  Rollouts the student already got right would receive the bare prompt in every
  condition, so they are not regenerated; the teacher runs on failures only.
- 8 rollouts per problem, temperature 0.7, `enable_thinking=False`, 0/1 scoring.

## What differs, and why

| theirs | here |
|---|---|
| peer solution = another correct rollout of the same model | contexts come from the dataset's reference solution |
| `feedback_only` = execution feedback | dropped: on math a verifier only says right/wrong |
| `correct_solution_processed` = model-extracted hints | dropped: model-generated text needs a leakage audit |
| — | `gold_step1`, `gold_steps`: a dose scale of true information |
| — | `wrong_solution`, `corrupted_solution`: misleading contexts |

Using the dataset instead of peer rollouts keeps every context deterministic,
reproducible and free of answer leakage, at the cost of being further from their
exact setup. The dose scale and the misleading contexts are the point of this repo:
their six conditions all carry correct information, so they cannot say what happens
when the privileged context is wrong.

## Conditions

| name | context |
|---|---|
| `none` | nothing: the ruler, measures plain resampling |
| `gold_step1` | first reasoning step |
| `gold_steps` | every step but the last, so the final number is withheld |
| `gt_solution` | full reference solution, answer included |
| `wrong_solution` | the full solution of another problem (fixed shuffle, seeded) |
| `corrupted_solution` | this problem's solution, every computed result shifted by one small delta, propagated so the text stays self-consistent |

`gold_steps` drops the last step because GSM8K usually states the answer inline in
it ("48+24 = 72"), so removing only the `#### 72` line would not withhold anything.

## Metrics

- `recovery` = share of the student's failed rollouts the self-teacher fixes.
  Preferred over the raw gap: the gap is diluted by rollouts that were already
  correct and that receive no context.
- `vs none` = recovery minus the `none` condition, with a bootstrap CI over problems.
  Paired, since all conditions are evaluated on the same failed rollouts.
- `copies the context` = share of rollouts returning the number the context points
  at. Only defined where the context states a final answer.
