# Design notes: from Tufa's setup (code) to ours (math)

Source: https://github.com/Tufalabs/opsd-predictive-law
Files read: `README.md`, `eval_self_teacher_gap_1step.sbatch`,
`experiments/rich_feedback/run_eval_self_teacher_gap_1step.sh`,
`verl/trainer/ppo/ray_trainer.py` (`_resolve_self_teacher_inputs`),
`verl/trainer/config/actor/actor.yaml` (templates).

## How they measure the gap

- 131 LiveCodeBench v6 problems, 8 rollouts each, sparse 0/1 score.
- One training step with `lr=0`. Weights are frozen; the step exists only to go
  through the trainer path: student rollout -> reprompt build -> teacher rollout.
- Metrics read off W&B: `critic/score/mean` (student) and
  `self-teacher-priv<METHOD>/score/mean` (teacher).
- `gap = teacher - student`. In the paper, improvement is reported as delta mean@4.
- `enable_thinking: false`; `<think>` traces are stripped from demonstrations.

## Teacher templates (verbatim from their config)

```
reprompt_template:  {prompt}{solution}{feedback}\n\nCorrectly solve the original question.
solution_template:  \nCorrect solution:\n\n{successful_previous_attempt}
feedback_template:  \nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}
```

`hint_extraction_template` asks for "5-10 short, specific key hints" from a correct
solution, without giving away the answer. `static_teacher_template` is an expert
preamble for competitive programming.

## Gating rules that matter

- `dont_reprompt_on_self_success=True`: privileged context goes only to rollouts the
  student got wrong. The gap is measured where the student fails.
- A peer solution is another rollout of the same model that happened to be correct,
  not the dataset's ground truth.
- `environment_feedback_only_without_solution=True`: in `all`, feedback shows up only
  when there is no peer solution to show.

## Condition mapping

| theirs | on code | here | on math |
|---|---|---|---|
| `none` | bare prompt | `none` | bare prompt |
| `static_teacher1` | CP expert preamble | `static_math` | math expert preamble (rewritten) |
| `feedback_only` | execution errors, failed tests | `feedback_binary` | "your answer X is incorrect" |
| — | — | `feedback_diag` | plus where the reasoning breaks (model-generated) |
| `correct_solution_processed` | 5-10 hints from a peer solution | `hints` | same template |
| `correct_solution_only` | full peer solution | `peer_solution` | correct peer rollout |
| `all` | peer solution + feedback | `all` | same |
| — | — | `gt_solution` | dataset solution: ceiling, not one of theirs |

`feedback_binary` vs `feedback_diag` isolates how much of the value of environment
feedback comes from it being diagnostic. On code that is free; on math it is not.

## Our setup

- Colab T4, Qwen3-1.7B (fallback 0.6B), vLLM, `enable_thinking=False`.
- GSM8K test, 100-200 problems, 8 rollouts at temperature 0.7, 512 max tokens.
- Scoring: final number after `####`.
- `gap = acc(teacher) - acc(student)` over the same N*n rollouts, bootstrap CI at the
  problem level.

## Out of scope (stated in the README)

- No OPSD training, so no final improvement and no test of the law itself.
- One model, one dataset, one seed.
- The expert preamble is rewritten, so that condition is not identical to theirs.
