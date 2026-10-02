# Plan

## Goal

Measure the predictor behind the OPSD law (student vs self-teacher gap, before any
training) on GSM8K instead of LiveCodeBench v6.

Reference: He, Sieber & Saponati, "A Predictive Law for On-Policy Self-Distillation
From World Feedback", RLxF @ ICML 2026, arXiv:2605.30070.

## Scope

- In: accuracy with and without privileged context, per condition, and the resulting gap.
- Out: OPSD training, so no final improvement.
- Claim: *if* the law transfers beyond code, *then* the best configuration on math is
  the one with the largest gap here.

## Setup

- Colab, T4 GPU. Fallback to Qwen3-0.6B if memory or time is tight.
- Qwen3-1.7B, vLLM, `enable_thinking=False`.
- GSM8K test, first 100-200 problems.
- 8 rollouts per problem, temperature 0.7, 512 max tokens.
- `gap = acc(condition) - acc(student)`, bootstrap CI at the problem level.

Conditions and their mapping to Tufa's `CONSTRUCTION_METHOD`: see `DESIGN.md`.

## Steps

1. Read their prompts, set up the repo, run the free conditions
   (`none`, `peer_solution`, `gt_solution`), smoke test on 10 problems, eyeball prompts
   for answer leakage.
2. Add `hints`, `feedback_binary`, `feedback_diag`. Full run. Table, plot, CIs.
   Optional: repeat on 0.6B to check the ordering is stable.
3. README with results and limitations. Publish. Write the email.

## Risks

- All gaps near zero because GSM8K is too easy: check the `gt_solution` ceiling first;
  if so, restrict to problems the student fails, or drop to 0.6B.
- Answer format not respected: tolerant parser, tuned after the smoke test.
- T4 too slow: fewer problems or the smaller model.

## Deliverable

Public repo, `figures/gaps.png`, `results/gaps.csv`, honest README, one paragraph for
the email.
