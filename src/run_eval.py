"""Measure the student vs self-teacher gap on GSM8K, without training.

Mirrors their 1-step eval with lr=0:
  1. student rollouts: n samples per problem, 0/1 score
  2. privileged context, built from the dataset, for failed rollouts only
     (dont_reprompt_on_self_success=True)
  3. teacher rollout on those, scored the same way
  4. recovery = share of failed rollouts the self-teacher fixes

Rollouts for samples the student already got right are identical across
conditions (they all get the bare prompt), so they are generated once, as the
`none` condition, and reused. That makes the comparison paired and cuts the cost
by roughly the student's accuracy.

Usage:
  python src/run_eval.py --n-problems 300 --n-samples 8 --model Qwen/Qwen3-1.7B
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CONTEXTS = ROOT / "data" / "contexts"


def parse_answer(text):
    """Final number: look for '#### x' first, then the last number in the text."""
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"(-?[\d,]*\.?\d+)", text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def correct(text, gold):
    got = parse_answer(text)
    return got is not None and gold is not None and abs(got - gold) < 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n-problems", type=int, default=300)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default=None,
                    help="comma separated; defaults to every condition")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import conditions as C
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    random.seed(args.seed)
    conds = args.conditions.split(",") if args.conditions else list(C.CONDITIONS)
    if "none" not in conds:
        conds = ["none"] + conds  # the ruler is mandatory

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n_problems))
    questions = [r["question"] for r in ds]
    answers = [r["answer"] for r in ds]
    golds = [parse_answer(a) for a in answers]

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="auto", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True)

    def chat(prompts, n=1, max_tokens=None):
        """enable_thinking=False, as in their eval."""
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for p in prompts]
        sp = SamplingParams(n=n, temperature=args.temperature, top_p=0.95,
                            max_tokens=max_tokens or args.max_tokens, seed=args.seed)
        outs = llm.generate(texts, sp)
        return [[o.text for o in out.outputs] for out in outs]

    n_s, n_q = args.n_samples, len(questions)

    # ---- 1. student --------------------------------------------------------
    print("[1/3] student: %d problems x %d samples" % (n_q, n_s))
    student = chat([C.student_prompt(q) for q in questions], n=n_s)
    ok = [[correct(t, golds[i]) for t in student[i]] for i in range(n_q)]
    student_acc = sum(sum(r) for r in ok) / (n_q * n_s)
    failed = [(i, j) for i in range(n_q) for j in range(n_s) if not ok[i][j]]
    print("      student acc = %.3f, failed rollouts = %d" % (student_acc, len(failed)))

    # ---- 2. privileged contexts, from the dataset only ---------------------
    # the distractor for wrong_solution: a fixed shuffle, so it is reproducible
    perm = list(range(n_q))
    random.Random(args.seed).shuffle(perm)
    other = [perm[i] if perm[i] != i else (i + 1) % n_q for i in range(n_q)]

    contexts = {}   # condition -> per problem text or None
    ctx_answer = {}  # condition -> the number the context points at, if any
    for cond in conds:
        contexts[cond] = [C.build_context(cond, answers[i],
                                          other_answer=answers[other[i]],
                                          seed=args.seed + i)
                          for i in range(n_q)]
        ctx_answer[cond] = [C.context_answer(c) if c else None for c in contexts[cond]]

    CONTEXTS.mkdir(parents=True, exist_ok=True)
    with open(CONTEXTS / ("%s_contexts.json" % args.tag), "w", encoding="utf-8") as f:
        json.dump({"gold": golds, "distractor_of": other,
                   "contexts": contexts, "ctx_answer": ctx_answer},
                  f, indent=2, ensure_ascii=False)

    # ---- 3. teacher rollouts ----------------------------------------------
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "raw").mkdir(exist_ok=True)
    summary = {"model": args.model, "n_problems": n_q, "n_samples": n_s,
               "seed": args.seed, "student_acc": student_acc,
               "n_failed": len(failed), "conditions": {}}
    examples = {}

    for cond in conds:
        prompts = [C.teacher_prompt(cond, questions[i], contexts[cond][i])
                   for i, j in failed]
        n_ctx = sum(1 for i, j in failed if contexts[cond][i])
        print("[3/3] teacher '%s': %d failed rollouts (%d with context)" % (
            cond, len(prompts), n_ctx))
        outs = chat(prompts, n=1)

        rows, fixed, followed, n_follow_ctx = [], 0, 0, 0
        for (i, j), o in zip(failed, outs):
            text = o[0]
            c = correct(text, golds[i])
            fixed += c
            ca = ctx_answer[cond][i]
            got = parse_answer(text)
            follows = (ca is not None and got is not None
                       and abs(got - ca) < 1e-4 and not c)
            if ca is not None:
                n_follow_ctx += 1
                followed += follows
            rows.append({"problem": i, "sample": j, "condition": cond,
                         "used_context": bool(contexts[cond][i]),
                         "correct": bool(c), "student_correct": False,
                         "answer": got, "ctx_answer": ca,
                         "followed_context": bool(follows)})

        recovery = fixed / len(rows) if rows else float("nan")
        follow_rate = followed / n_follow_ctx if n_follow_ctx else float("nan")
        summary["conditions"][cond] = {
            "recovery": recovery, "n_with_context": n_ctx,
            "follow_wrong_rate": follow_rate, "n_follow_checked": n_follow_ctx}
        print("      recovery=%.3f   follows the context's answer=%.3f" % (
            recovery, follow_rate))

        examples[cond] = [{"problem": i, "prompt": p}
                          for (i, j), p in zip(failed, prompts)
                          if contexts[cond][i]][:3]
        with open(RESULTS / "raw" / ("%s_%s.jsonl" % (args.tag, cond)), "w",
                  encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    with open(RESULTS / ("%s_summary.json" % args.tag), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(RESULTS / ("%s_prompt_examples.json" % args.tag), "w",
              encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print("\nwrote results/%s_summary.json" % args.tag)


if __name__ == "__main__":
    main()
