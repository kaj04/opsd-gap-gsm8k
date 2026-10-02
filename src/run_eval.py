"""Measure the student vs self-teacher gap on GSM8K, without training.

Mirrors their 1-step eval with lr=0:
  1. student rollouts: n samples per problem, 0/1 score
  2. build privileged context, only for failed rollouts
     (dont_reprompt_on_self_success=True)
  3. teacher rollout for every sample (bare prompt where there is no context)
  4. gap = acc(teacher) - acc(student) over the same N*n rollouts

Usage:
  python src/run_eval.py --n-problems 100 --n-samples 8 --model Qwen/Qwen3-1.7B
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


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
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default=",".join([
        "none", "static_math", "feedback_binary", "feedback_diag",
        "hints", "peer_solution", "all", "gt_solution"]))
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import conditions as C
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    random.seed(args.seed)
    conds = args.conditions.split(",")

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n_problems))
    questions = [r["question"] for r in ds]
    gt_solutions = [r["answer"] for r in ds]
    golds = [parse_answer(a) for a in gt_solutions]

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

    n_s = args.n_samples
    n_q = len(questions)

    # ---- 1. student --------------------------------------------------------
    print("[1/4] student: %d problems x %d samples" % (n_q, n_s))
    student = chat([C.student_prompt(q) for q in questions], n=n_s)
    ok = [[correct(t, golds[i]) for t in student[i]] for i in range(n_q)]
    student_acc = sum(sum(r) for r in ok) / (n_q * n_s)
    print("      student acc = %.3f" % student_acc)

    # ---- 2. privileged context ---------------------------------------------
    # peer = a correct rollout other than this one (cf. _get_solution)
    peers = [[None] * n_s for _ in range(n_q)]
    for i in range(n_q):
        succ = [j for j in range(n_s) if ok[i][j]]
        for j in range(n_s):
            cand = [k for k in succ if k != j]
            peers[i][j] = student[i][cand[0]] if cand else None

    # binary feedback: failed rollouts only
    fb_bin = [[None if ok[i][j] else
               C.BINARY_FEEDBACK.format(answer=parse_answer(student[i][j]))
               for j in range(n_s)] for i in range(n_q)]

    # diagnostic feedback: one generation per failed rollout
    fb_diag = [[None] * n_s for _ in range(n_q)]
    if "feedback_diag" in conds:
        idx = [(i, j) for i in range(n_q) for j in range(n_s) if not ok[i][j]]
        print("[2/4] diagnostic feedback for %d failed rollouts" % len(idx))
        if idx:
            outs = chat([C.DIAG_FEEDBACK_TEMPLATE.format(problem=questions[i],
                                                         attempt=student[i][j],
                                                         reference=gt_solutions[i])
                         for i, j in idx], n=1, max_tokens=160)
            for (i, j), o in zip(idx, outs):
                fb_diag[i][j] = o[0].strip()

    # hints extracted from a peer solution (once per problem)
    hints = [None] * n_q
    if "hints" in conds:
        have = [i for i in range(n_q) if any(ok[i])]
        print("[3/4] hint extraction for %d problems with a correct peer" % len(have))
        if have:
            outs = chat([C.HINT_EXTRACTION_TEMPLATE.format(
                problem=questions[i],
                solution=student[i][[j for j in range(n_s) if ok[i][j]][0]])
                for i in have], n=1, max_tokens=320)
            for i, o in zip(have, outs):
                hints[i] = o[0].strip()

    # ---- 3. teacher, per condition -----------------------------------------
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "raw").mkdir(exist_ok=True)
    summary = {"model": args.model, "n_problems": n_q, "n_samples": n_s,
               "seed": args.seed, "student_acc": student_acc, "conditions": {}}
    prompt_examples = {}  # real teacher prompts, kept for inspection

    for cond in conds:
        prompts, meta = [], []
        for i in range(n_q):
            base = C.student_prompt(questions[i])
            for j in range(n_s):
                fb = fb_diag[i][j] if cond == "feedback_diag" else fb_bin[i][j]
                if ok[i][j]:
                    p = base  # gating: no context for rollouts already correct
                else:
                    p = C.teacher_prompt(cond, questions[i], peer=peers[i][j],
                                         gt=gt_solutions[i], feedback=fb,
                                         hints=hints[i])
                prompts.append(p)
                meta.append((i, j, p != base))
        n_ctx = sum(m[2] for m in meta)
        # keep up to 3 real prompts with context, for later inspection
        prompt_examples[cond] = [
            {"problem": meta[k][0], "sample": meta[k][1], "prompt": prompts[k]}
            for k in range(len(prompts)) if meta[k][2]][:3]
        print("[4/4] teacher '%s': %d rollouts (%d with context)" % (cond, len(prompts), n_ctx))
        outs = chat(prompts, n=1)
        acc, rows = 0, []
        for (i, j, used), o in zip(meta, outs):
            c = correct(o[0], golds[i])
            acc += c
            rows.append({"problem": i, "sample": j, "condition": cond,
                         "used_context": used, "correct": bool(c),
                         "student_correct": bool(ok[i][j])})
        acc /= len(prompts)
        summary["conditions"][cond] = {"teacher_acc": acc,
                                       "gap": acc - student_acc,
                                       "n_with_context": n_ctx}
        print("      acc=%.3f  gap=%+.3f" % (acc, acc - student_acc))
        with open(RESULTS / "raw" / ("%s_%s.jsonl" % (args.tag, cond)), "w",
                  encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    out_path = RESULTS / ("%s_summary.json" % args.tag)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    ex_path = RESULTS / ("%s_prompt_examples.json" % args.tag)
    with open(ex_path, "w", encoding="utf-8") as f:
        json.dump(prompt_examples, f, indent=2, ensure_ascii=False)
    print("\nwrote: %s and %s" % (out_path, ex_path))


if __name__ == "__main__":
    main()
