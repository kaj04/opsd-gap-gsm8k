"""Table and plot of recovery per condition, with bootstrap confidence intervals.

recovery  = share of the student's failed rollouts the self-teacher fixes
vs none   = recovery minus the `none` condition, which is plain resampling
follows   = share of the rollouts where the model returned the number the
            privileged context points at, instead of the right one

Usage: python src/analyze.py --tag run
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

LABELS = {
    "none": "none (resampling)",
    "gold_step1": "first step only",
    "gold_steps": "all steps, no answer",
    "gt_solution": "full solution (ceiling)",
    "wrong_solution": "another problem's solution",
    "corrupted_solution": "corrupted solution",
}
MISLEADING = ("wrong_solution", "corrupted_solution")


def load(tag, cond):
    path = RESULTS / "raw" / ("%s_%s.jsonl" % (tag, cond))
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def by_problem(rows, field="correct"):
    d = {}
    for r in rows:
        d.setdefault(r["problem"], []).append(float(r[field]))
    return d


def bootstrap_diff(rows, base_rows, n_boot=4000, seed=0):
    """Recovery, and its difference from `none`, bootstrapped over problems."""
    rng = np.random.default_rng(seed)
    cur, base = by_problem(rows), by_problem(base_rows)
    problems = sorted(set(cur) & set(base))
    c = np.array([np.mean(cur[p]) for p in problems])
    b = np.array([np.mean(base[p]) for p in problems])
    draws = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, len(problems), len(problems))
        draws[k] = c[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return c.mean(), c.mean() - b.mean(), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    summary = json.load(open(RESULTS / ("%s_summary.json" % args.tag), encoding="utf-8"))
    conds = list(summary["conditions"].keys())
    base_rows = load(args.tag, "none")

    rows = []
    for cond in conds:
        raw = load(args.tag, cond)
        rec, diff, lo, hi = bootstrap_diff(raw, base_rows)
        rows.append({
            "condition": cond,
            "label": LABELS.get(cond, cond),
            "recovery": rec, "diff": diff, "lo": lo, "hi": hi,
            "follow": summary["conditions"][cond]["follow_wrong_rate"],
            "n_ctx": summary["conditions"][cond]["n_with_context"],
        })
    rows.sort(key=lambda r: r["diff"])

    with open(RESULTS / "gaps.csv", "w", encoding="utf-8") as f:
        f.write("condition,recovery,diff_vs_none,ci_lo,ci_hi,follow_context,n_with_context\n")
        for r in rows:
            f.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n" % (
                r["condition"], r["recovery"], r["diff"], r["lo"], r["hi"],
                r["follow"], r["n_ctx"]))

    print("%s, %d problems x %d samples" % (
        summary["model"], summary["n_problems"], summary["n_samples"]))
    print("student acc %.3f, %d failed rollouts\n" % (
        summary["student_acc"], summary["n_failed"]))
    print("%-28s %8s %9s %20s %8s" % (
        "condition", "recovery", "vs none", "95% CI on vs none", "follows"))
    for r in rows:
        follow = "     -  " if np.isnan(r["follow"]) else "%8.3f" % r["follow"]
        print("%-28s %8.3f %+9.3f   [%+.3f, %+.3f] %s" % (
            r["label"], r["recovery"], r["diff"], r["lo"], r["hi"], follow))
    print("\nrecovery = share of failed rollouts the self-teacher fixes")
    print("follows  = share where the model returned the number the context points at")

    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(rows))
    vals = [r["diff"] for r in rows]
    err = np.clip(np.array([[v - r["lo"] for v, r in zip(vals, rows)],
                            [r["hi"] - v for v, r in zip(vals, rows)]]), 0, None)
    colors = []
    for r in rows:
        if r["condition"] in MISLEADING:
            colors.append("#C44E52")      # misleading
        elif r["condition"] in ("none", "gt_solution"):
            colors.append("#BBBBBB")      # ruler and ceiling
        else:
            colors.append("#4C72B0")
    ax.barh(y, vals, xerr=err, color=colors, ecolor="#333", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.axvline(0, color="#666", lw=0.8)
    ax.set_xlabel("recovery above plain resampling")
    ax.set_title("Privileged context on GSM8K\n%s, %d problems x %d samples, "
                 "%d failed rollouts" % (
                     summary["model"], summary["n_problems"],
                     summary["n_samples"], summary["n_failed"]))
    fig.tight_layout()
    fig.savefig(FIGURES / "gaps.png", dpi=160)
    print("\nwrote results/gaps.csv and figures/gaps.png")


if __name__ == "__main__":
    main()
