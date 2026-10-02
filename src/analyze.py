"""Gap table and plot, with bootstrap confidence intervals.

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
    "none": "none (baseline)",
    "static_math": "expert preamble",
    "feedback_binary": "binary feedback",
    "feedback_diag": "diagnostic feedback",
    "hints": "peer hints",
    "peer_solution": "peer solution",
    "all": "peer + feedback",
    "gt_solution": "gold solution (ceiling)",
}


def load(tag, cond):
    path = RESULTS / "raw" / ("%s_%s.jsonl" % (tag, cond))
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def bootstrap_gap(teacher, student, n_boot=2000, seed=0):
    """Bootstrap at the problem level: rollouts of the same problem are correlated."""
    rng = np.random.default_rng(seed)
    problems = sorted({r["problem"] for r in teacher})
    by_p = {p: ([], []) for p in problems}
    for r in teacher:
        by_p[r["problem"]][0].append(r["correct"])
    for r in student:
        by_p[r["problem"]][1].append(r["correct"])
    t = np.array([np.mean(by_p[p][0]) for p in problems])
    s = np.array([np.mean(by_p[p][1]) for p in problems])
    point = t.mean() - s.mean()
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(problems), len(problems))
        draws.append(t[idx].mean() - s[idx].mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    summary = json.load(open(RESULTS / ("%s_summary.json" % args.tag), encoding="utf-8"))
    conds = list(summary["conditions"].keys())

    # the reference student is the student_correct column, identical across files
    student_rows = [{"problem": r["problem"], "correct": r["student_correct"]}
                    for r in load(args.tag, conds[0])]

    rows = []
    for cond in conds:
        raw = load(args.tag, cond)
        teacher_rows = [{"problem": r["problem"], "correct": r["correct"]} for r in raw]
        gap, lo, hi = bootstrap_gap(teacher_rows, student_rows)
        # recovery rate: of the rollouts the student got wrong, how many does the
        # self-teacher fix? Denominator is every failed rollout, so conditions are
        # comparable and `none` gives the plain-resampling baseline.
        failed = [r for r in raw if not r["student_correct"]]
        recovery = sum(r["correct"] for r in failed) / len(failed) if failed else float("nan")
        n_ctx_failed = sum(1 for r in failed if r["used_context"])
        rows.append({
            "condition": cond,
            "label": LABELS.get(cond, cond),
            "teacher_acc": summary["conditions"][cond]["teacher_acc"],
            "gap": gap, "ci_lo": lo, "ci_hi": hi,
            "recovery": recovery, "n_failed": len(failed),
            "n_ctx_failed": n_ctx_failed,
            "n_with_context": summary["conditions"][cond]["n_with_context"],
        })

    # `none` is the honest reference: same sampling path as the other conditions,
    # no privileged context. Report everything relative to it as well.
    base = next((r for r in rows if r["condition"] == "none"), None)
    for r in rows:
        r["gap_vs_none"] = r["gap"] - base["gap"] if base else float("nan")
        r["recovery_vs_none"] = r["recovery"] - base["recovery"] if base else float("nan")

    rows.sort(key=lambda r: r["gap"])

    csv_path = RESULTS / "gaps.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("condition,teacher_acc,gap,ci_lo,ci_hi,gap_vs_none,"
                "recovery,recovery_vs_none,n_failed,n_ctx_failed\n")
        for r in rows:
            f.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d\n" % (
                r["condition"], r["teacher_acc"], r["gap"], r["ci_lo"], r["ci_hi"],
                r["gap_vs_none"], r["recovery"], r["recovery_vs_none"],
                r["n_failed"], r["n_ctx_failed"]))

    print("student acc = %.3f   (%s, %d problems x %d samples)" % (
        summary["student_acc"], summary["model"],
        summary["n_problems"], summary["n_samples"]))
    print("%-26s %7s %8s %9s %9s %9s %6s" % (
        "condition", "acc", "gap", "vs none", "recovery", "vs none", "ctx"))
    for r in rows:
        print("%-26s %7.3f %+8.3f %+9.3f %9.3f %+9.3f %6d" % (
            r["label"], r["teacher_acc"], r["gap"], r["gap_vs_none"],
            r["recovery"], r["recovery_vs_none"], r["n_ctx_failed"]))
    print("\nrecovery = share of the student's failed rollouts the self-teacher fixes")
    print("vs none  = same number minus `none`, which is plain resampling")
    print("ctx      = failed rollouts that actually received privileged context")
    print("failed rollouts: %d" % rows[0]["n_failed"])

    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(rows))
    gaps = [r["gap"] for r in rows]
    err = np.array([[r["gap"] - r["ci_lo"] for r in rows],
                    [r["ci_hi"] - r["gap"] for r in rows]])
    ax.barh(y, gaps, xerr=err, color="#4C72B0", ecolor="#333", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.axvline(0, color="#666", lw=0.8)
    ax.set_xlabel("gap = acc(self-teacher) - acc(student)")
    ax.set_title("Gap by privileged context on GSM8K\n%s, %d problems x %d samples" % (
        summary["model"], summary["n_problems"], summary["n_samples"]))
    fig.tight_layout()
    fig.savefig(FIGURES / "gaps.png", dpi=160)
    print("\nwrote: %s and %s" % (csv_path, FIGURES / "gaps.png"))


if __name__ == "__main__":
    main()
