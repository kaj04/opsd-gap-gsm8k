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


def bootstrap_recovery(raw, base_raw, on_context_only=False, n_boot=2000, seed=0):
    """Recovery rate and its difference from the `none` condition, bootstrapped at
    the problem level over the student's failed rollouts."""
    rng = np.random.default_rng(seed)

    def by_problem(rows):
        d = {}
        for r in rows:
            if r["student_correct"]:
                continue
            if on_context_only and not r["used_context"]:
                continue
            d.setdefault(r["problem"], []).append(r["correct"])
        return d

    cur, base = by_problem(raw), by_problem(base_raw)
    problems = sorted(set(cur) & set(base))
    if not problems:
        return float("nan"), float("nan"), float("nan"), 0
    c = np.array([np.mean(cur[p]) for p in problems])
    b = np.array([np.mean(base[p]) for p in problems])
    n = sum(len(cur[p]) for p in problems)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(problems), len(problems))
        draws.append(c[idx].mean() - b[idx].mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return c.mean(), lo, hi, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    summary = json.load(open(RESULTS / ("%s_summary.json" % args.tag), encoding="utf-8"))
    conds = list(summary["conditions"].keys())

    # the reference student is the student_correct column, identical across files
    student_rows = [{"problem": r["problem"], "correct": r["student_correct"]}
                    for r in load(args.tag, conds[0])]

    base_raw = load(args.tag, "none") if "none" in conds else None

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
        # bootstrap CI on the recovery difference vs `none`, and the same number
        # restricted to failed rollouts that actually received context
        if base_raw is not None:
            _, rlo, rhi, _ = bootstrap_recovery(raw, base_raw)
            rec_ctx, clo, chi, n_ctx_used = bootstrap_recovery(
                raw, base_raw, on_context_only=True)
        else:
            rlo = rhi = rec_ctx = clo = chi = float("nan")
            n_ctx_used = 0
        rows.append({
            "rec_ci_lo": rlo, "rec_ci_hi": rhi,
            "recovery_on_ctx": rec_ctx, "ctx_ci_lo": clo, "ctx_ci_hi": chi,
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

    rows.sort(key=lambda r: r["recovery"])

    csv_path = RESULTS / "gaps.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("condition,teacher_acc,gap,gap_vs_none,recovery,recovery_vs_none,"
                "rec_ci_lo,rec_ci_hi,recovery_on_ctx,ctx_ci_lo,ctx_ci_hi,"
                "n_failed,n_ctx_failed\n")
        for r in rows:
            f.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d\n" % (
                r["condition"], r["teacher_acc"], r["gap"], r["gap_vs_none"],
                r["recovery"], r["recovery_vs_none"], r["rec_ci_lo"], r["rec_ci_hi"],
                r["recovery_on_ctx"], r["ctx_ci_lo"], r["ctx_ci_hi"],
                r["n_failed"], r["n_ctx_failed"]))

    print("student acc = %.3f   (%s, %d problems x %d samples)" % (
        summary["student_acc"], summary["model"],
        summary["n_problems"], summary["n_samples"]))
    print("%-24s %8s %9s %20s %10s %6s" % (
        "condition", "recovery", "vs none", "95% CI on vs none",
        "on ctx", "ctx"))
    for r in rows:
        print("%-24s %8.3f %+9.3f   [%+.3f, %+.3f] %10.3f %6d" % (
            r["label"], r["recovery"], r["recovery_vs_none"],
            r["rec_ci_lo"], r["rec_ci_hi"], r["recovery_on_ctx"],
            r["n_ctx_failed"]))
    print("\nrecovery = share of the student's failed rollouts the self-teacher fixes")
    print("vs none  = minus `none`, which is plain resampling. CI is bootstrapped")
    print("           over problems, so it covers the difference, not the level")
    print("on ctx   = same recovery, restricted to failures that got context")
    print("ctx      = how many failed rollouts actually received context")
    print("failed rollouts: %d" % rows[0]["n_failed"])

    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(rows))
    vals = [r["recovery_vs_none"] for r in rows]
    err = np.array([[v - r["rec_ci_lo"] for v, r in zip(vals, rows)],
                    [r["rec_ci_hi"] - v for v, r in zip(vals, rows)]])
    err = np.clip(err, 0, None)
    colors = ["#BBBBBB" if r["condition"] in ("none", "gt_solution") else "#4C72B0"
              for r in rows]
    ax.barh(y, vals, xerr=err, color=colors, ecolor="#333", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.axvline(0, color="#666", lw=0.8)
    ax.set_xlabel("recovery above plain resampling\n"
                  "(share of failed rollouts the self-teacher fixes, minus `none`)")
    ax.set_title("Privileged context on GSM8K: how much it beats resampling\n"
                 "%s, %d problems x %d samples" % (
        summary["model"], summary["n_problems"], summary["n_samples"]))
    fig.tight_layout()
    fig.savefig(FIGURES / "gaps.png", dpi=160)
    print("\nwrote: %s and %s" % (csv_path, FIGURES / "gaps.png"))


if __name__ == "__main__":
    main()
