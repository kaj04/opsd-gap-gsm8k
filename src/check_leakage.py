"""Check whether generated contexts leak the final answer.

A hint or a piece of feedback is allowed to point at the method or at the broken
step. It must not contain the gold number, otherwise the self-teacher is just
reading the answer off the prompt.

Usage: python src/check_leakage.py --tag run [--show 5]
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTEXTS = ROOT / "data" / "contexts"


def numbers(text):
    return {n.replace(",", "") for n in re.findall(r"-?[\d,]*\.?\d+", text)}


def fmt(gold):
    """The gold answer as it would plausibly be written in text."""
    out = {str(gold)}
    if float(gold).is_integer():
        out.add(str(int(gold)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--show", type=int, default=3, help="leaking examples to print")
    args = ap.parse_args()

    data = json.load(open(CONTEXTS / ("%s_contexts.json" % args.tag), encoding="utf-8"))
    gold = data["gold"]

    for field in ("hints", "feedback_diag"):
        items = data.get(field)
        if not items:
            continue
        flat = []
        if field == "hints":
            flat = [(i, None, t) for i, t in enumerate(items) if t]
        else:
            flat = [(i, j, t) for i, row in enumerate(items)
                    for j, t in enumerate(row) if t]
        if not flat:
            print("%s: nothing generated\n" % field)
            continue

        leaks = [(i, j, t) for i, j, t in flat
                 if gold[i] is not None and numbers(t) & fmt(gold[i])]
        print("%s: %d generated, %d contain the gold number (%.1f%%)" % (
            field, len(flat), len(leaks), 100 * len(leaks) / len(flat)))
        for i, j, t in leaks[:args.show]:
            print("  - problem %d%s | gold %s" % (
                i, "" if j is None else " sample %d" % j, gold[i]))
            print("    %s" % t.replace("\n", " ")[:300])
        print()

    print("A leak rate above ~10% means the prompt needs tightening.")


if __name__ == "__main__":
    main()
