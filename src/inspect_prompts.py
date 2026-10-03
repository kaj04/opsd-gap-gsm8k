"""Print one example teacher prompt per condition, on a real GSM8K item. No GPU.

Use it to sanity-check the contexts before spending GPU time.
Usage: python src/inspect_prompts.py [--index 0]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conditions as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    item, other = ds[args.index], ds[args.index + 1]

    for cond in C.CONDITIONS:
        ctx = C.build_context(cond, item["answer"], other_answer=other["answer"])
        prompt = C.teacher_prompt(cond, item["question"], ctx)
        print("=" * 78)
        print("CONDITION: %s" % cond)
        print("=" * 78)
        print(prompt)
        print()
        gold = C.context_answer(item["answer"])
        points_at = C.context_answer(ctx) if ctx else None
        print(">>> gold answer: %s | the context points at: %s" % (gold, points_at))
        print()


if __name__ == "__main__":
    main()
