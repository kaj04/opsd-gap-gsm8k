"""Print one example prompt per condition. No GPU needed.

Use it to check for answer leakage before spending GPU time.
Usage: python src/inspect_prompts.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conditions as C  # noqa: E402

QUESTION = ("Natalia sold clips to 48 of her friends in April, and then she sold "
            "half as many clips in May. How many clips did Natalia sell altogether "
            "in April and May?")
GT = ("Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
      "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
      "#### 72")
PEER = ("She sold 48 clips in April. In May she sold 48 / 2 = 24 clips. "
        "In total 48 + 24 = 72.\n#### 72")
FEEDBACK_BIN = C.BINARY_FEEDBACK.format(answer=96.0)
FEEDBACK_DIAG = ("The first error is in the May step: you doubled the April amount "
                 "instead of halving it.")
HINTS = ("1. Read 'half as many' as a division, not a multiplication.\n"
         "2. Compute the May amount before the total.\n"
         "3. The question asks for the sum of both months.")


def main():
    for cond in C.CONDITIONS:
        fb = FEEDBACK_DIAG if cond == "feedback_diag" else FEEDBACK_BIN
        p = C.teacher_prompt(cond, QUESTION, peer=PEER, gt=GT, feedback=fb, hints=HINTS)
        print("=" * 78)
        print("CONDITION: %s" % cond)
        print("=" * 78)
        print(p)
        print()
        leaked = "72" in p.replace(QUESTION, "")
        print(">>> leaks the final answer (72)? %s" % ("YES" if leaked else "no"))
        print()


if __name__ == "__main__":
    main()
