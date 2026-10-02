"""Self-teacher prompt construction for GSM8K.

Templates are copied verbatim from Tufalabs/opsd-predictive-law
(verl/trainer/config/actor/actor.yaml). The only rewritten one is the expert
preamble, which in their repo is specific to competitive programming.
"""

# --- verbatim from the Tufa repo ---------------------------------------------
REPROMPT_TEMPLATE = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question."

SOLUTION_TEMPLATE = "\n\nCorrect solution:\n\n{successful_previous_attempt}"

FEEDBACK_TEMPLATE = (
    "\n\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}"
)

HINT_EXTRACTION_TEMPLATE = """You are extracting 5-10 short, specific key hints from a correct solution to a problem. The hints should guide a solver toward the correct approach without giving away the full answer.

Problem:
{problem}

Correct solution:
{solution}

Output the 5-10 hints as a numbered list, one per line, with no extra commentary."""

# --- rewritten for math (flagged in the README) ------------------------------
STATIC_TEACHER_TEMPLATE = """You are an expert at grade-school math word problems. You master:
- Translating a word problem into explicit quantities and a sequence of arithmetic steps.
- Multi-step reasoning: rates, ratios, percentages, unit conversions, totals and differences.
- Keeping track of what each intermediate number means before combining it with another.

Approach EVERY problem with this methodology:

1. Read the problem twice. List every quantity given, with its unit and meaning.
2. Identify what is being asked, and which quantities it depends on.
3. Work forward one step at a time. After each step, state what the new number means.
4. Do the arithmetic carefully; never combine two numbers with different meanings.
5. Sanity-check the final number against the story (order of magnitude, sign, units).

{prompt}"""

DIAG_FEEDBACK_TEMPLATE = """A student solved this problem incorrectly. Point out, in at most two sentences, the first step where the reasoning or the arithmetic breaks down. Do NOT reveal the correct final answer.

Problem:
{problem}

Student attempt:
{attempt}

Correct reference solution (for your eyes only, do not quote the final answer):
{reference}"""

# the math equivalent of environment feedback: a binary verifier
BINARY_FEEDBACK = "Your previous answer was {answer}, which is incorrect."

ANSWER_INSTRUCTION = (
    "\n\nSolve the problem step by step. End your reply with the final numeric answer "
    "on its own line, in the form: #### <number>"
)

CONDITIONS = [
    "none",            # bare prompt: baseline, gap should be ~0
    "static_math",     # expert preamble (theirs: static_teacher1)
    "feedback_binary", # binary verifier (theirs: feedback_only)
    "feedback_diag",   # diagnostic feedback (added here)
    "hints",           # 5-10 hints from a peer solution (theirs: correct_solution_processed)
    "peer_solution",   # full peer solution (theirs: correct_solution_only)
    "all",             # peer solution + feedback (theirs: all)
    "gt_solution",     # dataset solution: ceiling, not one of their conditions
]


def student_prompt(question: str) -> str:
    return question + ANSWER_INSTRUCTION


def teacher_prompt(condition: str, question: str, *, peer=None, gt=None,
                   feedback=None, hints=None) -> str:
    """Build the teacher prompt. Falls back to the bare student prompt when the
    privileged context is unavailable, mirroring Tufa's gating (mask False means
    no signal)."""
    base = student_prompt(question)

    if condition == "none":
        return base
    if condition == "static_math":
        return STATIC_TEACHER_TEMPLATE.format(prompt=base)

    solution_block, feedback_block = "", ""

    if condition == "feedback_binary" or condition == "feedback_diag":
        if not feedback:
            return base
        feedback_block = FEEDBACK_TEMPLATE.format(feedback_raw=feedback)
    elif condition == "hints":
        if not hints:
            return base
        solution_block = SOLUTION_TEMPLATE.format(successful_previous_attempt=hints)
    elif condition == "peer_solution":
        if not peer:
            return base
        solution_block = SOLUTION_TEMPLATE.format(successful_previous_attempt=peer)
    elif condition == "gt_solution":
        if not gt:
            return base
        solution_block = SOLUTION_TEMPLATE.format(successful_previous_attempt=gt)
    elif condition == "all":
        # environment_feedback_only_without_solution=True: feedback only shows up
        # when there is no peer solution to show
        if peer:
            solution_block = SOLUTION_TEMPLATE.format(successful_previous_attempt=peer)
        elif feedback:
            feedback_block = FEEDBACK_TEMPLATE.format(feedback_raw=feedback)
        else:
            return base
    else:
        raise ValueError(f"unknown condition: {condition}")

    return REPROMPT_TEMPLATE.format(prompt=base, solution=solution_block,
                                    feedback=feedback_block)
