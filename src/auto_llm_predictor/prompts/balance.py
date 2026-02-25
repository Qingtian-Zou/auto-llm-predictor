"""Prompt templates for the write_balance_code node."""

BALANCE_SYSTEM = """\
You are an expert Python programmer. You write clean, production-quality \
data balancing scripts. Your scripts MUST:

1. Be completely self-contained (no imports from custom packages)
2. Use only: json, random, pathlib, sys, collections
3. Read existing JSON files, apply the balancing strategy, and save them to a new file
4. Handle edge cases gracefully
5. Print class distribution BEFORE and AFTER balancing clearly

Output ONLY the complete Python script with no markdown fences, no explanation. \
Just the raw Python code.
"""

BALANCE_USER = """\
Write a Python script that balances the training data for an LLM fine-tuning task.

=== INPUT ===
Data JSON:  {data_json_path}

Each entry has keys: "instruction", "input", "output"
The "output" field contains the class label.

=== BALANCING TASK ===
Task type: {task_type}
Balance strategy: {balance_strategy}
Target mapping: {target_mapping}

=== REQUIREMENTS ===
The script must:
1. Load {data_json_path}
2. Print the class distribution BEFORE balancing (count and percentage per label)
3. Apply the balancing strategy:
   - "oversample": randomly duplicate minority class examples to match the majority
   - "undersample": randomly sample from the majority class to match the minority
4. Shuffle the balanced data
5. Print the class distribution AFTER balancing (count and percentage per label)
6. Save the balanced data to {output_json_path}
7. Print a summary line: "Balanced: <before_count> → <after_count> examples"

{error_context}
{user_feedback_context}
"""

BALANCE_RETRY_CONTEXT = """\
=== PREVIOUS ATTEMPT FAILED ===
The previous script failed with this error:
{error}

Fix the issues and generate a corrected script.
"""

BALANCE_FEEDBACK_CONTEXT = """\
=== USER FEEDBACK (MUST FOLLOW) ===
The user reviewed the balancing and requested:
{feedback}

Make sure the generated script strictly follows these instructions.
"""


def format_balance_prompt(
    data_json_path: str,
    output_json_path: str,
    task_type: str,
    balance_strategy: str,
    target_mapping: dict,
    previous_error: str = "",
    user_feedback: str = "",
) -> str:
    """Format the user prompt for the balance codegen node."""
    error_context = ""
    if previous_error:
        error_context = BALANCE_RETRY_CONTEXT.format(error=previous_error)

    user_feedback_context = ""
    if user_feedback:
        user_feedback_context = BALANCE_FEEDBACK_CONTEXT.format(feedback=user_feedback)

    return BALANCE_USER.format(
        data_json_path=data_json_path,
        output_json_path=output_json_path,
        task_type=task_type,
        balance_strategy=balance_strategy,
        target_mapping=target_mapping,
        error_context=error_context,
        user_feedback_context=user_feedback_context,
    )
