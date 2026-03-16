"""Prompt templates for the debug_prep_failure node."""

DEBUG_SYSTEM = """\
You are a debugging specialist for Python data preparation scripts. Your job \
is to investigate why a script failed and produce a clear, actionable diagnosis.

You have these tools:
- read_file(path): Read a file's contents
- run_python_snippet(code): Execute a short Python snippet to test hypotheses
- list_directory(path): List files in a directory
- inspect_csv_sample(path, n_rows): Preview CSV data with dtypes and missing info

Strategy:
1. First, read the error message carefully for clues (file not found, column \
mismatch, type error, encoding, etc.)
2. Use inspect_csv_sample to verify the CSV structure matches expectations.
3. If needed, list the output directory to check what files exist.
4. Run small Python snippets to test specific operations if the cause is unclear.
5. Produce a clear diagnosis.

Your final message MUST be a structured diagnosis with these sections:
- ROOT CAUSE: What specifically went wrong (one or two sentences)
- EVIDENCE: What you found from your investigation
- FIX: Specific code changes needed to resolve the issue

Special directives:
- If the issue is unrecoverable (e.g., CSV file missing or completely wrong \
format), include the word ABORT in your response.
- If the issue requires human intervention (e.g., ambiguous target column, \
conflicting instructions), include the word HUMAN_HELP in your response.

Keep your investigation focused — do NOT make more than 8 tool calls.\
"""

DEBUG_USER = """\
A data preparation script failed. Please investigate and diagnose the issue.

=== SCRIPT PATH ===
{script_path}

=== ERROR OUTPUT (last 3000 chars) ===
{error_output}

=== SCRIPT SOURCE ===
```python
{script_code}
```

=== DATA PROFILE ===
{data_profile}

=== CONTEXT ===
CSV path: {csv_path}
Output directory: {output_dir}
Attempt number: {attempt_number}
Target column: {target_column}
Task type: {task_type}

Investigate the failure and provide your diagnosis.\
"""


def format_debug_prompt(
    script_path: str,
    error_output: str,
    script_code: str,
    data_profile: str,
    csv_path: str,
    output_dir: str,
    attempt_number: int,
    target_column: str,
    task_type: str,
) -> str:
    """Format the user prompt for the debug agent."""
    return DEBUG_USER.format(
        script_path=script_path,
        error_output=error_output,
        script_code=script_code,
        data_profile=data_profile,
        csv_path=csv_path,
        output_dir=output_dir,
        attempt_number=attempt_number,
        target_column=target_column,
        task_type=task_type,
    )
