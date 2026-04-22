"""Prompt templates for the write_prep_code node."""

CODEGEN_SYSTEM = """\
You are an expert Python programmer. You write clean, production-quality \
data preparation scripts. Your scripts MUST:

1. Be completely self-contained (no imports from custom packages)
2. Use only: pandas, numpy, json, pickle, sklearn, random, pathlib, argparse, sys, os, math
3. Expose a stable command-line interface (argparse) so the same script can \
be invoked at training time and at inference time without source edits
4. Apply data-dependent transformations correctly: fit on training data only, \
then apply (without refitting) to test data and to any future inference CSV
5. Persist all fitted transformers to disk so inference can reload them
6. Print progress messages to stdout
7. For classification tasks, preserve ALL target classes exactly as specified \
in the target_mapping — never merge, drop, or simplify classes unless \
specifically instructed otherwise. For regression tasks, target_mapping will \
be empty; output the raw target value.

Output ONLY the complete Python script with no markdown fences, no \
explanation. Just the raw Python code.
"""

CODEGEN_USER = """\
Write a Python script that converts CSV data into LlamaFactory-compatible \
Alpaca JSON format for fine-tuning, with a stable CLI so it can be reused \
for inference.

=== DATASET PROFILE ===
{data_profile}

=== PREPARATION PLAN ===
Target column: {target_column}
Task type: {task_type}
Target mapping: {target_mapping}
Selected features: {selected_features}
Instruction template: {instruction_template}
Input format: {input_format}
Output format: {output_format}
Data cleaning steps: {data_cleaning_steps}

=== INPUT FILES ===
Main (training) CSV: {csv_path}
{test_csv_section}

=== OUTPUT REQUIREMENTS ===
The script must save files to the directory passed via ``--output-dir``. \
At training time the orchestrator will pass: ``--output-dir {output_data_dir}``.

=== REQUIRED COMMAND-LINE INTERFACE ===
Implement argparse with EXACTLY these flags:

    --input-csv PATH       (required) Path to the training CSV in training mode,
                           or the new CSV to score in --predict-only mode.
    --test-csv PATH        (optional) Path to the test CSV. Ignored in
                           --predict-only mode.
    --output-dir PATH      (required) Directory where outputs are written.
    --predict-only         (flag)     Inference mode. See below.

=== TWO OPERATING MODES ===

Training mode (default — no ``--predict-only``):
  1. Load --input-csv as the *training* DataFrame and --test-csv as the *test*
     DataFrame (if provided).
  2. Apply all data cleaning and fit any data-dependent transformer
     (LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler,
     MinMaxScaler, SimpleImputer, vocabularies, target-encoded categories,
     learned bins, etc.) using the TRAINING DataFrame ONLY. Do NOT call
     `.fit(...)` or `.fit_transform(...)` on the test DataFrame.
  3. Apply the SAME fitted instances (via `.transform(...)`) to the test
     DataFrame. Constants derived from data (means, medians, modes, min/max,
     class lists, quantile cutoffs) must come from training and be reused
     verbatim for test.
  4. Stateless row-level operations (string casts, simple arithmetic on a
     single value, target_mapping lookups) may be applied independently to
     either DataFrame.
  5. Convert each row of the training DataFrame to an Alpaca dict
     {{"instruction": ..., "input": ..., "output": ...}} and write
     ``all_data.json`` (list of dicts).
  6. Convert each row of the test DataFrame the same way and write
     ``test_data.json``.
  7. Pickle every fitted transformer (and any data-derived constants) to
     ``<output-dir>/transformers.pkl``. The recommended structure is a single
     dict, e.g. ``{{"label_encoder": le, "scaler": scaler, "feature_columns": cols}}``.
     If no transformers were fitted (purely stateless preprocessing), still
     write the file with an empty dict so downstream code can rely on it
     existing.
  8. Write ``dataset_info.json``:
     {{"train": {{"file_name": "all_data.json"}}, "test": {{"file_name": "test_data.json"}}}}

Predict-only mode (``--predict-only`` flag set):
  1. Load ``<output-dir>/transformers.pkl``. If it does not exist, exit with
     a clear error message and a non-zero status code — do NOT silently
     re-fit.
  2. Load --input-csv as the inference DataFrame. Ignore --test-csv if given.
  3. Apply the loaded transformers via ``.transform(...)`` only. Never call
     ``.fit(...)`` or ``.fit_transform(...)`` in this mode.
  4. Convert each row to an Alpaca dict the same way as training mode and
     write ``all_data.json`` only. Do NOT write ``test_data.json``,
     ``dataset_info.json``, or re-pickle transformers.

=== ADDITIONAL RULES ===
- Per row, build the example dict with:
  * "instruction": the instruction template (same for every row)
  * "input": selected features formatted as readable text
  * "output": the target label (for classification: mapped using
    target_mapping, MUST preserve ALL classes; for regression: the raw
    numeric value as a string). In ``--predict-only`` mode the target
    column may be absent — emit an empty string for "output" in that case.
- DO NOT randomly split data into train/test — splitting has already
  happened upstream of this script.
- DO NOT apply any class balancing (oversampling/undersampling) — that is
  handled by a separate step downstream.
- Print summary statistics (number of rows processed, class distribution
  where applicable) per CSV.
- Handle edge cases gracefully: missing values in features, unexpected
  types, target column absent in predict-only mode.

{error_context}
{user_feedback_context}
"""

_TEST_CSV_SECTION = """\
Test CSV: {test_csv_path}
The test CSV has been split off from the training data upstream and shares
the same schema. Process it via the same fitted transformers — see the
training-mode rules below."""

CODEGEN_RETRY_CONTEXT = """\
=== PREVIOUS ATTEMPT FAILED ===
The previous script was:
```python
{previous_code}
```

It failed with this error:
{error}

Fix the issues and generate a corrected script.
"""

CODEGEN_FEEDBACK_CONTEXT = """\
=== USER FEEDBACK (MUST FOLLOW) ===
The user reviewed a previous version and requested:
{feedback}

Make sure the generated script strictly follows these instructions.
"""

CODEGEN_DEBUG_CONTEXT = """\
=== DEBUG AGENT DIAGNOSIS ===
A debugging agent investigated the failure and found:
{diagnosis}

{tool_summary_section}\
Use this diagnosis to fix the specific issues identified. The diagnosis \
is more reliable than the raw error output because the agent inspected \
the actual files and data.
"""


def format_codegen_prompt(
    csv_path: str,
    data_profile: str,
    target_column: str,
    task_type: str,
    target_mapping: dict,
    selected_features: list[str],
    instruction_template: str,
    input_format: str,
    output_format: str,
    data_cleaning_steps: list[str],
    output_data_dir: str,
    test_csv_path: str = "",
    previous_error: str = "",
    previous_code: str = "",
    user_feedback: str = "",
    debug_diagnosis: str = "",
    debug_tool_summary: str = "",
) -> str:
    """Format the user prompt for the codegen node."""
    error_context = ""
    if previous_error:
        error_context = CODEGEN_RETRY_CONTEXT.format(
            previous_code=previous_code, error=previous_error,
        )

    if debug_diagnosis:
        tool_summary_section = (
            f"Tools used during investigation:\n{debug_tool_summary}\n\n"
            if debug_tool_summary else ""
        )
        error_context += "\n" + CODEGEN_DEBUG_CONTEXT.format(
            diagnosis=debug_diagnosis,
            tool_summary_section=tool_summary_section,
        )

    user_feedback_context = ""
    if user_feedback:
        user_feedback_context = CODEGEN_FEEDBACK_CONTEXT.format(feedback=user_feedback)

    if test_csv_path:
        test_csv_section = _TEST_CSV_SECTION.format(test_csv_path=test_csv_path)
    else:
        test_csv_section = (
            "Test CSV: (none — only --input-csv will be passed; do not write "
            "test_data.json in this case)"
        )

    return CODEGEN_USER.format(
        csv_path=csv_path,
        data_profile=data_profile,
        target_column=target_column,
        task_type=task_type,
        target_mapping=target_mapping,
        selected_features=selected_features,
        instruction_template=instruction_template,
        input_format=input_format,
        output_format=output_format,
        data_cleaning_steps=data_cleaning_steps,
        output_data_dir=output_data_dir,
        test_csv_section=test_csv_section,
        error_context=error_context,
        user_feedback_context=user_feedback_context,
    )
