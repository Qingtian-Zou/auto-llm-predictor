"""Prompt templates for the write_prep_code node."""

CODEGEN_SYSTEM = """\
You are an expert Python programmer. You write clean, production-quality \
data preparation scripts. Your scripts MUST:

1. Be completely self-contained (no imports from custom packages)
2. Use only: pandas, numpy, json, sklearn, random, pathlib, argparse, sys
3. Read the CSV, apply the preparation plan, and produce output files
4. Handle edge cases (missing values, unexpected types) gracefully
5. Print progress messages to stdout

Output ONLY the complete Python script with no markdown fences, no explanation. \
Just the raw Python code.
"""

CODEGEN_USER = """\
Write a Python script that converts CSV data into LlamaFactory-compatible \
Alpaca JSON format for fine-tuning.

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
Main CSV: {csv_path}
{test_csv_section}

=== OUTPUT REQUIREMENTS ===
The script must save files to the directory: {output_data_dir}

{output_files_section}

The script should:
- Load the CSV from {csv_path}
- Apply data cleaning (drop missing targets, handle NaN features)
- For each row, create an example dict:
  * "instruction": the instruction template (same for every row)
  * "input": selected features formatted as readable text
  * "output": the target label (mapped using target_mapping)
- DO NOT split data — splitting is handled by a separate step
- Save the JSON files
- Print summary statistics (total samples, class distribution)

{error_context}
{user_feedback_context}
"""

_TEST_CSV_SECTION = """\
Test CSV:  {test_csv_path}
(A separate test set is provided. Process it using the same logic as the main CSV.)"""

_OUTPUT_SINGLE = """\
1. all_data.json — list of dicts with keys: "instruction", "input", "output"
2. dataset_info.json — LlamaFactory dataset registry:
   {{"train": {{"file_name": "all_data.json"}}}}"""

_OUTPUT_WITH_TEST = """\
1. all_data.json — list of dicts from the main CSV, keys: "instruction", "input", "output"
2. test_data.json — list of dicts from the test CSV, same format
3. dataset_info.json — LlamaFactory dataset registry:
   {{"train": {{"file_name": "all_data.json"}}, "test": {{"file_name": "test_data.json"}}}}"""

CODEGEN_RETRY_CONTEXT = """\
=== PREVIOUS ATTEMPT FAILED ===
The previous script failed with this error:
{error}

Fix the issues and generate a corrected script.
"""

CODEGEN_FEEDBACK_CONTEXT = """\
=== USER FEEDBACK (MUST FOLLOW) ===
The user reviewed a previous version and requested:
{feedback}

Make sure the generated script strictly follows these instructions.
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
    user_feedback: str = "",
) -> str:
    """Format the user prompt for the codegen node."""
    error_context = ""
    if previous_error:
        error_context = CODEGEN_RETRY_CONTEXT.format(error=previous_error)

    user_feedback_context = ""
    if user_feedback:
        user_feedback_context = CODEGEN_FEEDBACK_CONTEXT.format(feedback=user_feedback)

    # Build sections based on whether a test CSV is provided
    if test_csv_path:
        test_csv_section = _TEST_CSV_SECTION.format(test_csv_path=test_csv_path)
        output_files_section = _OUTPUT_WITH_TEST
    else:
        test_csv_section = "(No separate test CSV — splitting handled by a later step)"
        output_files_section = _OUTPUT_SINGLE

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
        output_files_section=output_files_section,
        error_context=error_context,
        user_feedback_context=user_feedback_context,
    )
