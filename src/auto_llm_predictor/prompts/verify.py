"""Prompt definitions for verifying prepared data generation."""

VERIFY_DATA_SYSTEM = """\
You are an expert AI data scientist verifying datasets for fine-tuning Language Models.
Your job is to examine samples of newly generated JSON data and ensure they are well-formed \
and structurally coherent.
"""

VERIFY_DATA_USER = """
Below are samples from the prepared datasets for an LLM fine-tuning task.

=== PREPARATION PLAN ===
{task_description}

=== USER-CONFIRMED SETTINGS ===
Target column:    {target_column}
Target mapping:   {target_mapping}
Selected features ({num_selected}): {selected_features}
Dropped features:  {dropped_features}

=== TRAINING DATA SAMPLES (all_data.json) ===
{train_samples}

=== TEST DATA SAMPLES (test_data.json) ===
{test_samples}

Please critically evaluate these samples against the following requirements:
1. Every entry MUST strictly adhere to the Alpaca format: strictly containing "instruction", "input", and "output" keys.
2. Each example must have exactly one output label (not multi-label), and it must match one of the labels defined in the target mapping above.
3. The "input" field MUST only reference the selected features listed above — dropped features must NOT appear.
4. If both train and test samples are provided, they MUST share the exact same JSON structure, instruction formatting, terminology, and label style.
5. There should not be any extra arbitrary keys outside of instruction, input, output.
6. The input should be properly formatted (e.g., sensible key-value pairs or text).

Provide a brief, concise paragraph summarizing your findings. Note any anomalies, inconsistencies, or deviations from the requirements.
If everything looks perfect and ready for training, say so clearly. Do not output anything other than your helpful critique.
"""
