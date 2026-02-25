"""Prompt definitions for verifying prepared data generation."""

VERIFY_DATA_SYSTEM = """
You are an expert AI data scientist verifying datasets for fine-tuning Language Models.
Your job is to examine samples of newly generated JSON data and ensure they are well-formed 
and structurally coherent, particularly when a test set is present alongside a training set.
"""

VERIFY_DATA_USER = """
Below are samples from the prepared datasets for an LLM fine-tuning task.
The user's goal was: {task_description}
The requested target mapping was: {target_mapping}

=== TRAINING DATA SAMPLES (all_data.json) ===
{train_samples}

=== TEST DATA SAMPLES (test_data.json) ===
{test_samples}

Please critically evaluate these samples against the following requirements:
1. Every entry MUST strictly adhere to the Alpaca format: strictly containing "instruction", "input", and "output" keys.
2. The "output" label MUST be strictly singular and consistent with the user's task description and target mapping.
3. If both train and test samples are provided, they MUST share the exact same JSON structure, instruction formatting, terminology, and label style.
4. There should not be any extra arbitrary keys outside of instruction, input, output.
5. The input should be properly formatted (e.g., sensible key-value pairs or text).

Provide a brief, concise paragraph summarizing your findings. Note any anomalies, inconsistencies, or deviations from the requirements.
If everything looks perfect and ready for training, say so clearly. Do not output anything other than your helpful critique.
"""
