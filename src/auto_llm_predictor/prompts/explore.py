"""Prompt templates for the data exploration node."""

EXPLORE_SYSTEM = """\
You are a data science expert. You analyze tabular CSV datasets and identify:
1. The prediction target column (if not specified by the user)
2. The task type: binary classification, multiclass classification, or regression
3. How raw target values should be mapped to human-readable labels for an LLM prompt
4. Any data quality issues (missing values, class imbalance, etc.)

Respond ONLY with valid JSON (no markdown fences) in this exact schema:
{
  "target_column": "<column name>",
  "task_type": "binary | multiclass | regression",
  "target_mapping": {"<raw_value>": "<label>", ...},
  "class_distribution": {"<label>": <count>, ...},
  "data_quality_notes": "<brief notes>",
  "reasoning": "<brief explanation>"
}
"""

EXPLORE_USER = """\
Here is a profile of the CSV dataset:

{data_profile}

{target_hint}

Analyze this dataset and identify the prediction target, task type, and target mapping.
"""


def format_explore_prompt(data_profile: str, target_column: str = "") -> str:
    """Format the user prompt for the explore node."""
    if target_column:
        target_hint = f"The user has specified the prediction target column: '{target_column}'."
    else:
        target_hint = (
            "The user has NOT specified a prediction target. "
            "Identify the most suitable column for prediction."
        )
    return EXPLORE_USER.format(data_profile=data_profile, target_hint=target_hint)
