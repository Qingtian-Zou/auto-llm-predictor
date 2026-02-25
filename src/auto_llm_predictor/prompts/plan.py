"""Prompt templates for the plan_preparation node."""

PLAN_SYSTEM = """\
You are a data engineering expert specializing in preparing tabular data for \
LLM fine-tuning. Given a dataset profile and target analysis, you create a \
detailed data preparation plan.

The goal is to convert a CSV file into LlamaFactory-compatible Alpaca JSON format:
{
  "instruction": "<system-level task description>",
  "input": "<patient/sample data as readable text>",
  "output": "<predicted label>"
}

Respond ONLY with valid JSON (no markdown fences) in this exact schema:
{
  "selected_features": ["col1", "col2", ...],
  "dropped_features": ["colX", "colY", ...],
  "feature_selection_reason": "<why these features were chosen and others dropped>",
  "instruction_template": "<the instruction text to use in every example>",
  "input_format": "<description of how to format each row's features as text>",
  "output_format": "<description of how to format the target label>",
  "target_mapping": {"<raw_value>": "<label>"},
  "test_ratio": 0.2,
  "balance_strategy": "none | oversample | undersample",
  "data_cleaning_steps": ["step1", "step2", ...],
  "reasoning": "<brief explanation>"
}
"""

PLAN_USER = """\
Here is the dataset profile:

{data_profile}

Target analysis:
- Target column: {target_column}
- Task type: {task_type}
- Target mapping: {target_mapping}
- Class distribution: {class_distribution}
- Data quality notes: {data_quality_notes}

The dataset has {n_columns} columns total.

{feature_instructions}

Create a detailed data preparation plan.
"""


def format_plan_prompt(
    data_profile: str,
    target_column: str,
    task_type: str,
    target_mapping: dict,
    class_distribution: dict | None = None,
    data_quality_notes: str = "",
    n_columns: int = 0,
    pre_selected_features: list[str] | None = None,
) -> str:
    """Format the user prompt for the plan node."""
    import json
    
    if pre_selected_features:
        features_json = json.dumps(pre_selected_features)
        # Note: If ensemble was used, nodes/plan.py has the dropped_features natively,
        # but the LLM still needs to fulfill the JSON schema. We tell it to leave dropped_features empty
        # because the pipeline handles merging it later if pre-selected.
        feature_instructions = (
            f"IMPORTANT: Features have already been pre-selected by an ensemble feature "
            f"selection algorithm (variance filtering + correlation + mutual information "
            f"+ Random Forest importance). You MUST use these exact features in the "
            f"'selected_features' field of your response:\n"
            f"{features_json}\n\n"
            f"For 'dropped_features', please output an empty list `[]` as the system handles it.\n"
            f"Do NOT add or remove features. Focus only on deciding the instruction "
            f"template, input format, output format, test ratio, balance strategy, "
            f"and data cleaning steps."
        )
    else:
        feature_instructions = (
            "If there are many feature columns (e.g. gene expression, omics data), "
            "select the most relevant ones and explicitly list the ones you omit in 'dropped_features'. "
            "For datasets with fewer columns (< 50), include all non-target columns as features "
            "and leave 'dropped_features' empty."
        )

    return PLAN_USER.format(
        data_profile=data_profile,
        target_column=target_column,
        task_type=task_type,
        target_mapping=target_mapping,
        class_distribution=class_distribution or {},
        data_quality_notes=data_quality_notes,
        n_columns=n_columns,
        feature_instructions=feature_instructions,
    )
