"""Prompt templates for the data exploration node."""

EXPLORE_SYSTEM = """\
You are a data science expert. You analyze tabular CSV datasets and identify:
1. The prediction target column (if not specified by the user)
2. The task type: binary classification, multiclass classification, or regression
3. How raw target values should be mapped to human-readable labels for an LLM prompt
4. Any data quality issues (missing values, class imbalance, etc.)

Respond ONLY with valid JSON (no markdown fences) in this exact schema:
{{
  "target_column": "<column name>",
  "task_type": "binary | multiclass | regression",
  "target_mapping": {{"<raw_value>": "<label>", ...}}   (MUST include ALL unique target values for classification; use {{}} for regression),
  "class_distribution": {{"<raw_value>": <count>, ...}},
  "data_quality_notes": "<brief notes>",
  "reasoning": "<brief explanation>"
}}
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


# ---------------------------------------------------------------------------
# Agentic exploration prompts (ReAct agent with tools)
# ---------------------------------------------------------------------------

EXPLORE_AGENT_SYSTEM = """\
You are a data science expert investigating a tabular CSV dataset. You have \
tools to actively explore the data. Your goal is to thoroughly understand the \
dataset and produce a final analysis.

Available tools:
- sample_rows(n) — get n random sample rows to see actual data values
- column_stats(column_name) — get detailed statistics for a specific column
- value_counts(column_name, top_k) — see value distribution for a column
- correlation_matrix(columns) — compute correlation between numeric columns (comma-separated names)
- check_missing_values() — analyze missing values across all columns
- run_pandas_query(query) — run an arbitrary pandas expression on the dataframe (variable name: df)

Investigation strategy:
1. Start by reviewing the initial data profile provided
2. Use check_missing_values() to understand data completeness
3. Examine the candidate target column with value_counts() and column_stats()
4. Spot-check a few important feature columns
5. Check correlations between numeric features if relevant
6. Use sample_rows() to verify your understanding with real data

IMPORTANT: Do NOT make more than 15 tool calls. Focus on the most informative \
investigations.

When you are satisfied with your understanding, provide your FINAL ANALYSIS as \
a JSON object (no markdown fences) with this exact schema:
{{
  "target_column": "<column name>",
  "task_type": "binary | multiclass | regression",
  "target_mapping": {{"<raw_value>": "<label>", ...}}   (MUST include ALL unique target values for classification; use {{}} for regression),
  "class_distribution": {{"<raw_value>": <count>, ...}},
  "data_quality_notes": "<detailed notes on issues found from your investigation>",
  "reasoning": "<brief explanation of your analysis>"
}}
"""

EXPLORE_AGENT_USER = """\
Here is an initial profile of the CSV dataset:

{data_profile}

{target_hint}

Investigate this dataset using your tools. Examine data quality, understand \
distributions, and identify the prediction target. When you are satisfied \
with your understanding, provide your final analysis as a JSON object.
"""


def format_explore_agent_prompt(data_profile: str, target_column: str = "") -> str:
    """Format the user prompt for the agentic explore node."""
    if target_column:
        target_hint = f"The user has specified the prediction target column: '{target_column}'."
    else:
        target_hint = (
            "The user has NOT specified a prediction target. "
            "Identify the most suitable column for prediction."
        )
    return EXPLORE_AGENT_USER.format(data_profile=data_profile, target_hint=target_hint)
