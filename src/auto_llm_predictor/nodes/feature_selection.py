"""Node: select_features — Ensemble feature selection for high-dimensional data.

Implements the methodology from ClinicalOmicsDB:
  1. Variance filtering — remove near-constant features
  2. Correlation ranking — Pearson correlation with target
  3. Mutual information ranking — non-linear dependency with target
  4. Random Forest importance — ensemble tree-based importance
  5. Average-rank aggregation — combine all methods by mean rank

For low-dimensional datasets (< 50 columns), this node is skipped
and feature selection is delegated to the LLM in plan_preparation.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from auto_llm_predictor.state import PipelineState

logger = logging.getLogger(__name__)

# Threshold: datasets with >= this many feature columns trigger ensemble selection
HIGH_DIM_THRESHOLD = 50


# ── Individual ranking methods ────────────────────────────────────


def _variance_filter(X: pd.DataFrame, threshold: float = 0.01) -> list[str]:
    """Remove features with variance below *threshold*."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
    return X.columns[selector.get_support()].tolist()


def _correlation_ranking(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Rank features by absolute Pearson correlation with target."""
    y_enc = y.copy()
    if y_enc.dtype == "object":
        le = LabelEncoder()
        y_enc = pd.Series(le.fit_transform(y_enc.astype(str)), index=y.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = X.corrwith(y_enc).abs().fillna(0)
    return corr.sort_values(ascending=False)


def _mutual_info_ranking(
    X: pd.DataFrame, y: pd.Series, task_type: str
) -> pd.Series:
    """Rank features by mutual information with target."""
    y_enc = y.copy()
    if y_enc.dtype == "object" or task_type != "regression":
        le = LabelEncoder()
        y_enc = le.fit_transform(y_enc.astype(str))

    mi_func = mutual_info_regression if task_type == "regression" else mutual_info_classif
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = mi_func(X, y_enc, random_state=42, n_jobs=-1)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)


def _random_forest_ranking(
    X: pd.DataFrame, y: pd.Series, task_type: str, n_estimators: int = 100
) -> pd.Series:
    """Rank features by Random Forest importance."""
    y_enc = y.copy()
    if y_enc.dtype == "object" or task_type != "regression":
        le = LabelEncoder()
        y_enc = le.fit_transform(y_enc.astype(str))
        rf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )
    else:
        rf = RandomForestRegressor(
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf.fit(X, y_enc)

    importances = rf.feature_importances_
    return pd.Series(importances, index=X.columns).sort_values(ascending=False)


# ── Ensemble aggregation ──────────────────────────────────────────


def _ensemble_select(
    rankings: dict[str, pd.Series], n_features: int
) -> list[str]:
    """Select top features using average-rank aggregation across methods."""
    all_features: set[str] = set()
    for ranking in rankings.values():
        all_features.update(ranking.index.tolist())

    feature_ranks: dict[str, float] = {}
    for feature in all_features:
        ranks = []
        for ranking in rankings.values():
            if feature in ranking.index:
                rank_pos = list(ranking.index).index(feature)
                ranks.append(rank_pos)
        feature_ranks[feature] = float(np.mean(ranks))

    sorted_features = sorted(feature_ranks.items(), key=lambda x: x[1])
    return [f[0] for f in sorted_features[:n_features]]


# ── Main node ─────────────────────────────────────────────────────


def select_features(state: PipelineState) -> dict:
    """Run ensemble feature selection on high-dimensional datasets.

    Writes: selected_features, feature_rankings_path, messages
    """
    csv_path = state["csv_path"]
    target_column = state["target_column"]
    task_type = state.get("task_type", "binary")
    output_dir = Path(state["output_dir"])

    df = pd.read_csv(csv_path, low_memory=False)

    # Identify numeric feature columns (exclude target and low-cardinality metadata)
    exclude_cols = {target_column}
    # Auto-detect likely metadata/ID columns
    for col in df.columns:
        if df[col].dtype == "object" and df[col].nunique() > 0.5 * len(df):
            exclude_cols.add(col)  # Likely ID column
        if col.lower() in (
            "id", "sample", "patient_id", "sample_id", "index",
            "study", "source", "source_file", "trial_id",
        ):
            exclude_cols.add(col)

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Separate numeric features for statistical methods
    numeric_features = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    logger.info(
        "Feature selection: %d total columns, %d feature candidates, %d numeric",
        len(df.columns), len(feature_cols), len(numeric_features),
    )

    if len(numeric_features) < HIGH_DIM_THRESHOLD:
        # Low-dimensional: pass all features, let the LLM decide in plan_preparation
        logger.info("Low-dimensional data (%d features). Skipping ensemble selection.", len(numeric_features))
        return {
            "selected_features": feature_cols,
            "messages": [
                HumanMessage(
                    content=f"[select_features] Low-dimensional data ({len(feature_cols)} features). "
                    "All features passed to planning stage."
                ),
            ],
        }

    # ── High-dimensional: run ensemble feature selection ──────
    n_features = min(100, len(numeric_features) // 2)
    logger.info("High-dimensional data. Running ensemble selection for top %d features.", n_features)

    # Drop rows with missing target
    df = df.dropna(subset=[target_column])

    X = df[numeric_features].copy()
    y = df[target_column].copy()

    # Fill missing values with median
    X = X.fillna(X.median())

    # Step 1: Variance filtering
    logger.info("Step 1/4: Variance filtering...")
    high_var = _variance_filter(X)
    logger.info("  %d → %d features after variance filter", len(numeric_features), len(high_var))
    X_filtered = X[high_var]

    # Step 2–4: Ranking methods
    rankings: dict[str, pd.Series] = {}

    logger.info("Step 2/4: Correlation ranking...")
    rankings["correlation"] = _correlation_ranking(X_filtered, y)

    logger.info("Step 3/4: Mutual information ranking...")
    rankings["mutual_info"] = _mutual_info_ranking(X_filtered, y, task_type)

    logger.info("Step 4/4: Random Forest importance...")
    rankings["random_forest"] = _random_forest_ranking(X_filtered, y, task_type)

    # Step 5: Ensemble aggregation
    selected = _ensemble_select(rankings, n_features)
    logger.info("Selected %d features via ensemble ranking.", len(selected))

    # Also include any non-numeric feature columns (e.g. categorical)
    non_numeric_features = [c for c in feature_cols if c not in numeric_features]
    all_selected = non_numeric_features + selected
    dropped_features = [c for c in feature_cols if c not in all_selected]

    # Save detailed rankings for transparency
    rankings_dir = output_dir / "feature_selection"
    rankings_dir.mkdir(parents=True, exist_ok=True)

    # Feature list
    features_path = rankings_dir / "selected_features.txt"
    features_path.write_text("\n".join(all_selected))

    # Detailed rankings CSV
    rankings_data = {"feature": selected}
    for method, ranking in rankings.items():
        rankings_data[f"{method}_rank"] = [
            list(ranking.index).index(f) if f in ranking.index else -1
            for f in selected
        ]
    rankings_df = pd.DataFrame(rankings_data)
    rankings_csv_path = rankings_dir / "feature_rankings.csv"
    rankings_df.to_csv(rankings_csv_path, index=False)

    logger.info("Saved feature rankings to %s", rankings_csv_path)

    # Build summary for LLM context
    top_5 = selected[:5]
    method_summary = ", ".join(
        f"{m}: top={list(r.index)[0]}" for m, r in rankings.items()
    )

    return {
        "selected_features": all_selected,
        "dropped_features": dropped_features,
        "messages": [
            HumanMessage(
                content=f"[select_features] Ensemble selection: {len(numeric_features)} candidates "
                f"→ {len(high_var)} after variance filter → top {len(selected)} by avg rank. "
                f"Dropped {len(dropped_features)} features. "
                f"Methods: {method_summary}. "
                f"Top 5: {top_5}. "
                f"Rankings saved to {rankings_csv_path}"
            ),
        ],
    }


def check_feature_complexity(state: PipelineState) -> str:
    """Conditional edge: decide whether to run ensemble feature selection.

    Returns 'select_features' for high-dimensional data,
    'plan_preparation' for low-dimensional data.
    """
    csv_path = state["csv_path"]
    df = pd.read_csv(csv_path, nrows=0, low_memory=False)
    n_cols = len(df.columns)

    if n_cols >= HIGH_DIM_THRESHOLD:
        return "select_features"
    return "plan_preparation"
