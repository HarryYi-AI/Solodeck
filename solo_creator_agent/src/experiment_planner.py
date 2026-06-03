from __future__ import annotations

import pandas as pd

from .causal_experiment import design_ab_test
from .causal_estimator import estimate_sample_size, stratified_regression_effect


def suggest_experiment_from_effect(
    contents: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str],
    platform: str | None = None,
    topic: str | None = None,
    language: str = "中文",
) -> dict:
    df = contents.copy()
    if platform:
        df = df[df["platform"].eq(platform)]
    if topic:
        df = df[df["topic"].eq(topic)]
    effect = stratified_regression_effect(df, treatment_col, outcome_col, covariates, n_boot=80)
    std = float(df[outcome_col].std()) if outcome_col in df.columns and len(df) > 1 else 0.0
    min_n = estimate_sample_size(effect.get("effect", 0), std)
    values = list(df[treatment_col].dropna().unique()) if treatment_col in df.columns else []
    treatment = str(values[0]) if values else "new_strategy"
    control = str(values[1]) if len(values) > 1 else "current_strategy"
    plan = design_ab_test(
        "验证内容策略增量" if language == "中文" else "Validate incremental content strategy effect",
        platform or "selected platform",
        topic or "selected topic",
        treatment,
        control,
        outcome_col,
        language=language,
    )
    plan.update({
        "estimated_effect": effect.get("effect", 0),
        "effect_ci": [effect.get("ci_low", 0), effect.get("ci_high", 0)],
        "suggested_min_total_sample": min_n,
        "finding_type": "实验建议",
        "data_confidence": effect.get("confidence", "low"),
    })
    return plan


def weekly_experiment_plan(contents: pd.DataFrame, language: str = "中文") -> list[dict]:
    candidates = [
        ("title_style", "views", ["platform", "topic", "followers_before", "ad_spend"]),
        ("cover_style", "completion_rate", ["platform", "topic", "duration_sec"]),
        ("platform", "consultations", ["topic", "followers_before", "ad_spend"]),
        ("content_type", "revenue", ["platform", "topic", "production_hours", "ad_spend"]),
    ]
    plans = []
    for treatment, outcome, covariates in candidates:
        if treatment in contents.columns and outcome in contents.columns:
            plans.append(suggest_experiment_from_effect(contents, treatment, outcome, covariates, language=language))
    return plans
