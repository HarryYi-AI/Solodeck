from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .causal_estimator import _bootstrap_ci, _design_matrix, _to_binary, iptw_effect, propensity_score_matching


def fixed_effect_estimate(df: pd.DataFrame, treatment_col: str, outcome_col: str, fixed_effect_cols: list[str], covariates: list[str] | None = None) -> dict:
    data = df.dropna(subset=[treatment_col, outcome_col]).copy()
    if data.empty or data[treatment_col].nunique() < 2:
        return {"method": "fixed_effect_ols", "effect_estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "sample_size": len(data), "warnings": ["缺少对照组或样本为空。"]}
    t = _to_binary(data[treatment_col]).rename("treatment")
    x = pd.concat([t, _design_matrix(data, (fixed_effect_cols or []) + (covariates or []))], axis=1).astype(float)
    y = data[outcome_col].astype(float)
    model = LinearRegression().fit(x, y)
    effect = float(model.coef_[0])
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(120):
        sample = data.sample(n=len(data), replace=True, random_state=int(rng.integers(0, 1_000_000)))
        bt = _to_binary(sample[treatment_col]).rename("treatment")
        if bt.nunique() < 2:
            continue
        bx = pd.concat([bt, _design_matrix(sample, (fixed_effect_cols or []) + (covariates or []))], axis=1).astype(float)
        boots.append(float(LinearRegression().fit(bx, sample[outcome_col].astype(float)).coef_[0]))
    ci_low, ci_high = _bootstrap_ci(boots)
    return {"method": "fixed_effect_ols", "effect_estimate": effect, "ci_low": ci_low, "ci_high": ci_high, "sample_size": len(data), "warnings": ["固定效应估计仍依赖已观测混杂变量是否充分。"]}


def estimate_incremental_effect(df: pd.DataFrame, treatment_col: str, outcome_col: str, covariates: list[str], group_fixed_effect: list[str] | None = None, time_col: str | None = None) -> dict:
    fixed = fixed_effect_estimate(df, treatment_col, outcome_col, group_fixed_effect or [], covariates)
    psm = propensity_score_matching(df, treatment_col, outcome_col, covariates)
    iptw = iptw_effect(df, treatment_col, outcome_col, covariates)
    return {
        "effect_estimate": fixed["effect_estimate"],
        "ci_low": fixed["ci_low"],
        "ci_high": fixed["ci_high"],
        "method": "fixed_effect_ols_with_psm_iptw",
        "sample_size": fixed["sample_size"],
        "warnings": fixed["warnings"] + (["建议先小范围验证。"] if fixed["sample_size"] < 30 else []),
        "robustness_checks": {"psm_effect": psm.get("effect", 0), "iptw_effect": iptw.get("effect", 0)},
    }


def paired_variant_effect(df: pd.DataFrame, pair_id_col: str, treatment_col: str, outcome_col: str) -> dict:
    rows = []
    for pid, group in df.groupby(pair_id_col):
        if group[treatment_col].nunique() < 2:
            continue
        t = _to_binary(group[treatment_col])
        treated = group.loc[t.eq(1), outcome_col].astype(float).mean()
        control = group.loc[t.eq(0), outcome_col].astype(float).mean()
        rows.append(float(treated - control))
    ci_low, ci_high = _bootstrap_ci(rows)
    return {"method": "paired_difference", "mean_difference": float(np.mean(rows)) if rows else 0.0, "ci_low": ci_low, "ci_high": ci_high, "pair_count": len(rows), "warning": "建议继续收集配对样本。" if len(rows) < 8 else ""}


def estimate_feature_upgrade_effect(products_df: pd.DataFrame, beta_tests_df: pd.DataFrame, feature_name: str, outcome: str) -> dict:
    if not beta_tests_df.empty and feature_name:
        df = beta_tests_df[beta_tests_df["feature_name"].fillna("").astype(str).str.contains(feature_name, case=False, regex=False)].copy()
        if not df.empty and outcome in df.columns:
            return paired_variant_effect(df.assign(pair_id=df.get("user_segment", "all")), "pair_id", "test_group", outcome)
    if products_df.empty:
        return {"method": "no_data", "mean_difference": 0.0, "warning": "没有产品或内测数据。"}
    df = products_df.copy()
    if "is_new_version" in df.columns and outcome in df.columns:
        return estimate_incremental_effect(df, "is_new_version", outcome, ["price", "cost"], ["series_id", "platform"])
    return {"method": "not_available", "mean_difference": 0.0, "warning": "缺少可用结果指标。"}


def generate_incremental_insight(result: dict, language: str = "中文") -> str:
    effect = result.get("effect_estimate", result.get("mean_difference", 0.0))
    low = result.get("ci_low", 0.0)
    high = result.get("ci_high", 0.0)
    n = result.get("sample_size", result.get("pair_count", 0))
    if language == "中文":
        direction = "正向" if effect > 0 else "负向或不明显"
        return f"策略增量方向为{direction}，估计值 {effect:.2f}，区间 [{low:.2f}, {high:.2f}]，样本 {n}。建议先小范围验证。"
    direction = "positive" if effect > 0 else "negative or unclear"
    return f"This is an exploratory quasi-experimental estimate. Direction: {direction}; effect {effect:.2f}, interval [{low:.2f}, {high:.2f}], sample {n}. Validate before scaling."
