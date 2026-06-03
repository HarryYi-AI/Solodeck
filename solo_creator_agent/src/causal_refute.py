from __future__ import annotations

import numpy as np
import pandas as pd

from .causal_estimator import _design_matrix, _to_binary, stratified_regression_effect


def placebo_shuffle_test(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str] | None = None,
    n_iter: int = 40,
) -> dict:
    base = stratified_regression_effect(data, treatment_col, outcome_col, covariates or [], n_boot=40)
    rng = np.random.default_rng(7)
    effects = []
    df = data.copy()
    for _ in range(n_iter):
        shuffled = df[treatment_col].sample(frac=1, replace=False, random_state=int(rng.integers(0, 1_000_000))).to_numpy()
        tmp = df.copy()
        tmp[treatment_col] = shuffled
        effects.append(stratified_regression_effect(tmp, treatment_col, outcome_col, covariates or [], n_boot=10)["effect"])
    p_like = float(np.mean(np.abs(effects) >= abs(base["effect"]))) if effects else 1.0
    passed = p_like < 0.1
    return {
        "test": "placebo_shuffle",
        "base_effect": base["effect"],
        "placebo_mean": float(np.mean(effects)) if effects else 0.0,
        "p_like": p_like,
        "passed": passed,
        "confidence": "medium" if passed and len(data) >= 40 else "low",
        "interpretation": "如果随机打乱 treatment 后仍常出现类似效果，原结论不稳。",
    }


def subsample_stability_test(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str] | None = None,
    n_iter: int = 30,
    frac: float = 0.7,
) -> dict:
    rng = np.random.default_rng(11)
    effects = []
    for _ in range(n_iter):
        sample = data.sample(frac=frac, replace=False, random_state=int(rng.integers(0, 1_000_000)))
        effects.append(stratified_regression_effect(sample, treatment_col, outcome_col, covariates or [], n_boot=10)["effect"])
    signs = np.sign(effects)
    sign_stability = float(max(np.mean(signs >= 0), np.mean(signs <= 0))) if effects else 0.0
    return {
        "test": "subsample_stability",
        "effect_mean": float(np.mean(effects)) if effects else 0.0,
        "effect_std": float(np.std(effects)) if effects else 0.0,
        "sign_stability": sign_stability,
        "passed": sign_stability >= 0.75,
        "confidence": "medium" if sign_stability >= 0.75 and len(data) >= 40 else "low",
    }


def covariate_balance(data: pd.DataFrame, treatment_col: str, covariates: list[str], treatment_value=None) -> pd.DataFrame:
    df = data.dropna(subset=[treatment_col]).copy()
    t = _to_binary(df[treatment_col], treatment_value=treatment_value)
    x = _design_matrix(df, covariates)
    rows = []
    for col in x.columns:
        treated = x.loc[t.eq(1), col].astype(float)
        control = x.loc[t.eq(0), col].astype(float)
        pooled = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2) if len(treated) > 1 and len(control) > 1 else 0
        smd = float((treated.mean() - control.mean()) / pooled) if pooled else 0.0
        rows.append({
            "covariate": col,
            "treated_mean": float(treated.mean()) if len(treated) else 0.0,
            "control_mean": float(control.mean()) if len(control) else 0.0,
            "standardized_mean_diff": smd,
            "balanced": abs(smd) < 0.2,
        })
    return pd.DataFrame(rows).sort_values("standardized_mean_diff", key=lambda s: s.abs(), ascending=False)


def refute_suite(data: pd.DataFrame, treatment_col: str, outcome_col: str, covariates: list[str] | None = None) -> dict:
    covariates = covariates or []
    placebo = placebo_shuffle_test(data, treatment_col, outcome_col, covariates)
    subsample = subsample_stability_test(data, treatment_col, outcome_col, covariates)
    balance = covariate_balance(data, treatment_col, covariates) if covariates else pd.DataFrame()
    passed_count = int(placebo["passed"]) + int(subsample["passed"])
    return {
        "summary": {
            "passed": passed_count,
            "total": 2 + (0 if balance.empty else 1),
            "confidence": "medium" if passed_count >= 2 else "low",
            "finding_type": "反驳检验",
        },
        "placebo": placebo,
        "subsample": subsample,
        "balance": balance,
    }
