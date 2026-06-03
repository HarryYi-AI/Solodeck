from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .causal_estimator import did_effect, estimate_sample_size, iptw_effect, propensity_score_matching, stratified_regression_effect
from .causal_refute import refute_suite
from .experiment_planner import suggest_experiment_from_effect
from .variable_mapper import map_variables


def identify_estimand(question: str, data: pd.DataFrame, variables: dict[str, Any] | None = None, language: str = "中文") -> dict:
    variables = variables or {}
    treatment = variables.get("treatment")
    outcome = variables.get("outcome")
    covariates = variables.get("confounders") or variables.get("covariates") or []
    variable_map = map_variables(data, use_llm=False, language=language)
    warnings = []
    if not treatment:
        warnings.append("缺少明确策略变量 treatment。" if language == "中文" else "Missing treatment variable.")
    if not outcome:
        warnings.append("缺少明确结果变量 outcome。" if language == "中文" else "Missing outcome variable.")
    if treatment and treatment in data.columns and data[treatment].nunique(dropna=True) < 2:
        warnings.append("策略变量只有一个取值，无法形成对照。" if language == "中文" else "Treatment has only one value; no comparison group.")
    if outcome and outcome in data.columns and not pd.api.types.is_numeric_dtype(data[outcome]):
        warnings.append("结果变量不是数值型，需先转成可比较指标。" if language == "中文" else "Outcome is not numeric.")
    return {
        "question": question,
        "treatment": treatment,
        "outcome": outcome,
        "confounders": covariates,
        "estimand_type": "backdoor_adjustment" if treatment and outcome and covariates else "exploratory_comparison",
        "variable_map": variable_map,
        "warnings": warnings,
        "finding_type": "识别性检查" if language == "中文" else "Identification check",
    }


def estimate_effect_IPW(data: pd.DataFrame, treatment: str, outcome: str, confounders: list[str], treatment_value=None) -> dict:
    result = iptw_effect(data, treatment, outcome, confounders, treatment_value=treatment_value)
    result["finding_type"] = "策略增量估计"
    return result


def estimate_effect_PSM(data: pd.DataFrame, treatment: str, outcome: str, confounders: list[str], treatment_value=None) -> dict:
    result = propensity_score_matching(data, treatment, outcome, confounders, treatment_value=treatment_value)
    result["finding_type"] = "策略增量估计"
    return result


def estimate_effect_DR(data: pd.DataFrame, treatment: str, outcome: str, confounders: list[str], treatment_value=None) -> dict:
    regression = stratified_regression_effect(data, treatment, outcome, confounders, treatment_value=treatment_value, n_boot=120)
    ipw = iptw_effect(data, treatment, outcome, confounders, treatment_value=treatment_value)
    effects = [item.get("effect", 0.0) for item in [regression, ipw] if item.get("confidence") != "insufficient"]
    effect = float(np.mean(effects)) if effects else 0.0
    spread = float(np.std(effects)) if len(effects) > 1 else 0.0
    return {
        "method": "doubly_robust",
        "effect": effect,
        "ci_low": effect - 1.96 * spread,
        "ci_high": effect + 1.96 * spread,
        "sample_size": regression.get("sample_size", len(data)),
        "confidence": "low" if len(data) < 80 else "medium",
        "components": {"regression": regression, "iptw": ipw},
        "finding_type": "策略增量估计",
        "warning": "建议先小范围验证，再放大投入。",
    }


def estimate_effect_DML(data: pd.DataFrame, treatment: str, outcome: str, confounders: list[str], treatment_value=None) -> dict:
    fallback = estimate_effect_DR(data, treatment, outcome, confounders, treatment_value=treatment_value)
    fallback["method"] = "dml"
    fallback["warning"] = "建议先小范围验证，再放大投入。"
    return fallback


def time_series_causal(data: pd.DataFrame, treatment: str, outcome: str, time_col: str = "publish_time", post_value=None, treatment_value=None) -> dict:
    result = did_effect(data, treatment, outcome, time_col, post_value=post_value, treatment_value=treatment_value)
    result["finding_type"] = "时间序列增量估计"
    if result.get("confidence") == "low":
        result["warning"] = "建议延长观察周期。"
    return result


def CATE_estimation(data: pd.DataFrame, treatment: str, outcome: str, confounders: list[str], subgroup_col: str = "platform") -> pd.DataFrame:
    if subgroup_col not in data.columns:
        return pd.DataFrame()
    rows = []
    for value, group in data.groupby(subgroup_col):
        if len(group) < 6:
            continue
        result = stratified_regression_effect(group, treatment, outcome, [c for c in confounders if c != subgroup_col], n_boot=40)
        rows.append({
            "subgroup": value,
            "method": result["method"],
            "effect": result["effect"],
            "ci_low": result.get("ci_low", 0),
            "ci_high": result.get("ci_high", 0),
            "sample_size": result["sample_size"],
            "confidence": result["confidence"],
        })
    return pd.DataFrame(rows).sort_values("effect", ascending=False) if rows else pd.DataFrame()


def run_causal_closed_loop(
    data: pd.DataFrame,
    question: str,
    treatment: str,
    outcome: str,
    confounders: list[str],
    language: str = "中文",
) -> dict:
    estimand = identify_estimand(question, data, {"treatment": treatment, "outcome": outcome, "confounders": confounders}, language=language)
    regression = stratified_regression_effect(data, treatment, outcome, confounders, n_boot=120)
    psm = estimate_effect_PSM(data, treatment, outcome, confounders)
    ipw = estimate_effect_IPW(data, treatment, outcome, confounders)
    dr = estimate_effect_DR(data, treatment, outcome, confounders)
    refutes = refute_suite(data, treatment, outcome, confounders)
    cate = CATE_estimation(data, treatment, outcome, confounders)
    std = float(data[outcome].std()) if outcome in data.columns and len(data) > 1 else 0.0
    sample_size = estimate_sample_size(regression.get("effect", 0), std)
    plan = suggest_experiment_from_effect(data, treatment, outcome, confounders, language=language)
    plan["suggested_min_total_sample"] = max(plan.get("suggested_min_total_sample", 0), sample_size)
    return {
        "estimand": estimand,
        "estimates": {
            "regression": regression,
            "psm": psm,
            "ipw": ipw,
            "dr": dr,
        },
        "refutation": refutes,
        "cate": cate,
        "experiment_plan": plan,
        "conclusion_type": "策略增量估计" if language == "中文" else "Strategy lift estimate",
    }


def run_dowhy_analysis(*args, **kwargs) -> dict:
    return {"status": "not_installed", "fallback": "Use run_causal_closed_loop. DoWhy can be enabled in production dependencies."}


def run_causalpy_analysis(*args, **kwargs) -> dict:
    return {"status": "not_installed", "fallback": "Use time_series_causal. CausalPy can be enabled for DID/ITS/Synthetic Control."}


def run_pymc_marketing_mmm(*args, **kwargs) -> dict:
    return {"status": "not_installed", "fallback": "Use platform strategy and revenue attribution until PyMC-Marketing is installed."}
