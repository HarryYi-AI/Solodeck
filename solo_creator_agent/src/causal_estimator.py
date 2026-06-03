from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors


def _to_binary(series: pd.Series, treatment_value=None) -> pd.Series:
    if treatment_value is not None:
        return series.eq(treatment_value).astype(int)
    if series.dtype == bool:
        return series.astype(int)
    values = list(series.dropna().unique())
    if len(values) == 2:
        return series.eq(values[0]).astype(int)
    return pd.get_dummies(series, dummy_na=False).iloc[:, 0].astype(int)


def _design_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [c for c in columns if c in df.columns]
    if not available:
        return pd.DataFrame(index=df.index)
    return pd.get_dummies(df[available], drop_first=True).fillna(0)


def _bootstrap_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.percentile(values, 100 * alpha / 2)), float(np.percentile(values, 100 * (1 - alpha / 2)))


def stratified_regression_effect(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str] | None = None,
    treatment_value=None,
    n_boot: int = 300,
) -> dict:
    df = data.dropna(subset=[treatment_col, outcome_col]).copy()
    if df.empty or df[treatment_col].nunique() < 2:
        return {"method": "stratified_regression", "effect": 0.0, "ci_low": 0.0, "ci_high": 0.0, "sample_size": len(df), "confidence": "insufficient"}
    t = _to_binary(df[treatment_col], treatment_value=treatment_value)
    x = pd.concat([t.rename("treatment"), _design_matrix(df, covariates or [])], axis=1).astype(float)
    y = df[outcome_col].astype(float)
    model = LinearRegression().fit(x, y)
    effect = float(model.coef_[0])
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(df.index.to_numpy(), size=len(df), replace=True)
        boot_df = df.loc[idx].reset_index(drop=True)
        boot_t = _to_binary(boot_df[treatment_col], treatment_value=treatment_value)
        if boot_t.nunique() < 2:
            continue
        boot_x = pd.concat([boot_t.rename("treatment"), _design_matrix(boot_df, covariates or [])], axis=1).astype(float)
        boot_y = boot_df[outcome_col].astype(float)
        boots.append(float(LinearRegression().fit(boot_x, boot_y).coef_[0]))
    ci_low, ci_high = _bootstrap_ci(boots)
    return {
        "method": "stratified_regression",
        "treatment": treatment_col,
        "outcome": outcome_col,
        "effect": effect,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "sample_size": int(len(df)),
        "confidence": "low" if len(df) < 30 else "medium",
        "interpretation": "估计因果效应，仍依赖已控制混杂变量是否充分。",
    }


def propensity_score_matching(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str],
    treatment_value=None,
) -> dict:
    df = data.dropna(subset=[treatment_col, outcome_col]).copy()
    t = _to_binary(df[treatment_col], treatment_value=treatment_value)
    if df.empty or t.nunique() < 2:
        return {"method": "psm", "effect": 0.0, "sample_size": len(df), "confidence": "insufficient"}
    x = _design_matrix(df, covariates).astype(float)
    if x.empty:
        return {"method": "psm", "effect": 0.0, "sample_size": len(df), "confidence": "insufficient", "warning": "缺少协变量，无法估计倾向得分。"}
    ps = LogisticRegression(max_iter=1000).fit(x, t).predict_proba(x)[:, 1]
    treated_idx = np.where(t.to_numpy() == 1)[0]
    control_idx = np.where(t.to_numpy() == 0)[0]
    nn = NearestNeighbors(n_neighbors=1).fit(ps[control_idx].reshape(-1, 1))
    matched_control = control_idx[nn.kneighbors(ps[treated_idx].reshape(-1, 1), return_distance=False).ravel()]
    y = df[outcome_col].astype(float).to_numpy()
    effect = float(np.mean(y[treated_idx] - y[matched_control])) if len(treated_idx) else 0.0
    return {
        "method": "psm",
        "treatment": treatment_col,
        "outcome": outcome_col,
        "effect": effect,
        "sample_size": int(len(df)),
        "treated_size": int(len(treated_idx)),
        "control_size": int(len(control_idx)),
        "confidence": "low" if len(df) < 50 else "medium",
        "interpretation": "倾向得分匹配估计，用于判断策略增量。",
    }


def iptw_effect(data: pd.DataFrame, treatment_col: str, outcome_col: str, covariates: list[str], treatment_value=None) -> dict:
    df = data.dropna(subset=[treatment_col, outcome_col]).copy()
    t = _to_binary(df[treatment_col], treatment_value=treatment_value)
    if df.empty or t.nunique() < 2:
        return {"method": "iptw", "effect": 0.0, "sample_size": len(df), "confidence": "insufficient"}
    x = _design_matrix(df, covariates).astype(float)
    if x.empty:
        return {"method": "iptw", "effect": 0.0, "sample_size": len(df), "confidence": "insufficient", "warning": "缺少协变量，无法估计权重。"}
    ps = np.clip(LogisticRegression(max_iter=1000).fit(x, t).predict_proba(x)[:, 1], 0.02, 0.98)
    y = df[outcome_col].astype(float).to_numpy()
    t_arr = t.to_numpy()
    w_t = t_arr / ps
    w_c = (1 - t_arr) / (1 - ps)
    effect = float(np.sum(w_t * y) / np.sum(w_t) - np.sum(w_c * y) / np.sum(w_c))
    return {"method": "iptw", "treatment": treatment_col, "outcome": outcome_col, "effect": effect, "sample_size": int(len(df)), "confidence": "low" if len(df) < 50 else "medium"}


def did_effect(data: pd.DataFrame, treatment_col: str, outcome_col: str, time_col: str, post_value=None, treatment_value=None) -> dict:
    df = data.dropna(subset=[treatment_col, outcome_col, time_col]).copy()
    if df.empty:
        return {"method": "did", "effect": 0.0, "sample_size": 0, "confidence": "insufficient"}
    treated = _to_binary(df[treatment_col], treatment_value=treatment_value)
    if post_value is None:
        post = pd.to_datetime(df[time_col]) >= pd.to_datetime(df[time_col]).median()
    else:
        post = df[time_col].eq(post_value)
    y = df[outcome_col].astype(float)
    means = {}
    for a, label_a in [(1, "treated"), (0, "control")]:
        for b, label_b in [(1, "post"), (0, "pre")]:
            mask = treated.eq(a) & post.eq(bool(b))
            means[f"{label_a}_{label_b}"] = float(y[mask].mean()) if mask.any() else 0.0
    effect = (means["treated_post"] - means["treated_pre"]) - (means["control_post"] - means["control_pre"])
    return {"method": "did", "effect": float(effect), "cell_means": means, "sample_size": int(len(df)), "confidence": "low" if len(df) < 40 else "medium"}


def dml_effect(data: pd.DataFrame, treatment_col: str, outcome_col: str, covariates: list[str], treatment_value=None) -> dict:
    """Lightweight DML fallback: residualize outcome and treatment, then regress residuals.

    This is intentionally dependency-light for demos. A production build can swap in EconML DML.
    """
    df = data.dropna(subset=[treatment_col, outcome_col]).copy()
    t = _to_binary(df[treatment_col], treatment_value=treatment_value)
    if df.empty or t.nunique() < 2:
        return {"method": "dml_fallback", "effect": 0.0, "sample_size": len(df), "confidence": "insufficient"}
    x = _design_matrix(df, covariates).astype(float)
    if x.empty:
        return {"method": "dml_fallback", "effect": 0.0, "sample_size": len(df), "confidence": "insufficient", "warning": "缺少协变量，无法残差化。"}
    y = df[outcome_col].astype(float)
    y_model = RandomForestRegressor(n_estimators=24, min_samples_leaf=4, random_state=42, n_jobs=1).fit(x, y)
    t_model = RandomForestRegressor(n_estimators=24, min_samples_leaf=4, random_state=43, n_jobs=1).fit(x, t.astype(float))
    y_res = y.to_numpy() - y_model.predict(x)
    t_res = t.astype(float).to_numpy() - t_model.predict(x)
    denom = float(np.dot(t_res, t_res))
    effect = float(np.dot(t_res, y_res) / denom) if denom else 0.0
    return {"method": "dml_fallback", "treatment": treatment_col, "outcome": outcome_col, "effect": effect, "sample_size": int(len(df)), "confidence": "low" if len(df) < 60 else "medium", "interpretation": "轻量 DML 估计；正式版可替换 EconML。"}


def causal_forest_effect(data: pd.DataFrame, treatment_col: str, outcome_col: str, covariates: list[str], treatment_value=None) -> dict:
    """T-learner random-forest fallback returning ATE and simple CATE table."""
    df = data.dropna(subset=[treatment_col, outcome_col]).copy()
    t = _to_binary(df[treatment_col], treatment_value=treatment_value)
    if df.empty or t.nunique() < 2:
        return {"method": "causal_forest_fallback", "ate": 0.0, "sample_size": len(df), "confidence": "insufficient", "cate": []}
    x = _design_matrix(df, covariates).astype(float)
    if x.empty:
        return {"method": "causal_forest_fallback", "ate": 0.0, "sample_size": len(df), "confidence": "insufficient", "cate": []}
    y = df[outcome_col].astype(float)
    model_t = RandomForestRegressor(n_estimators=32, min_samples_leaf=4, random_state=44, n_jobs=1).fit(x[t.eq(1)], y[t.eq(1)])
    model_c = RandomForestRegressor(n_estimators=32, min_samples_leaf=4, random_state=45, n_jobs=1).fit(x[t.eq(0)], y[t.eq(0)])
    cate_values = model_t.predict(x) - model_c.predict(x)
    cate = []
    for col in [c for c in ["platform", "topic", "account_id"] if c in df.columns]:
        for value, idx in df.groupby(col).groups.items():
            cate.append({"segment_col": col, "segment": str(value), "cate": float(np.mean(cate_values[list(idx)])), "sample_size": int(len(idx))})
    return {"method": "causal_forest_fallback", "ate": float(np.mean(cate_values)), "sample_size": int(len(df)), "confidence": "low" if len(df) < 80 else "medium", "cate": cate[:30], "interpretation": "随机森林 T-learner 近似 CATE；正式版可替换 EconML CausalForest。"}


def target_trial_emulation(data: pd.DataFrame, treatment_col: str, outcome_col: str, time_col: str, covariates: list[str] | None = None) -> dict:
    df = data.dropna(subset=[treatment_col, outcome_col, time_col]).copy()
    if df.empty:
        return {"method": "target_trial_fallback", "effect": 0.0, "sample_size": 0, "confidence": "insufficient"}
    df = df.sort_values(time_col)
    result = stratified_regression_effect(df, treatment_col, outcome_col, (covariates or []) + [c for c in ["platform", "topic", "account_id"] if c in df.columns])
    result["method"] = "target_trial_fallback"
    result["trial_window"] = f"{df[time_col].min()} -> {df[time_col].max()}"
    result["interpretation"] = "按目标试验思路整理窗口、干预和结果；仍需检查交换性与重叠性。"
    return result


def interrupted_time_series(data: pd.DataFrame, time_col: str, outcome_col: str, intervention_time=None) -> dict:
    df = data.dropna(subset=[time_col, outcome_col]).copy()
    if len(df) < 6:
        return {"method": "its_fallback", "level_change": 0.0, "trend_change": 0.0, "sample_size": len(df), "confidence": "insufficient"}
    df = df.sort_values(time_col).reset_index(drop=True)
    df["t"] = np.arange(len(df))
    cutoff = pd.to_datetime(intervention_time) if intervention_time is not None else pd.to_datetime(df[time_col]).median()
    df["post"] = (pd.to_datetime(df[time_col]) >= cutoff).astype(int)
    df["post_t"] = df["post"] * df["t"]
    x = df[["t", "post", "post_t"]].astype(float)
    y = df[outcome_col].astype(float)
    model = LinearRegression().fit(x, y)
    return {"method": "its_fallback", "level_change": float(model.coef_[1]), "trend_change": float(model.coef_[2]), "sample_size": int(len(df)), "confidence": "low" if len(df) < 20 else "medium", "intervention_time": str(cutoff)}


def marginal_structural_model(data: pd.DataFrame, treatment_col: str, outcome_col: str, covariates: list[str], treatment_value=None) -> dict:
    """Stabilized-weight MSM fallback using weighted least squares style regression."""
    df = data.dropna(subset=[treatment_col, outcome_col]).copy()
    t = _to_binary(df[treatment_col], treatment_value=treatment_value)
    if df.empty or t.nunique() < 2:
        return {"method": "msm_fallback", "effect": 0.0, "sample_size": len(df), "confidence": "insufficient"}
    x = _design_matrix(df, covariates).astype(float)
    if x.empty:
        return {"method": "msm_fallback", "effect": 0.0, "sample_size": len(df), "confidence": "insufficient"}
    ps = np.clip(LogisticRegression(max_iter=1000).fit(x, t).predict_proba(x)[:, 1], 0.02, 0.98)
    p_t = float(t.mean())
    weights = np.where(t.eq(1), p_t / ps, (1 - p_t) / (1 - ps))
    design = pd.DataFrame({"treatment": t.astype(float)})
    y = df[outcome_col].astype(float)
    model = LinearRegression().fit(design, y, sample_weight=weights)
    return {"method": "msm_fallback", "effect": float(model.coef_[0]), "sample_size": int(len(df)), "confidence": "low" if len(df) < 80 else "medium", "weight_max": float(np.max(weights))}


def estimate_ab_effect(ab_tests: pd.DataFrame) -> pd.DataFrame:
    from .causal_experiment import analyze_ab_test

    out = analyze_ab_test(ab_tests)
    if not out.empty:
        out["finding_type"] = "AB Test 结果"
    return out


def cross_platform_increment(contents: pd.DataFrame, outcome_col: str = "views") -> pd.DataFrame:
    group_cols = ["title", "topic"]
    rows = []
    for keys, group in contents.groupby(group_cols):
        if group["platform"].nunique() < 2:
            continue
        baseline = float(group[outcome_col].mean())
        for _, row in group.iterrows():
            rows.append({
                "title": keys[0],
                "topic": keys[1],
                "platform": row["platform"],
                "outcome": outcome_col,
                "value": float(row[outcome_col]),
                "increment_vs_same_content_avg": float(row[outcome_col] - baseline),
            })
    return pd.DataFrame(rows)


def estimate_sample_size(effect: float, std: float, power: float = 0.8) -> int:
    if effect == 0 or std == 0:
        return 0
    z_alpha = 1.96
    z_beta = 0.84 if power <= 0.8 else 1.28
    return int(np.ceil(2 * ((z_alpha + z_beta) * std / abs(effect)) ** 2))
