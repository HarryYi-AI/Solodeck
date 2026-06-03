import numpy as np
import pandas as pd


def _bootstrap_diff(treatment: np.ndarray, control: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    if len(treatment) == 0 or len(control) == 0:
        return 0.0, 0.0
    diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        t = rng.choice(treatment, size=len(treatment), replace=True)
        c = rng.choice(control, size=len(control), replace=True)
        diffs.append(t.mean() - c.mean())
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def analyze_ab_test(ab_tests: pd.DataFrame, language: str = "中文") -> pd.DataFrame:
    zh = language == "中文"
    rows = []
    for keys, group in ab_tests.groupby(["experiment_id", "outcome_metric"]):
        treatment = group[group["group"].eq("treatment")]["outcome_value"].astype(float).to_numpy()
        control = group[group["group"].eq("control")]["outcome_value"].astype(float).to_numpy()
        treatment_mean = float(treatment.mean()) if len(treatment) else 0.0
        control_mean = float(control.mean()) if len(control) else 0.0
        absolute_lift = treatment_mean - control_mean
        relative_lift = absolute_lift / control_mean if control_mean else 0.0
        ci_low, ci_high = _bootstrap_diff(treatment, control)
        low_conf = len(treatment) < 5 or len(control) < 5
        conclusion = (
            "先小范围验证。" if low_conf else ("实验组表现更好。" if ci_low > 0 else "暂未看到稳定提升。")
        ) if zh else (
            "Validate on a small scale first." if low_conf else ("Treatment performs better." if ci_low > 0 else "No stable lift observed.")
        )
        rows.append({
            "experiment_id": keys[0],
            "outcome_metric": keys[1],
            "treatment_mean": treatment_mean,
            "control_mean": control_mean,
            "absolute_lift": absolute_lift,
            "relative_lift": relative_lift,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "conclusion": conclusion,
        })
    return pd.DataFrame(rows)


def stratified_effect(contents: pd.DataFrame, treatment_col: str, outcome_col: str, strata_cols: list[str]) -> dict:
    rows = []
    weighted_sum = 0.0
    total_weight = 0
    for strata, group in contents.groupby(strata_cols):
        values = group[treatment_col].dropna().unique()
        if len(values) < 2:
            continue
        treatment_value, control_value = values[0], values[1]
        t = group[group[treatment_col].eq(treatment_value)][outcome_col].mean()
        c = group[group[treatment_col].eq(control_value)][outcome_col].mean()
        diff = float(t - c)
        weight = len(group)
        weighted_sum += diff * weight
        total_weight += weight
        rows.append({
            "strata": strata if isinstance(strata, tuple) else (strata,),
            "treatment_value": treatment_value,
            "control_value": control_value,
            "treatment_mean": float(t),
            "control_mean": float(c),
            "difference": diff,
            "sample_size": weight,
            "confidence": "待验证" if weight < 6 else "较稳定",
        })
    return {"strata_effects": rows, "weighted_effect": weighted_sum / total_weight if total_weight else 0.0}


def causal_readiness_check(data: pd.DataFrame, language: str = "中文") -> dict:
    zh = language == "中文"
    columns = set(data.columns)
    warnings = []
    score = 0
    if {"group", "treatment_name"}.intersection(columns):
        score += 20
    else:
        warnings.append("缺少明确 treatment/group 字段。" if zh else "Missing explicit treatment/group fields.")
    if "group" in columns and data["group"].nunique() >= 2:
        score += 20
    else:
        warnings.append("缺少 control 或 treatment 对照。" if zh else "Missing control or treatment comparison.")
    if len(data) >= 30:
        score += 20
    else:
        warnings.append("样本量不足，建议至少 30 条观测。" if zh else "Sample size is small; at least 30 observations are recommended.")
    if {"date", "publish_time"}.intersection(columns):
        score += 20
    else:
        warnings.append("缺少前后时间字段。" if zh else "Missing time fields.")
    if {"platform", "topic"}.issubset(columns):
        score += 20
    else:
        warnings.append("缺少可控协变量，例如 platform/topic。" if zh else "Missing controllable covariates such as platform/topic.")
    return {"readiness_score": score, "warning": warnings, "interpretation": "分数越高，越适合直接放大实验。" if zh else "Higher scores mean the test is more ready to scale."}


def design_ab_test(goal: str, platform: str, topic: str, treatment: str, control: str, metric: str, language: str = "中文") -> dict:
    if language != "中文":
        return {
            "hypothesis": f"When publishing {topic} on {platform}, {treatment} will improve {metric} compared with {control}.",
            "treatment_group": treatment,
            "control_group": control,
            "controlled_variables": ["platform", "topic", "publishing time", "content length", "cover style", "posting interval"],
            "primary_metric": metric,
            "secondary_metrics": ["views", "favorite_rate", "follow_rate", "conversion_rate", "revenue"],
            "duration_suggestion": "Run for 2 consecutive weeks or at least 6-10 similar posts.",
            "minimum_sample_suggestion": "At least 10 posts per group; validate longer if sample size is smaller.",
            "execution_plan": [
                "Keep platform and topic consistent to avoid platform effects.",
                "Randomly assign title/cover/hook variants to treatment and control.",
                "Record publishing time, production hours, promotion and pinning as covariates.",
                "Collect metrics at fixed windows: 24h, 72h and 7d after publishing.",
                "Use mean difference and bootstrap intervals to choose the next action.",
            ],
        }
    return {
        "hypothesis": f"在{platform}发布{topic}内容时，{treatment}相较于{control}能提升{metric}。",
        "treatment_group": treatment,
        "control_group": control,
        "controlled_variables": ["platform", "topic", "发布时间段", "内容长度", "封面风格", "发布时间间隔"],
        "primary_metric": metric,
        "secondary_metrics": ["views", "favorite_rate", "follow_rate", "conversion_rate", "revenue"],
        "duration_suggestion": "连续 2 周或至少覆盖 6-10 条同类内容。",
        "minimum_sample_suggestion": "每组至少 10 条内容；不足时延长观察周期。",
        "execution_plan": [
            "确定同一平台和同一主题，避免混入平台差异。",
            "随机分配标题/封面/开头钩子到实验组和对照组。",
            "记录发布时间、制作时长、投放或置顶等协变量。",
            "发布后 24h/72h/7d 固定时间采集指标。",
            "用均值差异和区间判断下一步动作。",
        ],
    }


def run_causalpy_analysis(*args, **kwargs) -> dict:
    return {"status": "not_implemented", "fallback": "当前版本使用 bootstrap/分层对比。未来可接入 CausalPy 做 DID、ITS、Synthetic Control。"}


def run_pymc_marketing_mmm(*args, **kwargs) -> dict:
    return {"status": "not_implemented", "fallback": "当前版本使用平台商业效率对比。未来可接入 PyMC-Marketing 做 MMM、CLV、预算优化。"}
