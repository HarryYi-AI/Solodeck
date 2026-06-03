from __future__ import annotations

import pandas as pd


def design_beta_test(product_or_feature: str, target_metric: str, user_segments: list[str] | str, constraints: dict | None = None) -> dict:
    segments = user_segments if isinstance(user_segments, list) else [x.strip() for x in str(user_segments).split(",") if x.strip()]
    constraints = constraints or {}
    min_sample = int(constraints.get("sample_size", 40))
    return {
        "test_goal": f"验证 {product_or_feature} 是否提升 {target_metric}",
        "treatment_group": f"提前体验 {product_or_feature}",
        "control_group": "继续使用当前版本/当前款式",
        "target_users": segments or ["核心目标用户"],
        "sample_size_suggestion": f"建议至少 {min_sample} 人，实验组/对照组各半；资源有限时先做 10+10 小组验证。",
        "duration": constraints.get("duration", "7-14 天"),
        "feedback_questions": generate_feedback_form("product", product_or_feature),
        "success_criteria": f"{target_metric} 提升方向稳定，且高严重性负反馈不超过 15%。",
        "stop_or_scale_rule": "若转化/留存提升且反馈风险可控，小范围扩大；若负反馈集中在价格/性能，修改后再测；样本不足则继续收集。",
    }


def generate_feedback_form(product_type: str, feature_name: str) -> list[str]:
    return [
        f"你是否理解 {feature_name} 的作用？",
        "哪一点最吸引你？",
        "哪一点阻碍你购买或继续使用？",
        "你愿意为它支付多少？",
        "你会推荐给谁？",
        "你最希望下一版改进什么？",
        "请给 1-5 分评分。",
        "其他开放反馈。",
    ]


def beta_test_readiness_check(products_df: pd.DataFrame, feedback_df: pd.DataFrame, beta_tests_df: pd.DataFrame) -> dict:
    score = 0
    warnings = []
    if not products_df.empty:
        score += 20
    else:
        warnings.append("缺少产品/功能对象。")
    if not feedback_df.empty:
        score += 20
    else:
        warnings.append("缺少结构化反馈。")
    if not beta_tests_df.empty and beta_tests_df["test_group"].nunique() >= 2:
        score += 30
    else:
        warnings.append("缺少实验组/对照组。")
    if not beta_tests_df.empty and len(beta_tests_df) >= 40:
        score += 20
    else:
        warnings.append("样本量偏小。")
    if not beta_tests_df.empty and {"converted", "retained_7d", "rating"}.intersection(beta_tests_df.columns):
        score += 10
    else:
        warnings.append("缺少结果指标。")
    if not beta_tests_df.empty and beta_tests_df.get("user_segment", pd.Series()).nunique() <= 1:
        warnings.append("目标用户可能过窄，存在选择偏差。")
    return {"readiness_score": min(score, 100), "warnings": warnings}


def recommend_next_validation_step(current_evidence: dict) -> dict:
    score = current_evidence.get("readiness_score", 0)
    lift = current_evidence.get("lift", 0)
    negative_rate = current_evidence.get("negative_rate", 0)
    if score >= 70 and lift > 0 and negative_rate < 0.2:
        action = "小范围扩大推广"
    elif negative_rate >= 0.35:
        action = "修改后再测"
    elif score < 50:
        action = "收集更多反馈"
    elif lift <= 0:
        action = "暂停该功能"
    else:
        action = "继续小范围测试"
    return {"recommended_action": action, "reason": "优先用最低成本验证，确认有效后再扩大。"}
