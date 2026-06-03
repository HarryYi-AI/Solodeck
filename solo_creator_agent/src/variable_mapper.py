from __future__ import annotations

from typing import Any

import pandas as pd

from .llm_agent import call_llm, llm_configured


ROLE_HINTS = {
    "treatment": {"title_style", "cover_style", "publish_hour", "hour", "weekday", "platform", "topic", "content_type", "is_sponsored"},
    "outcome": {"views", "impressions", "likes", "favorites", "comments", "shares", "completion_rate", "new_followers", "consultations", "conversions", "revenue", "rpm"},
    "confounder": {"followers_before", "production_hours", "duration_sec", "ad_spend", "topic", "platform", "content_type"},
    "mediator": {"views", "completion_rate", "likes", "favorites", "comments", "shares", "consultations"},
    "context": {"content_id", "title", "publish_time", "campaign_id", "brand_name"},
}


def heuristic_variable_map(columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        roles = [role for role, names in ROLE_HINTS.items() if col in names]
        if not roles:
            if col.endswith("_rate"):
                roles = ["outcome"]
            elif col.endswith("_id") or col in {"date", "deadline"}:
                roles = ["context"]
            else:
                roles = ["context"]
        rows.append({
            "variable": col,
            "role": roles[0],
            "all_possible_roles": ", ".join(roles),
            "semantic": _semantic_label(col),
            "confidence": "medium" if roles[0] != "context" else "low",
            "source": "heuristic",
        })
    return pd.DataFrame(rows)


def _semantic_label(col: str) -> str:
    labels = {
        "title_style": "标题风格，可作为内容策略 treatment",
        "cover_style": "封面风格，可作为内容策略 treatment",
        "publish_time": "发布时间，用于提取 weekday/hour",
        "platform": "平台，既可能是 treatment，也可能是分层 context",
        "topic": "主题，可能影响策略选择和结果",
        "followers_before": "发布前账号规模，是重要混杂变量",
        "ad_spend": "投放费用，是重要混杂变量",
        "views": "曝光/播放结果，也可能是转化前的中介变量",
        "new_followers": "增长结果指标",
        "consultations": "商业兴趣结果指标",
        "conversions": "成交结果指标",
        "revenue": "商业结果指标",
    }
    return labels.get(col, col.replace("_", " "))


def llm_variable_map(columns: list[str], sample_rows: list[dict[str, Any]], language: str = "中文") -> pd.DataFrame:
    if not llm_configured():
        return heuristic_variable_map(columns)
    prompt = """
你是数据科学家，请为内容经营分析数据表做变量语义理解。
只输出 JSON 数组，每个元素包含：
variable, role, all_possible_roles, semantic, confidence, source。
role 必须是 treatment/outcome/confounder/mediator/collider/context 之一。
注意：不要把相关性解释成因果；平台、主题、粉丝规模、投放费用常是混杂变量。
"""
    try:
        import json

        text = call_llm(prompt, {"columns": columns, "sample_rows": sample_rows[:5]}, language=language, profile="basic", temperature=0.1)
        parsed = json.loads(text)
        return pd.DataFrame(parsed)
    except Exception:
        return heuristic_variable_map(columns)


def map_variables(df: pd.DataFrame, use_llm: bool = False, language: str = "中文") -> pd.DataFrame:
    columns = list(df.columns)
    if use_llm:
        return llm_variable_map(columns, df.head(5).to_dict("records"), language=language)
    return heuristic_variable_map(columns)
