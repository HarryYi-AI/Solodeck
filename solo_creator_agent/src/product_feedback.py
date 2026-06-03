from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


ISSUE_KEYWORDS = {
    "pricing": ["贵", "价格", "付费", "price", "expensive"],
    "performance": ["慢", "卡", "延迟", "slow", "lag"],
    "usability": ["不会", "难用", "复杂", "confusing", "hard"],
    "feature_request": ["希望", "能不能", "增加", "want", "request"],
    "design": ["外观", "可爱", "颜色", "design", "cute"],
    "trust": ["担心", "不信任", "隐私", "trust", "privacy"],
    "quality": ["质量", "坏", "不稳定", "quality"],
}


def classify_feedback(feedback_df: pd.DataFrame) -> pd.DataFrame:
    df = feedback_df.copy()
    if df.empty:
        return df
    for col in ["feedback_text", "issue_type", "sentiment", "severity"]:
        if col not in df.columns:
            df[col] = ""
    issue_types, sentiments, severities = [], [], []
    for text in df["feedback_text"].fillna("").astype(str):
        lower = text.lower()
        issue = "emotional_value"
        for key, words in ISSUE_KEYWORDS.items():
            if any(w in text or w in lower for w in words):
                issue = key
                break
        neg = any(w in text or w in lower for w in ["不", "慢", "贵", "担心", "bad", "slow", "expensive"])
        pos = any(w in text or w in lower for w in ["喜欢", "可爱", "有用", "愿意", "love", "useful", "cute"])
        sentiment = "negative" if neg and not pos else "positive" if pos else "neutral"
        severity = "high" if issue in {"pricing", "performance", "trust"} and sentiment == "negative" else "medium" if sentiment == "negative" else "low"
        issue_types.append(issue)
        sentiments.append(sentiment)
        severities.append(severity)
    df["issue_type"] = df["issue_type"].where(df["issue_type"].astype(str).str.len().gt(0), issue_types)
    df["sentiment"] = df["sentiment"].where(df["sentiment"].astype(str).str.len().gt(0), sentiments)
    df["severity"] = df["severity"].where(df["severity"].astype(str).str.len().gt(0), severities)
    df["feature_request_flag"] = df["issue_type"].eq("feature_request")
    df["pricing_issue_flag"] = df["issue_type"].eq("pricing")
    df["trust_issue_flag"] = df["issue_type"].eq("trust")
    df["usability_issue_flag"] = df["issue_type"].eq("usability")
    df["purchase_intent_flag"] = df["feedback_text"].fillna("").astype(str).str.contains("购买|下单|愿意付|buy|pay", case=False, regex=True)
    return df


def feedback_topic_clustering(feedback_df: pd.DataFrame) -> pd.DataFrame:
    df = classify_feedback(feedback_df)
    if df.empty:
        return pd.DataFrame()
    text = df["feedback_text"].fillna("").astype(str)
    k = min(5, max(1, len(df) // 4))
    if len(df) < 4:
        df["topic_cluster"] = "topic_0"
    else:
        matrix = TfidfVectorizer(max_features=300, token_pattern=r"(?u)\b\w+\b").fit_transform(text)
        df["topic_cluster"] = [f"topic_{x}" for x in KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(matrix)]
    rows = []
    for topic, group in df.groupby("topic_cluster"):
        rows.append({
            "topic": topic,
            "evidence_count": int(len(group)),
            "top_issues": ", ".join(group["issue_type"].value_counts().head(3).index.astype(str)),
            "representative_feedback": group.iloc[0]["feedback_text"],
            "affected_user_segments": ", ".join(group.get("user_segment", pd.Series()).dropna().astype(str).unique()[:4]),
            "related_product_or_content": ", ".join((group.get("related_product_id", pd.Series()).fillna("").astype(str) + group.get("related_content_id", pd.Series()).fillna("").astype(str)).unique()[:4]),
        })
    return pd.DataFrame(rows).sort_values("evidence_count", ascending=False)


def feedback_to_roadmap(feedback_df: pd.DataFrame, products_df: pd.DataFrame | None = None, contents_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = classify_feedback(feedback_df)
    if df.empty:
        return pd.DataFrame()
    severity_weight = {"low": 1, "medium": 2, "high": 3}
    rows = []
    for issue, group in df.groupby("issue_type"):
        score = len(group) * group["severity"].map(severity_weight).fillna(1).mean()
        priority = "high" if score >= 8 else "medium" if score >= 4 else "low"
        rows.append({
            "issue": issue,
            "evidence_count": int(len(group)),
            "affected_segment": ", ".join(group.get("user_segment", pd.Series()).dropna().astype(str).unique()[:3]),
            "business_impact": "可能影响购买/留存" if issue in {"pricing", "performance", "trust", "usability"} else "影响满意度或传播",
            "suggested_action": {
                "pricing": "测试分层定价或首单优惠。",
                "performance": "优先优化响应速度并设置可量化指标。",
                "feature_request": "挑选高频请求进入下一版小范围内测。",
                "design": "保留高好感设计元素，测试颜色/材质变体。",
            }.get(issue, "先补充证据，再做低成本修改。"),
            "priority": priority,
            "expected_metric_to_watch": "conversion_rate, retained_7d, avg_rating",
        })
    return pd.DataFrame(rows).sort_values(["priority", "evidence_count"], ascending=[True, False])


def sentiment_revenue_link(feedback_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    df = classify_feedback(feedback_df)
    if df.empty or products_df.empty:
        return pd.DataFrame()
    merged = df.merge(products_df, left_on="related_product_id", right_on="product_id", how="left")
    return merged.groupby("sentiment", as_index=False).agg(feedback_count=("feedback_id", "count"), avg_revenue=("revenue", "mean"), avg_conversion=("conversions", "mean"), avg_rating=("avg_rating", "mean"))


def beta_feedback_effect(beta_tests_df: pd.DataFrame, feedback_df: pd.DataFrame) -> dict:
    if beta_tests_df.empty:
        return {"warning": "没有内测分组数据。"}
    rows = []
    for outcome in ["activated", "retained_7d", "converted", "revenue", "rating"]:
        if outcome not in beta_tests_df.columns:
            continue
        t = beta_tests_df[beta_tests_df["test_group"].eq("treatment")][outcome].astype(float)
        c = beta_tests_df[beta_tests_df["test_group"].eq("control")][outcome].astype(float)
        if len(t) and len(c):
            rows.append({"outcome": outcome, "treatment_mean": float(t.mean()), "control_mean": float(c.mean()), "lift": float(t.mean() - c.mean()), "warning": "建议继续观察" if len(t) + len(c) < 50 else ""})
    return {"effects": rows}


def generate_feedback_report(feedback_df: pd.DataFrame) -> str:
    df = classify_feedback(feedback_df)
    roadmap = feedback_to_roadmap(df)
    if df.empty:
        return "# 用户反馈报告\n\n暂无反馈数据。"
    pos = df[df["sentiment"].eq("positive")]["feedback_text"].head(3).tolist()
    neg = df[df["sentiment"].eq("negative")]["feedback_text"].head(3).tolist()
    return "\n".join([
        "# 用户反馈报告",
        "",
        f"- 反馈总数：{len(df)}",
        f"- 负面反馈占比：{df['sentiment'].eq('negative').mean():.1%}",
        f"- 最多问题类型：{df['issue_type'].value_counts().index[0]}",
        "",
        "## 用户最喜欢什么",
        *[f"- {x}" for x in pos],
        "",
        "## 用户最不满意什么",
        *[f"- {x}" for x in neg],
        "",
        "## 下一版优先事项",
        roadmap.head(5).to_markdown(index=False),
        "",
        "反馈是证据，不是最终市场结论；如果内测用户不是随机邀请，需要继续控制选择偏差。",
    ])
