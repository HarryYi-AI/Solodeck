import numpy as np
import pandas as pd


def _rate(num, den):
    return np.where(den > 0, num / den, 0.0)


def _sample_title(topic: str, style: str, language: str = "中文") -> str:
    if language != "中文":
        templates = {
            "pain_point": f"Why your {topic} content is not converting: fix these 3 issues",
            "tutorial": f"A practical beginner guide to {topic}: follow this workflow",
            "number": f"7 steps I used to turn {topic} into stable revenue",
            "story": f"A real case: how {topic} brought the first paid conversion",
            "contrast": f"{topic} is not failing because of effort, but because of the path",
            "result_oriented": f"How to use {topic} to increase consultation conversion",
            "question": f"Is {topic} still worth doing? Read the data first",
        }
        return templates.get(style, f"{topic}: a reusable growth experiment for next week")
    templates = {
        "pain_point": f"为什么你做{topic}总是没转化？这 3 个坑先改",
        "tutorial": f"{topic}从 0 到 1 实操指南：照着做一遍",
        "number": f"我用 7 个步骤把{topic}做成稳定收入",
        "story": f"一个真实案例：{topic}如何带来第一笔成交",
        "contrast": f"{topic}做不起来，不是努力不够，而是路径错了",
        "result_oriented": f"用{topic}把咨询转化提升 30% 的方法",
        "question": f"{topic}到底值不值得继续做？看这组数据",
    }
    return templates.get(style, f"{topic}下周选题：一个可复用的增长实验")


def title_style_analysis(contents: pd.DataFrame) -> dict:
    df = contents.copy()
    df["favorite_rate"] = _rate(df["favorites"], df["views"])
    df["follow_rate"] = _rate(df["new_followers"], df["views"])
    df["conversion_rate"] = _rate(df["conversions"], df["views"])
    grouped = df.groupby("title_style").agg(
        content_count=("content_id", "count"),
        views=("views", "mean"),
        click_proxy=("views", "mean"),
        favorite_rate=("favorite_rate", "mean"),
        follow_rate=("follow_rate", "mean"),
        conversion_rate=("conversion_rate", "mean"),
        revenue=("revenue", "mean"),
    ).reset_index().sort_values("views", ascending=False)
    return {
        "table": grouped,
        "best_for_growth": grouped.sort_values("follow_rate", ascending=False).head(1).to_dict("records"),
        "best_for_favorite": grouped.sort_values("favorite_rate", ascending=False).head(1).to_dict("records"),
        "best_for_conversion": grouped.sort_values("conversion_rate", ascending=False).head(1).to_dict("records"),
        "confidence": "待验证" if len(df) < 20 else "较稳定",
    }


def topic_strategy_analysis(contents: pd.DataFrame) -> dict:
    df = contents.copy()
    df["like_rate"] = _rate(df["likes"], df["views"])
    df["favorite_rate"] = _rate(df["favorites"], df["views"])
    df["comment_rate"] = _rate(df["comments"], df["views"])
    df["follow_rate"] = _rate(df["new_followers"], df["views"])
    grouped = df.groupby("topic").agg(
        content_count=("content_id", "count"),
        views=("views", "sum"),
        like_rate=("like_rate", "mean"),
        favorite_rate=("favorite_rate", "mean"),
        comment_rate=("comment_rate", "mean"),
        follow_rate=("follow_rate", "mean"),
        consultations=("consultations", "sum"),
        conversions=("conversions", "sum"),
        revenue=("revenue", "sum"),
        production_hours=("production_hours", "sum"),
    ).reset_index()
    grouped["revenue_per_hour"] = _rate(grouped["revenue"], grouped["production_hours"])
    return {
        "table": grouped.sort_values("revenue", ascending=False),
        "growth_topics": grouped.sort_values("follow_rate", ascending=False).head(3).to_dict("records"),
        "monetization_topics": grouped.sort_values("revenue_per_hour", ascending=False).head(3).to_dict("records"),
        "engagement_topics": grouped.sort_values("comment_rate", ascending=False).head(3).to_dict("records"),
        "course_topics": grouped.sort_values("favorite_rate", ascending=False).head(3).to_dict("records"),
    }


def platform_strategy_analysis(contents: pd.DataFrame, revenues: pd.DataFrame) -> dict:
    df = contents.copy()
    df["follow_efficiency"] = _rate(df["new_followers"], df["views"])
    df["commercial_efficiency"] = _rate(df["revenue"], df["views"]) * 1000
    grouped = df.groupby("platform").agg(
        content_count=("content_id", "count"),
        exposure_efficiency=("views", "mean"),
        growth_efficiency=("follow_efficiency", "mean"),
        commercial_efficiency=("commercial_efficiency", "mean"),
        consultations=("consultations", "sum"),
        conversions=("conversions", "sum"),
        production_hours=("production_hours", "sum"),
    ).reset_index()
    revenue_by_platform = revenues.groupby("platform")["amount"].sum() if not revenues.empty else pd.Series(dtype=float)
    grouped["revenue"] = grouped["platform"].map(revenue_by_platform).fillna(0) + df.groupby("platform")["revenue"].sum().reindex(grouped["platform"]).values
    grouped["revenue_per_hour"] = _rate(grouped["revenue"], grouped["production_hours"])
    return {
        "table": grouped.sort_values("revenue", ascending=False),
        "platform_roles": {
            "曝光平台": grouped.sort_values("exposure_efficiency", ascending=False).head(1)["platform"].tolist(),
            "涨粉平台": grouped.sort_values("growth_efficiency", ascending=False).head(1)["platform"].tolist(),
            "转化平台": grouped.sort_values("conversions", ascending=False).head(1)["platform"].tolist(),
            "高价值平台": grouped.sort_values("revenue_per_hour", ascending=False).head(1)["platform"].tolist(),
        },
    }


def publish_time_analysis(contents: pd.DataFrame) -> pd.DataFrame:
    df = contents.copy()
    df["publish_time"] = pd.to_datetime(df["publish_time"])
    df["weekday"] = df["publish_time"].dt.day_name()
    df["hour"] = df["publish_time"].dt.hour
    grouped = df.groupby(["platform", "weekday", "hour"]).agg(
        sample_size=("content_id", "count"),
        avg_views=("views", "mean"),
        avg_followers=("new_followers", "mean"),
        avg_revenue=("revenue", "mean"),
    ).reset_index()
    grouped["confidence"] = np.where(grouped["sample_size"] < 3, "待验证", "较稳定")
    return grouped.sort_values(["platform", "avg_views"], ascending=[True, False])


def weekly_topic_plan(contents: pd.DataFrame, revenues: pd.DataFrame, n: int = 7, language: str = "中文") -> list[dict]:
    topic_result = topic_strategy_analysis(contents)
    platform_result = platform_strategy_analysis(contents, revenues)
    title_result = title_style_analysis(contents)
    topic_table = topic_result["table"]
    platform_table = platform_result["table"]
    title_table = title_result["table"]
    objectives = ["growth", "engagement", "conversion", "monetization"]
    objective_labels_zh = {
        "growth": "拉新增长",
        "engagement": "互动讨论",
        "conversion": "咨询转化",
        "monetization": "商业变现",
    }
    plans = []
    for i in range(n):
        if objectives[i % 4] == "growth":
            topic = topic_table.sort_values("follow_rate", ascending=False).iloc[i % len(topic_table)]
        elif objectives[i % 4] == "engagement":
            topic = topic_table.sort_values("comment_rate", ascending=False).iloc[i % len(topic_table)]
        elif objectives[i % 4] == "conversion":
            topic = topic_table.sort_values("conversions", ascending=False).iloc[i % len(topic_table)]
        else:
            topic = topic_table.sort_values("revenue_per_hour", ascending=False).iloc[i % len(topic_table)]
        platform = platform_table.iloc[i % len(platform_table)]["platform"]
        style = title_table.iloc[i % len(title_table)]["title_style"]
        objective = objectives[i % 4]
        objective_label = objective_labels_zh.get(objective, objective)
        plans.append({
            "suggested_topic": topic["topic"],
            "suggested_platform": platform,
            "suggested_title_style": style,
            "objective": objective,
            "reason": f"“{topic['topic']}”过去在{objective_label}上表现较好，适合下周小范围验证。" if language == "中文" else f"{topic['topic']} has stronger historical {objective} performance, so it is worth validating next week.",
            "sample_title": _sample_title(topic["topic"], style, language=language),
        })
    return plans
