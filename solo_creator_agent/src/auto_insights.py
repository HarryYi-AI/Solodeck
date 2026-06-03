from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from .text_structured import extract_text_features


def _rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float) / denominator.replace(0, np.nan).astype(float)


def prepare_insight_frame(contents: pd.DataFrame) -> pd.DataFrame:
    df = extract_text_features(contents)
    if "publish_time" in df.columns:
        dt = pd.to_datetime(df["publish_time"], errors="coerce")
        df["weekday"] = dt.dt.dayofweek
        df["hour"] = dt.dt.hour
        df["time_slot"] = pd.cut(
            df["hour"].fillna(0),
            bins=[-1, 5, 11, 17, 22, 24],
            labels=["late_night", "morning", "afternoon", "evening", "late_night"],
            ordered=False,
        ).astype(str)
    else:
        df["weekday"] = -1
        df["hour"] = -1
        df["time_slot"] = "unknown"
    for col in ["views", "likes", "favorites", "comments", "new_followers", "consultations", "conversions", "revenue", "cost"]:
        if col not in df.columns:
            df[col] = 0
    df["favorite_rate"] = _rate(df["favorites"], df["views"]).fillna(0)
    df["follow_rate"] = _rate(df["new_followers"], df["views"]).fillna(0)
    df["conversion_rate"] = _rate(df["conversions"], df["views"]).fillna(0)
    df["profit"] = df["revenue"].astype(float) - df["cost"].astype(float)
    return df


def _confidence(n: int) -> str:
    if n >= 60:
        return "medium"
    if n >= 20:
        return "low"
    return "exploratory"


def _lift(best: float, baseline: float) -> float:
    if baseline == 0 or np.isnan(baseline):
        return 0.0
    return float((best - baseline) / abs(baseline))


def generate_auto_insights(contents: pd.DataFrame, language: str = "中文", max_items: int = 8) -> list[dict]:
    df = prepare_insight_frame(contents)
    zh = language == "中文"
    insights: list[dict] = []
    if df.empty:
        return insights

    top = df.sort_values("views", ascending=False).head(1).iloc[0]
    insights.append({
        "kind": "extreme_sample",
        "title": "最高播放内容" if zh else "Top-viewed content",
        "finding": (
            f"《{top['title']}》在 {top['platform']} 获得 {int(top['views']):,} 次播放/阅读。"
            if zh
            else f"“{top['title']}” on {top['platform']} reached {int(top['views']):,} views."
        ),
        "action": "把它拆成同主题系列，并复用标题结构和发布平台。" if zh else "Turn it into a series and reuse the title structure and platform fit.",
        "confidence": _confidence(len(df)),
        "priority": "high",
    })

    for group_col, metric, label_zh, label_en in [
        ("text_language", "views", "语言与播放量", "Language vs views"),
        ("has_number", "favorite_rate", "数字标题与收藏率", "Number titles vs save rate"),
        ("time_slot", "follow_rate", "发布时间与转粉率", "Publishing time vs follow rate"),
        ("inferred_title_style", "revenue", "标题风格与收入", "Title style vs revenue"),
    ]:
        if group_col not in df.columns or df[group_col].nunique(dropna=True) < 2:
            continue
        grouped = df.groupby(group_col, as_index=False).agg(
            sample_size=("content_id", "count") if "content_id" in df.columns else (metric, "count"),
            metric_value=(metric, "mean"),
        )
        grouped = grouped[grouped["sample_size"].ge(2)]
        if grouped.empty:
            continue
        best = grouped.sort_values("metric_value", ascending=False).iloc[0]
        baseline = float(grouped["metric_value"].mean())
        lift = _lift(float(best["metric_value"]), baseline)
        insights.append({
            "kind": "group_lift",
            "dimension": group_col,
            "metric": metric,
            "title": label_zh if zh else label_en,
            "finding": (
                f"{best[group_col]} 组的 {metric} 平均值最高，相对组均值约高 {lift:.0%}，样本 {int(best['sample_size'])} 条。"
                if zh
                else f"{best[group_col]} has the highest average {metric}, about {lift:.0%} above group average across {int(best['sample_size'])} samples."
            ),
            "action": "放进下周小实验，验证后再放大。" if zh else "Put it into next week's small test before scaling.",
            "confidence": _confidence(int(grouped["sample_size"].sum())),
            "priority": "medium" if abs(lift) < 0.2 else "high",
        })

    if df[["favorite_rate", "follow_rate"]].dropna().shape[0] >= 5:
        corr = float(df["favorite_rate"].corr(df["follow_rate"]))
        insights.append({
            "kind": "relationship",
            "title": "收藏率与转粉率关系" if zh else "Save rate and follow rate",
            "finding": (
                f"两者相关系数约 {corr:.2f}。收藏型内容需要补充转粉入口。"
                if zh
                else f"The correlation is about {corr:.2f}. Save-heavy content needs a stronger follow CTA."
            ),
            "action": "把收藏高但转粉低的内容加上关注理由或资料领取入口。" if zh else "Add follow rationale or lead magnets to high-save, low-follow content.",
            "confidence": _confidence(len(df)),
            "priority": "medium",
        })

    return insights[:max_items]


def insight_markdown(insights: list[dict], language: str = "中文") -> str:
    if not insights:
        return "暂无可用洞察。" if language == "中文" else "No insights available."
    lines = []
    for item in insights:
        lines.append(f"- **{item['title']}**：{item['finding']} {item['action']}" if language == "中文" else f"- **{item['title']}**: {item['finding']} {item['action']}")
    return "\n".join(lines)


def insight_figures(contents: pd.DataFrame) -> dict[str, object]:
    df = prepare_insight_frame(contents)
    figures: dict[str, object] = {}
    if not df.empty:
        figures["top_views"] = px.bar(df.sort_values("views", ascending=False).head(10), x="title", y="views", color="platform", title="Top Content by Views")
        figures["save_follow"] = px.scatter(df, x="favorite_rate", y="follow_rate", size="views", color="platform", hover_name="title", title="Save Rate vs Follow Rate")
        if "inferred_title_style" in df.columns:
            style = df.groupby("inferred_title_style", as_index=False).agg(views=("views", "mean"), revenue=("revenue", "mean"), sample_size=("title", "count"))
            figures["title_style"] = px.bar(style, x="inferred_title_style", y="revenue", color="views", title="Revenue by Inferred Title Style")
    return figures


def metric_snapshot(contents: pd.DataFrame) -> pd.DataFrame:
    df = prepare_insight_frame(contents)
    numeric = [c for c in ["views", "likes", "favorites", "comments", "new_followers", "consultations", "conversions", "revenue", "favorite_rate", "follow_rate", "conversion_rate", "profit"] if c in df.columns]
    rows = []
    for col in numeric:
        series = df[col].astype(float)
        rows.append({
            "metric": col,
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.5)),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
        })
    return pd.DataFrame(rows)
