import pandas as pd

from .revenue_analysis import content_commercial_value, topic_business_summary


def revenue_recommendations(contents: pd.DataFrame, revenues: pd.DataFrame, language: str = "中文") -> list[dict]:
    zh = language == "中文"
    value_df = content_commercial_value(contents, revenues, language=language)
    topic_df = topic_business_summary(contents)
    df = contents.copy()
    df["favorite_rate"] = df["favorites"] / df["views"].clip(lower=1)
    df["consultation_rate"] = df["consultations"] / df["views"].clip(lower=1)
    df["conversion_per_consultation"] = df["conversions"] / df["consultations"].clip(lower=1)
    df["revenue_per_hour"] = df["revenue"] / df["production_hours"].clip(lower=0.1)

    recs = []
    if not topic_df.empty:
        top = topic_df.iloc[0]
        recs.append({
            "title": f"主推“{top['topic']}”" if zh else f"Prioritize “{top['topic']}”",
            "reason": f"收入 ¥{top['total_revenue']:.0f}，效率 ¥{top['revenue_per_hour']:.0f}/小时。" if zh else f"Revenue ¥{top['total_revenue']:.0f}; efficiency ¥{top['revenue_per_hour']:.0f}/hour.",
            "action": "下周发 2 条同痛点内容，结尾加咨询入口。" if zh else "Publish 2 posts on the same pain point with a clear CTA.",
            "priority": "high",
        })

    high_fav = df.groupby("topic").agg(favorite_rate=("favorite_rate", "mean"), revenue=("revenue", "sum")).reset_index()
    fav_target = high_fav[(high_fav["favorite_rate"] >= high_fav["favorite_rate"].quantile(0.75)) & (high_fav["revenue"] <= high_fav["revenue"].median())]
    for _, row in fav_target.head(2).iterrows():
        recs.append({
            "title": f"把“{row['topic']}”产品化" if zh else f"Productize “{row['topic']}”",
            "reason": f"收藏率 {row['favorite_rate']:.2%}，收入偏低。" if zh else f"Save rate {row['favorite_rate']:.2%}; revenue is underused.",
            "action": "做清单/模板/小课，并在内容里放领取入口。" if zh else "Package it as a checklist/template/course with a lead CTA.",
            "priority": "medium",
        })

    consult = df.groupby("topic").agg(consultations=("consultations", "sum"), conversions=("conversions", "sum")).reset_index()
    consult["conversion_per_consultation"] = consult["conversions"] / consult["consultations"].clip(lower=1)
    weak_sales = consult[(consult["consultations"] >= consult["consultations"].quantile(0.75)) & (consult["conversion_per_consultation"] <= consult["conversion_per_consultation"].median())]
    for _, row in weak_sales.head(2).iterrows():
        recs.append({
            "title": f"提升“{row['topic']}”成交" if zh else f"Improve conversion for “{row['topic']}”",
            "reason": f"咨询 {int(row['consultations'])} 次，成交率 {row['conversion_per_consultation']:.2%}。" if zh else f"{int(row['consultations'])} consultations; conversion {row['conversion_per_consultation']:.2%}.",
            "action": "补案例、价格锚点、FAQ 和预约按钮。" if zh else "Add cases, price anchors, FAQs and booking buttons.",
            "priority": "high",
        })

    low_roi = df.groupby("topic").agg(production_hours=("production_hours", "mean"), revenue_per_hour=("revenue_per_hour", "mean")).reset_index()
    low_roi = low_roi[(low_roi["production_hours"] >= low_roi["production_hours"].quantile(0.75)) & (low_roi["revenue_per_hour"] <= low_roi["revenue_per_hour"].median())]
    for _, row in low_roi.head(2).iterrows():
        recs.append({
            "title": f"压缩“{row['topic']}”投入" if zh else f"Reduce effort on “{row['topic']}”",
            "reason": f"制作 {row['production_hours']:.1f} 小时，收益 ¥{row['revenue_per_hour']:.0f}/小时。" if zh else f"{row['production_hours']:.1f}h production; ¥{row['revenue_per_hour']:.0f}/hour.",
            "action": "先模板化或短内容验证，再做重内容。" if zh else "Use templates or short posts before deep production.",
            "priority": "medium",
        })

    for _, row in value_df.head(2).iterrows():
        recs.append({
            "title": f"复用《{row['title']}》" if zh else f"Repurpose “{row['title']}”",
            "reason": f"评分 {row['commercial_score']:.1f}，咨询 {int(row['consultations'])}，成交 {int(row['conversions'])}。" if zh else f"Score {row['commercial_score']:.1f}; consultations {int(row['consultations'])}; conversions {int(row['conversions'])}.",
            "action": "拆成案例、清单和转化内容各 1 条。" if zh else "Split into one case, one checklist and one conversion post.",
            "priority": "high" if row["commercial_score"] >= 80 else "medium",
        })

    return recs[:8]
