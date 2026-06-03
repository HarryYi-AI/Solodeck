import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def _safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    return np.where(den > 0, num / den, 0.0)


def calculate_rpm(contents: pd.DataFrame) -> pd.DataFrame:
    df = contents.copy()
    df["rpm"] = np.where(df["views"] > 0, df["revenue"] / df["views"] * 1000, 0.0)
    return df[["content_id", "title", "platform", "topic", "views", "revenue", "rpm"]]


def classify_revenue_summary(revenues: pd.DataFrame) -> pd.DataFrame:
    if revenues.empty:
        return pd.DataFrame(columns=["revenue_type", "amount", "share"])
    grouped = revenues.groupby("revenue_type", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    total = grouped["amount"].sum()
    grouped["share"] = np.where(total > 0, grouped["amount"] / total, 0)
    grouped["mom"] = np.nan
    grouped["yoy"] = np.nan
    return grouped


def platform_revenue_summary(revenues: pd.DataFrame) -> pd.DataFrame:
    grouped = revenues.groupby("platform", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    total = grouped["amount"].sum()
    grouped["share"] = np.where(total > 0, grouped["amount"] / total, 0)
    return grouped


def pending_payment_summary(revenues: pd.DataFrame, campaigns: pd.DataFrame) -> dict:
    pending_revenue = revenues.loc[revenues["status"].eq("pending"), "amount"].sum() if not revenues.empty else 0.0
    unpaid_campaigns = campaigns[campaigns["payment_status"].isin(["unpaid", "overdue"])] if not campaigns.empty else campaigns
    overdue_campaigns = campaigns[campaigns["payment_status"].eq("overdue")] if not campaigns.empty else campaigns
    return {
        "pending_revenue_amount": float(pending_revenue),
        "pending_campaign_amount": float(unpaid_campaigns["price"].sum()) if not unpaid_campaigns.empty else 0.0,
        "total_pending_amount": float(pending_revenue + (unpaid_campaigns["price"].sum() if not unpaid_campaigns.empty else 0)),
        "overdue_campaigns": overdue_campaigns[["campaign_id", "brand_name", "campaign_name", "price", "deadline", "payment_status"]].to_dict("records") if not overdue_campaigns.empty else [],
    }


def content_commercial_value(contents: pd.DataFrame, revenues: pd.DataFrame | None = None, language: str = "中文") -> pd.DataFrame:
    zh = language == "中文"
    df = contents.copy()
    if revenues is not None and not revenues.empty and "content_id" in revenues.columns:
        mapped = revenues.dropna(subset=["content_id"]).groupby("content_id")["amount"].sum()
        df["revenue"] = df["revenue"].fillna(0) + df["content_id"].map(mapped).fillna(0)

    df["favorite_rate"] = _safe_rate(df["favorites"], df["views"])
    df["follow_rate"] = _safe_rate(df["new_followers"], df["views"])
    df["conversion_rate"] = _safe_rate(df["conversions"], df["views"])
    df["revenue_per_hour"] = np.where(df["production_hours"] > 0, df["revenue"] / df["production_hours"], 0)

    features = df[["views", "favorite_rate", "follow_rate", "consultations", "conversions", "revenue", "revenue_per_hour"]].fillna(0)
    if len(df) > 1 and features.max().sum() > 0:
        scaled = MinMaxScaler().fit_transform(features)
        weights = np.array([0.12, 0.14, 0.16, 0.14, 0.18, 0.18, 0.08])
        df["commercial_score"] = (scaled @ weights * 100).round(1)
    else:
        df["commercial_score"] = 50.0

    def insight(row):
        if row["revenue"] > df["revenue"].quantile(0.75):
            return "高收入内容，可复用选题结构并延展产品化服务。" if zh else "High-revenue content. Reuse the topic structure and extend it into productized services."
        if row["favorite_rate"] > df["favorite_rate"].quantile(0.75) and row["revenue"] < df["revenue"].median():
            return "收藏强但变现弱，适合转资料包、课程或清单模板。" if zh else "Strong saves but weak monetization. Turn it into a toolkit, course or checklist template."
        if row["consultations"] > df["consultations"].quantile(0.75) and row["conversions"] <= df["conversions"].median():
            return "咨询强但成交弱，需要优化承接页和私域话术。" if zh else "Strong consultations but weak conversions. Improve the landing page and sales follow-up."
        if row["production_hours"] > df["production_hours"].quantile(0.75) and row["revenue_per_hour"] < df["revenue_per_hour"].median():
            return "制作投入偏高，建议模板化或降低制作成本。" if zh else "Production investment is high. Use templates or lower production cost."
        return "表现稳定，可作为常规内容池。" if zh else "Stable performance. Keep it in the regular content pool."

    df["insight"] = df.apply(insight, axis=1)
    return df[["content_id", "title", "platform", "topic", "commercial_score", "revenue", "consultations", "conversions", "insight"]].sort_values("commercial_score", ascending=False)


def topic_business_summary(contents: pd.DataFrame) -> pd.DataFrame:
    df = content_commercial_value(contents)
    base = contents[["content_id", "production_hours"]].merge(df[["content_id", "commercial_score"]], on="content_id")
    merged = contents.merge(base[["content_id", "commercial_score"]], on="content_id", how="left")
    grouped = merged.groupby("topic").agg(
        content_count=("content_id", "count"),
        total_revenue=("revenue", "sum"),
        avg_commercial_value=("commercial_score", "mean"),
        avg_production_hours=("production_hours", "mean"),
        total_production_hours=("production_hours", "sum"),
    ).reset_index()
    grouped["revenue_per_hour"] = np.where(grouped["total_production_hours"] > 0, grouped["total_revenue"] / grouped["total_production_hours"], 0)
    return grouped.sort_values("total_revenue", ascending=False)
