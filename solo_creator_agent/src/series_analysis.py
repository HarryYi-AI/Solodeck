from __future__ import annotations

import numpy as np
import pandas as pd


def _rate(num, den):
    return num.astype(float) / den.replace(0, np.nan).astype(float)


def _trend_tail(group: pd.DataFrame, time_col: str, outcome_col: str) -> str:
    ordered = group.sort_values(time_col)
    vals = ordered[outcome_col].astype(float).tail(3).tolist()
    if len(vals) < 3:
        return "insufficient"
    if vals[0] > vals[1] > vals[2]:
        return "down"
    if vals[0] < vals[1] < vals[2]:
        return "up"
    return "mixed"


def content_series_performance(contents_df: pd.DataFrame) -> pd.DataFrame:
    if contents_df.empty or "series_id" not in contents_df.columns:
        return pd.DataFrame()
    df = contents_df[contents_df["series_id"].fillna("").astype(str).ne("")].copy()
    if df.empty:
        return pd.DataFrame()
    df["favorite_rate"] = _rate(df["favorites"], df["views"]).fillna(0)
    df["follow_rate"] = _rate(df["new_followers"], df["views"]).fillna(0)
    df["consultation_rate"] = _rate(df["consultations"], df["views"]).fillna(0)
    rows = []
    for sid, group in df.groupby("series_id"):
        trend = _trend_tail(group, "publish_time", "views")
        fav_trend = _trend_tail(group, "publish_time", "favorite_rate")
        fatigue = trend == "down" and fav_trend in {"down", "mixed"}
        rows.append({
            "series_id": sid,
            "content_count": int(len(group)),
            "total_views": int(group["views"].sum()),
            "avg_favorite_rate": float(group["favorite_rate"].mean()),
            "avg_follow_rate": float(group["follow_rate"].mean()),
            "avg_consultation_rate": float(group["consultation_rate"].mean()),
            "total_revenue": float(group["revenue"].sum()),
            "avg_revenue_per_content": float(group["revenue"].mean()),
            "recent_3_trend": trend,
            "fatigue_warning": bool(fatigue),
        })
    return pd.DataFrame(rows).sort_values("total_revenue", ascending=False)


def product_series_performance(products_df: pd.DataFrame) -> pd.DataFrame:
    if products_df.empty or "series_id" not in products_df.columns:
        return pd.DataFrame()
    df = products_df[products_df["series_id"].fillna("").astype(str).ne("")].copy()
    if df.empty:
        return pd.DataFrame()
    df["conversion_rate"] = _rate(df["conversions"], df["clicks"]).fillna(0)
    df["refund_rate"] = _rate(df["refund_count"], df["conversions"]).fillna(0)
    df["gross_margin"] = _rate(df["revenue"] - df["cost"] * df["conversions"], df["revenue"]).fillna(0)
    rows = []
    for sid, group in df.groupby("series_id"):
        strongest = group.sort_values(["revenue", "conversion_rate"], ascending=False).iloc[0]
        weakest = group.sort_values(["revenue", "conversion_rate"], ascending=True).iloc[0]
        recommendation = "keep_and_expand" if strongest["conversion_rate"] >= group["conversion_rate"].median() else "improve"
        if float(group["refund_rate"].mean()) > 0.12:
            recommendation = "improve_quality"
        rows.append({
            "series_id": sid,
            "product_count": int(len(group)),
            "total_revenue": float(group["revenue"].sum()),
            "avg_conversion_rate": float(group["conversion_rate"].mean()),
            "avg_rating": float(group["avg_rating"].mean()),
            "refund_rate": float(group["refund_rate"].mean()),
            "gross_margin": float(group["gross_margin"].mean()),
            "strongest_variant": strongest["product_name"],
            "weakest_variant": weakest["product_name"],
            "recommendation": recommendation,
        })
    return pd.DataFrame(rows).sort_values("total_revenue", ascending=False)


def marginal_gain_of_new_item(df: pd.DataFrame, group_col: str, time_col: str, outcome_col: str) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for group_id, group in df.dropna(subset=[group_col]).groupby(group_col):
        ordered = group.sort_values(time_col).reset_index(drop=True)
        for idx, row in ordered.iterrows():
            prev = ordered.iloc[:idx]
            historical_avg = float(prev[outcome_col].mean()) if not prev.empty else np.nan
            prev_value = float(prev.iloc[-1][outcome_col]) if not prev.empty else np.nan
            current = float(row[outcome_col])
            gain = current - historical_avg if not np.isnan(historical_avg) else 0.0
            lift = gain / abs(historical_avg) if historical_avg and not np.isnan(historical_avg) else 0.0
            rows.append({
                "item_id": row.get("content_id", row.get("product_id", idx)),
                "group": group_id,
                "current_value": current,
                "historical_avg": historical_avg,
                "previous_value": prev_value,
                "marginal_gain": gain,
                "relative_lift": lift,
                "confidence_level": "low" if len(prev) < 3 else "medium",
                "warning": "建议继续观察。" if len(prev) < 3 else "",
            })
    return pd.DataFrame(rows)


def cannibalization_check(df: pd.DataFrame, series_col: str, time_col: str, outcome_col: str) -> pd.DataFrame:
    if df.empty or series_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for sid, group in df.groupby(series_col):
        ordered = group.sort_values(time_col)
        if len(ordered) < 4:
            risk, explanation = "low", "样本不足，暂未发现明显内部蚕食。"
        else:
            first = float(ordered[outcome_col].head(2).mean())
            last = float(ordered[outcome_col].tail(2).mean())
            total_up = float(ordered[outcome_col].tail(2).sum()) > float(ordered[outcome_col].head(2).sum())
            if last < first * 0.72 and not total_up:
                risk, explanation = "high", "后续项目明显下降，系列总量也未增长，可能在消耗同一批注意力。"
            elif last < first * 0.85:
                risk, explanation = "medium", "单项表现下降，但系列总量仍可能增加，建议换角度继续验证。"
            else:
                risk, explanation = "low", "暂未看到明显蚕食，系列仍有增量空间。"
        rows.append({"series_id": sid, "cannibalization_risk": risk, "explanation": explanation})
    return pd.DataFrame(rows)


def recommend_series_strategy(contents_df: pd.DataFrame, products_df: pd.DataFrame) -> list[dict]:
    recs = []
    content_perf = content_series_performance(contents_df)
    for _, row in content_perf.head(5).iterrows():
        priority = "high" if not row["fatigue_warning"] and row["total_revenue"] > content_perf["total_revenue"].median() else "medium"
        action = "继续做系列，但下一条要换应用场景或难度层级。" if row["fatigue_warning"] else "保留系列节奏，并把高收藏内容转成资料包/咨询入口。"
        recs.append({"title": f"内容系列：{row['series_id']}", "reason": f"总收入 ¥{row['total_revenue']:.0f}，最近趋势 {row['recent_3_trend']}。", "action": action, "priority": priority})
    product_perf = product_series_performance(products_df)
    for _, row in product_perf.head(5).iterrows():
        recs.append({"title": f"产品系列：{row['series_id']}", "reason": f"最强变体是 {row['strongest_variant']}，退款率 {row['refund_rate']:.1%}。", "action": "保留强变体，弱变体先改定价/材料/功能再继续投放。", "priority": "high" if row["recommendation"] == "keep_and_expand" else "medium"})
    return recs
