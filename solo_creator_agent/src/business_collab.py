from datetime import date, timedelta

import numpy as np
import pandas as pd


PLATFORM_CPM = {
    "xiaohongshu": 80,
    "bilibili": 70,
    "douyin": 65,
    "wechat": 120,
    "zhihu": 90,
    "youtube": 85,
    "tiktok": 75,
    "instagram": 95,
    "substack": 140,
    "x": 60,
}


def campaign_dashboard(campaigns: pd.DataFrame) -> dict:
    return {
        "total_campaigns": int(len(campaigns)),
        "active": int(campaigns[~campaigns["status"].isin(["completed"])].shape[0]),
        "reviewing": int(campaigns[campaigns["status"].eq("reviewing")].shape[0]),
        "to_publish": int(campaigns[campaigns["status"].isin(["confirmed", "scripting", "reviewing"])].shape[0]),
        "pending_payment": int(campaigns[campaigns["payment_status"].isin(["unpaid", "overdue", "deposit_received"])].shape[0]),
        "pending_invoice": int(campaigns[campaigns["invoice_status"].eq("pending")].shape[0]),
        "pending_report": int(campaigns[(campaigns["status"].isin(["published", "reporting"])) & campaigns["report_status"].eq("not_started")].shape[0]),
    }


def pricing_suggestion(contents: pd.DataFrame, campaigns: pd.DataFrame, platform: str, deliverables: str, base_cost: float, usage_rights: bool = False, exclusive: bool = False, urgent: bool = False) -> dict:
    recent = contents[contents["platform"].eq(platform)].sort_values("publish_time", ascending=False).head(10)
    avg_views = recent["views"].mean() if not recent.empty else contents["views"].mean()
    cpm = PLATFORM_CPM.get(platform, 70)
    base_price = avg_views * cpm / 1000 + base_cost
    multiplier = 1.0
    factors = []
    if usage_rights:
        multiplier += 0.20
        factors.append("包含素材使用权 +20%")
    if exclusive:
        multiplier += 0.30
        factors.append("竞品排他 +30%")
    if urgent:
        multiplier += 0.15
        factors.append("加急交付 +15%")
    avg_revision = campaigns["revision_count"].mean() if not campaigns.empty else 0
    revision_risk = min(0.20, max(0, avg_revision - 1) * 0.07)
    multiplier += revision_risk
    mid = base_price * multiplier
    return {
        "platform": platform,
        "deliverables": deliverables,
        "avg_recent_views": float(avg_views),
        "cpm": cpm,
        "low": round(mid * 0.85, 0),
        "mid": round(mid, 0),
        "high": round(mid * 1.25, 0),
        "explanation": f"近 10 条同平台平均播放 {avg_views:.0f}，CPM 系数 {cpm}，制作成本 {base_cost:.0f}，修订风险 +{revision_risk:.0%}。{'；'.join(factors) if factors else '无额外授权/排他/加急加价。'}",
    }


def campaign_risk_alerts(campaigns: pd.DataFrame, language: str = "中文") -> list[dict]:
    zh = language == "中文"
    alerts = []
    today = date.today()
    for _, row in campaigns.iterrows():
        deadline = pd.to_datetime(row["deadline"]).date()
        if deadline <= today + timedelta(days=3) and row["status"] != "published" and row["status"] != "completed":
            alerts.append({"level": "high", "campaign_id": row["campaign_id"], "title": "交付截止临近" if zh else "Deadline approaching", "detail": f"{row['brand_name']} {row['campaign_name']} 距截止 {max((deadline - today).days, 0)} 天，当前状态 {row['status']}。" if zh else f"{row['brand_name']} {row['campaign_name']} is due in {max((deadline - today).days, 0)} days. Current status: {row['status']}."})
        if row["payment_status"] == "overdue":
            alerts.append({"level": "high", "campaign_id": row["campaign_id"], "title": "合作款逾期" if zh else "Payment overdue", "detail": f"{row['brand_name']} 待收 {row['price']:.0f} 元，建议今天跟进付款。" if zh else f"{row['brand_name']} has ¥{row['price']:.0f} pending. Follow up payment today."})
        if row["invoice_status"] == "pending":
            alerts.append({"level": "medium", "campaign_id": row["campaign_id"], "title": "发票待处理" if zh else "Invoice pending", "detail": f"{row['brand_name']} 发票状态为 pending。" if zh else f"{row['brand_name']} invoice status is pending."})
        if int(row["revision_count"]) > 2:
            alerts.append({"level": "medium", "campaign_id": row["campaign_id"], "title": "修订次数偏高" if zh else "Too many revisions", "detail": f"已修订 {row['revision_count']} 次，建议锁定需求边界。" if zh else f"{row['revision_count']} revisions so far. Lock scope and acceptance criteria."})
        if row["status"] == "published" and row["report_status"] == "not_started":
            alerts.append({"level": "low", "campaign_id": row["campaign_id"], "title": "待生成复盘报告" if zh else "Brand report needed", "detail": f"{row['campaign_name']} 已发布，需要给甲方发送数据复盘。" if zh else f"{row['campaign_name']} has been published. Send the brand performance report."})
    return alerts


def generate_brand_report(campaign: dict | pd.Series, content_metrics: dict | pd.Series) -> str:
    c = campaign.to_dict() if hasattr(campaign, "to_dict") else dict(campaign)
    m = content_metrics.to_dict() if hasattr(content_metrics, "to_dict") else dict(content_metrics)
    return f"""# {c.get('brand_name', '')} 合作复盘报告

## 合作概览
- 项目：{c.get('campaign_name', '')}
- 平台：{c.get('platform', '')}
- 交付物：{c.get('deliverables', '')}
- 合作金额：¥{float(c.get('price', 0)):.0f}

## 发布内容
- 内容标题：{m.get('title', '待补充')}
- 内容 ID：{c.get('related_content_id', '待补充')}

## 数据表现
- 播放/阅读：{int(m.get('views', 0))}
- 点赞：{int(m.get('likes', 0))}
- 收藏：{int(m.get('favorites', 0))}
- 评论：{int(m.get('comments', 0))}
- 分享：{int(m.get('shares', 0))}
- 咨询：{int(m.get('consultations', 0))}
- 转化：{int(m.get('conversions', 0))}

## 互动亮点
本次内容在目标人群中完成了基础曝光和互动，可结合评论区问题沉淀下一轮选题。

## 评论摘要
可补充评论文本，用于提炼用户关注点、疑虑和购买意图。

## 转化数据
当前记录转化 {int(m.get('conversions', 0))} 次，建议与品牌方核对站外成交与优惠码数据。

## 下次合作建议
建议延续表现较好的标题风格和发布时间段，并在内容中增加明确 CTA 与福利承接页。
"""
