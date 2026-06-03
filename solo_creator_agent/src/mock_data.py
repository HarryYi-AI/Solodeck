from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

PLATFORMS = [
    "xiaohongshu",
    "bilibili",
    "douyin",
    "wechat",
    "zhihu",
    "youtube",
    "tiktok",
    "instagram",
    "substack",
    "x",
]
TOPICS = ["AI工具", "个人IP", "商业案例", "知识付费", "副业增长", "职场效率", "内容方法论", "产品化服务"]
CONTENT_TYPES = ["tutorial", "story", "review", "listicle", "opinion", "case_study"]
TITLE_STYLES = ["pain_point", "tutorial", "number", "story", "contrast", "result_oriented", "question"]
COVER_STYLES = ["face_text", "screenshot", "minimal", "contrast", "case_result", "product_demo"]


def _title(topic: str, style: str, idx: int) -> str:
    templates = {
        "pain_point": f"你做{topic}没结果，通常卡在这 3 个问题",
        "tutorial": f"{topic}实操指南：从 0 到 1 完整流程",
        "number": f"提升{topic}转化的 7 个动作",
        "story": f"我用一个真实案例跑通了{topic}",
        "contrast": f"{topic}高手和新手的区别在哪里",
        "result_oriented": f"用{topic}拿到第一笔稳定收入",
        "question": f"{topic}还值得做吗？这是我的数据复盘",
    }
    return f"{templates.get(style, topic)} #{idx:02d}"


def generate_mock_contents(n: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = datetime.now() - timedelta(days=95)
    for i in range(n):
        platform = PLATFORMS[i % len(PLATFORMS)]
        topic = TOPICS[(i * 3 + rng.integers(0, 3)) % len(TOPICS)]
        style = TITLE_STYLES[(i * 2 + rng.integers(0, 2)) % len(TITLE_STYLES)]
        ctype = "case_study" if topic == "商业案例" and i % 2 == 0 else CONTENT_TYPES[i % len(CONTENT_TYPES)]

        base_views = {
            "xiaohongshu": 8500,
            "bilibili": 12000,
            "douyin": 18000,
            "wechat": 5200,
            "zhihu": 7000,
            "youtube": 16000,
            "tiktok": 26000,
            "instagram": 14000,
            "substack": 4200,
            "x": 9000,
        }[platform]
        if style == "pain_point":
            base_views *= 1.35
        if ctype == "case_study":
            base_views *= 1.18
        views = int(max(400, rng.normal(base_views, base_views * 0.32)))
        impressions = int(views * rng.uniform(1.15, 2.8))
        followers_before = int(rng.integers(800, 150000))

        favorite_rate = 0.035 + rng.normal(0, 0.008)
        if platform in ["xiaohongshu", "instagram"]:
            favorite_rate += 0.055
        if style == "tutorial":
            favorite_rate += 0.035
        if topic in ["AI工具", "内容方法论"]:
            favorite_rate += 0.018

        follow_rate = 0.006 + rng.normal(0, 0.002)
        if platform in ["douyin", "bilibili", "youtube", "tiktok"]:
            follow_rate += 0.005
        if topic in ["个人IP", "副业增长"]:
            follow_rate += 0.004

        consult_rate = 0.001 + rng.normal(0, 0.0005)
        if platform in ["wechat", "substack"]:
            consult_rate += 0.006
        if topic in ["商业案例", "产品化服务", "知识付费"]:
            consult_rate += 0.003

        conversion_rate = 0.18
        if platform in ["wechat", "substack"]:
            conversion_rate += 0.18
        if topic == "商业案例":
            conversion_rate += 0.16
        if topic == "知识付费":
            conversion_rate += 0.10

        consultations = int(max(0, views * max(0, consult_rate)))
        conversions = int(max(0, consultations * max(0.05, min(0.65, conversion_rate + rng.normal(0, 0.06)))))
        likes = int(views * max(0.01, 0.045 + rng.normal(0, 0.012)))
        favorites = int(views * max(0.005, favorite_rate))
        comments = int(views * max(0.001, 0.008 + rng.normal(0, 0.003)))
        shares = int(views * max(0.001, 0.006 + rng.normal(0, 0.002)))
        new_followers = int(views * max(0.001, follow_rate))
        completion_rate = float(np.clip(rng.normal(0.42, 0.12) + (0.12 if platform in ["bilibili", "youtube"] else 0), 0.05, 0.95))
        production_hours = float(max(1.0, rng.normal(4.5, 1.5)))
        if platform in ["bilibili", "youtube"]:
            production_hours += 3.0
        if ctype == "case_study":
            production_hours += 1.5

        revenue = conversions * rng.uniform(280, 900)
        if topic == "商业案例":
            revenue *= 2.0
        if topic == "知识付费":
            revenue *= 1.5
        if platform in ["wechat", "substack"]:
            revenue *= 1.45

        rows.append({
            "content_id": f"C{i + 1:04d}",
            "title": _title(topic, style, i + 1),
            "platform": platform,
            "topic": topic,
            "content_type": ctype,
            "publish_time": (start + timedelta(days=i, hours=int(rng.integers(8, 23)))).isoformat(timespec="seconds"),
            "title_style": style,
            "cover_style": COVER_STYLES[(i + rng.integers(0, 2)) % len(COVER_STYLES)],
            "duration_sec": int(rng.integers(40, 900)),
            "production_hours": round(production_hours, 1),
            "followers_before": followers_before,
            "impressions": impressions,
            "views": views,
            "likes": likes,
            "favorites": favorites,
            "comments": comments,
            "shares": shares,
            "completion_rate": round(completion_rate, 4),
            "new_followers": new_followers,
            "consultations": consultations,
            "conversions": conversions,
            "revenue": round(float(revenue), 2),
            "ad_spend": round(float(rng.choice([0, 0, 0, rng.uniform(100, 1200)])), 2),
            "is_sponsored": bool(i % 11 == 0),
        })
    df = pd.DataFrame(rows)
    extra = []
    math_titles = ["傅里叶变换入门", "3分钟理解傅里叶变换", "拉普拉斯变换怎么用", "拉马努金变换直觉", "小波变换和 FFT 区别", "快速傅里叶变换 FFT 实战"]
    for j, title in enumerate(math_titles):
        views = [28000, 22000, 15000, 9500, 8200, 12000][j]
        extra.append({
            **df.iloc[j].to_dict(),
            "content_id": f"MATH{j+1:03d}",
            "title": title,
            "platform": ["bilibili", "douyin", "wechat", "zhihu", "bilibili", "youtube"][j],
            "topic": "数学变换",
            "body": f"{title} 的可视化解释和应用场景",
            "tags": "mathematics, signal_processing, transformation",
            "series_id": "math_transform_series",
            "parent_content_id": "MATH001" if j in [1, 4, 5] else "",
            "content_similarity_group": "math_transform",
            "knowledge_domain": "mathematics",
            "difficulty_level": ["beginner", "beginner", "intermediate", "advanced", "advanced", "advanced"][j],
            "novelty_score": [0.72, 0.45, 0.68, 0.86, 0.78, 0.73][j],
            "duplication_risk": [0.22, 0.72, 0.28, 0.18, 0.25, 0.20][j],
            "user_fatigue_risk": [0.1, 0.28, 0.35, 0.48, 0.55, 0.42][j],
            "views": views,
            "favorites": int(views * [0.06, 0.05, 0.09, 0.14, 0.13, 0.12][j]),
            "new_followers": int(views * [0.012, 0.008, 0.009, 0.006, 0.006, 0.007][j]),
            "consultations": int(views * [0.001, 0.001, 0.002, 0.005, 0.004, 0.004][j]),
            "conversions": [2, 1, 3, 8, 5, 6][j],
            "revenue": [800, 300, 1500, 7800, 4200, 5200][j],
        })
    repeat_titles = ["AI工具效率提升指南", "AI工具效率提升的3个方法", "AI工具效率提升避坑", "AI工具效率提升复盘", "AI工具效率提升还值得做吗"]
    for j, title in enumerate(repeat_titles):
        views = [18000, 16800, 11200, 7600, 5200][j]
        extra.append({
            **df.iloc[j + 10].to_dict(),
            "content_id": f"REP{j+1:03d}",
            "title": title,
            "topic": "AI工具",
            "body": "同一主题连续发布，用于模拟重复疲劳风险",
            "tags": "AI_tools, productivity",
            "series_id": "ai_tools_repeat_series",
            "content_similarity_group": "ai_efficiency_repeat",
            "knowledge_domain": "AI_tools",
            "difficulty_level": "beginner",
            "novelty_score": max(0.2, 0.7 - j * 0.12),
            "duplication_risk": min(0.9, 0.35 + j * 0.12),
            "user_fatigue_risk": min(0.9, 0.2 + j * 0.15),
            "views": views,
            "favorites": int(views * max(0.03, 0.08 - j * 0.008)),
            "new_followers": int(views * max(0.003, 0.012 - j * 0.002)),
            "revenue": [1800, 2100, 1900, 1500, 1200][j],
        })
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


def generate_mock_products(seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    specs = [
        ("P-AI-001", "SoloBot 桌面陪伴机器人基础版", "hardware_device", "solobot", "plastic", "white", "minimal", "voice_interaction, alarm, desktop_decoration", 699, 310, False, ""),
        ("P-AI-002", "SoloBot 情绪陪伴版", "hardware_device", "solobot", "plastic", "green", "cute", "voice_interaction, emotion_companion, music_playback", 899, 360, True, "P-AI-001"),
        ("P-AI-003", "SoloBot 触摸互动版", "hardware_device", "solobot", "silicone", "white", "cute", "touch_interaction, emotion_companion, alarm", 999, 430, True, "P-AI-001"),
        ("P-AI-004", "SoloBot 闹钟轻量版", "hardware_device", "solobot", "plastic", "blue", "minimal", "alarm, desktop_decoration", 499, 220, False, "P-AI-001"),
        ("P-LAMP-001", "FoldDesk 台灯黑色款", "home_product", "desk_lamp", "aluminum", "black", "foldable", "eye_protection, dimming", 199, 88, False, ""),
        ("P-LAMP-002", "FoldDesk 台灯奶油白款", "home_product", "desk_lamp", "aluminum", "white", "foldable", "eye_protection, dimming", 219, 90, False, "P-LAMP-001"),
        ("P-CASE-001", "MagCase 透明手机壳", "accessory", "phone_case", "tpu", "clear", "simple", "anti_drop, magnetic", 59, 18, False, ""),
        ("P-CASE-002", "MagCase 彩色手机壳", "accessory", "phone_case", "tpu", "pink", "cute", "anti_drop, magnetic", 69, 20, False, "P-CASE-001"),
        ("P-NOTE-001", "FocusNote 周计划本", "accessory", "notebook", "paper", "green", "planner", "weekly_plan, habit_tracking", 49, 16, False, ""),
    ]
    for i, (pid, name, cat, sid, material, color, style, tags, price, cost, newv, parent) in enumerate(specs):
        views = int(rng.integers(3000, 28000))
        clicks = int(views * rng.uniform(0.05, 0.18))
        consultations = int(clicks * rng.uniform(0.08, 0.28))
        conv_base = 0.08 + (0.08 if "emotion_companion" in tags else 0) + (0.04 if material in ["silicone", "aluminum"] else 0) - (0.04 if price > 900 else 0)
        conversions = int(max(1, consultations * max(0.03, conv_base)))
        rows.append({
            "product_id": pid, "product_name": name, "category": cat, "series_id": sid,
            "launch_date": (date.today() - timedelta(days=80 - i * 7)).isoformat(),
            "price": price, "cost": cost, "material": material, "color": color, "style": style, "size": "standard", "weight": round(float(rng.uniform(0.1, 1.2)), 2),
            "feature_tags": tags, "target_user": "creator, solo_company" if cat == "hardware_device" else "office_user",
            "platform": PLATFORMS[i % 5], "views": views, "clicks": clicks, "consultations": consultations, "conversions": conversions,
            "revenue": round(float(conversions * price), 2), "refund_count": int(conversions * rng.uniform(0, 0.12)), "review_count": int(conversions * rng.uniform(0.4, 1.4)),
            "avg_rating": round(float(np.clip(4.0 + rng.normal(0, 0.35) + (0.35 if "touch_interaction" in tags else 0), 2.8, 5.0)), 2),
            "is_new_version": newv, "parent_product_id": parent,
        })
    return pd.DataFrame(rows)


def generate_mock_feedback(products: pd.DataFrame, contents: pd.DataFrame, seed: int = 13) -> pd.DataFrame:
    texts = [
        ("价格有点高，但情绪陪伴功能很吸引我", "pricing", "negative", "high"),
        ("外观很可爱，放在桌面很有陪伴感", "design", "positive", "low"),
        ("语音反应有点慢，希望下一版更快", "performance", "negative", "high"),
        ("触摸互动很惊喜，但我还不确定会不会长期使用", "emotional_value", "neutral", "medium"),
        ("希望增加日程提醒和更多音乐", "feature_request", "neutral", "medium"),
        ("傅里叶变换讲得清楚，但后面几期有点重复", "content_clarity", "negative", "medium"),
        ("高级数学内容收藏价值很高，适合做资料包", "emotional_value", "positive", "low"),
        ("手机壳颜色好看但转化不强，想看真实上手图", "trust", "negative", "medium"),
    ]
    rows = []
    for i in range(36):
        text, issue, sentiment, severity = texts[i % len(texts)]
        p = products.iloc[i % len(products)]
        c = contents.iloc[i % len(contents)]
        rows.append({
            "feedback_id": f"F{i+1:04d}", "user_id": f"U{i+1:04d}", "source_type": ["beta_test", "review", "comment", "survey"][i % 4],
            "related_content_id": c["content_id"] if i % 3 == 0 else "", "related_product_id": p["product_id"] if i % 3 != 0 else "",
            "user_segment": ["creator", "student", "office_user", "small_shop"][i % 4], "country": ["CN", "US", "SG"][i % 3], "platform": p["platform"],
            "feedback_text": text, "rating": [3, 5, 2, 4, 4][i % 5], "sentiment": sentiment, "issue_type": issue, "severity": severity,
            "created_at": (datetime.now() - timedelta(days=20 - i % 18)).isoformat(timespec="seconds"),
            "converted_after_feedback": bool(i % 4 == 0), "retained_after_feedback": bool(i % 3 != 0),
        })
    return pd.DataFrame(rows)


def generate_mock_beta_tests(products: pd.DataFrame, seed: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    product_ids = products["product_id"].head(4).tolist()
    for i in range(64):
        treatment = i % 2 == 0
        feature = ["emotion_companion", "touch_interaction", "alarm"][i % 3]
        activated = rng.random() < (0.72 if treatment and feature != "alarm" else 0.55)
        retained = rng.random() < (0.46 if treatment and feature == "emotion_companion" else 0.33)
        converted = rng.random() < (0.22 if treatment and feature == "emotion_companion" else 0.12)
        rows.append({
            "beta_test_id": f"BT{i+1:04d}", "product_id": product_ids[i % len(product_ids)], "feature_name": feature,
            "test_group": "treatment" if treatment else "control", "user_id": f"BU{i+1:04d}", "user_segment": ["creator", "office_user", "student"][i % 3],
            "invited_at": (datetime.now() - timedelta(days=18 - i % 12)).isoformat(timespec="seconds"),
            "experienced_at": (datetime.now() - timedelta(days=16 - i % 12)).isoformat(timespec="seconds") if activated else "",
            "feedback_submitted": bool(rng.random() < 0.72), "rating": round(float(np.clip(rng.normal(4.1 if treatment else 3.7, 0.7), 1, 5)), 1),
            "activated": bool(activated), "retained_7d": bool(retained), "converted": bool(converted),
            "revenue": float(899 if converted else 0), "notes": "mock beta test",
        })
    return pd.DataFrame(rows)


def generate_mock_revenues(contents: pd.DataFrame, n: int = 30, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    revenue_types = ["brand_ads", "platform_share", "course", "consulting", "membership", "affiliate", "reward", "service"]
    rows = []
    sampled = contents.sample(n=min(n, len(contents)), random_state=seed).reset_index(drop=True)
    for i in range(n):
        c = sampled.iloc[i % len(sampled)]
        rtype = revenue_types[(i + rng.integers(0, 3)) % len(revenue_types)]
        amount = max(180, c["revenue"] * rng.uniform(0.25, 0.9))
        if rtype in ["brand_ads", "consulting", "service"]:
            amount *= 1.7
        rows.append({
            "revenue_id": f"R{i + 1:04d}",
            "date": (pd.to_datetime(c["publish_time"]).date() + timedelta(days=int(rng.integers(1, 12)))).isoformat(),
            "amount": round(float(amount), 2),
            "revenue_type": rtype,
            "platform": c["platform"],
            "content_id": c["content_id"] if i % 4 != 0 else "",
            "client_name": f"{['青石', '北岸', '映川', '云谷', '晴山'][i % 5]}品牌" if rtype == "brand_ads" else "",
            "status": "pending" if i in [4, 17, 25] else "received",
            "note": "mock revenue",
        })
    return pd.DataFrame(rows)


def generate_mock_campaigns(contents: pd.DataFrame) -> pd.DataFrame:
    today = date.today()
    statuses = ["negotiating", "confirmed", "scripting", "reviewing", "published", "reporting", "completed", "published", "completed", "reviewing"]
    payment = ["unpaid", "deposit_received", "fully_paid", "unpaid", "overdue", "deposit_received", "fully_paid", "overdue", "fully_paid", "unpaid"]
    invoice = ["pending", "not_needed", "issued", "pending", "pending", "issued", "issued", "pending", "not_needed", "pending"]
    report = ["not_started", "not_started", "not_started", "not_started", "not_started", "not_started", "sent", "not_started", "sent", "generated"]
    rows = []
    related = contents.sort_values("revenue", ascending=False).head(10).reset_index(drop=True)
    for i in range(10):
        c = related.iloc[i]
        rows.append({
            "campaign_id": f"B{i + 1:03d}",
            "brand_name": f"{['青石AI', '北岸效率', '映川教育', '云谷SaaS', '晴山咖啡'][i % 5]}",
            "campaign_name": f"{c['topic']}合作推广",
            "platform": c["platform"],
            "deliverables": "1篇图文 + 1条短视频" if i % 2 else "1篇深度内容",
            "price": round(float(max(3000, c["views"] * 0.8 + c["revenue"] * 0.25)), 0),
            "deadline": (today + timedelta(days=[2, 5, 8, 1, -3, 10, -1, -5, 12, 3][i])).isoformat(),
            "status": statuses[i],
            "payment_status": payment[i],
            "invoice_status": invoice[i],
            "revision_count": [1, 0, 2, 3, 1, 4, 1, 2, 0, 3][i],
            "report_status": report[i],
            "related_content_id": c["content_id"],
        })
    return pd.DataFrame(rows)


def generate_mock_ab_tests(contents: pd.DataFrame, n: int = 20, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    candidates = contents.sample(n=min(n, len(contents)), random_state=seed).reset_index(drop=True)
    for i in range(n):
        group = "treatment" if i % 2 == 0 else "control"
        metric = "favorite_rate" if i < 10 else "conversion_rate"
        c = candidates.iloc[i]
        base = c["favorites"] / max(c["views"], 1) if metric == "favorite_rate" else c["conversions"] / max(c["views"], 1)
        lift = 1.22 if group == "treatment" else 1.0
        value = max(0, base * lift + rng.normal(0, base * 0.08 + 0.0002))
        rows.append({
            "experiment_id": "EXP-title-style-001" if i < 10 else "EXP-case-cta-002",
            "date": (pd.to_datetime(c["publish_time"]).date() + timedelta(days=1)).isoformat(),
            "platform": c["platform"],
            "topic": c["topic"],
            "treatment_name": "title_style" if i < 10 else "cta_position",
            "treatment_value": "pain_point" if i < 10 else "early_cta",
            "control_value": "tutorial" if i < 10 else "ending_cta",
            "outcome_metric": metric,
            "group": group,
            "content_id": c["content_id"],
            "outcome_value": round(float(value), 6),
            "covariates_json": json.dumps({"platform": c["platform"], "topic": c["topic"]}, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def generate_all(output_dir: str | Path = DATA_DIR) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    contents = generate_mock_contents()
    revenues = generate_mock_revenues(contents)
    campaigns = generate_mock_campaigns(contents)
    ab_tests = generate_mock_ab_tests(contents)
    products = generate_mock_products()
    feedback = generate_mock_feedback(products, contents)
    beta_tests = generate_mock_beta_tests(products)
    contents.to_csv(out / "mock_contents.csv", index=False)
    revenues.to_csv(out / "mock_revenues.csv", index=False)
    campaigns.to_csv(out / "mock_campaigns.csv", index=False)
    ab_tests.to_csv(out / "mock_ab_tests.csv", index=False)
    products.to_csv(out / "mock_products.csv", index=False)
    feedback.to_csv(out / "mock_feedback.csv", index=False)
    beta_tests.to_csv(out / "mock_beta_tests.csv", index=False)


if __name__ == "__main__":
    generate_all()
    print(f"Mock data generated in {DATA_DIR}")
