from __future__ import annotations

import math
import random
import textwrap
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo_upload_pack"
RNG = random.Random(20260603)


PLATFORMS = ["xiaohongshu", "bilibili", "douyin", "wechat"]
PLATFORM_ZH = {
    "xiaohongshu": "小红书",
    "bilibili": "B站",
    "douyin": "抖音",
    "wechat": "公众号/视频号",
}
TOPICS = ["商用机器人", "桌面陪伴机器人", "AI工具", "副业增长", "产品化服务", "商业案例"]
TITLE_STYLES = ["pain_point", "tutorial", "case_study", "number", "result_oriented", "question"]
STYLE_ZH = {
    "pain_point": "痛点标题",
    "tutorial": "教程标题",
    "case_study": "案例复盘",
    "number": "数字清单",
    "result_oriented": "结果导向",
    "question": "提问标题",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for item in candidates:
        try:
            return ImageFont.truetype(item, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str, size=24, fill="#111827", bold=False, max_width: int | None = None, line_gap=8):
    f = font(size, bold=bold)
    if not max_width:
        draw.text(xy, text, font=f, fill=fill)
        return
    lines = []
    current = ""
    for char in text:
        test = current + char
        if draw.textbbox((0, 0), test, font=f)[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = char
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=f, fill=fill)
        y += size + line_gap


def rounded(draw: ImageDraw.ImageDraw, xy, radius=18, fill="#ffffff", outline="#e5e7eb", width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def money(value: float) -> str:
    return f"¥{value:,.0f}"


def generate_contents() -> pd.DataFrame:
    rows = []
    start = datetime.now() - timedelta(days=45)
    content_group = 1
    for i in range(36):
        platform = PLATFORMS[i % len(PLATFORMS)]
        topic = TOPICS[i % len(TOPICS)]
        style = TITLE_STYLES[i % len(TITLE_STYLES)]
        base = 9000 + (PLATFORMS.index(platform) * 2600) + RNG.randint(-1200, 2000)
        if platform == "douyin":
            base *= 1.55
        if platform == "wechat":
            base *= 0.75
        if style == "pain_point":
            base *= 1.25
        if style == "tutorial":
            favorite_boost = 1.55
        else:
            favorite_boost = 1.0
        if topic in {"商用机器人", "商业案例"}:
            revenue_boost = 1.8
        else:
            revenue_boost = 1.0
        views = max(600, int(base))
        favorites = int(views * RNG.uniform(0.035, 0.105) * favorite_boost)
        consultations = int(views * RNG.uniform(0.0015, 0.006) * (1.8 if platform == "wechat" else 1.0))
        conversions = max(0, int(consultations * RNG.uniform(0.18, 0.42)))
        revenue = conversions * RNG.choice([299, 499, 899, 1299]) * revenue_boost
        group_id = f"G-{content_group:03d}"
        if i % 4 == 3:
            content_group += 1
        title_templates = {
            "pain_point": f"为什么你的{topic}内容有流量却没成交？",
            "tutorial": f"{topic}从 0 到 1 实操指南",
            "case_study": f"一个真实案例：我们如何卖出第一批{topic}",
            "number": f"提升{topic}转化的 7 个动作",
            "result_oriented": f"用{topic}拿到第一笔稳定收入",
            "question": f"{topic}现在还值得做吗？",
        }
        rows.append(
            {
                "content_id": f"C{i+1:03d}",
                "content_group_id": group_id,
                "account_id": f"A{1 + i % 3}",
                "title": title_templates[style],
                "platform": platform,
                "topic": topic,
                "content_type": "case_study" if style == "case_study" else "tutorial",
                "title_style": style,
                "cover_style": RNG.choice(["clean_product", "before_after", "data_card", "human_scene"]),
                "publish_time": (start + timedelta(days=i, hours=RNG.choice([9, 12, 18, 21]))).isoformat(),
                "duration_sec": RNG.randint(45, 420),
                "production_hours": round(RNG.uniform(1.2, 7.5), 1),
                "followers_before": RNG.choice([38000, 120000, 910000]),
                "views": views,
                "likes": int(views * RNG.uniform(0.025, 0.085)),
                "favorites": favorites,
                "comments": int(views * RNG.uniform(0.004, 0.025)),
                "shares": int(views * RNG.uniform(0.003, 0.02)),
                "new_followers": int(views * RNG.uniform(0.001, 0.006)),
                "consultations": consultations,
                "conversions": conversions,
                "revenue": round(revenue, 2),
                "cost": round(RNG.uniform(120, 1400), 2),
                "ad_spend": round(RNG.uniform(0, 800), 2),
                "series_id": "robot_growth" if topic in {"商用机器人", "桌面陪伴机器人"} else "creator_ops",
                "novelty_score": round(RNG.uniform(0.35, 0.92), 2),
                "duplication_risk": round(RNG.uniform(0.05, 0.55), 2),
                "user_fatigue_risk": round(RNG.uniform(0.08, 0.60), 2),
            }
        )
    return pd.DataFrame(rows)


def generate_revenues(contents: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rev_types = ["brand_ads", "course", "consulting", "service", "affiliate"]
    for i in range(20):
        row = contents.sample(1, random_state=100 + i).iloc[0]
        rows.append(
            {
                "revenue_id": f"R{i+1:03d}",
                "date": (pd.to_datetime(row["publish_time"]).date() + timedelta(days=RNG.randint(1, 12))).isoformat(),
                "amount": round(max(399, row["revenue"] * RNG.uniform(0.5, 1.3) + RNG.randint(300, 3000)), 2),
                "revenue_type": rev_types[i % len(rev_types)],
                "platform": row["platform"],
                "content_id": row["content_id"],
                "product_id": "" if i % 3 else f"P{1 + i % 8:03d}",
                "client_name": RNG.choice(["北岸科技", "青石智能", "云谷教育", "晴山生活", "明河机器人"]),
                "status": "pending" if i in {3, 9, 14} else "received",
                "note": RNG.choice(["内容推广服务费", "产品咨询成交", "资料包销售", "商用机器人试用定金"]),
            }
        )
    return pd.DataFrame(rows)


def generate_campaigns(contents: pd.DataFrame) -> pd.DataFrame:
    brands = ["青石智能", "北岸效率", "明河机器人", "云谷SaaS", "晴山生活", "鲸川硬件"]
    statuses = ["negotiating", "confirmed", "scripting", "reviewing", "published", "reporting", "completed"]
    rows = []
    for i in range(8):
        content = contents.iloc[(i * 3) % len(contents)]
        rows.append(
            {
                "campaign_id": f"BIZ{i+1:03d}",
                "brand_name": brands[i % len(brands)],
                "campaign_name": RNG.choice(["新品种草", "商用机器人推广", "AI工具联合内容", "效率产品深度评测"]),
                "platform": content["platform"],
                "deliverables": RNG.choice(["1篇图文 + 1条短视频", "1篇深度内容", "2条短视频 + 数据复盘"]),
                "price": RNG.choice([6800, 9800, 12800, 18800, 26000]),
                "deadline": (date.today() + timedelta(days=RNG.choice([2, 5, 8, 14, 21]))).isoformat(),
                "status": statuses[i % len(statuses)],
                "payment_status": "overdue" if i in {1, 5} else RNG.choice(["unpaid", "deposit_received", "fully_paid"]),
                "invoice_status": "pending" if i in {2, 6} else RNG.choice(["not_needed", "issued"]),
                "revision_count": RNG.choice([0, 1, 1, 2, 3]),
                "report_status": "not_started" if i in {3, 4, 6} else RNG.choice(["generated", "sent"]),
                "related_content_id": content["content_id"],
            }
        )
    return pd.DataFrame(rows)


def generate_products() -> pd.DataFrame:
    products = [
        ("P001", "SoloBot 情绪陪伴款", "hardware_device", "emotion_companion,touch_interaction", 1299),
        ("P002", "SoloBot 商务接待款", "hardware_device", "voice_interaction,desktop_decoration", 1899),
        ("P003", "SoloBot 教育陪练款", "hardware_device", "voice_interaction,weekly_plan", 1599),
        ("P004", "智能桌面提醒器", "home_product", "alarm,habit_tracking", 399),
        ("P005", "创作者周计划模板", "digital_product", "weekly_plan,habit_tracking", 99),
        ("P006", "机器人采购避坑清单", "digital_product", "case_study,checklist", 199),
    ]
    rows = []
    for idx, (pid, name, category, features, price) in enumerate(products):
        views = RNG.randint(12000, 90000)
        clicks = int(views * RNG.uniform(0.08, 0.22))
        consultations = int(clicks * RNG.uniform(0.08, 0.22))
        conversions = int(consultations * RNG.uniform(0.12, 0.38))
        rows.append(
            {
                "product_id": pid,
                "product_name": name,
                "category": category,
                "series_id": "solobot" if "SoloBot" in name else "creator_tools",
                "launch_date": (date.today() - timedelta(days=70 - idx * 9)).isoformat(),
                "price": price,
                "cost": round(price * RNG.uniform(0.28, 0.52), 2),
                "material": RNG.choice(["ABS", "铝合金", "硅胶", "电子资料"]),
                "color": RNG.choice(["雾白", "石墨黑", "松绿色", "浅灰"]),
                "style": RNG.choice(["简约", "商务", "可爱", "效率"]),
                "size": RNG.choice(["S", "M", "L"]),
                "weight": RNG.randint(80, 900),
                "feature_tags": features,
                "target_user": RNG.choice(["内容创作者", "办公室用户", "教育机构", "小团队老板"]),
                "platform": PLATFORMS[idx % len(PLATFORMS)],
                "views": views,
                "clicks": clicks,
                "consultations": consultations,
                "conversions": conversions,
                "revenue": round(conversions * price, 2),
                "refund_count": RNG.randint(0, max(1, conversions // 12)),
                "review_count": RNG.randint(15, 240),
                "avg_rating": round(RNG.uniform(4.1, 4.9), 1),
                "is_new_version": idx in {0, 2, 5},
                "parent_product_id": "" if idx < 3 else f"P{idx:03d}",
            }
        )
    return pd.DataFrame(rows)


def generate_feedback(products: pd.DataFrame) -> pd.DataFrame:
    texts = [
        ("emotion_companion", "喜欢它主动提醒我休息，情绪陪伴很自然", "positive", "emotional_value", "low"),
        ("emotion_companion", "价格有点高，但如果能接日历我愿意买", "neutral", "pricing", "medium"),
        ("voice_interaction", "语音反应偶尔慢，商务场景会有点尴尬", "negative", "performance", "high"),
        ("touch_interaction", "触摸互动很有记忆点，适合放办公室展示", "positive", "design", "low"),
        ("weekly_plan", "希望能自动生成周计划和复盘提醒", "neutral", "feature_request", "medium"),
        ("habit_tracking", "模板好用，但新手不知道第一步做什么", "negative", "usability", "medium"),
    ]
    rows = []
    for i in range(28):
        feature, text, sentiment, issue, severity = texts[i % len(texts)]
        product = products.iloc[i % len(products)]
        rows.append(
            {
                "feedback_id": f"F{i+1:03d}",
                "user_id": f"U{1000+i}",
                "source_type": RNG.choice(["beta_test", "survey", "comment", "customer_service"]),
                "related_content_id": "" if i % 2 else f"C{1 + i % 20:03d}",
                "related_product_id": product["product_id"],
                "user_segment": RNG.choice(["creator", "office_user", "small_shop", "education"]),
                "country": "CN",
                "platform": product["platform"],
                "feedback_text": text,
                "rating": RNG.randint(3, 5),
                "sentiment": sentiment,
                "issue_type": issue,
                "severity": severity,
                "created_at": (datetime.now() - timedelta(days=RNG.randint(1, 20))).isoformat(),
                "converted_after_feedback": RNG.choice([0, 1]),
                "retained_after_feedback": RNG.choice([0, 1, 1]),
            }
        )
    return pd.DataFrame(rows)


def generate_ab_tests(contents: pd.DataFrame) -> pd.DataFrame:
    rows = []
    experiments = [
        ("EXP-title-robot-001", "xiaohongshu", "商用机器人", "标题风格", "pain_point", "tutorial", "favorite_rate", 0.086, 0.052),
        ("EXP-cta-wechat-002", "wechat", "产品化服务", "咨询入口", "early_cta", "ending_cta", "conversion_rate", 0.006, 0.004),
        ("EXP-cover-douyin-003", "douyin", "桌面陪伴机器人", "封面风格", "product_scene", "plain_text", "follow_rate", 0.009, 0.006),
    ]
    idx = 1
    for exp_id, platform, topic, name, treatment, control, metric, t_mean, c_mean in experiments:
        for group, mean in [("treatment", t_mean), ("control", c_mean)]:
            for _ in range(6):
                rows.append(
                    {
                        "experiment_id": exp_id,
                        "date": (date.today() - timedelta(days=RNG.randint(1, 18))).isoformat(),
                        "platform": platform,
                        "topic": topic,
                        "treatment_name": name,
                        "treatment_value": treatment,
                        "control_value": control,
                        "outcome_metric": metric,
                        "group": group,
                        "content_id": contents.iloc[idx % len(contents)]["content_id"],
                        "outcome_value": round(max(0, RNG.gauss(mean, mean * 0.18)), 4),
                        "covariates_json": '{"account_id":"A1","same_topic":true}',
                    }
                )
                idx += 1
    return pd.DataFrame(rows)


def generate_beta_tests(products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(40):
        product = products.iloc[i % len(products)]
        treatment = i % 2 == 0
        base_retention = 0.72 if treatment else 0.58
        base_convert = 0.28 if treatment else 0.18
        rows.append(
            {
                "beta_test_id": f"BT{i+1:03d}",
                "product_id": product["product_id"],
                "feature_name": str(product["feature_tags"]).split(",")[0],
                "test_group": "treatment" if treatment else "control",
                "user_id": f"U{2000+i}",
                "user_segment": RNG.choice(["creator", "office_user", "small_shop"]),
                "invited_at": (datetime.now() - timedelta(days=RNG.randint(12, 25))).isoformat(),
                "experienced_at": (datetime.now() - timedelta(days=RNG.randint(1, 10))).isoformat(),
                "feedback_submitted": RNG.choice([0, 1, 1]),
                "rating": round(RNG.uniform(4.0, 4.9) if treatment else RNG.uniform(3.4, 4.5), 1),
                "activated": 1,
                "retained_7d": 1 if RNG.random() < base_retention else 0,
                "converted": 1 if RNG.random() < base_convert else 0,
                "revenue": product["price"] if RNG.random() < base_convert else 0,
                "notes": "实验组增加情绪陪伴/触摸互动提示" if treatment else "对照组保持原版本",
            }
        )
    return pd.DataFrame(rows)


def draw_dashboard_image(path: Path, contents: pd.DataFrame, revenues: pd.DataFrame):
    img = Image.new("RGB", (1280, 860), "#fbfaf7")
    d = ImageDraw.Draw(img)
    draw_text(d, (48, 40), "SoloDeck Demo Account · Operating Snapshot", 34, bold=True)
    draw_text(d, (48, 88), "Virtual screenshot for upload and AI extraction demo", 22, fill="#6b7280")
    cards = [
        ("Content", f"{len(contents)} posts", "#e9f7ef"),
        ("Revenue", money(revenues["amount"].sum() + contents["revenue"].sum()), "#fff7df"),
        ("Pending", money(revenues[revenues["status"].eq("pending")]["amount"].sum()), "#fff1f1"),
        ("Consults", f"{contents['consultations'].sum()}", "#eef4ff"),
    ]
    x = 48
    for title, value, fill in cards:
        rounded(d, (x, 130, x + 270, 250), fill=fill, outline="#e3ddd2")
        draw_text(d, (x + 24, 152), title, 22, fill="#6b7280")
        draw_text(d, (x + 24, 188), value, 32, bold=True)
        x += 300
    rounded(d, (48, 290, 780, 780), fill="#ffffff", outline="#e5e7eb")
    draw_text(d, (76, 318), "Platform Performance", 28, bold=True)
    platform = contents.groupby("platform").agg(播放=("views", "sum"), 咨询=("consultations", "sum"), 收入=("revenue", "sum")).reset_index()
    max_views = max(platform["播放"].max(), 1)
    y = 380
    for _, row in platform.iterrows():
        name = str(row["platform"]).replace("xiaohongshu", "Xiaohongshu").replace("bilibili", "Bilibili").replace("douyin", "Douyin").replace("wechat", "WeChat")
        draw_text(d, (76, y), name, 22)
        bar_w = int(460 * row["播放"] / max_views)
        d.rounded_rectangle((210, y + 4, 210 + bar_w, y + 28), radius=8, fill="#5fa58c")
        draw_text(d, (690, y), money(row["收入"]), 20, fill="#374151")
        y += 70
    rounded(d, (820, 290, 1230, 780), fill="#ffffff", outline="#e5e7eb")
    draw_text(d, (848, 318), "Next Best Actions", 28, bold=True)
    actions = [
        "Move robot product CTAs to WeChat first.",
        "Run pain-point vs tutorial titles on Xiaohongshu.",
        "Follow up 3 pending payments, then send reports."
    ]
    y = 380
    for i, action in enumerate(actions, 1):
        d.ellipse((850, y + 4, 876, y + 30), fill="#ef4444" if i == 1 else "#f3c04d")
        draw_text(d, (858, y + 5), str(i), 16, fill="#ffffff", bold=True)
        draw_text(d, (892, y), action, 21, max_width=310)
        y += 110
    img.save(path)


def draw_payment_image(path: Path, revenues: pd.DataFrame):
    pending = revenues[revenues["status"].eq("pending")].head(4)
    img = Image.new("RGB", (900, 620), "#f7f8fa")
    d = ImageDraw.Draw(img)
    draw_text(d, (44, 34), "Revenue and Pending Payments", 32, bold=True)
    draw_text(d, (44, 76), "Virtual payment screenshot for SoloDeck upload", 20, fill="#6b7280")
    y = 130
    for _, row in pending.iterrows():
        rounded(d, (44, y, 856, y + 104), fill="#ffffff", outline="#e5e7eb")
        draw_text(d, (70, y + 20), f"Client {row['client_name']}", 24, bold=True)
        platform = str(row["platform"]).replace("xiaohongshu", "Xiaohongshu").replace("bilibili", "Bilibili").replace("douyin", "Douyin").replace("wechat", "WeChat")
        draw_text(d, (70, y + 56), f"{platform} | content service | {row['date']}", 18, fill="#6b7280")
        draw_text(d, (690, y + 28), money(row["amount"]), 28, fill="#b42318", bold=True)
        draw_text(d, (690, y + 66), "Pending", 18, fill="#b42318")
        y += 126
    img.save(path)


def draw_feedback_image(path: Path):
    img = Image.new("RGB", (1000, 720), "#fffdf8")
    d = ImageDraw.Draw(img)
    draw_text(d, (46, 36), "User Test Feedback Notes", 34, bold=True)
    draw_text(d, (46, 84), "Business robot / desktop companion robot · virtual survey notes", 20, fill="#6b7280")
    comments = [
        ("Creator user", "I like the emotional reminder, but I need calendar sync.", "Feature"),
        ("Office user", "Voice response is sometimes slow for business demos.", "Performance"),
        ("Shop owner", "Price is acceptable, but after-sales terms need clarity.", "Trust"),
        ("Education buyer", "The coaching model is useful; add course reminders.", "Feature"),
    ]
    y = 142
    for user, text, tag in comments:
        rounded(d, (46, y, 954, y + 118), fill="#ffffff", outline="#e5e7eb")
        draw_text(d, (74, y + 20), user, 22, bold=True)
        d.rounded_rectangle((800, y + 18, 920, y + 48), radius=12, fill="#eef7f2")
        draw_text(d, (816, y + 23), tag, 18, fill="#17683a")
        draw_text(d, (74, y + 62), text, 24, max_width=760)
        y += 142
    img.save(path)


def draw_campaign_image(path: Path, campaigns: pd.DataFrame):
    img = Image.new("RGB", (1050, 700), "#fbfaf7")
    d = ImageDraw.Draw(img)
    draw_text(d, (48, 36), "Campaign Tracker", 34, bold=True)
    draw_text(d, (48, 82), "Virtual brand collaboration screenshot for payment and report tasks", 20, fill="#6b7280")
    headers = ["Brand", "Project", "Price", "Status", "Payment"]
    xs = [58, 220, 470, 640, 810]
    y = 138
    for x, h in zip(xs, headers):
        draw_text(d, (x, y), h, 20, bold=True, fill="#374151")
    y += 42
    for _, row in campaigns.head(6).iterrows():
        rounded(d, (48, y - 12, 1000, y + 58), fill="#ffffff", outline="#e5e7eb")
        values = [f"Brand {idx+1}" if (idx := int(str(row["campaign_id"]).replace("BIZ", ""))) else "Brand", "Robot launch", money(row["price"]), row["status"], row["payment_status"]]
        for x, val in zip(xs, values):
            fill = "#b42318" if str(val) == "overdue" else "#111827"
            draw_text(d, (x, y), str(val), 19, fill=fill, max_width=170)
        y += 86
    img.save(path)


def write_readme():
    text = """# SoloDeck 评委演示上传资料包

这是一套虚拟数据，专门用于现场演示“上传资料 → 系统读取 → 生成经营建议”。

推荐演示顺序：

1. 在 SoloDeck 登录或进入体验工作台。
2. 不要先点“载入演示数据”，先展示真实上传流程。
3. 在“添加资料”区域上传这些图片：
   - `01_dashboard_snapshot.png`：经营概览截图
   - `02_payment_pending.png`：待收款截图
   - `03_feedback_notes.png`：用户反馈截图
   - `04_campaign_tracker.png`：商务合作截图
4. 文本补充可以粘贴：
   `明天上午 9 点和青石智能复盘商用机器人推广数据，但复盘报告还没完成。`
5. 如果要展示完整分析能力，再到“设置与数据文件”上传 CSV：
   - `demo_contents.csv`
   - `demo_revenues.csv`
   - `demo_campaigns.csv`
   - `demo_products.csv`
   - `demo_feedback.csv`
   - `demo_ab_tests.csv`
   - `demo_beta_tests.csv`

建议讲法：

SoloDeck 不要求用户接平台接口。用户可以上传后台截图、收款截图、反馈截图和 CSV，系统会自动整理成经营行动：先处理什么、下周验证什么、哪个平台和产品值得继续放大。
"""
    (OUT / "README_演示说明.md").write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    contents = generate_contents()
    revenues = generate_revenues(contents)
    campaigns = generate_campaigns(contents)
    products = generate_products()
    feedback = generate_feedback(products)
    ab_tests = generate_ab_tests(contents)
    beta_tests = generate_beta_tests(products)

    contents.to_csv(OUT / "demo_contents.csv", index=False)
    revenues.to_csv(OUT / "demo_revenues.csv", index=False)
    campaigns.to_csv(OUT / "demo_campaigns.csv", index=False)
    products.to_csv(OUT / "demo_products.csv", index=False)
    feedback.to_csv(OUT / "demo_feedback.csv", index=False)
    ab_tests.to_csv(OUT / "demo_ab_tests.csv", index=False)
    beta_tests.to_csv(OUT / "demo_beta_tests.csv", index=False)

    draw_dashboard_image(OUT / "01_dashboard_snapshot.png", contents, revenues)
    draw_payment_image(OUT / "02_payment_pending.png", revenues)
    draw_feedback_image(OUT / "03_feedback_notes.png")
    draw_campaign_image(OUT / "04_campaign_tracker.png", campaigns)
    write_readme()

    print(f"Generated demo upload pack at: {OUT}")
    for path in sorted(OUT.iterdir()):
        print(path.name)


if __name__ == "__main__":
    main()
