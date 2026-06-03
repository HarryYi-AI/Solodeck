from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def recommendation_id(item: dict[str, Any]) -> str:
    raw = "|".join(str(item.get(key, "")) for key in ["title", "reason", "action"])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def infer_category(item: dict[str, Any]) -> str:
    text = " ".join(str(item.get(key, "")) for key in ["title", "reason", "action"]).lower()
    mapping = [
        ("商务", ["商务", "合作", "回款", "收款", "发票", "brand", "payment"]),
        ("产品", ["产品", "款式", "功能", "机器人", "内测", "版本", "feature", "product"]),
        ("内容", ["标题", "内容", "选题", "发布", "平台", "收藏", "content", "title"]),
        ("实验", ["实验", "验证", "对照", "胜出", "观察", "test", "experiment"]),
        ("收入", ["收入", "成交", "变现", "价格", "收益", "revenue"]),
    ]
    for category, keywords in mapping:
        if any(keyword in text for keyword in keywords):
            return category
    return "经营"


def record_feedback(history: list[dict] | None, item: dict[str, Any], feedback: str) -> list[dict]:
    out = list(history or [])
    out.append(
        {
            "id": recommendation_id(item),
            "category": infer_category(item),
            "feedback": feedback,
            "title": item.get("title", ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return out[-200:]


def feedback_weights(history: list[dict] | None) -> dict[str, float]:
    weights: dict[str, float] = defaultdict(lambda: 1.0)
    for row in history or []:
        category = row.get("category") or "经营"
        feedback = row.get("feedback")
        if feedback == "accepted":
            weights[category] += 0.20
        elif feedback == "dismissed":
            weights[category] -= 0.12
        elif feedback == "done":
            weights[category] += 0.28
    return {k: max(0.55, min(1.8, v)) for k, v in weights.items()}


def rank_recommendations(items: list[dict], history: list[dict] | None = None) -> list[dict]:
    weights = feedback_weights(history)
    priority_score = {"high": 3.0, "medium": 2.0, "low": 1.0}
    ranked = []
    for index, item in enumerate(items):
        category = infer_category(item)
        score = priority_score.get(item.get("priority", "medium"), 2.0) * weights.get(category, 1.0) - index * 0.03
        ranked.append(({**item, "category": category, "learning_score": score}, score))
    return [item for item, _ in sorted(ranked, key=lambda pair: pair[1], reverse=True)]


def learning_summary(history: list[dict] | None, lang: str = "中文") -> str:
    if not history:
        return "暂无偏好记录。" if lang == "中文" else "No preference history yet."
    weights = feedback_weights(history)
    top = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:3]
    if lang == "中文":
        return "系统会优先展示你更常采纳的建议类型：" + "、".join(f"{k}" for k, _ in top)
    return "SoloDeck prioritizes recommendation types you accept more often: " + ", ".join(k for k, _ in top)
