from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class KnowledgeItem:
    topic: str
    audience: str
    problem: str
    principle: str
    action_template: str
    metric: str


DEFAULT_KNOWLEDGE = [
    KnowledgeItem(
        topic="跨平台发布",
        audience="内容创作者",
        problem="同一内容在不同平台表现不同",
        principle="优先比较同一内容在不同平台的表现，避免把内容差异误认为平台差异。",
        action_template="同一主题先在两个常用平台各发 2-3 条，72 小时后比较收藏、咨询和成交。",
        metric="收藏率、咨询数、成交数",
    ),
    KnowledgeItem(
        topic="标题实验",
        audience="内容创作者",
        problem="不知道哪种标题值得继续用",
        principle="一次只改变标题风格，平台、主题和发布时间尽量保持一致。",
        action_template="连续 7-14 天做两组标题，每组 3-6 条；一组平均高出 10% 以上才放大。",
        metric="收藏率、转粉率、咨询率",
    ),
    KnowledgeItem(
        topic="内容疲劳",
        audience="内容创作者",
        problem="系列内容越做越像，曝光下降",
        principle="连续重复会降低新鲜感，但高收藏或高咨询说明仍有深度价值。",
        action_template="保留高价值系列，但每 3 条穿插 1 条入门内容或真实案例，恢复新用户入口。",
        metric="播放趋势、收藏率、咨询率",
    ),
    KnowledgeItem(
        topic="产品功能内测",
        audience="产品团队",
        problem="不知道新功能是否值得主推",
        principle="先看实验组和对照组差异，再看反馈里的阻碍因素。",
        action_template="找 30-50 个相似用户做小批内测，观察 7 日留存、购买意向和差评原因。",
        metric="7日留存、转化率、评分",
    ),
    KnowledgeItem(
        topic="商用机器人款式",
        audience="电商/硬件团队",
        problem="不知道哪种机器人款式更赚钱",
        principle="款式收入要和功能、价格、退款率一起看，不能只看销量。",
        action_template="把收入最高且退款率低的款式放到首屏，弱款先改功能卖点或定价后再投放。",
        metric="收入、成交、退款率、评分",
    ),
    KnowledgeItem(
        topic="商务回款",
        audience="一人公司",
        problem="商务合作容易漏收款和漏复盘",
        principle="交付、发票、复盘和回款必须放在同一个行动清单里。",
        action_template="先处理逾期款，再补齐发票和复盘报告；所有合作设置交付前、发布后、复盘后三个提醒。",
        metric="待收款金额、逾期天数、复盘状态",
    ),
    KnowledgeItem(
        topic="高收藏低变现",
        audience="知识付费创作者",
        problem="内容很多人收藏，但没有转成收入",
        principle="收藏代表复用需求，通常适合沉淀成资料包、模板或小课。",
        action_template="把高收藏主题拆成清单模板、资料包或训练营预售页，在相关内容结尾测试入口。",
        metric="收藏率、咨询数、成交数",
    ),
    KnowledgeItem(
        topic="高咨询低成交",
        audience="一人公司",
        problem="有人咨询但最终成交少",
        principle="咨询到成交之间通常卡在承接页、价格解释或私域话术。",
        action_template="检查落地页第一屏、价格说明和常见问题；用 10 个咨询用户测试一句话卖点。",
        metric="咨询率、成交率、客单价",
    ),
]


def _corpus(items: Iterable[KnowledgeItem]) -> list[str]:
    return [
        " ".join([item.topic, item.audience, item.problem, item.principle, item.action_template, item.metric])
        for item in items
    ]


def query_knowledge(query: str, items: list[KnowledgeItem] | None = None, top_k: int = 3) -> list[dict]:
    base = items or DEFAULT_KNOWLEDGE
    if not query.strip() or not base:
        return []
    corpus = _corpus(base)
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked = scores.argsort()[::-1][:top_k]
    rows = []
    for idx in ranked:
        item = asdict(base[int(idx)])
        item["score"] = float(scores[int(idx)])
        rows.append(item)
    return rows


def knowledge_for_cards(cards: list[dict], lang: str = "中文", top_k: int = 3) -> list[dict]:
    if not cards:
        return []
    query = " ".join(
        str(card.get(key, ""))
        for card in cards[:6]
        for key in ["title", "insight", "action", "metric"]
    )
    rows = query_knowledge(query, top_k=top_k)
    if lang != "中文":
        return rows
    return rows


def knowledge_cards(cards: list[dict], lang: str = "中文") -> list[dict]:
    rows = knowledge_for_cards(cards, lang=lang, top_k=3)
    if lang != "中文":
        return [
            {
                "title": item["topic"],
                "reason": item["principle"],
                "action": item["action_template"],
                "priority": "medium",
            }
            for item in rows
        ]
    return [
        {
            "title": item["topic"],
            "reason": item["principle"],
            "action": item["action_template"],
            "priority": "medium",
        }
        for item in rows
    ]


def knowledge_frame(cards: list[dict], lang: str = "中文") -> pd.DataFrame:
    rows = knowledge_for_cards(cards, lang=lang, top_k=5)
    df = pd.DataFrame(rows)
    if lang == "中文" and not df.empty:
        df = df.rename(
            columns={
                "topic": "主题",
                "audience": "适用对象",
                "problem": "常见问题",
                "principle": "判断原则",
                "action_template": "行动模板",
                "metric": "观察指标",
                "score": "匹配度",
            }
        )
    return df
