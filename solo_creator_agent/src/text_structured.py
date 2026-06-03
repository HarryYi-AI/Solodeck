from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


ZH_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*|[\u4e00-\u9fff]{2,}")
NUMBER_RE = re.compile(r"\d+")
QUESTION_RE = re.compile(r"[?？]|^(why|how|what|when|where|can|should)\b", re.I)
ACTION_WORDS_ZH = ("做", "学", "拆", "复盘", "提升", "降低", "生成", "变现", "拿到", "避开")
ACTION_WORDS_EN = ("build", "make", "learn", "grow", "improve", "reduce", "launch", "sell", "avoid")
POSITIVE_WORDS = ("增长", "提升", "稳定", "高效", "赚钱", "复利", "growth", "better", "profitable", "efficient")
NEGATIVE_WORDS = ("卡", "失败", "焦虑", "低效", "亏", "风险", "stuck", "failed", "risk", "slow", "broken")


def detect_language(text: str) -> str:
    text = text or ""
    if not text.strip():
        return "unknown"
    zh_chars = len(ZH_RE.findall(text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    if zh_chars >= max(2, latin_chars * 0.35):
        return "zh"
    if latin_chars:
        return "en"
    return "unknown"


def infer_title_style(title: str) -> str:
    title = title or ""
    lower = title.lower()
    if QUESTION_RE.search(title):
        return "question"
    if NUMBER_RE.search(title):
        return "number"
    if any(word in title for word in ("为什么", "避坑", "别再", "卡在")) or any(word in lower for word in ("why", "stuck", "mistake", "avoid")):
        return "pain_point"
    if any(word in title for word in ("教程", "指南", "步骤", "方法")) or any(word in lower for word in ("guide", "tutorial", "how to")):
        return "tutorial"
    if any(word in title for word in ("对比", "区别", "还是")) or any(word in lower for word in ("vs", "versus", "compare")):
        return "contrast"
    if any(word in title for word in ("案例", "复盘", "我用")) or any(word in lower for word in ("case", "review")):
        return "story"
    if any(word in title for word in ("拿到", "提升", "增长", "变现")) or any(word in lower for word in ("grow", "increase", "earn")):
        return "result_oriented"
    return "tutorial"


def keyword_string(text: str, max_keywords: int = 5) -> str:
    tokens = [token.lower() for token in WORD_RE.findall(text or "") if len(token.strip()) >= 2]
    stop = {"this", "that", "with", "from", "your", "我的", "一个", "为什么", "怎么", "如何", "内容"}
    words = [token for token in tokens if token not in stop]
    return ", ".join([word for word, _ in Counter(words).most_common(max_keywords)])


def sentiment_score(text: str) -> float:
    lower = (text or "").lower()
    pos = sum(1 for word in POSITIVE_WORDS if word in lower or word in text)
    neg = sum(1 for word in NEGATIVE_WORDS if word in lower or word in text)
    return float(np.clip((pos - neg) / max(pos + neg, 1), -1, 1))


def _cluster_texts(texts: pd.Series, n_clusters: int | None = None) -> pd.Series:
    clean = texts.fillna("").astype(str)
    non_empty = clean.str.len().gt(0)
    labels = pd.Series(["topic_cluster_0"] * len(clean), index=clean.index)
    if non_empty.sum() < 6:
        return labels
    k = n_clusters or min(5, max(2, int(np.sqrt(non_empty.sum()))))
    try:
        vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
        matrix = vectorizer.fit_transform(clean[non_empty])
        if matrix.shape[1] < 2:
            return labels
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        clustered = model.fit_predict(matrix)
        labels.loc[non_empty] = [f"topic_cluster_{int(item)}" for item in clustered]
    except Exception:
        return labels
    return labels


def extract_text_features(contents_df: pd.DataFrame, use_cast: bool = False) -> pd.DataFrame:
    """Turn title/body/tags text into stable, analysis-ready variables.

    CAST can be wired in later; the fallback here is deterministic and fast enough
    for a live product demo.
    """
    df = contents_df.copy()
    if "title" not in df.columns:
        df["title"] = ""
    for col in ["body", "tags", "language"]:
        if col not in df.columns:
            df[col] = ""
    combined = (df["title"].fillna("") + " " + df["body"].fillna("") + " " + df["tags"].fillna("")).str.strip()
    df["text_language"] = df["language"].where(df["language"].astype(str).str.len().gt(0), combined.map(detect_language))
    df["title_length"] = df["title"].fillna("").astype(str).map(len)
    df["title_word_count"] = df["title"].fillna("").astype(str).map(lambda text: len(WORD_RE.findall(text)))
    df["has_number"] = df["title"].fillna("").astype(str).map(lambda text: bool(NUMBER_RE.search(text)))
    df["has_question"] = df["title"].fillna("").astype(str).map(lambda text: bool(QUESTION_RE.search(text)))
    df["has_action_verb"] = df["title"].fillna("").astype(str).map(
        lambda text: any(word in text for word in ACTION_WORDS_ZH) or any(word in text.lower() for word in ACTION_WORDS_EN)
    )
    df["inferred_title_style"] = df["title"].fillna("").astype(str).map(infer_title_style)
    df["text_keywords"] = combined.map(keyword_string)
    df["text_sentiment"] = combined.map(sentiment_score)
    df["text_topic_cluster"] = _cluster_texts(combined)

    if use_cast:
        try:
            import cast  # type: ignore  # noqa: F401
            df["text_parser"] = "cast_available_fallback_used"
        except Exception:
            df["text_parser"] = "local_fallback"
    else:
        df["text_parser"] = "local_fallback"
    return df


def text_feature_summary(contents_df: pd.DataFrame, language: str = "中文") -> list[dict]:
    df = extract_text_features(contents_df)
    zh = language == "中文"
    rows: list[dict] = []
    if "revenue" in df.columns and not df.empty:
        grouped = df.groupby("inferred_title_style", as_index=False).agg(
            content_count=("content_id", "count"),
            avg_views=("views", "mean"),
            avg_revenue=("revenue", "mean"),
            avg_favorite_rate=("favorites", lambda s: float((s / df.loc[s.index, "views"].replace(0, np.nan)).mean())),
        )
        best = grouped.sort_values(["avg_revenue", "avg_views"], ascending=False).head(1)
        if not best.empty:
            row = best.iloc[0]
            rows.append({
                "title": "标题文本已结构化" if zh else "Text features extracted",
                "reason": (
                    f"{row['inferred_title_style']} 风格当前平均收入最高，样本 {int(row['content_count'])} 条。"
                    if zh
                    else f"{row['inferred_title_style']} titles currently have the highest average revenue across {int(row['content_count'])} samples."
                ),
                "action": "把这个标题风格放进下周实验，并固定平台、主题和粉丝基数。" if zh else "Use this title style in next week's test while keeping platform, topic and follower base stable.",
                "priority": "medium",
            })
    return rows
