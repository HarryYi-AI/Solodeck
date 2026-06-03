from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _text_frame(df: pd.DataFrame, text_cols: list[str]) -> pd.Series:
    cols = [c for c in text_cols if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1)


def compute_text_similarity(df: pd.DataFrame, text_cols: list[str]) -> np.ndarray:
    text = _text_frame(df, text_cols)
    if len(text) == 0:
        return np.zeros((0, 0))
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        emb = SentenceTransformer("all-MiniLM-L6-v2").encode(text.tolist(), normalize_embeddings=True)
        return cosine_similarity(emb)
    except Exception:
        matrix = TfidfVectorizer(max_features=800, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b").fit_transform(text)
        return cosine_similarity(matrix)


def assign_similarity_groups(df: pd.DataFrame, threshold: float = 0.72, text_cols: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy().reset_index(drop=True)
    text_cols = text_cols or [c for c in ["title", "body", "tags", "product_name", "feature_tags", "category"] if c in out.columns]
    sim = compute_text_similarity(out, text_cols)
    parent = list(range(len(out)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if sim[i, j] >= threshold:
                union(i, j)
    roots = {root: idx + 1 for idx, root in enumerate(sorted({find(i) for i in range(len(out))}))}
    out["similarity_group"] = [f"SG-{roots[find(i)]:03d}" for i in range(len(out))]
    out["similarity_score"] = [float(np.max(np.delete(sim[i], i))) if len(out) > 1 else 0.0 for i in range(len(out))]
    return out


def detect_content_overlap(contents_df: pd.DataFrame) -> pd.DataFrame:
    df = assign_similarity_groups(contents_df, threshold=0.48, text_cols=["title", "body", "tags", "topic", "knowledge_domain"])
    rows = []
    for _, row in df.iterrows():
        group = df[df["similarity_group"].eq(row["similarity_group"])]
        similar_ids = [x for x in group.get("content_id", pd.Series(dtype=str)).tolist() if x != row.get("content_id")]
        same_series = bool(row.get("series_id")) and group.get("series_id", pd.Series()).eq(row.get("series_id")).any()
        difficulty = row.get("difficulty_level", "")
        score = float(row.get("similarity_score", 0))
        if score > 0.82 and not same_series:
            overlap = "duplicate"
        elif same_series:
            overlap = "same_series"
        elif difficulty == "advanced":
            overlap = "advanced_variant"
        elif row.get("parent_content_id"):
            overlap = "prerequisite"
        else:
            overlap = "parallel_topic"
        duplication = float(np.clip(max(score - 0.35, 0) / 0.65, 0, 1))
        novelty = float(np.clip(row.get("novelty_score", 1 - duplication) or (1 - duplication), 0, 1))
        fatigue = float(np.clip(row.get("user_fatigue_risk", duplication * (0.8 if same_series else 0.55)) or 0, 0, 1))
        rows.append({
            "content_id": row.get("content_id", ""),
            "title": row.get("title", ""),
            "series_id": row.get("series_id", ""),
            "similar_content_ids": ", ".join(similar_ids[:6]),
            "overlap_type": overlap,
            "duplication_risk": round(duplication, 3),
            "novelty_score": round(novelty, 3),
            "user_fatigue_risk": round(fatigue, 3),
            "explanation": "相似度高，建议换角度或合并为系列。" if overlap == "duplicate" else "属于同域/同系列线索，应观察边际增量而非只看单条播放。",
        })
    return pd.DataFrame(rows).sort_values(["duplication_risk", "user_fatigue_risk"], ascending=False)


def compute_differentiation_score(row: pd.Series) -> float:
    features = ["material", "color", "style", "size", "feature_tags", "target_user"]
    present = sum(1 for f in features if str(row.get(f, "")).strip())
    price_signal = min(abs(float(row.get("price", 0)) - float(row.get("cost", 0))) / max(float(row.get("price", 1)), 1), 1)
    return float(np.clip(0.08 * present + 0.35 * price_signal + (0.15 if row.get("is_new_version") else 0), 0, 1))


def detect_product_variants(products_df: pd.DataFrame) -> pd.DataFrame:
    if products_df.empty:
        return pd.DataFrame()
    df = assign_similarity_groups(products_df, threshold=0.4, text_cols=["product_name", "category", "feature_tags", "target_user"])
    rows = []
    for _, row in df.iterrows():
        tags = str(row.get("feature_tags", ""))
        if row.get("is_new_version") or row.get("parent_product_id"):
            vtype = "new_generation" if row.get("is_new_version") else "feature_upgrade"
        elif row.get("color"):
            vtype = "color_variant"
        elif row.get("material"):
            vtype = "material_variant"
        elif "bundle" in tags.lower():
            vtype = "bundle_variant"
        else:
            vtype = "price_variant"
        rows.append({
            "product_id": row.get("product_id", ""),
            "product_name": row.get("product_name", ""),
            "series_id": row.get("series_id", ""),
            "parent_product_id": row.get("parent_product_id", ""),
            "variant_features": ", ".join(str(row.get(c, "")) for c in ["material", "color", "style", "feature_tags"] if str(row.get(c, "")).strip()),
            "variant_type": vtype,
            "similarity_group": row.get("similarity_group"),
            "similarity_score": round(float(row.get("similarity_score", 0)), 3),
            "differentiation_score": round(compute_differentiation_score(row), 3),
        })
    return pd.DataFrame(rows).sort_values("differentiation_score", ascending=False)
