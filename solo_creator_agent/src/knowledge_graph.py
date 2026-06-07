from __future__ import annotations

import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go


TYPE_LABELS_ZH = {
    "platform": "平台",
    "account": "账号",
    "topic": "主题",
    "series": "内容系列",
    "content": "内容",
    "product": "产品",
    "feature": "功能",
    "segment": "用户人群",
    "issue": "反馈问题",
    "experiment": "实验",
    "treatment": "策略变量",
    "outcome": "结果指标",
    "metric": "指标",
}

REL_LABELS_ZH = {
    "posted_on": "发布在",
    "created_by": "属于账号",
    "about_topic": "围绕主题",
    "in_series": "属于系列",
    "sold_on": "销售平台",
    "has_feature": "包含功能",
    "targets": "面向人群",
    "feedback_about_product": "反馈指向产品",
    "feedback_about_content": "反馈指向内容",
    "mentions_issue": "提到问题",
    "revenue_from": "产生收入",
    "same_content_group": "同内容跨平台",
    "tests_strategy": "测试策略",
    "measures_outcome": "观察指标",
    "runs_on": "实验平台",
    "beta_tests_feature": "内测功能",
}

FEATURE_LABELS_ZH = {
    "emotion_companion": "情绪陪伴",
    "voice_interaction": "语音互动",
    "desktop_decoration": "桌面摆件",
    "touch_interaction": "触摸互动",
    "touch_control": "触摸控制",
    "music_playback": "音乐播放",
    "alarm": "提醒闹钟",
    "ambient_light": "氛围灯",
    "wireless_charging": "无线充电",
    "minimal_design": "极简设计",
    "timer": "计时器",
    "eye_protection": "护眼",
    "dimming": "调光",
    "anti_drop": "防摔",
    "magnetic": "磁吸",
    "weekly_plan": "周计划",
    "habit_tracking": "习惯追踪",
    "notion_template": "Notion 模板",
    "revenue_tracker": "收入追踪表",
    "ab_test_sheet": "实验记录表",
    "weekly_review": "周复盘",
    "title_formula": "标题公式",
    "soft_binding": "软装订",
    "thick_paper": "加厚纸张",
    "waterproof_cover": "防水封面",
    "content_clarity": "内容清晰度",
    "design": "外观设计",
    "emotional_value": "情绪价值",
    "feature_request": "功能需求",
    "performance": "性能体验",
    "pricing": "价格",
    "quality": "质量",
    "trust": "信任",
    "usability": "易用性",
}

PLATFORM_LABELS_ZH = {
    "xiaohongshu": "小红书",
    "bilibili": "B站",
    "douyin": "抖音",
    "wechat": "公众号/视频号",
    "zhihu": "知乎",
    "youtube": "YouTube",
    "tiktok": "TikTok",
    "instagram": "Instagram",
    "substack": "Substack",
    "x": "X / Twitter",
}


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _label(value: Any, lang: str) -> str:
    raw = _safe_text(value)
    if lang != "中文":
        return raw.replace("_", " ")
    return FEATURE_LABELS_ZH.get(raw, PLATFORM_LABELS_ZH.get(raw, raw.replace("_", " ")))


def _short(value: Any, limit: int = 26) -> str:
    text = _safe_text(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _add_node(nodes: dict[str, dict[str, Any]], node_id: str, label: str, node_type: str, size: float = 12, detail: str = "") -> None:
    if not node_id or node_id in nodes:
        return
    nodes[node_id] = {
        "id": node_id,
        "label": _short(label, 28),
        "type": node_type,
        "size": max(10, min(38, 10 + math.log1p(max(size, 0)) * 3)),
        "detail": detail,
    }


def _add_edge(edges: list[dict[str, Any]], source: str, target: str, relation: str, weight: float = 1, evidence: str = "") -> None:
    if not source or not target or source == target:
        return
    edges.append({
        "source": source,
        "target": target,
        "relation": relation,
        "weight": max(1.0, _safe_float(weight, 1)),
        "evidence": evidence,
    })


def _split_features(value: Any) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    return [item.strip() for item in text.replace(",", "|").replace("；", "|").replace(";", "|").split("|") if item.strip()]


def _top_rows(df: pd.DataFrame, score_cols: list[str], limit: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    score = pd.Series(0.0, index=out.index)
    for col in score_cols:
        if col in out.columns:
            score = score + pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["_kg_score"] = score
    return out.sort_values("_kg_score", ascending=False).head(limit).drop(columns=["_kg_score"], errors="ignore")


def build_knowledge_graph(
    contents: pd.DataFrame,
    products: pd.DataFrame,
    feedback: pd.DataFrame,
    revenues: pd.DataFrame,
    campaigns: pd.DataFrame | None = None,
    ab_tests: pd.DataFrame | None = None,
    beta_tests: pd.DataFrame | None = None,
    lang: str = "中文",
    max_content_nodes: int = 45,
    max_product_nodes: int = 35,
) -> dict[str, pd.DataFrame]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    content_rows = _top_rows(contents, ["revenue", "conversions", "consultations", "views"], max_content_nodes)
    for _, row in content_rows.iterrows():
        cid = _safe_text(row.get("content_id"))
        if not cid:
            continue
        content_node = f"content:{cid}"
        title = row.get("title") or cid
        _add_node(nodes, content_node, _short(title, 24), "content", row.get("revenue", row.get("views", 0)), _safe_text(title))

        platform = _safe_text(row.get("platform"))
        if platform:
            platform_node = f"platform:{platform}"
            _add_node(nodes, platform_node, _label(platform, lang), "platform", 22)
            _add_edge(edges, content_node, platform_node, "posted_on", row.get("views", 1))

        account = _safe_text(row.get("account_id"))
        if account:
            account_name = row.get("account_name") or account
            account_node = f"account:{account}"
            _add_node(nodes, account_node, _short(account_name, 20), "account", row.get("followers_before", 10))
            _add_edge(edges, content_node, account_node, "created_by", 1)

        topic = _safe_text(row.get("topic"))
        if topic:
            topic_node = f"topic:{topic}"
            _add_node(nodes, topic_node, _label(topic, lang), "topic", row.get("revenue", 10))
            _add_edge(edges, content_node, topic_node, "about_topic", row.get("favorites", 1))

        series = _safe_text(row.get("series_id"))
        if series:
            series_node = f"series:{series}"
            _add_node(nodes, series_node, _label(series, lang), "series", row.get("revenue", 10))
            _add_edge(edges, content_node, series_node, "in_series", 1)

        content_group = _safe_text(row.get("content_group_id"))
        if content_group:
            group_node = f"group:{content_group}"
            _add_node(nodes, group_node, _label(content_group, lang), "series", row.get("revenue", 10))
            _add_edge(edges, content_node, group_node, "same_content_group", row.get("views", 1))

    product_rows = _top_rows(products, ["revenue", "conversions", "consultations", "views"], max_product_nodes)
    for _, row in product_rows.iterrows():
        pid = _safe_text(row.get("product_id"))
        if not pid:
            continue
        product_node = f"product:{pid}"
        product_name = row.get("product_name") or pid
        _add_node(nodes, product_node, _short(product_name, 24), "product", row.get("revenue", 0), _safe_text(product_name))

        platform = _safe_text(row.get("platform"))
        if platform:
            platform_node = f"platform:{platform}"
            _add_node(nodes, platform_node, _label(platform, lang), "platform", 22)
            _add_edge(edges, product_node, platform_node, "sold_on", row.get("views", 1))

        segment = _safe_text(row.get("target_user"))
        if segment:
            segment_node = f"segment:{segment}"
            _add_node(nodes, segment_node, _label(segment, lang), "segment", row.get("conversions", 1))
            _add_edge(edges, product_node, segment_node, "targets", row.get("conversions", 1))

        for feature in _split_features(row.get("feature_tags")):
            feature_node = f"feature:{feature}"
            _add_node(nodes, feature_node, _label(feature, lang), "feature", row.get("conversions", 1))
            _add_edge(edges, product_node, feature_node, "has_feature", row.get("conversions", 1))

    if not feedback.empty:
        fb_rows = feedback.copy()
        severity_score = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        if "severity" in fb_rows.columns:
            fb_rows["_sev"] = fb_rows["severity"].map(severity_score).fillna(1)
            fb_rows = fb_rows.sort_values("_sev", ascending=False).head(80)
        for _, row in fb_rows.iterrows():
            issue = _safe_text(row.get("issue_type"))
            if issue:
                issue_node = f"issue:{issue}"
                _add_node(nodes, issue_node, _label(issue, lang), "issue", _safe_float(row.get("severity"), 1))
                product_id = _safe_text(row.get("related_product_id"))
                content_id = _safe_text(row.get("related_content_id"))
                if product_id and f"product:{product_id}" in nodes:
                    _add_edge(edges, f"product:{product_id}", issue_node, "mentions_issue", 1, row.get("feedback_text", ""))
                if content_id and f"content:{content_id}" in nodes:
                    _add_edge(edges, f"content:{content_id}", issue_node, "mentions_issue", 1, row.get("feedback_text", ""))

            segment = _safe_text(row.get("user_segment"))
            product_id = _safe_text(row.get("related_product_id"))
            if segment and product_id and f"product:{product_id}" in nodes:
                segment_node = f"segment:{segment}"
                _add_node(nodes, segment_node, _label(segment, lang), "segment", 12)
                _add_edge(edges, segment_node, f"product:{product_id}", "feedback_about_product", 1)

    if not revenues.empty:
        for _, row in revenues.head(120).iterrows():
            amount = _safe_float(row.get("amount"), 0)
            content_id = _safe_text(row.get("content_id"))
            product_id = _safe_text(row.get("product_id"))
            if amount <= 0:
                continue
            if content_id and f"content:{content_id}" in nodes:
                _add_edge(edges, f"content:{content_id}", "metric:revenue", "revenue_from", amount)
            if product_id and f"product:{product_id}" in nodes:
                _add_edge(edges, f"product:{product_id}", "metric:revenue", "revenue_from", amount)
        if any(edge["target"] == "metric:revenue" for edge in edges):
            _add_node(nodes, "metric:revenue", "收入" if lang == "中文" else "revenue", "metric", revenues.get("amount", pd.Series(dtype=float)).sum())

    if ab_tests is not None and not ab_tests.empty:
        for _, row in ab_tests.head(80).iterrows():
            exp_id = _safe_text(row.get("experiment_id"))
            if not exp_id:
                continue
            exp_node = f"experiment:{exp_id}"
            _add_node(nodes, exp_node, exp_id, "experiment", 16)
            platform = _safe_text(row.get("platform"))
            if platform:
                platform_node = f"platform:{platform}"
                _add_node(nodes, platform_node, _label(platform, lang), "platform", 16)
                _add_edge(edges, exp_node, platform_node, "runs_on", 1)
            treatment = _safe_text(row.get("treatment_value") or row.get("treatment_name"))
            if treatment:
                treatment_node = f"treatment:{treatment}"
                _add_node(nodes, treatment_node, _label(treatment, lang), "treatment", 14)
                _add_edge(edges, exp_node, treatment_node, "tests_strategy", 1)
            outcome = _safe_text(row.get("outcome_metric"))
            if outcome:
                outcome_node = f"outcome:{outcome}"
                _add_node(nodes, outcome_node, _label(outcome, lang), "outcome", 14)
                _add_edge(edges, exp_node, outcome_node, "measures_outcome", 1)
            content_id = _safe_text(row.get("content_id"))
            if content_id and f"content:{content_id}" in nodes:
                _add_edge(edges, exp_node, f"content:{content_id}", "feedback_about_content", 1)

    if beta_tests is not None and not beta_tests.empty:
        for _, row in beta_tests.head(80).iterrows():
            feature = _safe_text(row.get("feature_name"))
            product_id = _safe_text(row.get("product_id"))
            if not feature:
                continue
            feature_node = f"feature:{feature}"
            _add_node(nodes, feature_node, _label(feature, lang), "feature", 14)
            if product_id and f"product:{product_id}" in nodes:
                _add_edge(edges, f"product:{product_id}", feature_node, "beta_tests_feature", 1)

    node_df = pd.DataFrame(nodes.values())
    edge_df = pd.DataFrame(edges)
    if not edge_df.empty:
        edge_df = edge_df.groupby(["source", "target", "relation"], as_index=False).agg(
            weight=("weight", "sum"),
            evidence=("evidence", "first"),
        )
    return {"nodes": node_df, "edges": edge_df}


def feature_combination_query(products: pd.DataFrame, lang: str = "中文", top_n: int = 6) -> pd.DataFrame:
    if products.empty or "feature_tags" not in products.columns:
        return pd.DataFrame()
    rows = []
    for _, row in products.iterrows():
        features = _split_features(row.get("feature_tags"))
        if not features:
            continue
        combo = "|".join(sorted(features))
        views = _safe_float(row.get("views"), 0)
        conversions = _safe_float(row.get("conversions"), 0)
        rows.append({
            "功能组合": " + ".join(_label(feature, lang) for feature in sorted(features)) if lang == "中文" else " + ".join(sorted(features)),
            "平台": _label(row.get("platform"), lang),
            "产品数": 1,
            "收入": _safe_float(row.get("revenue"), 0),
            "成交": conversions,
            "转化率": conversions / views if views else 0,
            "样本标识": combo,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    grouped = df.groupby(["功能组合", "平台", "样本标识"], as_index=False).agg(
        产品数=("产品数", "sum"),
        收入=("收入", "sum"),
        成交=("成交", "sum"),
        转化率=("转化率", "mean"),
    )
    return grouped.sort_values(["成交", "收入"], ascending=False).head(top_n)


def series_exploration_query(contents: pd.DataFrame, lang: str = "中文", top_n: int = 6) -> pd.DataFrame:
    needed = {"series_id", "content_id"}
    if contents.empty or not needed.issubset(contents.columns):
        return pd.DataFrame()
    df = contents.copy()
    for col in ["revenue", "views", "favorites", "consultations", "conversions", "user_fatigue_risk", "duplication_risk"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    grouped = df.groupby("series_id", as_index=False).agg(
        内容数=("content_id", "nunique"),
        收入=("revenue", "sum"),
        平均播放=("views", "mean"),
        收藏=("favorites", "sum"),
        咨询=("consultations", "sum"),
        成交=("conversions", "sum"),
        疲劳风险=("user_fatigue_risk", "mean"),
        重复风险=("duplication_risk", "mean"),
    )
    grouped["判断"] = grouped.apply(
        lambda row: "穿插入门内容，避免连续重复" if row["疲劳风险"] >= 0.55 or row["重复风险"] >= 0.55 else "可继续观察表现",
        axis=1,
    )
    if lang != "中文":
        grouped = grouped.rename(columns={
            "内容数": "content_count",
            "收入": "revenue",
            "平均播放": "avg_views",
            "收藏": "favorites",
            "咨询": "consultations",
            "成交": "conversions",
            "疲劳风险": "fatigue_risk",
            "重复风险": "duplication_risk",
            "判断": "note",
        })
    return grouped.sort_values("收入" if lang == "中文" else "revenue", ascending=False).head(top_n)


def platform_topic_query(contents: pd.DataFrame, revenues: pd.DataFrame, lang: str = "中文", top_n: int = 8) -> pd.DataFrame:
    if contents.empty or "platform" not in contents.columns or "topic" not in contents.columns:
        return pd.DataFrame()
    df = contents.copy()
    for col in ["views", "favorites", "consultations", "conversions", "revenue"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    grouped = df.groupby(["platform", "topic"], as_index=False).agg(
        内容数=("content_id", "nunique"),
        收入=("revenue", "sum"),
        平均播放=("views", "mean"),
        收藏=("favorites", "sum"),
        咨询=("consultations", "sum"),
        成交=("conversions", "sum"),
    )
    grouped["平台"] = grouped["platform"].map(lambda value: _label(value, lang))
    grouped["主题"] = grouped["topic"].map(lambda value: _label(value, lang))
    out = grouped[["平台", "主题", "内容数", "收入", "平均播放", "收藏", "咨询", "成交"]]
    if lang != "中文":
        out = out.rename(columns={
            "平台": "platform",
            "主题": "topic",
            "内容数": "content_count",
            "收入": "revenue",
            "平均播放": "avg_views",
            "收藏": "favorites",
            "咨询": "consultations",
            "成交": "conversions",
        })
    return out.sort_values("收入" if lang == "中文" else "revenue", ascending=False).head(top_n)


def graph_insight_cards(products: pd.DataFrame, contents: pd.DataFrame, feedback: pd.DataFrame, lang: str = "中文") -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    feature_df = feature_combination_query(products, lang, top_n=1)
    if not feature_df.empty:
        row = feature_df.iloc[0]
        cards.append({
            "title": f"高转化功能组合：{row['功能组合']}" if lang == "中文" else f"High-converting feature mix: {row['功能组合']}",
            "reason": f"成交 {row['成交']:.0f}，收入 ¥{row['收入']:.0f}。" if lang == "中文" else f"Conversions {row['成交']:.0f}, revenue {row['收入']:.0f}.",
            "action": "把它作为下一轮产品页主卖点，但仍需实验验证。" if lang == "中文" else "Use it as the next product-page selling point, then validate with an experiment.",
            "priority": "high",
        })
    series_df = series_exploration_query(contents, lang, top_n=1)
    if not series_df.empty:
        row = series_df.iloc[0]
        cards.append({
            "title": f"重点内容系列：{row['series_id']}" if lang == "中文" else f"Key content series: {row['series_id']}",
            "reason": f"收入 ¥{row['收入']:.0f}，内容 {row['内容数']:.0f} 条。" if lang == "中文" else f"Revenue {row['revenue']:.0f}, {row['content_count']:.0f} posts.",
            "action": row["判断"] if lang == "中文" else row["note"],
            "priority": "medium",
        })
    if not feedback.empty and "issue_type" in feedback.columns:
        issue = feedback["issue_type"].value_counts().head(1)
        if not issue.empty:
            issue_name = _label(issue.index[0], lang)
            cards.append({
                "title": f"反馈集中点：{issue_name}" if lang == "中文" else f"Feedback cluster: {issue_name}",
                "reason": f"出现 {int(issue.iloc[0])} 次。" if lang == "中文" else f"Mentioned {int(issue.iloc[0])} times.",
                "action": "进入产品迭代清单，结合收入和转化再排优先级。" if lang == "中文" else "Add it to the roadmap and prioritize with revenue/conversion evidence.",
                "priority": "medium",
            })
    return cards


def plot_knowledge_graph(nodes: pd.DataFrame, edges: pd.DataFrame, lang: str = "中文") -> go.Figure:
    if nodes.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    type_order = ["platform", "account", "topic", "series", "content", "product", "feature", "segment", "experiment", "treatment", "outcome", "issue", "metric"]
    color_map = {
        "platform": "#5FA58C",
        "account": "#8AA6C1",
        "topic": "#DDAA4C",
        "series": "#C8A2C8",
        "content": "#3B82F6",
        "product": "#EF6F6C",
        "feature": "#22A06B",
        "segment": "#7C3AED",
        "issue": "#F59E0B",
        "experiment": "#EC4899",
        "treatment": "#14B8A6",
        "outcome": "#6366F1",
        "metric": "#111827",
    }
    x_map = {node_type: idx for idx, node_type in enumerate(type_order)}
    positioned = nodes.copy().reset_index(drop=True)
    coords: dict[str, tuple[float, float]] = {}
    for node_type, group in positioned.groupby("type", sort=False):
        ids = group["id"].tolist()
        n = len(ids)
        x = x_map.get(node_type, len(type_order))
        for i, node_id in enumerate(ids):
            y = (n - 1) / 2 - i
            coords[node_id] = (x, y)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    hover_edges: list[str] = []
    if not edges.empty:
        for _, edge in edges.iterrows():
            source = edge["source"]
            target = edge["target"]
            if source not in coords or target not in coords:
                continue
            x0, y0 = coords[source]
            x1, y1 = coords[target]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            relation = REL_LABELS_ZH.get(edge["relation"], edge["relation"]) if lang == "中文" else edge["relation"]
            hover_edges.append(f"{relation}: {edge.get('weight', 1):.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.7, color="rgba(17,24,39,0.18)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    for node_type, group in positioned.groupby("type", sort=False):
        xs = [coords[node_id][0] for node_id in group["id"]]
        ys = [coords[node_id][1] for node_id in group["id"]]
        node_type_label = TYPE_LABELS_ZH.get(node_type, node_type) if lang == "中文" else node_type
        hover = [
            f"{node_type_label}: {row['label']}<br>{row.get('detail', '')}"
            for _, row in group.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=group["label"].tolist(),
            textposition="middle right",
            hovertext=hover,
            hoverinfo="text",
            marker=dict(
                size=group["size"].tolist(),
                color=color_map.get(node_type, "#64748B"),
                line=dict(width=1, color="white"),
                opacity=0.9,
            ),
            name=node_type_label,
        ))

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=10, r=10, t=35, b=10),
        title="知识图谱：实体关系解释" if lang == "中文" else "Knowledge Graph: Entity Relationships",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", y=-0.05),
        hovermode="closest",
    )
    return fig


def _node_lookup(nodes: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if nodes.empty:
        return {}
    return {row["id"]: row.to_dict() for _, row in nodes.iterrows()}


def local_graph_query(nodes: pd.DataFrame, edges: pd.DataFrame, keyword: str, lang: str = "中文", limit: int = 12) -> pd.DataFrame:
    """Return a small neighborhood for one content/product/feature/user segment."""
    if nodes.empty or edges.empty:
        return pd.DataFrame()
    keyword = _safe_text(keyword).lower()
    lookup = _node_lookup(nodes)
    if not keyword:
        seed_ids = nodes.sort_values("size", ascending=False).head(3)["id"].tolist()
    else:
        seed_ids = [
            row["id"]
            for _, row in nodes.iterrows()
            if keyword in _safe_text(row.get("id")).lower()
            or keyword in _safe_text(row.get("label")).lower()
            or keyword in _safe_text(row.get("detail")).lower()
        ][:5]
    if not seed_ids:
        return pd.DataFrame()
    related = edges[(edges["source"].isin(seed_ids)) | (edges["target"].isin(seed_ids))].copy().head(limit)
    rows = []
    for _, edge in related.iterrows():
        source = lookup.get(edge["source"], {"label": edge["source"], "type": ""})
        target = lookup.get(edge["target"], {"label": edge["target"], "type": ""})
        relation = REL_LABELS_ZH.get(edge["relation"], edge["relation"]) if lang == "中文" else edge["relation"]
        rows.append({
            "起点": source.get("label", edge["source"]),
            "关系": relation,
            "终点": target.get("label", edge["target"]),
            "强度": edge.get("weight", 1),
            "说明": "局部关系用于解释线索，不直接证明因果。" if lang == "中文" else "Local graph context explains evidence but does not prove causality.",
        })
    out = pd.DataFrame(rows)
    if lang != "中文" and not out.empty:
        out = out.rename(columns={"起点": "source", "关系": "relation", "终点": "target", "强度": "weight", "说明": "note"})
    return out


def global_graph_query(
    contents: pd.DataFrame,
    products: pd.DataFrame,
    feedback: pd.DataFrame,
    revenues: pd.DataFrame,
    lang: str = "中文",
) -> dict[str, pd.DataFrame]:
    """Global GraphRAG-style retrieval over series, platforms, features and feedback."""
    return {
        "platform_topic": platform_topic_query(contents, revenues, lang=lang, top_n=8),
        "feature_combinations": feature_combination_query(products, lang=lang, top_n=8),
        "series": series_exploration_query(contents, lang=lang, top_n=8),
        "feedback": feedback_issue_query(feedback, products, lang=lang, top_n=8),
    }


def feedback_issue_query(feedback: pd.DataFrame, products: pd.DataFrame, lang: str = "中文", top_n: int = 8) -> pd.DataFrame:
    if feedback.empty or "issue_type" not in feedback.columns:
        return pd.DataFrame()
    df = feedback.copy()
    df["issue_type"] = df["issue_type"].fillna("")
    if "sentiment" not in df.columns:
        df["sentiment"] = ""
    if "severity" not in df.columns:
        df["severity"] = ""
    grouped = df.groupby(["issue_type", "sentiment", "severity"], as_index=False).agg(
        反馈数=("feedback_id", "count") if "feedback_id" in df.columns else ("issue_type", "count"),
        关联产品数=("related_product_id", "nunique") if "related_product_id" in df.columns else ("issue_type", "count"),
    )
    grouped["问题"] = grouped["issue_type"].map(lambda value: _label(value, lang))
    out = grouped[["问题", "sentiment", "severity", "反馈数", "关联产品数"]].sort_values("反馈数", ascending=False).head(top_n)
    if lang != "中文":
        out = out.rename(columns={"问题": "issue", "反馈数": "feedback_count", "关联产品数": "related_product_count"})
    return out


def graph_rag_answer(
    query: str,
    scope: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    contents: pd.DataFrame,
    products: pd.DataFrame,
    feedback: pd.DataFrame,
    revenues: pd.DataFrame,
    lang: str = "中文",
) -> dict[str, Any]:
    """A lightweight GraphRAG layer: retrieve graph context, then produce templated answers."""
    local = scope in {"local", "局部查询"}
    if local:
        evidence = local_graph_query(nodes, edges, query, lang=lang)
        if evidence.empty:
            answer = "没有找到匹配实体。可以输入内容标题、产品名、功能名或用户人群。" if lang == "中文" else "No matching entity found. Try a title, product, feature or segment."
        else:
            first = evidence.iloc[0]
            answer = (
                f"找到与“{query or first['起点']}”相关的 {len(evidence)} 条关系。优先查看“{first['起点']}—{first['关系']}—{first['终点']}”。"
                if lang == "中文"
                else f"Found {len(evidence)} related graph links. Start with {first['source']} - {first['relation']} - {first['target']}."
            )
        return {"answer": answer, "evidence": evidence, "scope": "local"}

    global_context = global_graph_query(contents, products, feedback, revenues, lang=lang)
    feature = global_context["feature_combinations"].head(1)
    platform = global_context["platform_topic"].head(1)
    series = global_context["series"].head(1)
    parts = []
    if not platform.empty:
        row = platform.iloc[0]
        parts.append(f"{row['平台']}的“{row['主题']}”收入最高" if lang == "中文" else f"{row['platform']} / {row['topic']} leads revenue")
    if not feature.empty:
        row = feature.iloc[0]
        parts.append(f"高转化功能组合是“{row['功能组合']}”" if lang == "中文" else f"top feature mix is {row['功能组合']}")
    if not series.empty:
        row = series.iloc[0]
        parts.append(f"重点系列是“{row['series_id']}”" if lang == "中文" else f"key series is {row['series_id']}")
    answer = "；".join(parts) + ("。这些是图谱检索线索，放大前仍要看实验或增量估计。" if lang == "中文" else ". These are graph retrieval signals; validate with experiments or lift estimates before scaling.")
    return {"answer": answer, "evidence": global_context, "scope": "global"}


def strategy_constraint_checks(cards: list[dict[str, Any]], contents: pd.DataFrame, ab_tests: pd.DataFrame, beta_tests: pd.DataFrame, lang: str = "中文") -> list[dict[str, Any]]:
    """KAG-style logical checks for suggestions. It does not estimate causality."""
    checked: list[dict[str, Any]] = []
    max_followers = 0.0
    if not contents.empty and "followers_before" in contents.columns:
        max_followers = pd.to_numeric(contents["followers_before"], errors="coerce").fillna(0).max()
    has_experiment = not ab_tests.empty or not beta_tests.empty
    for card in cards:
        item = dict(card)
        action = _safe_text(item.get("action") or item.get("detail"))
        sample_size = _safe_float(item.get("sample_size"), 0)
        warnings = []
        if ("放大" in action or "扩大" in action or "scale" in action.lower()) and not has_experiment:
            warnings.append("没有实验记录时，不建议直接放大；先做小范围验证。" if lang == "中文" else "No experiment record; validate before scaling.")
        if 0 < sample_size < 20:
            warnings.append("样本少于 20，只能作为探索线索。" if lang == "中文" else "Sample size below 20; treat as exploratory.")
        if max_followers >= 500_000 and any(token in action for token in ["各发 6", "发 12", "连续发布"]):
            warnings.append("大号账号不建议高频硬测，先用低流量时段或小号验证。" if lang == "中文" else "Large accounts should avoid high-frequency tests; validate in low-risk slots first.")
        item["constraint_warnings"] = warnings
        if warnings:
            item["priority"] = "medium" if item.get("priority") == "high" else item.get("priority", "medium")
            item["confidence"] = "需先满足实验前置条件" if lang == "中文" else "Needs experiment prerequisites"
            item["action"] = f"{action}（{warnings[0]}）" if action else warnings[0]
        checked.append(item)
    return checked
