from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.agent_orchestrator import find_synthetic_data_dir, run_agent_suite
from src.data_loader import load_all
from src.knowledge_graph import graph_rag_answer, plot_knowledge_graph
from src.mock_data import generate_all
from src.revenue_analysis import pending_payment_summary, platform_revenue_summary
from src.strategy_analysis import weekly_topic_plan


st.set_page_config(page_title="SoloDeck Modern", page_icon="▰", layout="wide", initial_sidebar_state="collapsed")


PLATFORM_LABELS = {
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

STYLE_LABELS = {
    "pain_point": "痛点标题",
    "tutorial": "教程标题",
    "number": "数字清单",
    "story": "故事型",
    "contrast": "对比型",
    "result_oriented": "结果导向",
    "question": "提问型",
}


def load_demo_data():
    synthetic = find_synthetic_data_dir(ROOT.parent)
    data_dir = synthetic or ROOT / "data"
    if not (data_dir / "mock_contents.csv").exists():
        generate_all(data_dir)
    return load_all(data_dir)


def platform_name(value: str) -> str:
    return PLATFORM_LABELS.get(str(value), str(value))


def pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.0%}"


def safe_rate(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def metric_lifts(contents: pd.DataFrame, revenues: pd.DataFrame, days: int) -> dict[str, dict]:
    df = contents.copy()
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    latest = df["publish_time"].max()
    if pd.isna(latest):
        latest = pd.Timestamp.today()
    current_start = latest - timedelta(days=days)
    previous_start = latest - timedelta(days=days * 2)
    current = df[df["publish_time"].between(current_start, latest)]
    previous = df[df["publish_time"].between(previous_start, current_start)]
    if current.empty:
        current = df.tail(max(1, min(len(df), days * 2)))
    if previous.empty:
        previous = df.head(max(1, min(len(df), days * 2)))

    cur_views = current["views"].sum()
    prev_views = previous["views"].sum()
    cur_consult = safe_rate(current["consultations"].sum(), cur_views)
    prev_consult = safe_rate(previous["consultations"].sum(), prev_views)
    cur_fav = safe_rate(current["favorites"].sum(), cur_views)
    prev_fav = safe_rate(previous["favorites"].sum(), prev_views)
    cur_revenue = float(current["revenue"].sum())
    prev_revenue = float(previous["revenue"].sum())

    return {
        "consult": {"label": "咨询率", "value": cur_consult, "lift": safe_rate(cur_consult - prev_consult, abs(prev_consult) or 1), "icon": "message"},
        "favorite": {"label": "收藏率", "value": cur_fav, "lift": safe_rate(cur_fav - prev_fav, abs(prev_fav) or 1), "icon": "bookmark"},
        "views": {"label": "播放量", "value": cur_views, "lift": safe_rate(cur_views - prev_views, abs(prev_views) or 1), "icon": "play"},
        "revenue": {"label": "收入", "value": cur_revenue, "lift": safe_rate(cur_revenue - prev_revenue, abs(prev_revenue) or 1), "icon": "yen"},
    }


def trend_frame(contents: pd.DataFrame) -> pd.DataFrame:
    df = contents.copy()
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df = df.dropna(subset=["publish_time"])
    if df.empty:
        return pd.DataFrame()
    df["date"] = df["publish_time"].dt.date
    grouped = df.groupby("date", as_index=False).agg(
        views=("views", "sum"),
        favorites=("favorites", "sum"),
        consultations=("consultations", "sum"),
        conversions=("conversions", "sum"),
        revenue=("revenue", "sum"),
    )
    grouped["咨询率"] = grouped["consultations"] / grouped["views"].replace(0, pd.NA)
    grouped["收藏率"] = grouped["favorites"] / grouped["views"].replace(0, pd.NA)
    grouped["成交率"] = grouped["conversions"] / grouped["consultations"].replace(0, pd.NA)
    out = grouped.tail(8).copy()
    numeric_cols = ["views", "favorites", "consultations", "conversions", "revenue", "咨询率", "收藏率", "成交率"]
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out


def svg_icon(kind: str) -> str:
    icons = {
        "message": '<path d="M5 6.5A6.5 6.5 0 0 1 11.5 0h1A6.5 6.5 0 0 1 19 6.5v1A6.5 6.5 0 0 1 12.5 14H9l-5 3v-4.2A6.5 6.5 0 0 1 0 7.5v-1A6.5 6.5 0 0 1 5 6.5Z" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="6.5" cy="7" r="1" fill="currentColor"/><circle cx="10" cy="7" r="1" fill="currentColor"/><circle cx="13.5" cy="7" r="1" fill="currentColor"/>',
        "bookmark": '<path d="M4 2.5A2.5 2.5 0 0 1 6.5 0h7A2.5 2.5 0 0 1 16 2.5V19l-6-3.7L4 19V2.5Z" fill="none" stroke="currentColor" stroke-width="1.8"/>',
        "play": '<circle cx="10" cy="10" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M8 6.5 14 10l-6 3.5v-7Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>',
        "yen": '<circle cx="10" cy="10" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M6.5 5.5 10 10l3.5-4.5M10 10v5M7 10h6M7 13h6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
    }
    return f'<svg viewBox="0 0 20 20" aria-hidden="true">{icons.get(kind, icons["message"])}</svg>'


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], header { display:none !important; }
        .stApp {
            background: #eef4f1;
            color: #143126;
        }
        .block-container {
            max-width: 1180px;
            padding: 0.9rem 1rem 1.4rem;
        }
        [data-testid="stVerticalBlock"] { gap: 0.75rem; }
        .sd-shell {
            min-height: calc(100vh - 36px);
            border-radius: 26px;
            overflow: hidden;
            background: #fbfefd;
            box-shadow: 0 28px 70px rgba(18, 57, 48, .17);
            border: 1px solid rgba(255,255,255,.74);
        }
        .sd-sidebar {
            height: calc(100vh - 36px);
            min-height: 760px;
            background:
                radial-gradient(circle at 30% 10%, rgba(111, 210, 177, .34), transparent 24%),
                linear-gradient(160deg, #0f765e 0%, #145845 48%, #0c4337 100%);
            color: #f7fffb;
            padding: 26px 16px;
        }
        .sd-brand {
            display: flex;
            gap: 10px;
            align-items: center;
            font-weight: 760;
            font-size: 18px;
            margin-bottom: 34px;
        }
        .sd-logo {
            width: 28px; height: 28px; border-radius: 8px;
            background: linear-gradient(135deg, #eafff4, #87d7b6);
            position: relative;
            box-shadow: inset 0 -5px 0 rgba(18, 92, 72, .22);
        }
        .sd-logo:after {
            content: "";
            position: absolute;
            width: 8px; height: 8px;
            right: -2px; top: 4px;
            border-radius: 99px;
            background: #f6ce6d;
        }
        .sd-nav-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 13px 14px;
            margin: 7px 0;
            border-radius: 12px;
            color: rgba(255,255,255,.84);
            font-weight: 650;
            font-size: 15px;
        }
        .sd-nav-item.active {
            background: rgba(255,255,255,.18);
            color: #fff;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.12);
        }
        .sd-nav-icon {
            width: 22px; height: 22px; border-radius: 7px;
            display:inline-flex; align-items:center; justify-content:center;
            border: 1.3px solid rgba(255,255,255,.76);
            font-size: 12px;
        }
        .sd-main {
            min-height: calc(100vh - 36px);
            padding: 34px 38px 30px;
            background:
              radial-gradient(circle at 88% 9%, rgba(95,165,140,.10), transparent 18%),
              linear-gradient(180deg, #ffffff 0%, #fbfdfc 100%);
        }
        .sd-title-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
            margin-bottom: 26px;
        }
        .sd-title {
            font-size: 30px;
            line-height: 1.1;
            font-weight: 760;
            letter-spacing: 0;
            color: #12251e;
        }
        .sd-actions { display:flex; gap: 12px; align-items:center; }
        .sd-select, .sd-button {
            border: 1px solid #d9e5e0;
            background: #fff;
            border-radius: 10px;
            height: 42px;
            padding: 0 16px;
            display:inline-flex;
            align-items:center;
            gap:8px;
            font-weight: 680;
            color:#18382e;
            box-shadow: 0 8px 20px rgba(21, 78, 63, .06);
        }
        .sd-button.primary {
            color:#0b6d53;
        }
        .sd-card-grid {
            display:grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 18px;
            margin-bottom: 30px;
        }
        .sd-metric-card {
            border: 1px solid #dbe8e4;
            border-radius: 16px;
            padding: 20px 20px 18px;
            min-height: 156px;
            background: rgba(255,255,255,.86);
            box-shadow: 0 13px 35px rgba(18, 57, 48, .07);
        }
        .sd-metric-card svg {
            width: 38px;
            height: 38px;
            color: #0f7b5f;
            margin-bottom: 13px;
        }
        .sd-lift {
            font-size: 34px;
            line-height: 1;
            color: #087b5b;
            font-weight: 780;
            margin-bottom: 14px;
        }
        .sd-label {
            color: #18342b;
            font-size: 17px;
            font-weight: 710;
        }
        .sd-up {
            color: #0a8d67;
            font-weight: 780;
            padding-left: 4px;
        }
        .sd-section-title {
            font-size: 23px;
            font-weight: 760;
            color:#12251e;
            margin: 8px 0 10px;
        }
        .sd-panel {
            border: 1px solid #deebe7;
            border-radius: 18px;
            background: rgba(255,255,255,.88);
            padding: 18px;
            box-shadow: 0 16px 38px rgba(18, 57, 48, .07);
        }
        .sd-action-card {
            border: 1px solid #dfebe7;
            border-radius: 14px;
            padding: 15px 16px;
            background: #ffffff;
            margin-bottom: 12px;
        }
        .sd-badge {
            display:inline-flex;
            align-items:center;
            border-radius: 999px;
            padding: 4px 9px;
            background:#edf8f3;
            color:#087b5b;
            font-size: 12px;
            font-weight: 760;
            margin-bottom: 7px;
        }
        .sd-action-title {
            color:#10251f;
            font-size: 16px;
            font-weight: 750;
            margin-bottom: 5px;
        }
        .sd-muted { color:#64746e; font-size: 13.5px; line-height:1.55; }
        .sd-mini-grid {
            display:grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }
        .sd-mini {
            border-radius: 14px;
            border: 1px solid #e1ebe8;
            padding: 14px;
            background: #fff;
        }
        .sd-mini-label { color:#62746d; font-size: 13px; }
        .sd-mini-value { color:#15362c; font-weight: 780; font-size: 21px; margin-top: 5px; }
        .stPlotlyChart {
            border-radius: 18px;
        }
        @media (max-width: 980px) {
            .sd-sidebar { height:auto; min-height: auto; border-radius: 20px 20px 0 0; }
            .sd-main { padding: 24px 20px; }
            .sd-card-grid, .sd-mini-grid { grid-template-columns: 1fr 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(active: str) -> None:
    items = [
        ("概览", "home"),
        ("内容分析", "chart"),
        ("受众分析", "user"),
        ("渠道分析", "channel"),
        ("转化分析", "flow"),
        ("设置", "gear"),
    ]
    icon_text = {"home": "⌂", "chart": "↗", "user": "◇", "channel": "▤", "flow": "✣", "gear": "⚙"}
    nav = "".join(
        f'<div class="sd-nav-item {"active" if label == active else ""}"><span class="sd-nav-icon">{icon_text[key]}</span><span>{label}</span></div>'
        for label, key in items
    )
    st.markdown(
        f"""
        <div class="sd-sidebar">
          <div class="sd-brand"><span class="sd-logo"></span><span>SoloDeck</span></div>
          {nav}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(lifts: dict[str, dict]) -> None:
    order = ["consult", "favorite", "views", "revenue"]
    html = ['<div class="sd-card-grid">']
    for key in order:
        item = lifts[key]
        html.append(
            f"""
            <div class="sd-metric-card">
              {svg_icon(item["icon"])}
              <div class="sd-lift">{pct(item["lift"])}</div>
              <div class="sd-label">{item["label"]}<span class="sd-up">↑</span></div>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_trend_chart(trend: pd.DataFrame) -> None:
    fig = go.Figure()
    if not trend.empty:
        x = pd.to_datetime(trend["date"]).dt.strftime("%-m/%-d")
        fig.add_trace(go.Scatter(
            x=x,
            y=trend["咨询率"] * 100,
            mode="lines+markers",
            name="咨询率",
            line=dict(color="#087b5b", width=3),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=trend["收藏率"] * 100,
            mode="lines+markers",
            name="收藏率",
            line=dict(color="#087b5b", width=2.4, dash="dash"),
            marker=dict(size=7),
        ))
    fig.update_layout(
        height=315,
        margin=dict(l=8, r=8, t=8, b=8),
        template="plotly_white",
        legend=dict(orientation="h", y=1.12, x=0.62),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="转化率", ticksuffix="%", range=[0, max(7, float((trend[["咨询率", "收藏率"]].max().max() * 120) if not trend.empty else 7))]),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", color="#18342b"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_actions(cards: list[dict]) -> None:
    html = []
    for card in cards[:3]:
        title = card.get("title", "建议")
        reason = card.get("reason") or card.get("insight", "")
        action = card.get("action", "")
        html.append(
            f"""
            <div class="sd-action-card">
              <span class="sd-badge">建议</span>
              <div class="sd-action-title">{title}</div>
              <div class="sd-muted">{reason}</div>
              <div class="sd-muted">{action}</div>
            </div>
            """
        )
    st.markdown("".join(html), unsafe_allow_html=True)


def render_causal_and_graph(agent: dict, contents: pd.DataFrame, products: pd.DataFrame, feedback: pd.DataFrame, revenues: pd.DataFrame) -> None:
    causal = agent["modules"].get("causal_estimator_agent", {})
    kg_agent = agent["modules"].get("knowledge_graph_agent", {})
    ate = causal.get("ate", [])
    kg_nodes = kg_agent.get("nodes", pd.DataFrame())
    kg_edges = kg_agent.get("edges", pd.DataFrame())

    st.markdown('<div class="sd-section-title">因果与图谱解释</div>', unsafe_allow_html=True)
    left, right = st.columns([0.42, 0.58], gap="large")
    with left:
        if ate:
            cards = []
            for item in ate[:3]:
                question = str(item.get("question", "策略增量")).replace("title_style -> views", "标题风格影响播放").replace("same_content_cross_platform -> revenue", "同内容跨平台收入差异").replace("advanced_methods_available", "专业模型可用")
                effect = item.get("effect_estimate", item.get("mean_difference", item.get("effect", 0)))
                confidence = item.get("confidence", "探索性")
                cards.append(
                    f"""
                    <div class="sd-action-card">
                      <span class="sd-badge">因果估计</span>
                      <div class="sd-action-title">{question}</div>
                      <div class="sd-muted">估计增量：{float(effect or 0):,.2f}；可信度：{confidence}</div>
                    </div>
                    """
                )
            st.markdown("".join(cards), unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="sd-action-card">
                  <span class="sd-badge">因果估计</span>
                  <div class="sd-action-title">等待更多实验数据</div>
                  <div class="sd-muted">系统会在有 AB Test、跨平台或内测数据后估计策略增量。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        if not kg_nodes.empty:
            answer = graph_rag_answer("", "全局查询", kg_nodes, kg_edges, contents, products, feedback, revenues, "中文")
            st.markdown(
                f"""
                <div class="sd-panel">
                  <span class="sd-badge">知识图谱</span>
                  <div class="sd-action-title">先解释关系，再做因果判断</div>
                  <div class="sd-muted">{answer["answer"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(plot_knowledge_graph(kg_nodes.head(80), kg_edges.head(220), "中文"), width="stretch", config={"displayModeBar": False})
        else:
            st.markdown('<div class="sd-panel"><div class="sd-muted">暂无图谱数据。</div></div>', unsafe_allow_html=True)


def render_overview(contents, revenues, campaigns, ab_tests, products, feedback, beta_tests) -> None:
    period = st.session_state.get("period", "7天")
    days = int(period.replace("天", ""))
    agent = run_agent_suite(contents, revenues, campaigns, ab_tests, products, feedback, beta_tests, lang="中文")
    lifts = metric_lifts(contents, revenues, days)
    trend = trend_frame(contents)
    kg_agent = agent["modules"].get("knowledge_graph_agent", {})
    kg_nodes = kg_agent.get("nodes", pd.DataFrame())
    kg_edges = kg_agent.get("edges", pd.DataFrame())

    st.markdown(
        f"""
        <div class="sd-title-row">
          <div class="sd-title">策略效果总览</div>
          <div class="sd-actions">
            <div class="sd-select">{period}⌄</div>
            <div class="sd-button primary">＋ 新建策略</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_metric_cards(lifts)

    left, right = st.columns([0.62, 0.38], gap="large")
    with left:
        st.markdown('<div class="sd-section-title">趋势分析</div>', unsafe_allow_html=True)
        render_trend_chart(trend)
    with right:
        st.markdown('<div class="sd-section-title">下一步行动</div>', unsafe_allow_html=True)
        render_actions(agent["cards"])

    lower_left, lower_right = st.columns([0.5, 0.5], gap="large")
    with lower_left:
        st.markdown('<div class="sd-section-title">图谱摘要</div>', unsafe_allow_html=True)
        if not kg_nodes.empty:
            answer = graph_rag_answer("", "全局查询", kg_nodes, kg_edges, contents, products, feedback, revenues, "中文")
            st.markdown(
                f"""
                <div class="sd-panel">
                  <span class="sd-badge">GraphRAG</span>
                  <div class="sd-action-title">经营关系检索</div>
                  <div class="sd-muted">{answer["answer"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="sd-panel"><div class="sd-muted">暂无图谱数据。</div></div>', unsafe_allow_html=True)
    with lower_right:
        st.markdown('<div class="sd-section-title">经营状态</div>', unsafe_allow_html=True)
        pending = pending_payment_summary(revenues, campaigns)
        revenue_by_platform = platform_revenue_summary(revenues)
        top_platform = platform_name(revenue_by_platform.iloc[0]["platform"]) if not revenue_by_platform.empty else "暂无"
        st.markdown(
            f"""
            <div class="sd-mini-grid">
              <div class="sd-mini"><div class="sd-mini-label">收入主阵地</div><div class="sd-mini-value">{top_platform}</div></div>
              <div class="sd-mini"><div class="sd-mini-label">待收款</div><div class="sd-mini-value">¥{pending["total_pending_amount"]:,.0f}</div></div>
              <div class="sd-mini"><div class="sd-mini-label">实验记录</div><div class="sd-mini-value">{len(ab_tests)}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_causal_and_graph(agent, contents, products, feedback, revenues)


def render_content(contents, revenues) -> None:
    plan = weekly_topic_plan(contents, revenues, n=5, language="中文")
    st.markdown('<div class="sd-title-row"><div class="sd-title">内容分析</div><div class="sd-actions"><div class="sd-button primary">生成排期</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sd-panel">', unsafe_allow_html=True)
    if plan:
        for item in plan:
            title_style = STYLE_LABELS.get(item["suggested_title_style"], item["suggested_title_style"])
            st.markdown(
                f"""
                <div class="sd-action-card">
                  <span class="sd-badge">{platform_name(item["suggested_platform"])}</span>
                  <div class="sd-action-title">{item["sample_title"]}</div>
                  <div class="sd-muted">主题：{item["suggested_topic"]}；标题风格：{title_style}；目标：{item["objective"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_placeholder(title: str, desc: str) -> None:
    st.markdown(
        f"""
        <div class="sd-title-row"><div class="sd-title">{title}</div></div>
        <div class="sd-panel">
          <span class="sd-badge">即将展开</span>
          <div class="sd-action-title">{title}</div>
          <div class="sd-muted">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()
    contents, revenues, campaigns, ab_tests, products, feedback, beta_tests = load_demo_data()
    if "modern_page" not in st.session_state:
        st.session_state.modern_page = "概览"
    if "period" not in st.session_state:
        st.session_state.period = "7天"

    st.markdown('<div class="sd-shell">', unsafe_allow_html=True)
    sidebar, main_area = st.columns([0.19, 0.81], gap="small")
    with sidebar:
        render_sidebar(st.session_state.modern_page)
    with main_area:
        st.markdown('<div class="sd-main">', unsafe_allow_html=True)
        if st.session_state.modern_page == "概览":
            render_overview(contents, revenues, campaigns, ab_tests, products, feedback, beta_tests)
        elif st.session_state.modern_page == "内容分析":
            render_content(contents, revenues)
        elif st.session_state.modern_page == "受众分析":
            render_placeholder("受众分析", "这里会展示用户人群、反馈问题和产品偏好的图谱关系。")
        elif st.session_state.modern_page == "渠道分析":
            render_placeholder("渠道分析", "这里会展示不同平台的收入、咨询、收藏和转化差异。")
        elif st.session_state.modern_page == "转化分析":
            render_placeholder("转化分析", "这里会展示 AB Test、因果增量估计和可放大策略。")
        else:
            render_placeholder("设置", "这里会接入上传、账号、API 和数据源设置。")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
