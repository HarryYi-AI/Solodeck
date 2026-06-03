from __future__ import annotations

import os
import sys
import hashlib
import html
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.auto_insights import generate_auto_insights, insight_figures, metric_snapshot
from src.agent_orchestrator import analysis_to_json, find_synthetic_data_dir, run_agent_suite
from src.auth import init_auth_db, login_user, register_user, user_count
from src.business_collab import campaign_dashboard, campaign_risk_alerts, generate_brand_report, pricing_suggestion
from src.beta_test_planner import beta_test_readiness_check, design_beta_test, generate_feedback_form, recommend_next_validation_step
from src.calendar_export import tasks_to_ics
from src.causal_interface import run_causal_closed_loop
from src.causal_estimator import cross_platform_increment, iptw_effect, propensity_score_matching, stratified_regression_effect
from src.causal_experiment import analyze_ab_test, causal_readiness_check, design_ab_test
from src.causal_refute import refute_suite
from src.data_loader import load_all, read_uploaded_csv
from src.experiment_planner import weekly_experiment_plan
from src.llm_agent import extract_records_from_uploads, generate_ai_business_advice, interpret_experiment, llm_configured, model_status, polish_report
from src.mock_data import generate_all
from src.recommendation_engine import revenue_recommendations
from src.report_generator import generate_strategy_report, generate_weekly_business_report
from src.revenue_analysis import calculate_rpm, classify_revenue_summary, content_commercial_value, pending_payment_summary, platform_revenue_summary, topic_business_summary
from src.incremental_effect import estimate_feature_upgrade_effect, estimate_incremental_effect, generate_incremental_insight
from src.knowledge_base import knowledge_cards, knowledge_frame
from src.product_feedback import beta_feedback_effect, classify_feedback, feedback_to_roadmap, feedback_topic_clustering, generate_feedback_report, sentiment_revenue_link
from src.recommendation_learning import learning_summary, rank_recommendations, record_feedback, recommendation_id
from src.series_analysis import cannibalization_check, content_series_performance, marginal_gain_of_new_item, product_series_performance, recommend_series_strategy
from src.similarity_engine import detect_content_overlap, detect_product_variants
from src.strategy_analysis import platform_strategy_analysis, publish_time_analysis, title_style_analysis, topic_strategy_analysis, weekly_topic_plan
from src.text_structured import extract_text_features, text_feature_summary
from src.user_storage import init_user_storage, list_user_files, load_latest_user_dataset, load_user_workspace, save_uploaded_file, save_user_dataset, save_user_workspace
from src.variable_mapper import map_variables
from src.workflow_engine import build_workflow_trace, workflow_markdown


st.set_page_config(page_title="SoloDeck", page_icon="▰", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    :root {color-scheme: light;}
    #MainMenu, footer {visibility: hidden;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}
    header, [data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background:
          radial-gradient(circle at 12% 0%, rgba(95,165,140,0.10), transparent 28%),
          linear-gradient(180deg, #fffdf8 0%, #faf8f3 46%, #fbfaf7 100%) !important;
        color: #111827 !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background: #ffffff !important;
        color: #111827 !important;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: min(1440px, calc(100vw - 44px));
    }
    .setup-box {border: 1px solid #e6e8eb; border-radius: 8px; padding: 1rem; background: #fff;}
    h1 {font-size: 1.7rem !important; letter-spacing: 0; font-weight: 650 !important;}
    h2, h3 {letter-spacing: 0; font-weight: 620 !important;}
    p, label, li {color: #111827;}
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #111827 !important;
    }
    input, textarea, [data-baseweb="input"], [data-baseweb="textarea"] {
        background: #ffffff !important;
        color: #111827 !important;
        border-color: #d1d5db !important;
    }
    [data-baseweb="select"] > div, [data-baseweb="tag"] {
        background: #ffffff !important;
        color: #111827 !important;
        border-color: #d1d5db !important;
    }
    [data-baseweb="tag"] {
        padding-left: 10px !important;
        padding-right: 8px !important;
        min-width: fit-content !important;
    }
    [data-baseweb="tag"] span, [data-baseweb="tag"] div {
        color: #111827 !important;
        overflow: visible !important;
    }
    button {border-radius: 12px !important;}
    div.stButton > button[kind="secondary"] {
        border-color: #d8ded6 !important;
        background: #fffefc !important;
        color: #111827 !important;
    }
    div.stButton > button:hover {
        border-color: #86b79f !important;
        color: #143b30 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e9e4dc !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 30px rgba(36, 45, 38, 0.035) !important;
        background: #fffefc !important;
    }
    .soft-card {
        border: 1px solid #e9e4dc;
        border-radius: 14px;
        padding: 14px 16px;
        background: #fffefc;
        margin-bottom: 10px;
        box-shadow: 0 8px 22px rgba(36, 45, 38, 0.035);
    }
    .risk-band {
        border-left: 3px solid #ef4444;
        background: #fffafa;
    }
    .priority-card {
        border: 1px solid #e9e4dc;
        border-radius: 16px;
        padding: 16px 18px;
        background: #fffefc;
        margin-bottom: 12px;
        box-shadow: 0 12px 28px rgba(36, 45, 38, 0.04);
    }
    .priority-card.p0 {border-left: 4px solid #ef4444;}
    .priority-card.p1 {border-left: 4px solid #d69b2d;}
    .priority-card.p2 {border-left: 4px solid #7c8a7a;}
    .priority-row {display:flex; align-items:center; gap:8px; margin-bottom:8px;}
    .priority-chip {
        display:inline-flex;
        align-items:center;
        justify-content:center;
        min-width: 34px;
        height: 24px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .priority-chip.p0 {background:#fff1f1; color:#b42318; border:1px solid #ffc7c7;}
    .priority-chip.p1 {background:#fff8e6; color:#8a5a00; border:1px solid #f5d48a;}
    .priority-chip.p2 {background:#f4f6f8; color:#4b5563; border:1px solid #d7dce3;}
    .priority-title {font-weight:700; font-size:1.02rem; color:#111827;}
    .priority-body {color:#374151; line-height:1.65;}
    .soft-card-title {
        font-weight: 650;
        font-size: 1rem;
        margin-bottom: 6px;
        color: #111827;
    }
    .soft-muted {
        color: #6b7280;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 2px 9px;
        font-size: 0.78rem;
        margin-right: 6px;
        border: 1px solid #d7dce3;
        color: #374151;
        background: #f8fafc;
    }
    .badge-danger {border-color: #f4b4b4; color: #9f1d1d; background: #fff7f7;}
    .badge-warn {border-color: #f1d592; color: #7a5200; background: #fffaf0;}
    .badge-ok {border-color: #b7dfc5; color: #17683a; background: #f4fbf6;}
    .app-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 10px 0 16px;
        border-bottom: 1px solid #eee7dc;
        margin-bottom: 16px;
    }
    .brand {
        display:flex;
        align-items:center;
        gap:12px;
        font-weight: 760;
        font-size: 1.12rem;
        color:#111827;
    }
    .brand-mark {
        width: 28px;
        height: 28px;
        border-radius: 9px;
        background: linear-gradient(145deg, #5fa58c, #d8e7cf);
        display:inline-block;
        position: relative;
        box-shadow: inset 0 -7px 0 rgba(255,255,255,0.42);
    }
    .brand-mark:after {
        content: "";
        position: absolute;
        right: -4px;
        top: 5px;
        width: 9px;
        height: 9px;
        border-radius: 999px;
        background: #f4c86b;
    }
    .nav-sub {
        color:#7a746b;
        font-size:0.88rem;
        margin-top:2px;
    }
    .capability-grid {
        display:grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
    }
    .capability {
        border: 1px solid #e9e4dc;
        border-radius: 14px;
        padding: 12px 14px;
        background: #fffefc;
    }
    .capability b {
        display:block;
        margin-bottom:4px;
    }
    .guide-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin: 10px 0 12px;
    }
    .guide-step {
        border: 1px solid #e9e4dc;
        border-radius: 14px;
        background: #fffefc;
        padding: 10px 12px;
        min-height: 66px;
        box-shadow: 0 8px 22px rgba(36, 45, 38, 0.035);
    }
    .guide-index {
        display:inline-flex;
        width:24px;
        height:24px;
        align-items:center;
        justify-content:center;
        border-radius:999px;
        background:#e8f1ea;
        color:#2f6f5b;
        font-weight:720;
        margin-right:8px;
    }
    .guide-title {
        color:#111827;
        font-weight:700;
    }
    .guide-copy {
        margin-top:7px;
        color:#6b7280;
        font-size:0.86rem;
        line-height:1.45;
    }
    .landing {
        border: 1px solid #e9e4dc;
        border-radius: 22px;
        background: rgba(255,254,252,0.88);
        padding: 28px;
        margin: 14px 0 16px;
        box-shadow: 0 18px 48px rgba(36,45,38,0.055);
    }
    .landing h1 {
        font-size: 2.25rem !important;
        line-height: 1.12 !important;
        margin: 0 0 10px !important;
    }
    .landing-copy {
        max-width: 680px;
        color: #5d6672;
        font-size: 1rem;
        line-height: 1.7;
    }
    .feature-grid {
        display:grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin-top: 16px;
    }
    .feature-card {
        border: 1px solid #e9e4dc;
        border-radius: 16px;
        padding: 14px;
        background: #fffefc;
        min-height: 104px;
    }
    .feature-card b {
        display:block;
        color:#111827;
        margin-bottom: 6px;
    }
    .feature-card span {
        color:#6b7280;
        font-size:0.9rem;
        line-height:1.5;
    }
    .onboarding {
        border: 1px solid #dfe8df;
        border-radius: 16px;
        background: #f8fcf8;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .done-card {
        opacity: 0.72;
        filter: grayscale(0.15);
    }
    .status-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 12px 0 14px;
    }
    .status-pill {
        border: 1px solid #e9e4dc;
        border-radius: 14px;
        background: rgba(255,254,252,0.86);
        padding: 10px 12px;
        min-height: 58px;
    }
    .status-label {
        color:#7a746b;
        font-size:0.78rem;
        margin-bottom:3px;
    }
    .status-value {
        color:#111827;
        font-size:1.1rem;
        font-weight:720;
    }
    [data-testid="stExpander"] {
        border-color: #ece7df !important;
        border-radius: 13px !important;
        background: rgba(255,254,252,0.78) !important;
    }
    [data-baseweb="tab-list"] {
        gap: 6px !important;
    }
    [data-baseweb="tab"] {
        border-radius: 999px !important;
        padding: 7px 12px !important;
        background: #fffefc !important;
        border: 1px solid #eee7dc !important;
        height: auto !important;
    }
    @media (max-width: 900px) {
        .capability-grid {grid-template-columns: 1fr 1fr;}
        .guide-strip {grid-template-columns: 1fr;}
        .feature-grid {grid-template-columns: 1fr 1fr;}
        .status-strip {grid-template-columns: 1fr 1fr;}
    }
    @media (max-width: 640px) {
        .capability-grid {grid-template-columns: 1fr;}
        .block-container {padding-left: 1rem; padding-right: 1rem; max-width: 100vw;}
        .status-strip {grid-template-columns: 1fr;}
        .feature-grid {grid-template-columns: 1fr;}
    }
    hr {margin: 1.25rem 0;}
    .flow-step {
        border: 1px solid #e5e7ea;
        border-radius: 8px;
        padding: 12px;
        background: #fff;
        height: 100%;
    }
    .flow-step b {display: block; margin-bottom: 4px;}
    .small-muted {color: #667085; font-size: 0.9rem;}
    .top-controls {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 10px;
        margin: -4px 0 16px;
    }
    div[data-testid="column"]:has(.account-anchor) {
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 0 !important;
    }
    div[data-testid="column"]:has(.account-anchor) button {
        width: auto !important;
        min-height: 36px !important;
        padding: 0.38rem 0.82rem !important;
        border-radius: 999px !important;
        background: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #111827 !important;
        font-weight: 560 !important;
        white-space: nowrap !important;
    }
    div[data-testid="column"]:has(.language-anchor) [data-testid="stRadio"] {
        margin-top: -4px;
    }
    div[data-testid="column"]:has(.language-anchor) label {
        font-size: 0.9rem !important;
    }
    div[data-testid="column"]:has(.language-anchor) [role="radiogroup"] {
        gap: 4px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def cached_load(data_dir_override: str | None = None):
    data_dir = Path(data_dir_override) if data_dir_override else ROOT / "data"
    if not (data_dir / "mock_contents.csv").exists():
        generate_all(data_dir)
    return load_all(data_dir)


def empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy()


def use_example_data() -> bool:
    return bool(st.session_state.get("guest_mode")) or bool(st.session_state.get("use_example_data"))


def metric_card(label: str, value, help_text: str | None = None):
    st.metric(label, value, help=help_text)


def render_html(markup: str) -> None:
    """Render custom UI HTML without Markdown treating indented tags as code."""
    st.html(str(markup).strip())


def suggestion_cards(items: list[dict], lang: str = "中文"):
    for index, item in enumerate(items):
        priority = item.get("priority") or item.get("level") or "medium"
        rank = "p0" if priority == "high" or index == 0 else "p1" if priority == "medium" or index == 1 else "p2"
        label = ("首要" if rank == "p0" else "重要" if rank == "p1" else "可做") if lang == "中文" else ("Now" if rank == "p0" else "Important" if rank == "p1" else "Later")
        title = item.get("title", "建议" if lang == "中文" else "Recommendation")
        reason = item.get("reason", "")
        action = item.get("action") or item.get("detail") or ""
        if lang != "中文":
            title, reason, action = englishize_text(title), englishize_text(reason), englishize_text(action)
        render_html(
            f"""
            <div class="priority-card {rank}">
              <div class="priority-row"><span class="priority-chip {rank}">{label}</span><span class="priority-title">{safe_text(title)}</span></div>
              <div class="soft-muted">{reason}</div>
              <div class="priority-body">{action}</div>
            </div>
            """
        )


def agent_cards_to_suggestions(cards: list[dict], lang: str = "中文", limit: int = 6) -> list[dict]:
    suggestions = []
    for card in cards[:limit]:
        reason = card.get("insight", "")
        suggestions.append({
            "title": card.get("title", ""),
            "reason": reason,
            "action": card.get("action", ""),
            "priority": card.get("priority", "medium"),
        })
    return suggestions


def render_suggestion_feedback(items: list[dict], lang: str = "中文", key_prefix: str = "rec_feedback") -> None:
    zh = lang == "中文"
    if not items:
        return
    with st.expander("调整建议偏好" if zh else "Tune Recommendation Preferences", expanded=False):
        st.caption(
            "告诉 SoloDeck 哪类建议更适合你，后续会优先展示类似动作。"
            if zh
            else "Tell SoloDeck what fits you; similar actions will be prioritized later."
        )
        for idx, item in enumerate(items[:4]):
            rid = recommendation_id(item)
            cols = st.columns([0.58, 0.21, 0.21], vertical_alignment="center")
            cols[0].caption(item.get("title", ""))
            if cols[1].button("适合我" if zh else "Useful", key=f"{key_prefix}_accept_{rid}_{idx}", width="stretch"):
                st.session_state.recommendation_feedback = record_feedback(st.session_state.get("recommendation_feedback", []), item, "accepted")
                persist_workspace_for_current_user()
                st.rerun()
            if cols[2].button("先不看" if zh else "Skip", key=f"{key_prefix}_skip_{rid}_{idx}", width="stretch"):
                st.session_state.recommendation_feedback = record_feedback(st.session_state.get("recommendation_feedback", []), item, "dismissed")
                persist_workspace_for_current_user()
                st.rerun()
        st.caption(learning_summary(st.session_state.get("recommendation_feedback", []), lang))


def safe_text(value) -> str:
    return html.escape("" if value is None else str(value))


def require_invite_access() -> None:
    access_code = os.getenv("SOLODECK_ACCESS_CODE", "").strip()
    if not access_code or st.session_state.get("access_granted"):
        return
    render_html(
        """
        <div class="app-nav">
          <div>
            <div class="brand"><span class="brand-mark"></span>SoloDeck</div>
            <div class="nav-sub">Creator business intelligence agent</div>
          </div>
        </div>
        """
    )
    st.caption("输入体验码后进入工作台。")
    with st.container(border=True):
        code = st.text_input("体验码", type="password", placeholder="请输入邀请体验码")
        if st.button("进入 SoloDeck", width="stretch"):
            if code.strip() == access_code:
                st.session_state.access_granted = True
                st.rerun()
            else:
                st.error("体验码不正确，请检查后重试。")
    st.stop()


def action_brief(contents: pd.DataFrame, revenues: pd.DataFrame, campaigns: pd.DataFrame, lang: str = "中文") -> list[str]:
    zh = lang == "中文"
    recs = revenue_recommendations(contents, revenues, language=lang)
    risks = campaign_risk_alerts(campaigns, language=lang)
    plan = weekly_topic_plan(contents, revenues, n=1, language=lang)
    items = []
    if risks:
        items.append(f"先处理商务风险：{risks[0]['title']}，{risks[0]['detail']}" if zh else f"Handle business risk first: {risks[0]['title']}. {risks[0]['detail']}")
    if recs:
        items.append(f"本周主推动作：{recs[0]['title']}。{recs[0]['action']}" if zh else f"This week's main action: {recs[0]['title']}. {recs[0]['action']}")
    if plan:
        platform = platform_name(plan[0]["suggested_platform"]) if zh else plan[0]["suggested_platform"]
        items.append(f"下一条内容建议：{platform}｜{plan[0]['sample_title']}" if zh else f"Next content suggestion: {platform} | {plan[0]['sample_title']}")
    return items


def stable_id(prefix: str, *parts) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]}"


def init_todo_store() -> None:
    if "todo_items" not in st.session_state:
        st.session_state.todo_items = []
    if "done_todos" not in st.session_state:
        st.session_state.done_todos = {}
    if "import_nonce" not in st.session_state:
        st.session_state.import_nonce = 0
    if "last_import_message" not in st.session_state:
        st.session_state.last_import_message = ""
    if "latest_ai_advice" not in st.session_state:
        st.session_state.latest_ai_advice = ""
    if "workspace_loaded_user_id" not in st.session_state:
        st.session_state.workspace_loaded_user_id = None
    if "recommendation_feedback" not in st.session_state:
        st.session_state.recommendation_feedback = []


def add_todos(items: list[dict]) -> None:
    existing = {item.get("id") for item in st.session_state.todo_items}
    for item in items:
        if item.get("id") not in existing:
            st.session_state.todo_items.append(item)
            existing.add(item.get("id"))


def load_workspace_for_current_user(force: bool = False) -> None:
    user_id = st.session_state.get("user_id")
    if not user_id:
        return
    if not force and st.session_state.get("workspace_loaded_user_id") == user_id:
        return
    workspace = load_user_workspace(user_id)
    st.session_state.imported_records = workspace.get("imported_records") or {"contents": [], "revenues": [], "campaigns": []}
    for key in ["contents", "revenues", "campaigns"]:
        st.session_state.imported_records.setdefault(key, [])
    st.session_state.todo_items = workspace.get("todo_items") or []
    st.session_state.done_todos = workspace.get("done_todos") or {}
    st.session_state.recommendation_feedback = workspace.get("recommendation_feedback") or []
    st.session_state.workspace_loaded_user_id = user_id


def persist_workspace_for_current_user() -> None:
    user_id = st.session_state.get("user_id")
    if not user_id:
        return
    save_user_workspace(
        user_id,
        imported_records=st.session_state.get("imported_records", {"contents": [], "revenues": [], "campaigns": []}),
        todo_items=st.session_state.get("todo_items", []),
        done_todos=st.session_state.get("done_todos", {}),
        recommendation_feedback=st.session_state.get("recommendation_feedback", []),
    )


def todos_from_records(target_table: str, records: list[dict], lang: str) -> list[dict]:
    zh = lang == "中文"
    todos: list[dict] = []
    today = date.today().isoformat()
    for index, record in enumerate(records):
        if target_table == "campaigns":
            brand = record.get("brand_name") or record.get("client_name") or ("新品牌" if zh else "new brand")
            campaign = record.get("campaign_name") or record.get("deliverables") or ("商务合作" if zh else "campaign")
            payment = record.get("payment_status", "")
            invoice = record.get("invoice_status", "")
            report = record.get("report_status", "")
            deadline = record.get("deadline", "")
            if payment in {"overdue", "unpaid", "deposit_received", ""}:
                todos.append({
                    "id": stable_id("todo_campaign_payment", brand, campaign, deadline, index),
                    "priority": "high" if payment == "overdue" else "medium",
                    "source": "商务合作" if zh else "Campaign",
                    "title": f"跟进 {brand} 收款" if zh else f"Follow up payment from {brand}",
                    "detail": f"{campaign} 当前收款状态：{payment or '未确认'}，建议补充付款节点和提醒。" if zh else f"{campaign} payment status is {payment or 'unknown'}; add payment milestone and reminder.",
                })
            if invoice == "pending":
                todos.append({
                    "id": stable_id("todo_campaign_invoice", brand, campaign, deadline, index),
                    "priority": "medium",
                    "source": "票据" if zh else "Invoice",
                    "title": f"处理 {brand} 发票" if zh else f"Handle invoice for {brand}",
                    "detail": f"{campaign} 发票待处理，避免影响回款或复盘交付。" if zh else f"{campaign} invoice is pending; avoid payment/reporting delay.",
                })
            if report == "not_started":
                todos.append({
                    "id": stable_id("todo_campaign_report", brand, campaign, deadline, index),
                    "priority": "low",
                    "source": "复盘" if zh else "Report",
                    "title": f"生成 {brand} 合作复盘" if zh else f"Generate brand report for {brand}",
                    "detail": f"{campaign} 需要沉淀数据表现、互动亮点和下次合作建议。" if zh else f"{campaign} needs metrics, highlights and next-collab suggestions.",
                })
        elif target_table == "revenues":
            client = record.get("client_name") or record.get("platform") or ("客户" if zh else "client")
            amount = record.get("amount", 0)
            status = record.get("status", "")
            if status == "pending":
                todos.append({
                    "id": stable_id("todo_revenue_pending", client, amount, record.get("date"), index),
                    "priority": "high",
                    "source": "收入" if zh else "Revenue",
                    "title": f"确认 {client} 待收款" if zh else f"Confirm pending payment from {client}",
                    "detail": f"识别到 ¥{amount} 待收，建议今天核对到账、发票和合同备注。" if zh else f"Detected ¥{amount} pending; check payment, invoice and contract notes today.",
                })
        elif target_table == "contents":
            title = record.get("title") or ("新增内容" if zh else "new content")
            platform = platform_name(record.get("platform", "")) if zh else record.get("platform", "")
            todos.append({
                "id": stable_id("todo_content_review", title, platform, today, index),
                "priority": "medium",
                "source": "内容" if zh else "Content",
                "title": f"复盘「{title}」" if zh else f"Review “{title}”",
                "detail": f"已加入 {platform or '平台'} 内容数据，建议查看商业评分并决定是否复制选题。" if zh else f"Added {platform or 'platform'} content data; review commercial score and decide whether to replicate it.",
            })
    return todos


def todos_from_extracted_tasks(tasks: list[dict], lang: str) -> list[dict]:
    zh = lang == "中文"
    todos = []
    for index, task in enumerate(tasks or []):
        title = task.get("title") or task.get("task") or task.get("summary") or ("新待办" if zh else "New task")
        detail = task.get("detail") or task.get("description") or ""
        priority = task.get("priority") if task.get("priority") in {"high", "medium", "low"} else "medium"
        due_at = task.get("due_at") or task.get("deadline") or ""
        source = task.get("source") or ("输入资料" if zh else "Input")
        todos.append({
            "id": stable_id("todo_extracted", title, detail, due_at, index),
            "priority": priority,
            "source": source,
            "title": title,
            "detail": detail,
            "due_at": due_at,
        })
    return todos


def _infer_due_at_from_text(text: str) -> str:
    lower = text.lower()
    base = datetime.now()
    if "tomorrow" in lower or "明天" in text:
        base = base + timedelta(days=1)
    hour = None
    minute = 0
    match_24h = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", lower)
    if match_24h:
        hour = int(match_24h.group(1))
        minute = int(match_24h.group(2) or 0)
        suffix = match_24h.group(3)
        if suffix == "pm" and hour < 12:
            hour += 12
        if suffix == "am" and hour == 12:
            hour = 0
    elif "nine" in lower or "九点" in text or "9点" in text:
        hour = 9
    elif "morning" in lower or "早上" in text or "上午" in text:
        hour = 9
    if hour is None:
        hour = 9 if ("morning" in lower or "早上" in text or "上午" in text) else 18
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0).isoformat()


def todos_from_free_text(text: str, lang: str) -> list[dict]:
    if not text or not text.strip():
        return []
    lower = text.lower()
    task_keywords = ["tomorrow", "report", "meeting", "finish", "not completed", "汇报", "会议", "明天", "完成", "还没有完成", "财报"]
    if not any(keyword in lower or keyword in text for keyword in task_keywords):
        return []
    zh = lang == "中文"
    person_match = re.search(r"(?:with|to|和|给)\s*([A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,8})", text)
    person = person_match.group(1) if person_match else ("对方" if zh else "the stakeholder")
    if "financial" in lower or "财报" in text:
        subject = "财报"
        subject_en = "financial report"
    else:
        subject = "事项"
        subject_en = "work item"
    due_at = _infer_due_at_from_text(text)
    return [{
        "id": stable_id("todo_free_text", text, due_at),
        "priority": "high",
        "source": "文字输入" if zh else "Text Input",
        "title": f"准备并汇报{subject}给 {person}" if zh else f"Prepare and report the {subject_en} to {person}",
        "detail": f"从输入识别到：{text.strip()}。这更像待办事项，已直接加入任务中心。" if zh else f"Detected from input: {text.strip()}. This looks like a task, so it was added to Task Center.",
        "due_at": due_at,
    }]


def clean_import_message(record_count: int, todo_count: int, notes: str, validation: dict | None, lang: str) -> str:
    zh = lang == "中文"
    parts = []
    if record_count:
        parts.append(f"已加入 {record_count} 条结构化记录" if zh else f"Added {record_count} structured records")
    else:
        parts.append("未识别到可入库的结构化记录" if zh else "No valid structured records were added")
    if todo_count:
        parts.append(f"生成 {todo_count} 个待办" if zh else f"created {todo_count} tasks")
    if notes:
        normalized = re.sub(r"[；;。.\s]+$", "", str(notes).strip())
        if normalized:
            parts.append(normalized)
    if validation and validation.get("error_count"):
        reason = "部分字段缺失或类型不符合当前表结构，已保留为待办/说明，没有写入数据表" if zh else "Some fields did not match the selected table schema, so they were kept as tasks/notes instead of database rows"
        parts.append(reason)
    end = "。" if zh else "."
    return "；".join(parts) + end


def system_todos(contents: pd.DataFrame, revenues: pd.DataFrame, campaigns: pd.DataFrame, lang: str) -> list[dict]:
    zh = lang == "中文"
    todos: list[dict] = []
    for idx, alert in enumerate(campaign_risk_alerts(campaigns, language=lang)[:4]):
        level = alert.get("level", "medium")
        todos.append({
            "id": stable_id("todo_alert", alert.get("campaign_id"), alert.get("title"), idx),
            "priority": "high" if level == "high" else "medium" if level == "medium" else "low",
            "source": "风险" if zh else "Risk",
            "title": alert.get("title", ""),
            "detail": alert.get("detail", ""),
        })
    for idx, rec in enumerate(revenue_recommendations(contents, revenues, language=lang)[:2]):
        todos.append({
            "id": stable_id("todo_rec", rec.get("title"), idx),
            "priority": rec.get("priority", "medium"),
            "source": "建议" if zh else "Recommendation",
            "title": rec.get("title", ""),
            "detail": rec.get("action", ""),
        })
    return todos


def long_term_todos(contents: pd.DataFrame, revenues: pd.DataFrame, lang: str) -> list[dict]:
    zh = lang == "中文"
    todos: list[dict] = []
    for idx, rec in enumerate(revenue_recommendations(contents, revenues, language=lang)[2:6]):
        todos.append({
            "id": stable_id("todo_long_rec", rec.get("title"), idx, lang),
            "priority": "medium",
            "source": "长期策略" if zh else "Long-term",
            "title": rec.get("title", ""),
            "detail": rec.get("action", ""),
        })
    for idx, plan in enumerate(weekly_topic_plan(contents, revenues, n=4, language=lang)):
        platform = platform_name(plan["suggested_platform"]) if zh else plan["suggested_platform"]
        objective = OBJECTIVE_LABELS_ZH.get(plan["objective"], plan["objective"]) if zh else plan["objective"]
        todos.append({
            "id": stable_id("todo_long_plan", plan.get("sample_title"), idx, lang),
            "priority": "low",
            "source": "内容计划" if zh else "Content Plan",
            "title": f"{platform}｜{plan['sample_title']}",
            "detail": f"{objective}：{plan['reason']}",
        })
    return todos


def todos_from_ai_advice(advice: str, lang: str) -> list[dict]:
    zh = lang == "中文"
    lines = []
    for raw in advice.splitlines():
        line = raw.strip().lstrip("-*0123456789. ").strip()
        if len(line) >= 8 and not line.startswith("#"):
            lines.append(line)
    return [{
        "id": stable_id("todo_ai", line, idx, lang),
        "priority": "medium" if idx else "high",
        "source": "智能建议" if zh else "Smart Advice",
        "title": line[:42] + ("..." if len(line) > 42 else ""),
        "detail": "来自智能经营建议，优先执行这一项。" if zh else "From SoloDeck intelligence. Prioritize this action.",
    } for idx, line in enumerate(lines[:3])]


def render_task_section(items: list[dict], lang: str, limit: int = 9, key_prefix: str = "task") -> None:
    zh = lang == "中文"
    if not items:
        render_html(
            '<div class="soft-card"><span class="badge badge-ok">OK</span><span class="soft-muted">暂无待办，导入资料后会自动生成。</span></div>'
            if zh
            else '<div class="soft-card"><span class="badge badge-ok">OK</span><span class="soft-muted">No tasks yet. Imports will create tasks automatically.</span></div>'
        )
        return
    priority_order = {"high": 0, "medium": 1, "low": 2}
    active = [item for item in items if not st.session_state.done_todos.get(item["id"])]
    ordered = sorted(active, key=lambda item: priority_order.get(item.get("priority", "medium"), 1))
    if not ordered:
        render_html(
            '<div class="soft-card"><span class="badge badge-ok">OK</span><span class="soft-muted">当前任务已完成。</span></div>'
            if zh
            else '<div class="soft-card"><span class="badge badge-ok">OK</span><span class="soft-muted">All visible tasks are done.</span></div>'
        )
        return
    cols = st.columns(3)
    for index, item in enumerate(ordered[:limit]):
        priority = item.get("priority", "medium")
        badge_class = "badge-danger" if priority == "high" else "badge-warn" if priority == "medium" else "badge-ok"
        label = {"high": "紧急", "medium": "重要", "low": "可做"}.get(priority, "重要") if zh else priority.title()
        checked = st.session_state.done_todos.get(item["id"], False)
        source = item.get("source", "")
        title = item.get("title", "")
        detail = item.get("detail", "")
        if not zh:
            source, title, detail = englishize_text(source), englishize_text(title), englishize_text(detail)
        with cols[index % 3]:
            render_html(
                f"""
                <div class="soft-card">
                  <span class="badge {badge_class}">{safe_text(label)}</span><span class="badge">{safe_text(source)}</span>
                  <div class="soft-card-title">{safe_text(title)}</div>
                  <div class="soft-muted">{safe_text(detail)}</div>
                </div>
                """
            )
            widget_key = f"done_{key_prefix}_{item['id']}"
            st.checkbox("完成" if zh else "Done", value=checked, key=widget_key)
            new_value = st.session_state.get(widget_key, checked)
            if new_value != checked:
                st.session_state.done_todos[item["id"]] = new_value
                persist_workspace_for_current_user()
                st.rerun()
            st.session_state.done_todos[item["id"]] = new_value
    persist_workspace_for_current_user()


def render_done_tasks(items: list[dict], lang: str, limit: int = 12) -> None:
    zh = lang == "中文"
    done_items = [item for item in items if st.session_state.done_todos.get(item["id"])]
    if not done_items:
        st.caption("还没有已完成任务。" if zh else "No completed tasks yet.")
        return
    for item in done_items[:limit]:
        render_html(
            f"""
            <div class="soft-card done-card">
              <span class="badge badge-ok">{'已完成' if zh else 'Done'}</span>
              <div class="soft-card-title">{safe_text(item.get('title', ''))}</div>
            </div>
            """
        )


def render_task_plan(short_items: list[dict], long_items: list[dict], lang: str) -> None:
    zh = lang == "中文"
    short_active = [item for item in short_items if not st.session_state.done_todos.get(item["id"])]
    high_count = sum(1 for item in short_active if item.get("priority") == "high")
    st.markdown("### 3. 下一步行动" if zh else "### 3. Next Actions")
    st.caption(
        f"只看最该做的 3 件事。{high_count} 个紧急。"
        if zh
        else f"Only the top 3 actions. {high_count} urgent."
    )
    priority_order = {"high": 0, "medium": 1, "low": 2}
    primary_items = sorted(short_active, key=lambda item: priority_order.get(item.get("priority", "medium"), 1))[:3]
    render_task_section(primary_items or short_items[:3], lang, limit=3, key_prefix="primary")

    calendar_tasks = [{**item, "done": st.session_state.done_todos.get(item["id"], False)} for item in short_items]
    calendar_cols = st.columns([0.34, 0.66])
    ics = tasks_to_ics(calendar_tasks, calendar_name="SoloDeck 短期任务" if zh else "SoloDeck Short-Term Tasks")
    calendar_cols[0].download_button(
        "加入日历提醒" if zh else "Add to Calendar",
        data=ics,
        file_name="solodeck_tasks.ics",
        mime="text/calendar",
        width="stretch",
        help="下载 .ics 文件后，可用 Apple Calendar、Outlook、Windows 日历或 Google Calendar 导入。" if zh else "Download the .ics file and import it into Apple Calendar, Outlook, Windows Calendar or Google Calendar.",
    )
    with calendar_cols[1]:
        st.caption(
            "下载日历文件后，可加入系统日历并设置提醒。"
            if zh
            else "Download the calendar file to add these actions to your system calendar."
        )

    with st.expander("展开查看完整任务和长期计划" if zh else "Show Full Tasks and Long-Term Plan", expanded=False):
        st.markdown("#### 短期任务" if zh else "#### Short-Term Tasks")
        render_task_section(short_items, lang, limit=9, key_prefix="short_full")
        st.markdown("#### 已完成" if zh else "#### Completed")
        render_done_tasks(short_items + long_items, lang)
        st.markdown("#### 长期任务" if zh else "#### Long-Term Tasks")
        long_calendar_tasks = [{**item, "done": st.session_state.done_todos.get(item["id"], False)} for item in long_items]
        ics_long = tasks_to_ics(long_calendar_tasks, calendar_name="SoloDeck 长期任务" if zh else "SoloDeck Long-Term Tasks")
        st.download_button(
            "导入长期任务到日历" if zh else "Import Long-Term Tasks",
            data=ics_long,
            file_name="solodeck_long_tasks.ics",
            mime="text/calendar",
            width="stretch",
        )
        render_task_section(long_items, lang, limit=9, key_prefix="long_full")


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


def platform_name(platform: str) -> str:
    return PLATFORM_LABELS.get(platform, platform)


STYLE_LABELS_ZH = {
    "pain_point": "痛点型",
    "tutorial": "教程型",
    "number": "数字清单型",
    "story": "故事型",
    "contrast": "对比型",
    "result_oriented": "结果导向型",
    "question": "提问型",
}

EXPERIMENT_VALUE_LABELS_ZH = {
    "pain_point": "痛点标题",
    "tutorial": "教程标题",
    "early_cta": "前置咨询入口",
    "ending_cta": "结尾咨询入口",
    "case_study": "案例复盘",
    "listicle": "清单内容",
    "new_strategy": "新做法",
    "current_strategy": "当前做法",
    "treatment": "实验组",
    "control": "对照组",
    "emotion_companion": "情绪陪伴功能",
    "touch_interaction": "触摸互动功能",
    "alarm": "提醒闹钟功能",
    "voice_interaction": "语音互动",
    "music_playback": "音乐播放",
    "desktop_decoration": "桌面陪伴摆件",
    "eye_protection": "护眼",
    "dimming": "调光",
    "anti_drop": "防摔",
    "magnetic": "磁吸",
    "weekly_plan": "周计划",
    "habit_tracking": "习惯追踪",
}

EXPERIMENT_VALUE_LABELS_EN = {
    "pain_point": "pain-point title",
    "tutorial": "tutorial title",
    "early_cta": "early CTA",
    "ending_cta": "ending CTA",
    "case_study": "case study",
    "listicle": "listicle",
    "new_strategy": "new approach",
    "current_strategy": "current approach",
    "emotion_companion": "emotion companion",
    "touch_interaction": "touch interaction",
    "alarm": "alarm",
}

PRODUCT_CATEGORY_ZH = {
    "hardware_device": "商用/陪伴机器人",
    "home_product": "家居电商",
    "accessory": "配件电商",
}

TYPE_LABELS_ZH = {
    "brand_ads": "品牌广告",
    "platform_share": "平台分成",
    "course": "课程",
    "consulting": "咨询",
    "membership": "会员",
    "affiliate": "分销",
    "reward": "打赏",
    "service": "服务",
}

METRIC_LABELS_ZH = {
    "view_rate": "曝光效率",
    "like_rate": "点赞率",
    "favorite_rate": "收藏率",
    "follow_rate": "转粉率",
    "conversion_rate": "转化率",
    "revenue": "收入",
    "converted": "成交",
    "retained_7d": "7日留存",
    "activated": "激活",
    "rating": "评分",
}

OBJECTIVE_LABELS_ZH = {
    "growth": "拉新增长",
    "engagement": "互动讨论",
    "conversion": "咨询转化",
    "monetization": "商业变现",
}

TOPIC_LABELS_EN = {
    "AI工具": "AI Tools",
    "个人IP": "Personal Brand",
    "商业案例": "Business Cases",
    "知识付费": "Knowledge Products",
    "副业增长": "Side Business Growth",
    "职场效率": "Workplace Productivity",
    "内容方法论": "Content Methodology",
    "产品化服务": "Productized Services",
    "核心主题": "Core Topic",
}

BRAND_LABELS_EN = {
    "青石AI": "Qingshi AI",
    "北岸效率": "Northshore Productivity",
    "映川教育": "Yingchuan Education",
    "云谷SaaS": "Cloud Valley SaaS",
    "晴山咖啡": "Sunnyhill Coffee",
    "青石品牌": "Qingshi Brand",
    "北岸品牌": "Northshore Brand",
    "映川品牌": "Yingchuan Brand",
    "云谷品牌": "Cloud Valley Brand",
    "晴山品牌": "Sunnyhill Brand",
}


def englishize_text(value) -> str:
    text = "" if value is None else str(value)
    for raw, label in TOPIC_LABELS_EN.items():
        text = text.replace(raw, label)
    for raw, label in BRAND_LABELS_EN.items():
        text = text.replace(raw, label)
    replacements = {
        "你做": "Your ",
        "没结果，通常卡在这 3 个问题": " content is stuck on these 3 issues",
        "实操指南：从 0 到 1 完整流程": " practical guide: complete workflow from 0 to 1",
        "提升": "Improve ",
        "转化的 7 个动作": " conversion with 7 actions",
        "我用一个真实案例跑通了": "A real case that validated ",
        "高手和新手的区别在哪里": ": what separates advanced and beginner creators",
        "用": "Using ",
        "拿到第一笔稳定收入": " to earn the first stable revenue",
        "还值得做吗？这是我的数据复盘": ": is it still worth doing? A data review",
        "合作推广": " sponsorship campaign",
        "1篇图文 + 1条短视频": "1 post + 1 short video",
        "1篇深度内容": "1 in-depth post",
        "近 10 条同平台平均播放": "Average views of recent 10 same-platform posts:",
        "，CPM 系数": ", CPM coefficient:",
        "，制作成本": ", production cost:",
        "，修订风险 +": ", revision risk +",
        "无额外授权/排他/加急加价。": "No extra usage rights, exclusivity or rush fee.",
        "包含素材使用权 +20%": "Usage rights +20%",
        "竞品排他 +30%": "Exclusivity +30%",
        "加急交付 +15%": "Rush delivery +15%",
        "品牌合作定金 / 内容推广服务费": "Sponsorship deposit / content promotion service",
        "高收入内容，可复用选题结构并延展产品化服务。": "High-revenue content. Reuse the topic structure and extend it into productized services.",
        "收藏强但变现弱，适合转资料包、课程或清单模板。": "Strong saves but weak monetization. Turn it into a toolkit, course or checklist template.",
        "咨询强但成交弱，需要优化承接页和私域话术。": "Strong consultations but weak conversions. Improve the landing page and sales follow-up.",
        "制作投入偏高，建议模板化或降低制作成本。": "Production investment is high. Use templates or lower production cost.",
        "表现稳定，可作为常规内容池。": "Stable performance. Keep it in the regular content pool.",
    }
    for raw, label in replacements.items():
        text = text.replace(raw, label)
    return text


def localize_frame(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if lang != "中文":
        for col in ["topic", "suggested_topic", "title", "campaign_name", "brand_name", "deliverables", "insight", "sample_title", "reason", "note"]:
            if col in out.columns:
                out[col] = out[col].map(englishize_text).fillna(out[col])
        return out
    for col in ["platform", "suggested_platform"]:
        if col in out.columns:
            out[col] = out[col].map(platform_name).fillna(out[col])
    for col in ["title_style", "suggested_title_style"]:
        if col in out.columns:
            out[col] = out[col].map(STYLE_LABELS_ZH).fillna(out[col])
    if "revenue_type" in out.columns:
        out["revenue_type"] = out["revenue_type"].map(TYPE_LABELS_ZH).fillna(out["revenue_type"])
    if "outcome_metric" in out.columns:
        out["outcome_metric"] = out["outcome_metric"].map(METRIC_LABELS_ZH).fillna(out["outcome_metric"])
    if "objective" in out.columns:
        out["objective"] = out["objective"].map(OBJECTIVE_LABELS_ZH).fillna(out["objective"])
    return out


COLUMN_LABELS_ZH = {
    "content_id": "内容编号",
    "title": "标题",
    "platform": "平台",
    "topic": "主题",
    "title_style": "标题风格",
    "revenue_type": "收入类型",
    "amount": "金额",
    "share": "占比",
    "views": "播放/阅读",
    "revenue": "收入",
    "commercial_score": "商业评分",
    "consultations": "咨询",
    "conversions": "成交",
    "insight": "洞察",
    "content_count": "内容数",
    "total_revenue": "总收入",
    "avg_commercial_value": "平均商业价值",
    "avg_production_hours": "平均制作时长",
    "total_production_hours": "总制作时长",
    "revenue_per_hour": "单位时间收益",
    "favorite_rate": "收藏率",
    "follow_rate": "转粉率",
    "conversion_rate": "转化率",
    "comment_rate": "评论率",
    "like_rate": "点赞率",
    "outcome_metric": "结果指标",
    "relative_lift": "相对提升",
    "treatment_mean": "实验组平均",
    "control_mean": "对照组平均",
    "ci_low": "区间下限",
    "ci_high": "区间上限",
    "experiment_id": "实验编号",
    "issue": "问题类型",
    "evidence_count": "证据数量",
    "affected_segment": "影响人群",
    "business_impact": "业务影响",
    "suggested_action": "建议动作",
    "priority": "优先级",
    "expected_metric_to_watch": "观察指标",
    "absolute_lift": "绝对提升",
    "conclusion": "结论",
    "weekday": "星期",
    "hour": "小时",
    "sample_size": "样本量",
    "confidence": "置信度",
    "suggested_topic": "建议主题",
    "suggested_platform": "建议平台",
    "suggested_title_style": "建议标题风格",
    "objective": "目标",
    "reason": "原因",
    "sample_title": "示例标题",
    "campaign_id": "合作编号",
    "brand_name": "品牌",
    "campaign_name": "合作名称",
    "deliverables": "交付物",
    "price": "报价",
    "deadline": "截止日期",
    "status": "状态",
    "payment_status": "收款状态",
    "invoice_status": "发票状态",
    "revision_count": "修订次数",
    "report_status": "复盘状态",
    "related_content_id": "关联内容",
}


def display_frame(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    out = localize_frame(df, lang)
    if lang == "中文":
        out = out.rename(columns={k: v for k, v in COLUMN_LABELS_ZH.items() if k in out.columns})
    return out


def format_lift_table(rows: list[dict], lang: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if lang != "中文":
        return df
    question_map = {
        "advanced_methods_available": "可继续运行更完整的专业估计",
        "title_style -> views": "标题风格对播放量",
        "same_content_cross_platform -> revenue": "同内容跨平台收入差异",
    }
    method_map = {
        "专业增量模型": "专业模型可按需运行",
        "fast_fixed_effect": "快速增量估计",
        "paired_difference": "同组配对比较",
        "quick_user_test_lift": "内测分组比较",
        "quick_product_version_diff": "新旧版本比较",
        "no_data": "暂无可比数据",
    }
    out = pd.DataFrame()
    out["问题"] = df.get("question", pd.Series(dtype=str)).map(lambda x: question_map.get(str(x), str(x).replace(" -> revenue/converted", " 对收入/成交")))
    out["方法"] = df.get("method", pd.Series(dtype=str)).map(lambda x: method_map.get(str(x), str(x)))
    effect = df.get("effect_estimate", df.get("mean_difference", df.get("effect", pd.Series([None] * len(df)))))
    out["估计增量"] = pd.to_numeric(effect, errors="coerce").round(3)
    low = pd.to_numeric(df.get("ci_low", pd.Series([None] * len(df))), errors="coerce")
    high = pd.to_numeric(df.get("ci_high", pd.Series([None] * len(df))), errors="coerce")
    out["参考区间"] = [
        "" if pd.isna(l) or pd.isna(h) or (l == 0 and h == 0) else f"{l:.3f} ~ {h:.3f}"
        for l, h in zip(low, high)
    ]
    sample = df.get("sample_size", df.get("pair_count", pd.Series([None] * len(df))))
    out["样本量"] = sample
    out["说明"] = df.get("warnings", df.get("warning", df.get("interpretation", pd.Series([""] * len(df))))).map(
        lambda x: "；".join(x) if isinstance(x, list) else "" if pd.isna(x) else str(x)
    )
    return out


def format_experiment_table(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if lang != "中文":
        return df
    out = pd.DataFrame()
    out["实验"] = df.get("experiment_id", "")
    out["观察指标"] = df.get("outcome_metric", "").map(lambda x: METRIC_LABELS_ZH.get(str(x), str(x)))
    out["实验组平均"] = pd.to_numeric(df.get("treatment_mean", 0), errors="coerce").round(4)
    out["对照组平均"] = pd.to_numeric(df.get("control_mean", 0), errors="coerce").round(4)
    out["净提升"] = pd.to_numeric(df.get("absolute_lift", 0), errors="coerce").round(4)
    out["相对提升"] = pd.to_numeric(df.get("relative_lift", 0), errors="coerce").map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    out["参考区间"] = [
        f"{float(l):.4f} ~ {float(h):.4f}" if pd.notna(l) and pd.notna(h) else ""
        for l, h in zip(df.get("ci_low", pd.Series([None] * len(df))), df.get("ci_high", pd.Series([None] * len(df))))
    ]
    out["结论"] = df.get("conclusion", "")
    return out


def format_feedback_table(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if lang != "中文":
        return df
    issue_map = {
        "pricing": "价格疑虑",
        "performance": "性能体验",
        "usability": "使用门槛",
        "feature_request": "功能需求",
        "design": "外观设计",
        "trust": "信任与隐私",
        "quality": "质量稳定性",
        "emotional_value": "情绪价值",
    }
    priority_map = {"high": "高", "medium": "中", "low": "低"}
    out = pd.DataFrame()
    out["问题类型"] = df.get("issue", "").map(lambda x: issue_map.get(str(x), str(x)))
    out["证据数量"] = df.get("evidence_count", "")
    out["影响人群"] = df.get("affected_segment", "")
    out["业务影响"] = df.get("business_impact", "")
    out["建议动作"] = df.get("suggested_action", "")
    out["优先级"] = df.get("priority", "").map(lambda x: priority_map.get(str(x), str(x)))
    out["观察指标"] = df.get("expected_metric_to_watch", "").map(lambda x: str(x).replace("conversion_rate", "转化率").replace("retained_7d", "7日留存").replace("avg_rating", "平均评分"))
    return out


def money(value) -> str:
    try:
        return f"¥{float(value):,.0f}"
    except Exception:
        return "¥0"


def metric_label(metric: str, lang: str) -> str:
    return METRIC_LABELS_ZH.get(metric, metric) if lang == "中文" else metric.replace("_", " ")


def strategy_label(value, lang: str) -> str:
    text = "" if value is None else str(value)
    if lang == "中文":
        return EXPERIMENT_VALUE_LABELS_ZH.get(text, STYLE_LABELS_ZH.get(text, platform_name(text)))
    return EXPERIMENT_VALUE_LABELS_EN.get(text, text.replace("_", " "))


def feature_label(value, lang: str) -> str:
    parts = [x.strip() for x in str(value or "").split(",") if x.strip()]
    labels = [strategy_label(part, lang) for part in parts]
    return "、".join(labels) if lang == "中文" else ", ".join(labels)


def platform_cta_locations(platform: str, lang: str) -> str:
    zh = lang == "中文"
    locations_zh = {
        "wechat": "公众号菜单、文章结尾、视频号主页简介、私信自动回复",
        "xiaohongshu": "账号简介、置顶笔记、每篇笔记结尾、评论区置顶回复",
        "bilibili": "视频简介、置顶评论、合集简介、动态置顶",
        "douyin": "主页简介、橱窗/团购入口、视频结尾、置顶评论",
        "zhihu": "回答结尾、专栏简介、个人主页、私信欢迎语",
    }
    locations_en = {
        "wechat": "account menu, article ending, channel bio and auto-reply",
        "xiaohongshu": "bio, pinned post, post ending and pinned comment",
        "bilibili": "video description, pinned comment, playlist intro and pinned dynamic",
        "douyin": "profile bio, shop/deal entry, video ending and pinned comment",
        "zhihu": "answer ending, column intro, profile and welcome DM",
    }
    return (locations_zh if zh else locations_en).get(platform, "主页简介、内容结尾、置顶评论、私信自动回复" if zh else "profile bio, content ending, pinned comment and auto-reply")


def experiment_pace(contents: pd.DataFrame, platform: str, lang: str) -> dict:
    df = contents[contents.get("platform", pd.Series(dtype=str)).eq(platform)].copy() if not contents.empty and "platform" in contents.columns else pd.DataFrame()
    follower_base = float(pd.to_numeric(df.get("followers_before", pd.Series(dtype=float)), errors="coerce").max() or 0)
    if follower_base >= 500_000:
        total, days, per_group = 6, 14, 3
        cadence_zh = "大号节奏：两周完成，不要一天发完；每 2-3 天发 1 条。"
        cadence_en = "Large-account pace: finish over 14 days; do not post all at once. Publish 1 every 2-3 days."
    elif follower_base >= 100_000:
        total, days, per_group = 8, 10, 4
        cadence_zh = "中号节奏：10 天完成；每 1-2 天发 1 条。"
        cadence_en = "Mid-account pace: finish over 10 days; publish 1 every 1-2 days."
    else:
        total, days, per_group = 12, 7, 6
        cadence_zh = "轻量节奏：7 天完成；每天 1-2 条，不要连续刷屏。"
        cadence_en = "Light pace: finish over 7 days; publish 1-2 per day without flooding."
    return {
        "total": total,
        "days": days,
        "per_group": per_group,
        "cadence": cadence_zh if lang == "中文" else cadence_en,
        "follower_base": follower_base,
    }


def experiment_content_brief(topic: str, lang: str) -> str:
    if lang == "中文":
        return f"内容用同一主题“{topic}”，只换标题/入口；方向选：真实案例、踩坑清单、步骤教程。"
    return f"Use the same topic “{topic}” and change only the title/CTA; use case, mistakes and step-by-step formats."


def ab_test_action_cards(ab_tests: pd.DataFrame, lang: str, contents_df: pd.DataFrame | None = None) -> list[dict]:
    zh = lang == "中文"
    result = analyze_ab_test(ab_tests, language=lang)
    if result.empty or ab_tests.empty:
        return []
    cards = []
    for _, row in result.sort_values("relative_lift", ascending=False).head(4).iterrows():
        exp_id = row["experiment_id"]
        metric = row["outcome_metric"]
        raw = ab_tests[(ab_tests["experiment_id"].eq(exp_id)) & (ab_tests["outcome_metric"].eq(metric))]
        sample = raw.iloc[0] if not raw.empty else {}
        platform = platform_name(sample.get("platform", "")) if zh else sample.get("platform", "")
        raw_platform = sample.get("platform", "")
        topic = sample.get("topic", "")
        treatment = strategy_label(sample.get("treatment_value", "new"), lang)
        control = strategy_label(sample.get("control_value", "current"), lang)
        lift = float(row.get("relative_lift", 0))
        absolute = float(row.get("absolute_lift", 0))
        priority = "high" if lift > 0.12 else "medium" if lift > 0 else "low"
        pace = experiment_pace(contents_df if contents_df is not None else pd.DataFrame(), raw_platform, lang)
        content_brief = experiment_content_brief(topic, lang)
        if zh:
            title = f"{platform}｜{treatment} 对比 {control}"
            reason = f"{metric_label(metric, lang)}提升 {lift:.1%}，净增 {absolute:.3f}。"
            action = f"{pace['cadence']} 共 {pace['total']} 条：{pace['per_group']} 条用{treatment}，{pace['per_group']} 条用{control}。{content_brief} 每条发布 72 小时后记录{metric_label(metric, lang)}，一组平均高出 10% 以上才算胜出。"
            if lift <= 0:
                action = f"不要增加{treatment}。{pace['cadence']} 做 {pace['per_group'] * 2} 条新对比，改测标题/封面；每条发布 72 小时后看{metric_label(metric, lang)}。"
        else:
            title = f"{platform} | {treatment} vs {control}"
            reason = f"{metric_label(metric, lang)} lift {lift:.1%}; absolute lift {absolute:.3f}."
            action = f"{pace['cadence']} Run {pace['total']} posts: {pace['per_group']} with {treatment}, {pace['per_group']} with {control}. {content_brief} Track {metric_label(metric, lang)} 72h after each post; winner must average 10% higher."
            if lift <= 0:
                action = f"Do not scale {treatment}. {pace['cadence']} Run {pace['per_group'] * 2} new posts testing title/cover; compare {metric_label(metric, lang)} after 72h."
        cards.append({"title": title, "reason": reason, "action": action, "priority": priority})
    return cards


def platform_opportunity_cards(contents: pd.DataFrame, revenues: pd.DataFrame, lang: str) -> list[dict]:
    zh = lang == "中文"
    result = platform_strategy_analysis(contents, revenues)
    table = result.get("table", pd.DataFrame())
    if table.empty:
        return []
    cards = []
    top_revenue = table.sort_values("revenue", ascending=False).head(1).iloc[0]
    top_eff = table.sort_values("revenue_per_hour", ascending=False).head(1).iloc[0]
    top_conv = table.sort_values("conversions", ascending=False).head(1).iloc[0]
    rows = [
        (top_revenue, "平台收益最高", "Highest platform revenue", "把商单报价、产品页、咨询入口放到：{locations}。", "Place paid offers, product page and consultation CTA in: {locations}."),
        (top_eff, "单位时间收益最高", "Best revenue per hour", "选近 5 条高成交内容，复用标题结构、开头钩子和结尾入口，先做 3 条。", "Pick 5 high-conversion posts, reuse title structure, opening hook and ending CTA, then publish 3 more."),
        (top_conv, "成交最强平台", "Best conversion platform", "把购买路径压到 2 步：内容结尾点入口，私信/表单直接预约或付款。", "Reduce the buying path to 2 steps: content CTA, then DM/form to book or pay."),
    ]
    seen = set()
    seen_platforms = set()
    for row, title_zh, title_en, action_zh, action_en in rows:
        platform = row.get("platform", "")
        key = (title_zh, platform)
        if key in seen or platform in seen_platforms:
            continue
        seen.add(key)
        seen_platforms.add(platform)
        display_platform = platform_name(platform) if zh else platform
        locations = platform_cta_locations(platform, lang)
        revenue = float(row.get("revenue", 0))
        rph = float(row.get("revenue_per_hour", 0))
        conversions = int(row.get("conversions", 0))
        action = action_zh.format(locations=locations) if zh else action_en.format(locations=locations)
        cards.append({
            "title": f"{title_zh}：{display_platform}" if zh else f"{title_en}: {display_platform}",
            "reason": (f"收入 {money(revenue)}，单位时间 {money(rph)}/小时，成交 {conversions}。" if zh else f"Revenue {money(revenue)}, {money(rph)}/hour, {conversions} conversions."),
            "action": action,
            "priority": "high" if revenue > 0 or conversions > 0 else "medium",
        })
    return cards


def evidence_cards(contents: pd.DataFrame, revenues: pd.DataFrame, products: pd.DataFrame, ab_tests: pd.DataFrame, lang: str) -> list[dict]:
    zh = lang == "中文"
    cards: list[dict] = []
    if not ab_tests.empty:
        ab_result = analyze_ab_test(ab_tests, language=lang)
        if not ab_result.empty:
            row = ab_result.sort_values("relative_lift", ascending=False).iloc[0]
            raw = ab_tests[(ab_tests["experiment_id"].eq(row["experiment_id"])) & (ab_tests["outcome_metric"].eq(row["outcome_metric"]))]
            sample = raw.iloc[0] if not raw.empty else {}
            platform = platform_name(sample.get("platform", "")) if zh else sample.get("platform", "")
            treatment = strategy_label(sample.get("treatment_value", ""), lang)
            control = strategy_label(sample.get("control_value", ""), lang)
            metric = metric_label(row["outcome_metric"], lang)
            cards.append({
                "title": "依据：已有实验结果" if zh else "Evidence: Existing Test",
                "reason": f"{platform} 上，{treatment} 相比 {control}，{metric}提升 {float(row['relative_lift']):.1%}。" if zh else f"On {platform}, {treatment} beat {control} by {float(row['relative_lift']):.1%} on {metric}.",
                "action": "所以先复测这个变量，再决定是否放大。" if zh else "Retest this variable first, then decide whether to scale.",
                "priority": "high",
            })
    platform_cards = platform_opportunity_cards(contents, revenues, lang)
    if platform_cards:
        top = platform_cards[0]
        cards.append({
            "title": "依据：平台收益差异" if zh else "Evidence: Platform Gap",
            "reason": top["reason"],
            "action": "所以商单和产品入口优先放到表现最强的平台。" if zh else "Place paid offers and product CTAs on the strongest platform first.",
            "priority": "medium",
        })
    if not products.empty:
        df = products.copy()
        df["revenue"] = pd.to_numeric(df.get("revenue", 0), errors="coerce").fillna(0)
        df["conversions"] = pd.to_numeric(df.get("conversions", 0), errors="coerce").fillna(0)
        top = df.sort_values("revenue", ascending=False).head(1).iloc[0]
        category = PRODUCT_CATEGORY_ZH.get(str(top.get("category", "")), str(top.get("category", ""))) if zh else str(top.get("category", "")).replace("_", " ")
        cards.append({
            "title": "依据：产品款式表现" if zh else "Evidence: Product Model",
            "reason": f"{category}「{top.get('product_name', '')}」收入 {money(top.get('revenue', 0))}，成交 {int(top.get('conversions', 0))}。" if zh else f"{top.get('product_name', '')} revenue {money(top.get('revenue', 0))}, conversions {int(top.get('conversions', 0))}.",
            "action": "所以首页会优先推荐主推款式和对应卖点。" if zh else "So SoloDeck prioritizes the top model and its selling point.",
            "priority": "medium",
        })
    return cards[:3]


def product_variant_action_cards(products: pd.DataFrame, beta_tests: pd.DataFrame, lang: str) -> list[dict]:
    zh = lang == "中文"
    cards: list[dict] = []
    if not products.empty:
        df = products.copy()
        for col in ["views", "consultations", "conversions", "revenue", "price", "cost", "avg_rating"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        parent_revenue = df.set_index("product_id")["revenue"].to_dict() if "product_id" in df.columns else {}
        for _, row in df.sort_values("revenue", ascending=False).head(3).iterrows():
            platform = platform_name(row.get("platform", "")) if zh else row.get("platform", "")
            parent_id = str(row.get("parent_product_id", "") or "")
            parent_delta = float(row.get("revenue", 0)) - float(parent_revenue.get(parent_id, 0))
            conv_rate = float(row.get("conversions", 0)) / max(float(row.get("views", 0)), 1)
            feature = feature_label(str(row.get("feature_tags", "")).split(",")[0].strip(), lang) or ("核心卖点" if zh else "main feature")
            category = PRODUCT_CATEGORY_ZH.get(str(row.get("category", "")), str(row.get("category", ""))) if zh else str(row.get("category", "")).replace("_", " ")
            if zh:
                reason = f"收入 {money(row.get('revenue', 0))}，成交 {int(row.get('conversions', 0))}，转化率 {conv_rate:.2%}。"
                if parent_id and parent_delta:
                    reason += f" 相比基础款 {money(parent_delta)}。"
                action = f"{category}优先在{platform}推广，素材突出“{feature}”。"
            else:
                reason = f"Revenue {money(row.get('revenue', 0))}, {int(row.get('conversions', 0))} conversions, conversion rate {conv_rate:.2%}."
                if parent_id and parent_delta:
                    reason += f" {money(parent_delta)} vs parent model."
                action = f"Prioritize {platform}; highlight “{feature}”."
            cards.append({
                "title": f"主推款式：{row.get('product_name', '')}" if zh else f"Top model: {row.get('product_name', '')}",
                "reason": reason,
                "action": action,
                "priority": "high",
            })
    if not beta_tests.empty and {"feature_name", "test_group"}.issubset(beta_tests.columns):
        metric = "revenue" if "revenue" in beta_tests.columns else "converted"
        for feature, group in beta_tests.groupby("feature_name"):
            treatment = pd.to_numeric(group[group["test_group"].eq("treatment")][metric], errors="coerce").fillna(0)
            control = pd.to_numeric(group[group["test_group"].eq("control")][metric], errors="coerce").fillna(0)
            if treatment.empty or control.empty:
                continue
            lift = float(treatment.mean() - control.mean())
            display_feature = strategy_label(feature, lang)
            if zh:
                action = "继续扩大到 20+20 人。" if lift > 0 else "暂不放大，先改卖点或人群。"
                reason = f"{metric_label(metric, lang)}净增 {lift:.2f}。"
            else:
                action = "Expand to 20+20 users." if lift > 0 else "Do not scale yet; adjust positioning or segment."
                reason = f"{metric_label(metric, lang)} lift {lift:.2f}."
            cards.append({
                "title": f"功能实验：{display_feature}" if zh else f"Feature test: {display_feature}",
                "reason": reason,
                "action": action,
                "priority": "high" if lift > 0 else "medium",
            })
    return cards[:5]


def filter_by_platforms(contents: pd.DataFrame, revenues: pd.DataFrame, campaigns: pd.DataFrame, ab_tests: pd.DataFrame, selected: list[str], products: pd.DataFrame | None = None, feedback: pd.DataFrame | None = None, beta_tests: pd.DataFrame | None = None):
    if not selected:
        return contents, revenues, campaigns, ab_tests, products if products is not None else pd.DataFrame(), feedback if feedback is not None else pd.DataFrame(), beta_tests if beta_tests is not None else pd.DataFrame()
    content_ids = set(contents.loc[contents["platform"].isin(selected), "content_id"])
    filtered_contents = contents[contents["platform"].isin(selected)].copy()
    filtered_revenues = revenues[(revenues["platform"].isin(selected)) | (revenues["content_id"].isin(content_ids))].copy()
    filtered_campaigns = campaigns[(campaigns["platform"].isin(selected)) | (campaigns["related_content_id"].isin(content_ids))].copy()
    filtered_ab = ab_tests[(ab_tests["platform"].isin(selected)) | (ab_tests["content_id"].isin(content_ids))].copy()
    filtered_products = products[products["platform"].isin(selected)].copy() if products is not None and not products.empty and "platform" in products.columns else (products if products is not None else pd.DataFrame())
    product_ids = set(filtered_products.get("product_id", pd.Series(dtype=str)).astype(str).tolist()) if filtered_products is not None and not filtered_products.empty else set()
    filtered_feedback = feedback[
        (feedback.get("platform", pd.Series(index=feedback.index, dtype=str)).isin(selected))
        | (feedback.get("related_content_id", pd.Series(index=feedback.index, dtype=str)).isin(content_ids))
        | (feedback.get("related_product_id", pd.Series(index=feedback.index, dtype=str)).isin(product_ids))
    ].copy() if feedback is not None and not feedback.empty else pd.DataFrame()
    filtered_beta = beta_tests[beta_tests.get("product_id", pd.Series(index=beta_tests.index, dtype=str)).isin(product_ids)].copy() if beta_tests is not None and not beta_tests.empty and product_ids else (beta_tests if beta_tests is not None else pd.DataFrame())
    return filtered_contents, filtered_revenues, filtered_campaigns, filtered_ab, filtered_products, filtered_feedback, filtered_beta


def append_records(df: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    if not records:
        return df
    added = pd.DataFrame(records)
    for col in df.columns:
        if col not in added.columns:
            added[col] = "" if df[col].dtype == object else 0
    added = added[df.columns]
    return pd.concat([df, added], ignore_index=True)


def init_import_store() -> None:
    defaults = {"contents": [], "revenues": [], "campaigns": []}
    if "imported_records" not in st.session_state:
        st.session_state.imported_records = defaults
    for key, value in defaults.items():
        st.session_state.imported_records.setdefault(key, value)


def apply_import_store(contents: pd.DataFrame, revenues: pd.DataFrame, campaigns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contents = append_records(contents, st.session_state.imported_records.get("contents", []))
    revenues = append_records(revenues, st.session_state.imported_records.get("revenues", []))
    campaigns = append_records(campaigns, st.session_state.imported_records.get("campaigns", []))
    if "publish_time" in contents.columns:
        contents["publish_time"] = pd.to_datetime(contents["publish_time"], errors="coerce").fillna(pd.Timestamp.now())
    if "date" in revenues.columns:
        revenues["date"] = pd.to_datetime(revenues["date"], errors="coerce").dt.date
    if "deadline" in campaigns.columns:
        campaigns["deadline"] = pd.to_datetime(campaigns["deadline"], errors="coerce").dt.date
    return contents, revenues, campaigns


def concise_summary(contents: pd.DataFrame, revenues: pd.DataFrame, campaigns: pd.DataFrame, lang: str) -> str:
    total_revenue = float(revenues["amount"].sum() + contents["revenue"].sum())
    risks = campaign_risk_alerts(campaigns, language=lang)
    plan = weekly_topic_plan(contents, revenues, n=1, language=lang)
    topic = plan[0]["suggested_topic"] if plan else "核心主题"
    platform = platform_name(plan[0]["suggested_platform"]) if plan and lang == "中文" else (plan[0]["suggested_platform"] if plan else "main platform")
    if lang == "中文":
        risk_text = f"同时有 {len(risks)} 个商务事项需要跟进" if risks else "当前没有明显高风险商务事项"
        return f"本期累计经营收入约 ¥{total_revenue:,.0f}，下周建议优先在{platform}围绕“{topic}”继续发内容，{risk_text}。"
    risk_text = f"{len(risks)} business items need attention" if risks else "no major business risks are detected"
    return f"Total tracked revenue is about ¥{total_revenue:,.0f}. Next week, focus on {topic} on {platform}; {risk_text}."


def account_panel(lang: str) -> None:
    zh = lang == "中文"
    init_auth_db()
    if "signed_in" not in st.session_state:
        st.session_state.signed_in = False
    if "account_name" not in st.session_state:
        st.session_state.account_name = ""
    if "account_email" not in st.session_state:
        st.session_state.account_email = ""
    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    st.subheader("账户" if zh else "Account")
    st.caption(
        f"已注册用户：{user_count()}。登录后自动保存你的资料与任务。"
        if zh
        else f"Registered users: {user_count()}. Your uploads and tasks are saved after login."
    )
    if st.session_state.signed_in:
        st.caption(("已登录：" if zh else "Logged in: ") + st.session_state.account_name)
        recent_files = list_user_files(st.session_state.get("user_id"), limit=5)
        if recent_files:
            with st.expander("最近上传" if zh else "Recent Uploads", expanded=False):
                for file in recent_files:
                    size_kb = int(file.get("size_bytes", 0) / 1024)
                    st.caption(f"{file.get('original_name')} · {size_kb} KB")
        if st.button("退出登录" if zh else "Log out", width="stretch"):
            persist_workspace_for_current_user()
            st.session_state.signed_in = False
            st.session_state.account_name = ""
            st.session_state.account_email = ""
            st.session_state.user_id = None
            st.session_state.workspace_loaded_user_id = None
            st.session_state.imported_records = {"contents": [], "revenues": [], "campaigns": []}
            st.session_state.todo_items = []
            st.session_state.done_todos = {}
            st.rerun()
        return

    mode = st.radio("账户操作" if zh else "Account Action", ["登录", "注册"] if zh else ["Log in", "Sign up"], horizontal=True, label_visibility="collapsed")
    with st.form("account_form", clear_on_submit=False):
        email = st.text_input("手机号或邮箱" if zh else "Phone or Email", placeholder="184... / name@example.com")
        display_name = ""
        if mode in ["注册", "Sign up"]:
            display_name = st.text_input("昵称" if zh else "Display Name", placeholder="SoloDeck user")
        password = st.text_input("密码" if zh else "Password", type="password")
        submitted = st.form_submit_button(mode, width="stretch")
    if submitted:
        if mode in ["注册", "Sign up"]:
            result = register_user(email, password, display_name)
            if result["ok"]:
                st.session_state.signed_in = True
                st.session_state.user_id = result["user_id"]
                st.session_state.account_name = result["display_name"]
                st.session_state.account_email = result["email"]
                st.session_state.guest_mode = False
                st.session_state.use_example_data = False
                load_workspace_for_current_user(force=True)
                st.success("注册成功，已登录。" if zh else "Account created and logged in.")
                st.rerun()
            else:
                st.error(result["error"] if zh else result["error"])
        else:
            result = login_user(email, password)
            if result["ok"]:
                st.session_state.signed_in = True
                st.session_state.user_id = result["user_id"]
                st.session_state.account_name = result["display_name"]
                st.session_state.account_email = result["email"]
                st.session_state.guest_mode = False
                st.session_state.use_example_data = False
                load_workspace_for_current_user(force=True)
                st.success("已登录" if zh else "Logged in")
                st.rerun()
            else:
                st.error(result["error"] if zh else result["error"])


@st.dialog("SoloDeck Account")
def account_dialog(lang: str) -> None:
    account_panel(lang)


def require_login_access(lang: str) -> None:
    if os.getenv("SOLODECK_REQUIRE_LOGIN", "false").lower() != "true":
        return
    if st.session_state.get("signed_in"):
        return
    with st.container(border=True):
        st.info("请先登录或注册后继续使用 SoloDeck。" if lang == "中文" else "Log in or sign up to continue using SoloDeck.")
        account_panel(lang)
    st.stop()


def render_plan_cards(plan: list[dict], lang: str) -> None:
    for index, item in enumerate(plan, start=1):
        platform = platform_name(item["suggested_platform"]) if lang == "中文" else item["suggested_platform"]
        objective = OBJECTIVE_LABELS_ZH.get(item["objective"], item["objective"]) if lang == "中文" else item["objective"]
        style = STYLE_LABELS_ZH.get(item["suggested_title_style"], item["suggested_title_style"]) if lang == "中文" else item["suggested_title_style"]
        reason = item["reason"]
        if lang == "中文":
            for raw, label in OBJECTIVE_LABELS_ZH.items():
                reason = reason.replace(raw, label)
            title = item["sample_title"]
        else:
            title = englishize_text(item["sample_title"])
            reason = englishize_text(reason)
        render_html(
            f"""
            <div class="soft-card">
              <div><span class="badge">{index}</span><span class="badge">{platform}</span><span class="badge">{objective}</span><span class="badge">{style}</span></div>
              <div class="soft-card-title">{safe_text(title)}</div>
              <div class="soft-muted">{safe_text(reason)}</div>
            </div>
            """
        )


def render_alert_cards(alerts: list[dict], lang: str) -> None:
    if not alerts:
        render_html(
            '<div class="soft-card"><span class="badge badge-ok">OK</span><span class="soft-muted">当前没有明显高风险商务事项。</span></div>'
            if lang == "中文"
            else '<div class="soft-card"><span class="badge badge-ok">OK</span><span class="soft-muted">No major business risks detected.</span></div>'
        )
        return
    for alert in alerts:
        level = alert.get("level", "medium")
        badge_class = "badge-danger" if level == "high" else "badge-warn" if level == "medium" else "badge-ok"
        label = {"high": "紧急", "medium": "关注", "low": "提醒"}.get(level, "关注") if lang == "中文" else level.title()
        title = alert.get("title", "")
        detail = alert.get("detail", "")
        if lang != "中文":
            title, detail = englishize_text(title), englishize_text(detail)
        render_html(
            f"""
            <div class="soft-card {'risk-band' if level == 'high' else ''}">
              <span class="badge {badge_class}">{label}</span>
              <div class="soft-card-title">{safe_text(title)}</div>
              <div class="soft-muted">{safe_text(detail)}</div>
            </div>
            """
        )


def render_quick_guide(lang: str) -> None:
    zh = lang == "中文"
    steps = [
        ("选平台", "选常用渠道"),
        ("加资料", "上传或粘贴"),
        ("看行动", "先做前三项"),
    ] if zh else [
        ("Channels", "Pick active ones"),
        ("Data", "Upload or paste"),
        ("Actions", "Top 3 first"),
    ]
    html_steps = "".join(
        f"<div class='guide-step'><div><span class='guide-index'>{idx}</span><span class='guide-title'>{safe_text(title)}</span></div><div class='guide-copy'>{safe_text(copy)}</div></div>"
        for idx, (title, copy) in enumerate(steps, start=1)
    )
    render_html(f"<div class='guide-strip'>{html_steps}</div>")


def render_sidebar_help(lang: str) -> None:
    zh = lang == "中文"
    with st.sidebar:
        st.markdown("### SoloDeck")
        st.caption("你的经营便签和决策管家。" if zh else "Your business notepad and decision assistant.")
        with st.expander("使用指引" if zh else "Guide", expanded=False):
            render_quick_guide(lang)
        with st.expander("产品定位" if zh else "Positioning", expanded=False):
            if zh:
                st.markdown("Power BI 回答“数据发生了什么”；SoloDeck 回答“下周做什么、在哪里做、如何验证”。")
            else:
                st.markdown("Power BI answers what happened. SoloDeck answers what to do next, where to do it, and how to validate it.")
        with st.expander("分析能力" if zh else "Analysis", expanded=False):
            if zh:
                st.markdown("- 识别内容系列、标题风格、产品款式和反馈主题\n- 判断哪些做法值得继续验证\n- 生成实验方案、观察指标和停止规则")
            else:
                st.markdown("- Understands series, title styles, product models and feedback themes\n- Estimates strategy lift\n- Creates tests, metrics and stop rules")


def onboarding_seen() -> bool:
    if st.session_state.get("onboarding_seen"):
        return True
    user_id = st.session_state.get("user_id")
    if user_id:
        seen = bool(load_latest_user_dataset(user_id, "onboarding_seen", False))
        st.session_state.onboarding_seen = seen
        return seen
    return False


def mark_onboarding_seen() -> None:
    st.session_state.onboarding_seen = True
    user_id = st.session_state.get("user_id")
    if user_id:
        save_user_dataset(user_id, "onboarding_seen", True)


def render_onboarding(lang: str) -> None:
    if onboarding_seen():
        return
    zh = lang == "中文"
    render_html(
        f"""
        <div class="onboarding">
          <b>{'第一次使用，按这个顺序就行' if zh else 'First time here? Follow this path.'}</b>
          <div class="soft-muted">{'先选平台，再上传截图/CSV/文字。系统会把最重要的 3 件事放到最前面。' if zh else 'Pick channels, add screenshots/CSV/notes. SoloDeck puts the top 3 actions first.'}</div>
        </div>
        """
    )
    if st.button("跳过新手教程" if zh else "Skip Onboarding", key="skip_onboarding"):
        mark_onboarding_seen()
        st.rerun()


def render_public_landing(lang: str) -> None:
    zh = lang == "中文"
    headline = "把经营数据变成下一步行动" if zh else "Turn business data into next actions"
    copy = "Power BI 帮你看清数据，SoloDeck 帮你决定下一步怎么做。它把内容、产品、反馈和收入转成可执行经营动作。" if zh else "Power BI helps you see data. SoloDeck helps you decide the next move. It turns content, product, feedback and revenue data into executable actions."
    features = [
        ("语义理解", "识别标题风格、内容系列、产品款式和用户反馈。"),
        ("相似识别", "发现重复内容、产品变体和系列蚕食风险。"),
        ("增量判断", "估计策略是否带来收益，并给出验证方式。"),
        ("经营记忆", "保存资料、任务和复盘，持续沉淀账号规律。"),
    ] if zh else [
        ("Semantic Layer", "Understands titles, series, product models and feedback."),
        ("Similarity", "Finds duplicate content, variants and cannibalization risks."),
        ("Lift", "Estimates what may create revenue lift and how to validate it."),
        ("Memory", "Saves uploads, tasks and reviews to build account knowledge."),
    ]
    feature_html = "".join(
        f"<div class='feature-card'><b>{safe_text(title)}</b><span>{safe_text(desc)}</span></div>"
        for title, desc in features
    )
    render_html(
        f"""
        <div class="landing">
          <h1>{safe_text(headline)}</h1>
          <div class="landing-copy">{safe_text(copy)}</div>
          <div class="feature-grid">{feature_html}</div>
        </div>
        """
    )
    col1, col2, col3 = st.columns([0.2, 0.2, 0.6])
    if col1.button("登录 / 注册" if zh else "Log in / Sign up", key="landing_login", width="stretch"):
        account_dialog(lang)
    if col2.button("体验工作台" if zh else "Try Workspace", key="landing_guest", width="stretch"):
        st.session_state.guest_mode = True
        st.rerun()
    st.caption("登录后会保存你的上传文件、导入数据和任务状态。" if zh else "Log in to save uploads, imported records and task status.")


def render_status_strip(contents_df: pd.DataFrame, revenues_df: pd.DataFrame, campaigns_df: pd.DataFrame, products_df: pd.DataFrame, lang: str) -> None:
    zh = lang == "中文"
    total_revenue = float(revenues_df["amount"].sum() + contents_df["revenue"].sum()) if not contents_df.empty else 0.0
    risk_count = len(campaign_risk_alerts(campaigns_df, language=lang)) if not campaigns_df.empty else 0
    values = [
        ("内容" if zh else "Content", f"{len(contents_df)}"),
        ("收入" if zh else "Revenue", f"¥{total_revenue:,.0f}"),
        ("产品" if zh else "Products", f"{len(products_df)}"),
        ("待处理" if zh else "Risks", f"{risk_count}"),
    ]
    html_items = "".join(
        f"""
        <div class="status-pill">
          <div class="status-label">{safe_text(label)}</div>
          <div class="status-value">{safe_text(value)}</div>
        </div>
        """
        for label, value in values
    )
    render_html(f"<div class='status-strip'>{html_items}</div>")


require_invite_access()
init_auth_db()
init_user_storage()
init_import_store()
init_todo_store()
load_workspace_for_current_user()
synthetic_data_dir = find_synthetic_data_dir(ROOT)
example_data_dir = st.session_state.get("example_data_dir")
contents, revenues, campaigns, ab_tests, products, feedback, beta_tests = cached_load(example_data_dir)
if not use_example_data():
    contents, revenues, campaigns, ab_tests, products, feedback, beta_tests = (
        empty_like(contents),
        empty_like(revenues),
        empty_like(campaigns),
        empty_like(ab_tests),
        empty_like(products),
        empty_like(feedback),
        empty_like(beta_tests),
    )
contents, revenues, campaigns = apply_import_store(contents, revenues, campaigns)

language = st.session_state.get("language", "中文")
ZH = language == "中文"

brand_title = "SoloDeck"
brand_subtitle = "上传资料，直接得到下一步经营建议。" if ZH else "Upload materials. Get next actions."
render_html(
    f"""
    <div class="app-nav">
      <div>
        <div class="brand"><span class="brand-mark"></span>{brand_title}</div>
        <div class="nav-sub">{brand_subtitle}</div>
      </div>
    </div>
    """
)

control_left, control_right = st.columns([0.84, 0.16], vertical_alignment="center")
with control_left:
    render_html('<span class="language-anchor"></span>')
    language_choice = st.radio("语言", ["中文", "English"], horizontal=True, index=0 if ZH else 1, label_visibility="collapsed")
    if language_choice != language:
        st.session_state.language = language_choice
        st.rerun()
with control_right:
    render_html('<span class="account-anchor"></span>')
    account_label = ("已登录" if st.session_state.get("signed_in") else "登录 / 注册") if ZH else ("Logged in" if st.session_state.get("signed_in") else "Log in / Sign up")
    if st.button(account_label):
        account_dialog(language)

require_login_access(language)

if not st.session_state.get("signed_in") and not st.session_state.get("guest_mode"):
    render_public_landing(language)
    st.stop()

render_onboarding(language)
render_sidebar_help(language)

with st.expander("设置与数据文件" if ZH else "Settings and CSV Data", expanded=False):
    s1, s2, s3 = st.columns([0.28, 0.28, 0.44])
    ai_enabled = s1.toggle(
        "智能建议" if ZH else "Smart Advice",
        value=llm_configured(),
        disabled=not llm_configured(),
        help="自动生成经营建议、实验解读和报告摘要。" if ZH else "Generate business advice, experiment interpretation and report summaries.",
    )
    simple_mode = s2.toggle(
        "极简模式" if ZH else "Simple Mode",
        value=True,
        help="默认只展示结果和建议；关闭后查看完整指标、图表和专业模块。" if ZH else "Show recommendations first.",
    )
    if s3.button("载入演示数据" if ZH else "Load Example Data", width="stretch"):
        if synthetic_data_dir:
            st.session_state.example_data_dir = str(synthetic_data_dir)
        else:
            generate_all(ROOT / "data")
            st.session_state.example_data_dir = str(ROOT / "data")
        st.session_state.use_example_data = True
        st.cache_data.clear()
        st.rerun()
    if use_example_data():
        source_label = Path(st.session_state.get("example_data_dir", str(ROOT / "data"))).name
        st.caption(("已载入演示数据：内容、产品、收入、反馈、商务合作和实验记录。" if ZH else "Example data loaded: content, products, revenue, feedback, campaigns and experiments."))
    if llm_configured():
        st.caption("智能建议已开启。" if ZH else "Smart advice is enabled.")

    csv_cols = st.columns(4)
    content_file = csv_cols[0].file_uploader("内容 CSV" if ZH else "Contents CSV", type=["csv"])
    revenue_file = csv_cols[1].file_uploader("收入 CSV" if ZH else "Revenues CSV", type=["csv"])
    campaign_file = csv_cols[2].file_uploader("商务 CSV" if ZH else "Campaigns CSV", type=["csv"])
    ab_file = csv_cols[3].file_uploader("实验 CSV" if ZH else "Experiment CSV", type=["csv"])
    if content_file:
        save_uploaded_file(st.session_state.get("user_id"), content_file, category="contents_csv")
        contents = read_uploaded_csv(content_file, ["publish_time"])
        contents["publish_time"] = pd.to_datetime(contents["publish_time"])
    if revenue_file:
        save_uploaded_file(st.session_state.get("user_id"), revenue_file, category="revenues_csv")
        revenues = read_uploaded_csv(revenue_file, ["date"])
        revenues["date"] = pd.to_datetime(revenues["date"]).dt.date
    if campaign_file:
        save_uploaded_file(st.session_state.get("user_id"), campaign_file, category="campaigns_csv")
        campaigns = read_uploaded_csv(campaign_file, ["deadline"])
        campaigns["deadline"] = pd.to_datetime(campaigns["deadline"]).dt.date
    if ab_file:
        save_uploaded_file(st.session_state.get("user_id"), ab_file, category="ab_tests_csv")
        ab_tests = read_uploaded_csv(ab_file, ["date"])
        ab_tests["date"] = pd.to_datetime(ab_tests["date"]).dt.date
    extra_cols = st.columns(3)
    product_file = extra_cols[0].file_uploader("产品 CSV" if ZH else "Products CSV", type=["csv"])
    feedback_file = extra_cols[1].file_uploader("反馈 CSV" if ZH else "Feedback CSV", type=["csv"])
    beta_file = extra_cols[2].file_uploader("内测 CSV" if ZH else "User Tests CSV", type=["csv"])
    if product_file:
        save_uploaded_file(st.session_state.get("user_id"), product_file, category="products_csv")
        products = read_uploaded_csv(product_file, ["launch_date"])
        products["launch_date"] = pd.to_datetime(products["launch_date"]).dt.date
    if feedback_file:
        save_uploaded_file(st.session_state.get("user_id"), feedback_file, category="feedback_csv")
        feedback = read_uploaded_csv(feedback_file, ["created_at"])
        feedback["created_at"] = pd.to_datetime(feedback["created_at"], errors="coerce")
    if beta_file:
        save_uploaded_file(st.session_state.get("user_id"), beta_file, category="beta_tests_csv")
        beta_tests = read_uploaded_csv(beta_file, ["invited_at", "experienced_at"])
        beta_tests["invited_at"] = pd.to_datetime(beta_tests["invited_at"], errors="coerce")
        beta_tests["experienced_at"] = pd.to_datetime(beta_tests["experienced_at"], errors="coerce")

COMMON_PLATFORMS = ["xiaohongshu", "bilibili", "douyin", "wechat", "zhihu", "youtube", "tiktok", "instagram", "substack", "x"]
data_platforms = sorted(contents["platform"].dropna().unique().tolist()) if "platform" in contents.columns else []
all_platforms = sorted(set(data_platforms).union(COMMON_PLATFORMS))
default_platforms = [p for p in ["xiaohongshu", "bilibili", "douyin", "wechat", "zhihu"] if p in all_platforms]
if "selected_platforms" not in st.session_state:
    st.session_state.selected_platforms = default_platforms[:4] if default_platforms else all_platforms[:4]
if "pending_selected_platforms" in st.session_state:
    st.session_state.selected_platforms = [p for p in st.session_state.pending_selected_platforms if p in all_platforms]
    del st.session_state.pending_selected_platforms
default_selected_platforms = [p for p in st.session_state.selected_platforms if p in all_platforms]
setup_left, setup_right = st.columns([0.38, 0.62])
with setup_left:
    with st.container(border=True):
        st.markdown("### 1. 选择平台" if ZH else "### 1. Platforms")
        st.caption("选常用渠道。" if ZH else "Pick active channels.")
        pills_kwargs = {
            "label": "平台" if ZH else "Platforms",
            "options": all_platforms,
            "format_func": platform_name if ZH else lambda x: x,
            "selection_mode": "multi",
            "key": "selected_platforms",
            "help": "支持国内外主流内容平台。基础分析逻辑一致，但平台定位、收入效率和策略建议会随平台数据变化。" if ZH else "Supports mainstream Chinese and global creator platforms.",
        }
        if "selected_platforms" not in st.session_state:
            pills_kwargs["default"] = default_selected_platforms
        selected_platforms = st.pills(**pills_kwargs)
        setup_cols = st.columns(2)
        setup_cols[0].metric("平台" if ZH else "Platforms", len(selected_platforms))
        setup_cols[1].metric("内容" if ZH else "Content", int(contents[contents["platform"].isin(selected_platforms)]["content_id"].nunique()) if selected_platforms else len(contents))
        setup_cols_2 = st.columns(2)
        setup_cols_2[0].metric("收入记录" if ZH else "Revenue", int(revenues[revenues["platform"].isin(selected_platforms)].shape[0]) if selected_platforms else len(revenues))
        setup_cols_2[1].metric("商务合作" if ZH else "Campaigns", int(campaigns[campaigns["platform"].isin(selected_platforms)].shape[0]) if selected_platforms else len(campaigns))

with setup_right:
    with st.container(border=True):
        st.markdown("### 2. 添加资料" if ZH else "### 2. Add Data")
        st.caption("截图、文件或文字。" if ZH else "Files, screenshots or notes.")
        if st.session_state.last_import_message:
            st.success(st.session_state.last_import_message)
            st.session_state.last_import_message = ""
        import_nonce = st.session_state.import_nonce
        import_top = st.columns([0.34, 0.66])
        import_target_label = import_top[0].selectbox(
            "资料类型" if ZH else "Type",
            ["内容数据", "收入数据", "商务合作"] if ZH else ["contents", "revenues", "campaigns"],
            key="main_import_target",
        )
        import_target = {
            "内容数据": "contents",
            "收入数据": "revenues",
            "商务合作": "campaigns",
            "contents": "contents",
            "revenues": "revenues",
            "campaigns": "campaigns",
        }[import_target_label]
        import_files = import_top[1].file_uploader(
            "截图或文件" if ZH else "Screenshots or Files",
            type=["png", "jpg", "jpeg", "webp", "txt", "csv"],
            accept_multiple_files=True,
            key=f"main_import_files_{import_nonce}",
        )
        import_text = st.text_area(
            "文字补充" if ZH else "Text Notes",
            placeholder="粘贴后台数据、聊天报价、收款记录..." if ZH else "Paste backend stats, deal chats or payment notes...",
            height=82,
            key=f"main_import_text_{import_nonce}",
        )
        if st.button("读取并加入分析" if ZH else "Extract and Add", disabled=not ai_enabled, width="stretch"):
            if not import_files and not import_text.strip():
                st.warning("请先上传截图/文件或粘贴文字。" if ZH else "Upload files or paste text first.")
            else:
                with st.spinner("正在读取资料..." if ZH else "Extracting materials..."):
                    try:
                        extracted = extract_records_from_uploads(import_text, import_files, import_target, language=language)
                        records = extracted.get("records", [])
                        saved_file_ids = []
                        for uploaded in import_files or []:
                            saved = save_uploaded_file(st.session_state.get("user_id"), uploaded, category=import_target)
                            if saved:
                                saved_file_ids.append(saved["id"])
                        st.session_state.imported_records[import_target].extend(records)
                        imported_platforms = [record.get("platform") for record in records if record.get("platform")]
                        if imported_platforms:
                            current_platforms = set(st.session_state.get("selected_platforms", []))
                            current_platforms.update(imported_platforms)
                            st.session_state.pending_selected_platforms = sorted(current_platforms)
                        new_todos = (
                            todos_from_records(import_target, records, language)
                            + todos_from_extracted_tasks(extracted.get("tasks", []), language)
                        )
                        if not new_todos:
                            new_todos = todos_from_free_text(import_text, language)
                        add_todos(new_todos)
                        persist_workspace_for_current_user()
                        st.session_state.last_import_message = clean_import_message(
                            len(records),
                            len(new_todos),
                            extracted.get("notes", ""),
                            extracted.get("validation", {}),
                            language,
                        )
                        st.session_state.import_nonce += 1
                        st.rerun()
                    except Exception:
                        st.error("读取失败。请换一张更清晰的截图，或把关键文字粘贴到文本框。" if ZH else "Import failed. Try a clearer screenshot or paste the key text.")

contents, revenues, campaigns, ab_tests, products, feedback, beta_tests = filter_by_platforms(contents, revenues, campaigns, ab_tests, selected_platforms, products, feedback, beta_tests)

if contents.empty:
    render_status_strip(contents, revenues, campaigns, products, language)
    with st.container(border=True):
        st.markdown("### 你的工作区是空的" if ZH else "### Your Workspace Is Empty")
        st.caption(
            "新用户默认没有任何样例数据。上传截图、CSV 或文字后，SoloDeck 会生成行动建议；也可以在“设置与数据文件”里载入体验数据。"
            if ZH
            else "New users start with no sample data. Upload screenshots, CSVs or notes to generate actions, or load example data from Settings and CSV Data."
        )
    st.stop()

render_status_strip(contents, revenues, campaigns, products, language)
agent_result = run_agent_suite(contents, revenues, campaigns, ab_tests, products, feedback, beta_tests, lang=language)
workflow_trace = build_workflow_trace(contents, revenues, campaigns, ab_tests, products, feedback, beta_tests, agent_result, lang=language)
metadata_ground_truth = {}
if use_example_data():
    metadata_path = Path(st.session_state.get("example_data_dir", str(ROOT / "data"))) / "metadata_ground_truth.json"
    if metadata_path.exists():
        try:
            metadata_ground_truth = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata_ground_truth = {}

quick_actions = action_brief(contents, revenues, campaigns, language)
short_tasks = system_todos(contents, revenues, campaigns, language) + st.session_state.todo_items
long_tasks = long_term_todos(contents, revenues, language)
with st.container(border=True):
    render_task_plan(short_tasks, long_tasks, language)
    if st.session_state.latest_ai_advice:
        with st.expander("查看上次智能建议" if ZH else "View Latest Smart Advice", expanded=False):
            st.markdown(st.session_state.latest_ai_advice)
    if ai_enabled:
        if st.button("更新任务和经营建议" if ZH else "Update Tasks and Advice", width="stretch"):
            with st.spinner("正在分析经营数据..." if ZH else "Analyzing your business data..."):
                revenue_summary_preview = classify_revenue_summary(revenues).head(8)
                platform_summary_preview = platform_revenue_summary(revenues).head(8)
                value_preview = content_commercial_value(contents, revenues, language=language).head(8)
                topic_preview = topic_business_summary(contents).head(8)
                risk_preview = campaign_risk_alerts(campaigns, language=language)[:8]
                payload = {
                    "selected_platforms": [platform_name(p) if ZH else p for p in selected_platforms],
                    "metrics": {
                        "content_count": len(contents),
                        "total_revenue": float(revenues["amount"].sum() + contents["revenue"].sum()),
                        "pending_payment": float(pending_payment_summary(revenues, campaigns)["total_pending_amount"]),
                    },
                    "revenue_summary": revenue_summary_preview,
                    "platform_summary": platform_summary_preview,
                    "high_value_contents": value_preview,
                    "topic_summary": topic_preview,
                    "campaign_risks": risk_preview,
                    "rule_based_actions": quick_actions,
                }
                try:
                    advice = generate_ai_business_advice(payload, language=language)
                    st.session_state.latest_ai_advice = advice
                    add_todos(todos_from_ai_advice(advice, language))
                    st.rerun()
                except Exception as exc:
                    st.error("暂时未能生成智能建议，请稍后再试。" if ZH else "Smart advice is temporarily unavailable. Please try again.")

if simple_mode:
    st.divider()
    st.markdown("### 本周验证什么" if ZH else "### What to Validate This Week")
    st.caption("只保留最该做的动作；详细数据放在下面，可按需展开。" if ZH else "Only the most important actions are shown here. Details are below.")
    front_test_cards = (
        agent_cards_to_suggestions(agent_result["cards"], language, limit=4)
        + ab_test_action_cards(ab_tests, language, contents)
        + platform_opportunity_cards(contents, revenues, language)
        + product_variant_action_cards(products, beta_tests, language)
    )
    front_test_cards = rank_recommendations(front_test_cards, st.session_state.get("recommendation_feedback", []))
    suggestion_cards(front_test_cards[:4], language)
    render_suggestion_feedback(front_test_cards[:4], language, key_prefix="front")

    evidence_col, plan_col = st.columns(2)
    with evidence_col:
        with st.container(border=True):
            st.markdown("### 依据是什么" if ZH else "### Evidence")
            st.caption("建议来自实验、平台和产品数据。" if ZH else "Based on tests, platforms and product data.")
            suggestion_cards(evidence_cards(contents, revenues, products, ab_tests, language), language)
    with plan_col:
        with st.container(border=True):
            st.markdown("### 内容计划" if ZH else "### Content Plan")
            st.caption("本周可直接放进排期。" if ZH else "Ready to place into this week's calendar.")
            render_plan_cards(weekly_topic_plan(contents, revenues, n=3, language=language), language)

    with st.expander("系统如何得出建议" if ZH else "How SoloDeck Reached These Actions", expanded=False):
        st.markdown(workflow_markdown(workflow_trace, language))
        kb = knowledge_cards(front_test_cards, language)
        if kb:
            st.markdown("#### 经营知识依据" if ZH else "#### Operating Knowledge")
            suggestion_cards(kb, language)

    with st.expander("详细分析与下载" if ZH else "Detailed Analysis and Downloads", expanded=False):
        st.caption(
            "这里放完整证据、实验结果和可下载报告，日常使用不需要一直展开。"
            if ZH
            else "Full evidence, experiment results and downloadable reports live here."
        )
        st.markdown("#### 经营报告" if ZH else "#### Operating Report")
        st.markdown(agent_result["report"])
        st.markdown("#### 流程记录" if ZH else "#### Workflow Trace")
        st.dataframe(pd.DataFrame(workflow_trace["steps"]), width="stretch")
        kb_frame = knowledge_frame(front_test_cards, language)
        if not kb_frame.empty:
            st.markdown("#### 知识匹配" if ZH else "#### Knowledge Matches")
            st.dataframe(kb_frame, width="stretch")
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "下载 Markdown 报告" if ZH else "Download Markdown Report",
            agent_result["report"],
            file_name="solodeck_operating_report.md",
            mime="text/markdown",
            width="stretch",
            key="agent_md_download",
        )
        dl2.download_button(
            "下载 JSON 结果" if ZH else "Download JSON Results",
            analysis_to_json(agent_result),
            file_name="solodeck_agent_results.json",
            mime="application/json",
            width="stretch",
            key="agent_json_download",
        )
        ate = agent_result["modules"].get("causal_estimator_agent", {}).get("ate", [])
        cate = agent_result["modules"].get("causal_estimator_agent", {}).get("cate", [])
        if ate:
            st.markdown("#### 策略增量估计" if ZH else "#### Strategy Lift Estimates")
            st.dataframe(format_lift_table(ate, language), width="stretch", hide_index=True)
        if cate:
            st.markdown("#### 不同主题的效果差异" if ZH else "#### Segment Differences")
            st.dataframe(format_lift_table(cate, language).head(30), width="stretch", hide_index=True)
        ab_agent = agent_result["modules"].get("ab_test_agent", {})
        if hasattr(ab_agent.get("ab_results"), "empty") and not ab_agent["ab_results"].empty:
            st.markdown("#### 实验效果" if ZH else "#### Experiment Results")
            st.dataframe(format_experiment_table(ab_agent["ab_results"], language), width="stretch", hide_index=True)
        fb_agent = agent_result["modules"].get("feedback_analysis_agent", {})
        if hasattr(fb_agent.get("roadmap"), "empty") and not fb_agent["roadmap"].empty:
            st.markdown("#### 用户反馈重点" if ZH else "#### Feedback Priorities")
            st.dataframe(format_feedback_table(fb_agent["roadmap"], language), width="stretch", hide_index=True)
        if metadata_ground_truth:
            with st.expander("示例数据机制" if ZH else "Example Data Mechanisms", expanded=False):
                st.json(metadata_ground_truth)

    with st.expander("我想看指标、图表和专业分析" if ZH else "Show Metrics, Charts and Advanced Analysis"):
        revenue_summary = classify_revenue_summary(revenues)
        platform_summary = platform_revenue_summary(revenues)
        value_df = content_commercial_value(contents, revenues, language=language)
        topic_df = topic_business_summary(contents)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(display_frame(revenue_summary, language), width="stretch")
        with col2:
            st.dataframe(display_frame(platform_summary, language), width="stretch")
        st.dataframe(display_frame(value_df.head(10), language), width="stretch")
        st.dataframe(display_frame(topic_df.head(10), language), width="stretch")

    report = generate_weekly_business_report(contents, revenues, campaigns, language=language)
    with st.expander("生成经营周报" if ZH else "Generate Business Report"):
        st.markdown("#### 简短总结" if ZH else "#### Short Summary")
        st.write(concise_summary(contents, revenues, campaigns, language))
        if ai_enabled:
            if st.button("生成精简顾问周报" if ZH else "Generate Executive Report", width="stretch", key="simple_ai_report"):
                with st.spinner("正在生成周报..." if ZH else "Generating the report..."):
                    try:
                        report = polish_report(report, language=language)
                    except Exception as exc:
                        st.error("暂时未能生成周报，请稍后再试。" if ZH else "The report is temporarily unavailable. Please try again.")
        if not ZH:
            report = englishize_text(report)
        st.download_button("下载周报" if ZH else "Download Report", report, file_name="creator_agent_report.md", key="simple_download")
        with st.expander("查看详细周报" if ZH else "View Detailed Report"):
            st.markdown(report)

    st.stop()

tabs = st.tabs(
    ["经营总览", "策略与实验", "商务与收款", "相似分析", "系列增量", "用户反馈", "内测设计", "推广建议", "报告"]
    if ZH
    else ["Overview", "Strategy & Experiments", "Business", "Similarity", "Series Lift", "Feedback", "User Tests", "Launch Advice", "Reports"]
)

with tabs[0]:
    st.subheader("收益与商业价值看板" if ZH else "Revenue and Commercial Value")
    revenue_summary = classify_revenue_summary(revenues)
    platform_summary = platform_revenue_summary(revenues)
    pending = pending_payment_summary(revenues, campaigns)
    rpm_df = calculate_rpm(contents)
    value_df = content_commercial_value(contents, revenues, language=language)
    topic_df = topic_business_summary(contents)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总收入" if ZH else "Total Revenue", f"¥{revenues['amount'].sum() + contents['revenue'].sum():,.0f}")
    c2.metric("待收款" if ZH else "Pending Payment", f"¥{pending['total_pending_amount']:,.0f}")
    c3.metric("内容数" if ZH else "Content Count", f"{len(contents)}")
    c4.metric("平均 RPM", f"¥{rpm_df['rpm'].mean():.1f}")

    st.markdown("### 先看建议" if ZH else "### Recommendations First")
    suggestion_cards(revenue_recommendations(contents, revenues, language=language), language)

    with st.expander("查看指标与图表" if ZH else "View Metrics and Charts"):
        col1, col2 = st.columns(2)
        with col1:
            revenue_summary_display = localize_frame(revenue_summary, language)
            st.plotly_chart(px.pie(revenue_summary_display, names="revenue_type", values="amount", title="收入类型占比" if ZH else "Revenue Mix"), width="stretch")
            st.dataframe(display_frame(revenue_summary, language), width="stretch")
        with col2:
            platform_summary_display = localize_frame(platform_summary, language)
            st.plotly_chart(px.bar(platform_summary_display, x="platform", y="amount", title="平台收入" if ZH else "Revenue by Platform"), width="stretch")
            st.dataframe(display_frame(platform_summary, language), width="stretch")

        st.markdown("#### 内容商业价值" if ZH else "#### Content Commercial Value")
        st.info("商业评分综合曝光、收藏、转粉、咨询、成交、收入和制作效率。" if ZH else "The score combines reach, saves, follows, consultations, conversions, revenue and production efficiency.")
        value_display = localize_frame(value_df, language)
        topic_display = localize_frame(topic_df, language)
        st.plotly_chart(px.scatter(value_display, x="consultations", y="revenue", color="platform", size="commercial_score", hover_name="title", title="咨询量与收入" if ZH else "Consultations vs Revenue"), width="stretch")
        st.dataframe(display_frame(value_df, language), width="stretch")
        st.markdown("#### 主题商业汇总" if ZH else "#### Topic Business Summary")
        st.plotly_chart(px.bar(topic_display, x="topic", y="revenue_per_hour", color="total_revenue", title="主题单位时间收益" if ZH else "Revenue per Hour by Topic"), width="stretch")
        st.dataframe(display_frame(topic_df, language), width="stretch")

with tabs[1]:
    st.subheader("策略分析" if ZH else "Strategy Analysis")
    title_result = title_style_analysis(contents)
    topic_result = topic_strategy_analysis(contents)
    platform_result = platform_strategy_analysis(contents, revenues)
    time_df = publish_time_analysis(contents)
    plan = weekly_topic_plan(contents, revenues, n=7, language=language)

    st.markdown("### 本周应做的实验" if ZH else "### Tests to Run This Week")
    st.caption("按卡片执行即可：平台、数量、观察指标和胜出标准都在里面。" if ZH else "Follow the cards: platform, quantity, metric and win condition are included.")
    test_cards = ab_test_action_cards(ab_tests, language, contents) + platform_opportunity_cards(contents, revenues, language)
    suggestion_cards(test_cards[:5], language)

    st.markdown("### 下周选题计划" if ZH else "### Next Week Topic Plan")
    st.dataframe(display_frame(pd.DataFrame(plan), language), width="stretch")

    with st.expander("查看策略指标与平台定位" if ZH else "View Strategy Metrics and Platform Roles"):
        st.info(f"标题策略可信度：{title_result['confidence']}。建议用小实验验证后放大。" if ZH else f"Title strategy confidence: {title_result['confidence']}. Validate with a small test before scaling.")
        col1, col2 = st.columns(2)
        with col1:
            title_table_display = localize_frame(title_result["table"], language)
            st.plotly_chart(px.bar(title_table_display, x="title_style", y="views", title="标题风格曝光表现" if ZH else "Views by Title Style"), width="stretch")
            st.dataframe(display_frame(title_result["table"], language), width="stretch")
        with col2:
            topic_strategy_display = localize_frame(topic_result["table"], language)
            st.plotly_chart(px.bar(topic_strategy_display, x="topic", y="revenue_per_hour", title="主题单位时间收益" if ZH else "Revenue per Hour by Topic"), width="stretch")
            st.dataframe(display_frame(topic_result["table"], language), width="stretch")

        st.markdown("#### 平台定位" if ZH else "#### Platform Roles")
        if ZH:
            role_display = {k: [platform_name(v) for v in vals] for k, vals in platform_result["platform_roles"].items()}
            st.json(role_display)
        else:
            st.json(platform_result["platform_roles"])
        st.dataframe(display_frame(platform_result["table"], language), width="stretch")

        st.markdown("#### 发布时间段" if ZH else "#### Publishing Time")
        st.dataframe(display_frame(time_df.head(20), language), width="stretch")

with tabs[1]:
    st.divider()
    st.subheader("增量洞察：判断策略是否值得放大" if ZH else "Lift Insights: Decide What to Scale")
    st.caption(
        "系统用统计方法估计策略增量，并给出可执行的小实验。"
        if ZH
        else "SoloDeck estimates strategy lift with statistical methods and turns it into practical tests."
    )
    causal_cols = st.columns([0.3, 0.3, 0.4])
    treatment_col = causal_cols[0].selectbox(
        "策略变量" if ZH else "Treatment",
        [c for c in ["title_style", "cover_style", "platform", "topic", "content_type", "is_sponsored"] if c in contents.columns],
        key="causal_treatment",
    )
    outcome_col = causal_cols[1].selectbox(
        "结果指标" if ZH else "Outcome",
        [c for c in ["views", "favorites", "completion_rate", "new_followers", "consultations", "conversions", "revenue"] if c in contents.columns],
        key="causal_outcome",
    )
    covariates = causal_cols[2].multiselect(
        "控制变量" if ZH else "Covariates",
        [c for c in ["platform", "topic", "content_type", "followers_before", "production_hours", "duration_sec", "ad_spend"] if c in contents.columns and c != treatment_col],
        default=[c for c in ["platform", "topic", "followers_before", "ad_spend"] if c in contents.columns and c != treatment_col],
    )

    variable_map = map_variables(contents, use_llm=False, language=language)
    reg_effect = stratified_regression_effect(contents, treatment_col, outcome_col, covariates)
    psm_effect = propensity_score_matching(contents, treatment_col, outcome_col, covariates)
    iptw = iptw_effect(contents, treatment_col, outcome_col, covariates)
    refute = refute_suite(contents, treatment_col, outcome_col, covariates)
    effect_cards = st.columns(4)
    effect_cards[0].metric("分层回归效应" if ZH else "Regression Effect", f"{reg_effect['effect']:.2f}")
    effect_cards[1].metric("匹配后增量" if ZH else "PSM Effect", f"{psm_effect['effect']:.2f}")
    effect_cards[2].metric("加权后增量" if ZH else "IPTW Effect", f"{iptw['effect']:.2f}")
    effect_cards[3].metric("反驳通过" if ZH else "Refutes Passed", f"{refute['summary']['passed']}/{refute['summary']['total']}")
    st.info(
        (
            f"策略增量估计：样本 {reg_effect['sample_size']}，可信度 {reg_effect['confidence']}，区间约 [{reg_effect['ci_low']:.2f}, {reg_effect['ci_high']:.2f}]。"
            if ZH
            else f"Estimated strategy lift: sample size {reg_effect['sample_size']}, confidence {reg_effect['confidence']}, interval [{reg_effect['ci_low']:.2f}, {reg_effect['ci_high']:.2f}]."
        )
    )
    with st.expander("查看变量语义、反驳检验和跨平台增量" if ZH else "View Variable Semantics, Refutation and Cross-Platform Increment"):
        st.markdown("#### 变量语义理解" if ZH else "#### Variable Semantics")
        st.dataframe(variable_map, width="stretch")
        st.markdown("#### 协变量平衡性" if ZH else "#### Covariate Balance")
        if hasattr(refute["balance"], "empty") and not refute["balance"].empty:
            st.dataframe(refute["balance"], width="stretch")
        st.markdown("#### 反驳检验" if ZH else "#### Refutation Tests")
        st.json({k: v for k, v in refute.items() if k != "balance"})
        increment = cross_platform_increment(contents, outcome_col=outcome_col)
        if not increment.empty:
            st.markdown("#### 同内容跨平台增量" if ZH else "#### Same-Content Cross-Platform Increment")
            st.dataframe(display_frame(increment.head(20), language), width="stretch")

    experiment_candidates = weekly_experiment_plan(contents, language=language)
    if experiment_candidates:
        with st.expander("自动生成的下周实验候选" if ZH else "Auto-Generated Experiment Candidates"):
            st.dataframe(pd.DataFrame(experiment_candidates)[["hypothesis", "primary_metric", "estimated_effect", "suggested_min_total_sample", "data_confidence"]], width="stretch")

    st.divider()
    st.subheader("实验向导：用最小实验验证内容策略" if ZH else "Experiment Guide: Validate Content Strategy with Minimal Tests")
    st.info("你不需要懂统计学。这里的用法是：选择一个目标，只改变一个变量，连续发布两组内容，然后看实验组是否更好。" if ZH else "No statistics background required: pick one goal, change one variable, publish two groups, compare outcomes.")

    guide_cols = st.columns(5)
    steps = [
        ("1. 定目标", "例如提升收藏、咨询或成交。"),
        ("2. 选变量", "一次只测试标题、行动入口、封面或发布时间。"),
        ("3. 分两组", "实验组用新做法，对照组用旧做法。"),
        ("4. 固定条件", "平台、主题、内容长度尽量一致。"),
        ("5. 看结果", "优先放大表现更稳的一组。"),
    ] if ZH else [
        ("1. Goal", "Improve saves, consultations or sales."),
        ("2. Variable", "Test title, CTA, cover or timing."),
        ("3. Groups", "New approach vs old approach."),
        ("4. Controls", "Keep platform, topic and length similar."),
        ("5. Result", "Scale the steadier winner."),
    ]
    for col, (title, desc) in zip(guide_cols, steps):
        col.html(f"<div class='flow-step'><b>{title}</b><span class='small-muted'>{desc}</span></div>")

    readiness = causal_readiness_check(ab_tests, language=language)
    st.metric("实验准备度" if ZH else "Experiment Readiness", f"{readiness['readiness_score']}/100")
    if readiness["warning"]:
        st.warning("；".join(readiness["warning"]))
    st.caption("建议先验证，再放大投入。" if ZH else "Validate first, then scale investment.")

    ab_result = analyze_ab_test(ab_tests, language=language)
    if ab_result.empty:
        st.warning("当前平台暂无实验数据。可上传实验 CSV，或先使用系统生成的实验方案。" if ZH else "No experiment data for selected platforms. Upload an experiment CSV or use the generated plan first.")
    else:
        ab_display = localize_frame(ab_result, language)
        st.plotly_chart(px.bar(ab_display, x="experiment_id", y="relative_lift", color="outcome_metric", title="实验相对提升" if ZH else "Relative Lift"), width="stretch")
        st.dataframe(display_frame(ab_result, language), width="stretch")

    st.markdown("### 设计下周实验" if ZH else "### Design Next Week's Experiment")
    if ZH:
        presets = [
            "标题风格：痛点标题 对比 教程标题",
            "成交承接：前置行动入口 对比 结尾行动入口",
            "发布时间：晚上 9 点 对比 中午 12 点",
            "内容形式：案例复盘 对比 方法清单",
        ]
    else:
        presets = [
            "Title Style: Pain Point vs Tutorial",
            "CTA Placement: Early CTA vs Ending CTA",
            "Publishing Time: 9 PM vs 12 PM",
            "Format: Case Study vs Listicle",
        ]
    preset = st.selectbox("选择一个实验场景" if ZH else "Choose an Experiment Scenario", presets)
    col1, col2, col3 = st.columns(3)
    goal = col1.text_input("目标" if ZH else "Goal", "提升咨询转化" if ZH else "Improve consultation conversion")
    platform = col2.selectbox("平台" if ZH else "Platform", sorted(contents["platform"].unique()), format_func=platform_name if ZH else lambda x: x)
    topic = col3.selectbox("主题" if ZH else "Topic", sorted(contents["topic"].unique()))
    col4, col5, col6 = st.columns(3)
    default_treatment = {
        "标题风格：痛点标题 对比 教程标题": "痛点型标题",
        "成交承接：前置行动入口 对比 结尾行动入口": "正文前 30% 放咨询入口",
        "发布时间：晚上 9 点 对比 中午 12 点": "21:00 发布",
        "内容形式：案例复盘 对比 方法清单": "案例复盘",
        "Title Style: Pain Point vs Tutorial": "Pain point title",
        "CTA Placement: Early CTA vs Ending CTA": "CTA in first 30%",
        "Publishing Time: 9 PM vs 12 PM": "Publish at 21:00",
        "Format: Case Study vs Listicle": "Case study",
    }[preset]
    default_control = {
        "标题风格：痛点标题 对比 教程标题": "教程型标题",
        "成交承接：前置行动入口 对比 结尾行动入口": "结尾放咨询入口",
        "发布时间：晚上 9 点 对比 中午 12 点": "12:00 发布",
        "内容形式：案例复盘 对比 方法清单": "方法清单",
        "Title Style: Pain Point vs Tutorial": "Tutorial title",
        "CTA Placement: Early CTA vs Ending CTA": "CTA at ending",
        "Publishing Time: 9 PM vs 12 PM": "Publish at 12:00",
        "Format: Case Study vs Listicle": "Listicle",
    }[preset]
    treatment = col4.text_input("实验组", default_treatment)
    control = col5.text_input("对照组", default_control)
    metric_options = ["view_rate", "like_rate", "favorite_rate", "follow_rate", "conversion_rate", "revenue"]
    metric = col6.selectbox("主指标" if ZH else "Primary Metric", metric_options, format_func=lambda x: METRIC_LABELS_ZH.get(x, x) if ZH else x)
    design_metric = METRIC_LABELS_ZH.get(metric, metric) if ZH else metric
    design = design_ab_test(goal, platform_name(platform) if ZH else platform, topic, treatment, control, design_metric, language=language)
    with st.container(border=True):
        st.markdown("#### 可直接执行的实验方案" if ZH else "#### Ready-to-Run Experiment Plan")
        st.write(f"**{'假设' if ZH else 'Hypothesis'}：** {design['hypothesis']}")
        st.write(f"**{'实验组' if ZH else 'Treatment'}：** {design['treatment_group']}")
        st.write(f"**{'对照组' if ZH else 'Control'}：** {design['control_group']}")
        st.write(f"**{'主指标' if ZH else 'Primary Metric'}：** {METRIC_LABELS_ZH.get(design['primary_metric'], design['primary_metric']) if ZH else design['primary_metric']}")
        st.write(f"**{'周期' if ZH else 'Duration'}：** {design['duration_suggestion']}")
        st.write(f"**{'最低样本' if ZH else 'Minimum Sample'}：** {design['minimum_sample_suggestion']}")
        st.markdown("**执行清单：**" if ZH else "**Execution Checklist:**")
        for item in design["execution_plan"]:
            st.write(f"- {item}")
    if ai_enabled and not ab_result.empty:
        if st.button("解读实验结果" if ZH else "Interpret Experiment", width="stretch"):
            with st.spinner("正在解读实验结果..." if ZH else "Interpreting experiment results..."):
                try:
                    st.markdown(interpret_experiment(ab_result, design, language=language))
                except Exception as exc:
                    st.error("暂时未能解读实验结果，请稍后再试。" if ZH else "Experiment interpretation is temporarily unavailable. Please try again.")

with tabs[2]:
    st.subheader("商务合作与收款流程" if ZH else "Sponsorship and Payment Workflow")
    st.caption("管理报价、风险、复盘和收款信息。" if ZH else "Manage pricing, risks, reports and payment details.")
    dashboard = campaign_dashboard(campaigns)
    cols = st.columns(7)
    dashboard_labels = {
        "total_campaigns": "总合作数",
        "active": "进行中",
        "reviewing": "待审核",
        "to_publish": "待发布",
        "pending_payment": "待收款",
        "pending_invoice": "待开票",
        "pending_report": "待复盘",
    }
    for col, (k, v) in zip(cols, dashboard.items()):
        col.metric(dashboard_labels.get(k, k) if ZH else k.replace("_", " ").title(), v)

    alerts = campaign_risk_alerts(campaigns, language=language)
    st.markdown("### 风险提醒" if ZH else "### Risk Alerts")
    suggestion_cards(alerts, language)
    st.dataframe(display_frame(campaigns, language), width="stretch")

    st.markdown("### 报价建议" if ZH else "### Pricing Suggestion")
    p1, p2, p3, p4 = st.columns(4)
    price_platform = p1.selectbox("报价平台" if ZH else "Platform", sorted(contents["platform"].unique()), key="pricing_platform", format_func=platform_name if ZH else lambda x: x)
    deliverables = p2.text_input("交付物" if ZH else "Deliverables", "1篇深度内容 + 1条短视频" if ZH else "1 long-form post + 1 short video")
    base_cost = p3.number_input("制作成本" if ZH else "Production Cost", min_value=0, value=1200, step=100)
    usage_rights = p4.checkbox("素材使用权" if ZH else "Usage Rights")
    exclusive = st.checkbox("竞品排他" if ZH else "Exclusivity")
    urgent = st.checkbox("加急交付" if ZH else "Urgent Delivery")
    price_result = pricing_suggestion(contents, campaigns, price_platform, deliverables, base_cost, usage_rights, exclusive, urgent)
    price_cols = st.columns(3)
    price_cols[0].metric("保守报价" if ZH else "Low", f"¥{price_result['low']:,.0f}")
    price_cols[1].metric("建议报价" if ZH else "Mid", f"¥{price_result['mid']:,.0f}")
    price_cols[2].metric("高位报价" if ZH else "High", f"¥{price_result['high']:,.0f}")
    st.caption(price_result["explanation"] if ZH else englishize_text(price_result["explanation"]))

    st.markdown("### 收款信息" if ZH else "### Payment Info")
    alipay_account = os.getenv("CREATOR_ALIPAY_ACCOUNT", "").strip() or ("未配置" if ZH else "Not configured")
    selected_amount = st.number_input("本次收款金额" if ZH else "Payment Amount", min_value=0, value=int(price_result["mid"]), step=100)
    payment_note = st.text_input("转账备注" if ZH else "Payment Note", "品牌合作定金 / 内容推广服务费" if ZH else "Sponsorship deposit / content promotion service")
    with st.container(border=True):
        st.write("**收款方式：支付宝转账**" if ZH else "**Payment Method: Alipay Transfer**")
        st.write(f"{'收款账号' if ZH else 'Account'}：{alipay_account}")
        st.write(f"{'收款金额' if ZH else 'Amount'}：¥{selected_amount:,.0f}")
        st.write(f"{'备注' if ZH else 'Note'}：{payment_note}")
        st.caption("把这段收款信息发给合作方即可。" if ZH else "Send these payment details to the partner.")

    st.markdown("### 甲方复盘报告" if ZH else "### Brand Report")
    if campaigns.empty:
        st.info("当前平台暂无商务合作数据。可以先使用报价与收款信息。")
    else:
        campaign_id = st.selectbox("选择合作", campaigns["campaign_id"].tolist())
        selected = campaigns[campaigns["campaign_id"].eq(campaign_id)].iloc[0]
        related = contents[contents["content_id"].eq(selected["related_content_id"])]
        metrics = related.iloc[0] if not related.empty else {}
        brand_report = generate_brand_report(selected, metrics)
        st.markdown(brand_report if ZH else englishize_text(brand_report))

with tabs[3]:
    st.subheader("相似内容 / 产品分析" if ZH else "Similar Content / Product Analysis")
    overlap_df = detect_content_overlap(contents)
    variant_df = detect_product_variants(products)
    st.caption("区分重复、同系列、进阶和互补内容；产品侧识别颜色、材料、功能版本差异。" if ZH else "Separates duplicates, series, advanced variants and product versions.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 内容相似风险" if ZH else "#### Content Similarity Risk")
        if overlap_df.empty:
            st.info("暂无相似内容数据。" if ZH else "No content similarity data.")
        else:
            st.dataframe(display_frame(overlap_df.head(20), language), width="stretch")
    with col2:
        st.markdown("#### 产品变体差异" if ZH else "#### Product Variants")
        if variant_df.empty:
            st.info("暂无产品数据。" if ZH else "No product data.")
        else:
            st.dataframe(display_frame(variant_df.head(20), language), width="stretch")


with tabs[4]:
    st.subheader("系列增量分析" if ZH else "Series Incremental Analysis")
    content_series = content_series_performance(contents)
    product_series = product_series_performance(products)
    series_recs = recommend_series_strategy(contents, products)
    suggestion_cards(series_recs[:4], language)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 内容系列" if ZH else "#### Content Series")
        if not content_series.empty:
            st.dataframe(display_frame(content_series, language), width="stretch")
            gain = marginal_gain_of_new_item(contents, "series_id", "publish_time", "revenue")
            st.dataframe(display_frame(gain.tail(20), language), width="stretch")
    with col2:
        st.markdown("#### 产品系列" if ZH else "#### Product Series")
        if not product_series.empty:
            st.dataframe(display_frame(product_series, language), width="stretch")
            product_gain = marginal_gain_of_new_item(products, "series_id", "launch_date", "revenue")
            st.dataframe(display_frame(product_gain.tail(20), language), width="stretch")
    cannibal = cannibalization_check(contents, "series_id", "publish_time", "revenue")
    if not cannibal.empty:
        st.markdown("#### 蚕食风险" if ZH else "#### Cannibalization Risk")
        st.dataframe(display_frame(cannibal, language), width="stretch")


with tabs[5]:
    st.subheader("用户反馈分析" if ZH else "User Feedback Analysis")
    classified_feedback = classify_feedback(feedback)
    roadmap = feedback_to_roadmap(classified_feedback, products, contents)
    topics = feedback_topic_clustering(classified_feedback)
    if classified_feedback.empty:
        st.info("暂无反馈数据。可在设置里上传 feedback.csv。" if ZH else "No feedback data. Upload feedback.csv in settings.")
    else:
        f1, f2, f3 = st.columns(3)
        f1.metric("反馈数" if ZH else "Feedback", len(classified_feedback))
        f2.metric("负面占比" if ZH else "Negative", f"{classified_feedback['sentiment'].eq('negative').mean():.1%}")
        f3.metric("高优先级问题" if ZH else "High Priority", int(roadmap["priority"].eq("high").sum()) if not roadmap.empty else 0)
        st.markdown("#### 下一版优先事项" if ZH else "#### Roadmap")
        st.dataframe(display_frame(roadmap, language), width="stretch")
        st.markdown("#### 高频主题与原文证据" if ZH else "#### Topics and Evidence")
        st.dataframe(display_frame(topics, language), width="stretch")
        link = sentiment_revenue_link(classified_feedback, products)
        if not link.empty:
            st.caption("建议优先验证情绪变化最明显的主题。" if ZH else "Prioritize topics with the clearest sentiment shifts.")
            st.dataframe(display_frame(link, language), width="stretch")


with tabs[6]:
    st.subheader("内测实验设计器" if ZH else "User Test Planner")
    readiness = beta_test_readiness_check(products, feedback, beta_tests)
    st.metric("内测准备度" if ZH else "Test Readiness", f"{readiness['readiness_score']}/100")
    if readiness["warnings"]:
        st.warning("；".join(readiness["warnings"]))
    b1, b2, b3 = st.columns(3)
    feature = b1.text_input("产品/功能" if ZH else "Product / Feature", "SoloBot 情绪陪伴功能" if ZH else "SoloBot emotion companion")
    target_metric = b2.selectbox("目标指标" if ZH else "Target Metric", ["converted", "retained_7d", "revenue", "rating", "activated"])
    segments = b3.text_input("目标用户" if ZH else "Target Users", "creator, office_user")
    beta_plan = design_beta_test(feature, target_metric, segments, {"sample_size": 40})
    with st.container(border=True):
        st.write(f"**{'目标' if ZH else 'Goal'}：** {beta_plan['test_goal']}")
        st.write(f"**{'实验组' if ZH else 'Treatment'}：** {beta_plan['treatment_group']}")
        st.write(f"**{'对照组' if ZH else 'Control'}：** {beta_plan['control_group']}")
        st.write(f"**{'样本建议' if ZH else 'Sample'}：** {beta_plan['sample_size_suggestion']}")
        st.write(f"**{'决策规则' if ZH else 'Decision Rule'}：** {beta_plan['stop_or_scale_rule']}")
    st.markdown("#### 反馈问卷" if ZH else "#### Feedback Form")
    for q in generate_feedback_form("product", feature):
        st.write(f"- {q}")


with tabs[7]:
    st.subheader("新功能 / 新产品推广建议" if ZH else "New Feature / Product Launch Advice")
    st.markdown("### 产品款式收益机会" if ZH else "### Product Model Revenue Opportunities")
    product_cards = product_variant_action_cards(products, beta_tests, language)
    if product_cards:
        suggestion_cards(product_cards, language)
    else:
        st.info("暂无产品或内测数据。上传产品 CSV 后会自动判断主推款式。" if ZH else "No product or user-test data yet. Upload products CSV to identify top models.")

    feature_options = sorted(beta_tests["feature_name"].dropna().unique().tolist()) if not beta_tests.empty and "feature_name" in beta_tests.columns else ["emotion_companion"]
    selected_feature = st.selectbox("选择功能" if ZH else "Feature", feature_options)
    outcome = st.selectbox("结果指标" if ZH else "Outcome", ["converted", "retained_7d", "activated", "revenue", "rating"])
    effect = estimate_feature_upgrade_effect(products, beta_tests, selected_feature, outcome)
    insight = generate_incremental_insight(effect, language)
    render_html(f"<div class='priority-card p0'><div class='priority-title'>{safe_text('推广判断' if ZH else 'Launch Decision')}</div><div class='priority-body'>{safe_text(insight)}</div></div>")
    beta_effect = beta_feedback_effect(beta_tests, feedback)
    if beta_effect.get("effects"):
        st.dataframe(pd.DataFrame(beta_effect["effects"]), width="stretch")
    negative_rate = classify_feedback(feedback)["sentiment"].eq("negative").mean() if not feedback.empty else 0
    next_step = recommend_next_validation_step({"readiness_score": beta_test_readiness_check(products, feedback, beta_tests)["readiness_score"], "lift": effect.get("effect_estimate", effect.get("mean_difference", 0)), "negative_rate": negative_rate})
    st.success(f"{'推荐动作' if ZH else 'Recommended Action'}：{next_step['recommended_action']}。{next_step['reason']}")


with tabs[8]:
    st.subheader("周报生成" if ZH else "Report Generator")
    report_type = st.radio("报告类型" if ZH else "Report Type", ["经营周报", "策略报告"] if ZH else ["Business Report", "Strategy Report"], horizontal=True)
    if report_type in ["经营周报", "Business Report"]:
        report = generate_weekly_business_report(contents, revenues, campaigns, language=language)
    else:
        report = generate_strategy_report(contents, ab_tests, language=language)
    st.markdown("### 简短总结" if ZH else "### Short Summary")
    st.write(concise_summary(contents, revenues, campaigns, language))
    if ai_enabled:
        if st.button("生成精简顾问报告" if ZH else "Generate Executive Report", width="stretch"):
            with st.spinner("正在生成报告..." if ZH else "Generating the report..."):
                try:
                    report = polish_report(report, language=language)
                except Exception as exc:
                    st.error("暂时未能生成报告，请稍后再试。" if ZH else "The report is temporarily unavailable. Please try again.")
    if not ZH:
        report = englishize_text(report)
    st.download_button("下载报告" if ZH else "Download Report", report, file_name="creator_agent_report.md")
    with st.expander("查看报告全文" if ZH else "View Full Report", expanded=True):
        st.markdown(report)
