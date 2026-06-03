from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .business_collab import campaign_risk_alerts
from .causal_estimator import (
    _design_matrix,
    _to_binary,
    causal_forest_effect,
    did_effect,
    dml_effect,
    interrupted_time_series,
    iptw_effect,
    marginal_structural_model,
    propensity_score_matching,
    stratified_regression_effect,
    target_trial_emulation,
)
from .causal_experiment import analyze_ab_test
from .incremental_effect import estimate_feature_upgrade_effect, estimate_incremental_effect, fixed_effect_estimate, paired_variant_effect
from .product_feedback import beta_feedback_effect, classify_feedback, feedback_to_roadmap
from .revenue_analysis import pending_payment_summary, platform_revenue_summary, topic_business_summary
from .series_analysis import cannibalization_check, content_series_performance, product_series_performance
from .similarity_engine import detect_content_overlap, detect_product_variants
from .strategy_analysis import platform_strategy_analysis, title_style_analysis, topic_strategy_analysis
from .text_structured import extract_text_features


MODULE_NAMES_ZH = {
    "data_parser_agent": "资料整理",
    "causal_estimator_agent": "因果估计",
    "strategy_simulation_agent": "策略仿真",
    "feedback_analysis_agent": "用户反馈",
    "revenue_analysis_agent": "收入与商务",
    "ab_test_agent": "实验效果",
}

MODULE_NAMES_EN = {
    "data_parser_agent": "Data Parsing",
    "causal_estimator_agent": "Causal Estimation",
    "strategy_simulation_agent": "Strategy Simulation",
    "feedback_analysis_agent": "Feedback Priorities",
    "revenue_analysis_agent": "Revenue and Business",
    "ab_test_agent": "Experiment Analysis",
}


@dataclass
class InsightCard:
    agent: str
    finding_type: str
    title: str
    insight: str
    action: str
    priority: str = "medium"
    confidence: str = "exploratory"
    metric: str = ""
    effect: float | None = None
    sample_size: int | None = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _rate(num: pd.Series, den: pd.Series) -> pd.Series:
    return num.astype(float) / den.replace(0, np.nan).astype(float)


def _label(lang: str, zh: str, en: str) -> str:
    return zh if lang == "中文" else en


def _platform_label(platform: str, lang: str) -> str:
    if lang != "中文":
        return str(platform)
    return {
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
    }.get(str(platform), str(platform))


def _style_label(style: str, lang: str) -> str:
    if lang != "中文":
        return str(style).replace("_", " ")
    return {
        "pain_point": "痛点标题",
        "tutorial": "教程标题",
        "number": "数字清单标题",
        "story": "故事标题",
        "contrast": "对比标题",
        "result_oriented": "结果导向标题",
        "question": "提问标题",
        "case_study": "案例复盘",
        "listicle": "清单内容",
        "early_cta": "前置咨询入口",
        "ending_cta": "结尾咨询入口",
    }.get(str(style), str(style))


def _feature_label(feature: str, lang: str) -> str:
    if lang != "中文":
        return str(feature).replace("_", " ")
    return {
        "emotion_companion": "情绪陪伴功能",
        "touch_interaction": "触摸互动功能",
        "voice_interaction": "语音互动",
        "alarm": "提醒功能",
        "music_playback": "音乐播放",
        "desktop_decoration": "桌面陪伴摆件",
    }.get(str(feature), str(feature))


def _confidence(n: int, effect: float | None = None, low: float | None = None, high: float | None = None, lang: str = "中文") -> str:
    if n < 20:
        return _label(lang, "低：样本偏少，仅作探索", "Low: small sample, exploratory")
    if low is not None and high is not None and low <= 0 <= high:
        return _label(lang, "中：方向未完全稳定", "Medium: direction not fully stable")
    if n >= 60:
        return _label(lang, "较高：可进入小规模放大", "Higher: ready for limited scaling")
    return _label(lang, "中：建议继续验证", "Medium: continue validation")


def _fast_effect(data: pd.DataFrame, treatment_col: str, outcome_col: str, covariates: list[str]) -> dict[str, Any]:
    df = data.dropna(subset=[treatment_col, outcome_col]).copy()
    if df.empty or df[treatment_col].nunique() < 2:
        return {"method": "fast_fixed_effect", "effect_estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "sample_size": len(df), "warnings": ["缺少对照组或样本为空。"]}
    t = _to_binary(df[treatment_col]).rename("treatment")
    x = pd.concat([t, _design_matrix(df, covariates)], axis=1).astype(float)
    y = df[outcome_col].astype(float)
    model = LinearRegression().fit(x, y)
    effect = float(model.coef_[0])
    return {
        "method": "fast_fixed_effect",
        "effect_estimate": effect,
        "ci_low": effect,
        "ci_high": effect,
        "sample_size": int(len(df)),
        "warnings": ["快速估计用于工作台即时反馈；区间估计请在专业分析中查看。"],
    }


def _fast_feature_effect(products: pd.DataFrame, beta_tests: pd.DataFrame, feature: str, outcome: str) -> dict[str, Any]:
    if not beta_tests.empty and {"feature_name", "test_group", outcome}.issubset(beta_tests.columns):
        df = beta_tests[beta_tests["feature_name"].fillna("").astype(str).str.contains(feature, case=False, regex=False)].copy()
        if not df.empty and df["test_group"].nunique() >= 2:
            t = df[df["test_group"].eq("treatment")][outcome].astype(float)
            c = df[df["test_group"].eq("control")][outcome].astype(float)
            return {
            "method": "quick_user_test_lift",
                "mean_difference": float(t.mean() - c.mean()) if len(t) and len(c) else 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "pair_count": int(len(df)),
                "warning": "快速内测均值差；完整区间请在专业分析中查看。",
            }
    if not products.empty and {"is_new_version", outcome}.issubset(products.columns):
        new = products[products["is_new_version"].astype(bool)][outcome].astype(float)
        old = products[~products["is_new_version"].astype(bool)][outcome].astype(float)
        return {
            "method": "quick_product_version_diff",
            "mean_difference": float(new.mean() - old.mean()) if len(new) and len(old) else 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "pair_count": int(len(products)),
            "warning": "快速新旧版本均值差；需要继续内测验证。",
        }
    return {"method": "no_data", "mean_difference": 0.0, "pair_count": 0, "warning": "缺少产品或内测结果指标。"}


def find_synthetic_data_dir(project_root: Path) -> Path | None:
    candidates = [
        project_root.parent / "data" / "solodeck_synthetic",
        project_root.parent / "data",
        project_root / "data",
    ]
    required = {"mock_contents.csv", "mock_revenues.csv", "mock_campaigns.csv", "mock_ab_tests.csv"}
    for candidate in candidates:
        if candidate.exists() and required.issubset({p.name for p in candidate.glob("*.csv")}):
            return candidate
    return None


def data_parser_agent(
    contents: pd.DataFrame,
    products: pd.DataFrame,
    feedback: pd.DataFrame,
    revenues: pd.DataFrame,
    campaigns: pd.DataFrame,
    ab_tests: pd.DataFrame,
    beta_tests: pd.DataFrame,
    lang: str = "中文",
) -> dict[str, Any]:
    cards: list[InsightCard] = []
    total_rows = sum(len(x) for x in [contents, products, feedback, revenues, campaigns, ab_tests, beta_tests])
    available = []
    for name, df in [
        (_label(lang, "内容", "content"), contents),
        (_label(lang, "产品", "products"), products),
        (_label(lang, "反馈", "feedback"), feedback),
        (_label(lang, "收入", "revenue"), revenues),
        (_label(lang, "商务", "campaigns"), campaigns),
        (_label(lang, "实验", "experiments"), ab_tests),
        (_label(lang, "内测", "user tests"), beta_tests),
    ]:
        if not df.empty:
            available.append(f"{name} {len(df)}")
    if False and available:
        cards.append(InsightCard(
            agent="data_parser_agent",
            finding_type=_label(lang, "资料整理", "Data understanding"),
            title=_label(lang, "资料已整理", "Workspace is ready"),
            insight=_label(lang, f"已读取 {'、'.join(available)}，共 {total_rows} 条记录。", f"Loaded {', '.join(available)}; {total_rows} records in total."),
            action=_label(lang, "系统会根据这些资料生成经营动作。", "SoloDeck will turn these materials into actions."),
            priority="high",
            confidence="",
            sample_size=total_rows,
        ))
    text_features = extract_text_features(contents) if not contents.empty else pd.DataFrame()
    return {"cards": [asdict(c) for c in cards], "text_features": text_features}


def causal_estimator_agent(contents: pd.DataFrame, products: pd.DataFrame, beta_tests: pd.DataFrame, lang: str = "中文") -> dict[str, Any]:
    cards: list[InsightCard] = []
    ate_reports: list[dict[str, Any]] = []
    cate_reports: list[dict[str, Any]] = []

    if not contents.empty and {"title_style", "views"}.issubset(contents.columns):
        covariates = [c for c in ["platform", "topic", "account_id", "followers_before", "production_hours", "ad_spend"] if c in contents.columns]
        fixed = [c for c in ["account_id", "platform"] if c in contents.columns]
        result = _fast_effect(contents, "title_style", "views", fixed + covariates)
        ate_reports.append({"question": "title_style -> views", **result})
        ate_reports.append({
            "question": "advanced_methods_available",
            "method": "专业增量模型",
            "effect": 0.0,
            "sample_size": int(len(contents)),
            "confidence": "按需运行",
            "interpretation": "主工作台默认先给快速经营判断；需要时可继续运行更完整的专业估计。",
        })
        effect = _safe_float(result.get("effect_estimate"))
        cards.append(InsightCard(
            agent="causal_estimator_agent",
            finding_type=_label(lang, "估计因果", "Estimated causal effect"),
            title=_label(lang, "标题风格是否真的带来增量", "Does title style add lift?"),
            insight=_label(
                lang,
                f"控制账号、平台和主题后，标题风格对播放的估计增量为 {effect:.1f}。",
                f"After controlling account, platform and topic, estimated title-style lift on views is {effect:.1f}.",
            ),
            action=_label(lang, "不要只看均值排行；下周在同一账号、同一平台内做配对实验。", "Do not scale by averages alone; run a paired test within the same account and platform."),
            priority="high" if effect > 0 else "medium",
            confidence=_confidence(int(result.get("sample_size", 0)), effect, result.get("ci_low"), result.get("ci_high"), lang),
            metric="views",
            effect=effect,
            sample_size=int(result.get("sample_size", 0)),
        ))

    if not contents.empty and {"content_group_id", "platform", "revenue"}.issubset(contents.columns):
        paired = paired_variant_effect(contents, "content_group_id", "platform", "revenue")
        ate_reports.append({"question": "same_content_cross_platform -> revenue", **paired})
        diff = _safe_float(paired.get("mean_difference"))
        cards.append(InsightCard(
            agent="causal_estimator_agent",
            finding_type=_label(lang, "配对估计", "Paired estimate"),
            title=_label(lang, "同一内容跨平台差异", "Same-content platform lift"),
            insight=_label(lang, f"按同内容配对后，平台差异带来的平均收入差为 {diff:.1f}。", f"Within same-content pairs, average platform revenue difference is {diff:.1f}."),
            action=_label(lang, "优先复制同内容多平台发布，再比较 72 小时和 7 天窗口。", "Republish the same content across platforms first, then compare 72h and 7d windows."),
            priority="medium",
            confidence=_confidence(int(paired.get("pair_count", 0)), diff, paired.get("ci_low"), paired.get("ci_high"), lang),
            metric="revenue",
            effect=diff,
            sample_size=int(paired.get("pair_count", 0)),
        ))

    if not products.empty:
        feature = ""
        if not beta_tests.empty and "feature_name" in beta_tests.columns and not beta_tests["feature_name"].dropna().empty:
            feature = str(beta_tests["feature_name"].dropna().mode().iloc[0])
        if not feature and "feature_tags" in products.columns:
            feature = str(products["feature_tags"].dropna().astype(str).str.split(",").explode().mode().iloc[0]) if not products["feature_tags"].dropna().empty else ""
        if feature:
            feature_result = _fast_feature_effect(products, beta_tests, feature, "revenue" if "revenue" in beta_tests.columns else "converted")
            ate_reports.append({"question": f"{feature} -> revenue/converted", **feature_result})
            effect = _safe_float(feature_result.get("effect_estimate", feature_result.get("mean_difference")))
            cards.append(InsightCard(
                agent="causal_estimator_agent",
                finding_type=_label(lang, "功能增量", "Feature lift"),
                title=_label(lang, f"{_feature_label(feature, lang)}是否值得放大", f"Should {_feature_label(feature, lang)} scale?"),
                insight=_label(lang, f"当前估计增量为 {effect:.2f}，结果来自产品/内测数据。", f"Current estimated lift is {effect:.2f}, based on product/user-test data."),
                action=_label(lang, "先扩大到相似人群的小批内测，再决定是否主推。", "Expand to a similar small user-test cohort before making it the hero feature."),
                priority="high" if effect > 0 else "medium",
                confidence=_confidence(int(feature_result.get("sample_size", feature_result.get("pair_count", 0))), effect, feature_result.get("ci_low"), feature_result.get("ci_high"), lang),
                metric="revenue",
                effect=effect,
                sample_size=int(feature_result.get("sample_size", feature_result.get("pair_count", 0))),
            ))

    if not contents.empty and {"topic", "title_style", "revenue"}.issubset(contents.columns):
        top_topics = contents.groupby("topic")["revenue"].sum().sort_values(ascending=False).head(3).index
        for topic, group in contents[contents["topic"].isin(top_topics)].groupby("topic"):
            if len(group) >= 6 and group["title_style"].nunique() >= 2:
                res = _fast_effect(group, "title_style", "revenue", [c for c in ["platform", "account_id", "followers_before"] if c in group.columns])
                cate_reports.append({"segment": str(topic), "outcome": "revenue", **res})

    return {"cards": [asdict(c) for c in cards], "ate": ate_reports, "cate": cate_reports}


def strategy_simulation_agent(contents: pd.DataFrame, revenues: pd.DataFrame, products: pd.DataFrame, lang: str = "中文") -> dict[str, Any]:
    cards: list[InsightCard] = []
    simulations: list[dict[str, Any]] = []
    if not contents.empty:
        platform_result = platform_strategy_analysis(contents, revenues)["table"]
        topic_result = topic_strategy_analysis(contents)["table"]
        title_table = title_style_analysis(contents)["table"]
        if not platform_result.empty and not topic_result.empty and not title_table.empty:
            best_platform = platform_result.sort_values("revenue_per_hour", ascending=False).iloc[0]
            best_topic = topic_result.sort_values("revenue_per_hour", ascending=False).iloc[0]
            best_style = title_table.sort_values("conversion_rate", ascending=False).iloc[0]
            platform = _platform_label(best_platform["platform"], lang)
            topic = str(best_topic["topic"])
            style = _style_label(best_style["title_style"], lang)
            simulations.append({
                "strategy": "platform_topic_title",
                "platform": platform,
                "topic": topic,
                "title_style": style,
                "expected_metric": "revenue_per_hour",
                "estimated_value": _safe_float(best_topic["revenue_per_hour"]),
            })
            cards.append(InsightCard(
                agent="strategy_simulation_agent",
                finding_type=_label(lang, "策略仿真", "Strategy simulation"),
                title=_label(lang, "下周先测这一组组合", "Test this combination next"),
                insight=_label(lang, f"历史上 {platform} + {topic} + {style} 更接近高单位收益组合。", f"Historically, {platform} + {topic} + {style} is closer to a high revenue-per-hour mix."),
                action=_label(lang, f"连续 10-14 天发布 4-6 条同主题内容，每条只改一个变量，72 小时记录收藏/咨询/成交。", "Publish 4-6 same-topic items over 10-14 days, change only one variable each time, and record saves/consultations/sales after 72h."),
                priority="high",
                confidence=_label(lang, "相关性发现，需要实验验证", "Correlation finding; validate with an experiment"),
                metric="revenue_per_hour",
                effect=_safe_float(best_topic["revenue_per_hour"]),
                sample_size=int(best_topic.get("content_count", 0)),
            ))

    if not products.empty and {"feature_tags", "revenue"}.issubset(products.columns):
        exploded = products.assign(feature=products["feature_tags"].fillna("").astype(str).str.split(",")).explode("feature")
        feature_table = exploded.groupby("feature", as_index=False).agg(revenue=("revenue", "sum"), conversions=("conversions", "sum"), avg_rating=("avg_rating", "mean"))
        feature_table = feature_table[feature_table["feature"].astype(str).str.len().gt(0)].sort_values("revenue", ascending=False)
        if not feature_table.empty:
            top = feature_table.iloc[0]
            cards.append(InsightCard(
                agent="strategy_simulation_agent",
                finding_type=_label(lang, "产品策略", "Product strategy"),
                title=_label(lang, f"主推 {_feature_label(top['feature'], lang)}", f"Prioritize {_feature_label(top['feature'], lang)}"),
                insight=_label(lang, f"带该功能的产品累计收入 {top['revenue']:.0f}，成交 {int(top['conversions'])}。", f"Products with this feature generated {top['revenue']:.0f} revenue and {int(top['conversions'])} sales."),
                action=_label(lang, "商品页首屏突出该功能，并用 30 人内测验证购买意向。", "Feature it above the fold and validate purchase intent with a 30-user test."),
                priority="high",
                confidence=_label(lang, "相关性发现，需要内测验证", "Correlation finding; validate with user tests"),
                metric="revenue",
                effect=_safe_float(top["revenue"]),
                sample_size=int(len(products)),
            ))
    return {"cards": [asdict(c) for c in cards], "simulations": simulations}


def feedback_analysis_agent(feedback: pd.DataFrame, products: pd.DataFrame, contents: pd.DataFrame, lang: str = "中文") -> dict[str, Any]:
    cards: list[InsightCard] = []
    classified = classify_feedback(feedback)
    roadmap = feedback_to_roadmap(classified, products, contents)
    if not roadmap.empty:
        top = roadmap.iloc[0]
        cards.append(InsightCard(
            agent="feedback_analysis_agent",
            finding_type=_label(lang, "用户反馈", "Feedback"),
            title=_label(lang, f"下一版先处理：{top['issue']}", f"Fix first: {top['issue']}"),
            insight=_label(lang, f"{int(top['evidence_count'])} 条反馈指向该问题，业务影响：{top['business_impact']}。", f"{int(top['evidence_count'])} feedback items point to this issue. Business impact: {top['business_impact']}."),
            action=str(top["suggested_action"]),
            priority="high" if top["priority"] == "high" else "medium",
            confidence=_label(lang, "用户反馈证据", "User feedback evidence"),
            sample_size=int(top["evidence_count"]),
        ))
    return {"cards": [asdict(c) for c in cards], "roadmap": roadmap, "classified_feedback": classified}


def revenue_analysis_agent(contents: pd.DataFrame, revenues: pd.DataFrame, campaigns: pd.DataFrame, lang: str = "中文") -> dict[str, Any]:
    cards: list[InsightCard] = []
    platform = platform_revenue_summary(revenues)
    topic = topic_business_summary(contents) if not contents.empty else pd.DataFrame()
    pending = pending_payment_summary(revenues, campaigns) if not revenues.empty or not campaigns.empty else {"total_pending_amount": 0, "overdue_campaigns": []}
    risks = campaign_risk_alerts(campaigns, language=lang) if not campaigns.empty else []
    if not platform.empty:
        top = platform.sort_values("amount", ascending=False).iloc[0]
        cards.append(InsightCard(
            agent="revenue_analysis_agent",
            finding_type=_label(lang, "商业机会", "Commercial opportunity"),
            title=_label(lang, f"收入主阵地：{_platform_label(top['platform'], lang)}", f"Revenue anchor: {_platform_label(top['platform'], lang)}"),
            insight=_label(lang, f"该平台收入 {top['amount']:.0f}，占比 {top['share']:.1%}。", f"This platform generated {top['amount']:.0f}, {top['share']:.1%} of revenue."),
            action=_label(lang, "把商单落地页、咨询入口和复购入口优先放到这个平台。", "Put sponsorship landing pages, consultation CTA and repeat-purchase paths on this platform first."),
            priority="high",
            confidence=_label(lang, "收入流水证据", "Revenue evidence"),
            metric="revenue",
            effect=_safe_float(top["amount"]),
            sample_size=int(len(revenues)),
        ))
    if pending.get("total_pending_amount", 0):
        cards.append(InsightCard(
            agent="revenue_analysis_agent",
            finding_type=_label(lang, "现金流提醒", "Cash-flow alert"),
            title=_label(lang, "先处理待收款", "Handle receivables first"),
            insight=_label(lang, f"待收款合计 {pending['total_pending_amount']:.0f}。", f"Pending payments total {pending['total_pending_amount']:.0f}."),
            action=_label(lang, "今天发付款确认；逾期合作同步补发票、合同和复盘节点。", "Send payment confirmation today; for overdue deals, align invoice, contract and report milestones."),
            priority="high",
            confidence=_label(lang, "商务状态证据", "Business status evidence"),
            metric="pending_payment",
            effect=_safe_float(pending["total_pending_amount"]),
            sample_size=len(risks),
        ))
    return {"cards": [asdict(c) for c in cards], "platform_revenue": platform, "topic_business": topic, "pending": pending, "risks": risks}


def ab_test_agent(ab_tests: pd.DataFrame, beta_tests: pd.DataFrame, lang: str = "中文") -> dict[str, Any]:
    cards: list[InsightCard] = []
    ab_result = analyze_ab_test(ab_tests, language=lang) if not ab_tests.empty else pd.DataFrame()
    if not ab_result.empty:
        best = ab_result.sort_values("relative_lift", ascending=False).iloc[0]
        cards.append(InsightCard(
            agent="ab_test_agent",
            finding_type=_label(lang, "实验结果", "Experiment result"),
            title=_label(lang, f"{best['experiment_id']} 表现最好", f"{best['experiment_id']} performs best"),
            insight=_label(lang, f"{best['outcome_metric']} 相对提升 {best['relative_lift']:.1%}，置信区间 [{best['ci_low']:.3f}, {best['ci_high']:.3f}]。", f"{best['outcome_metric']} relative lift is {best['relative_lift']:.1%}, CI [{best['ci_low']:.3f}, {best['ci_high']:.3f}]."),
            action=_label(lang, "如果区间稳定高于 0，扩大到下一批；否则只保留为候选策略。", "If the interval stays above 0, scale to the next batch; otherwise keep it as a candidate."),
            priority="high" if best["ci_low"] > 0 else "medium",
            confidence=_confidence(int(len(ab_tests)), _safe_float(best["absolute_lift"]), best["ci_low"], best["ci_high"], lang),
            metric=str(best["outcome_metric"]),
            effect=_safe_float(best["absolute_lift"]),
            sample_size=int(len(ab_tests)),
        ))
    beta = beta_feedback_effect(beta_tests, pd.DataFrame()) if not beta_tests.empty else {"effects": []}
    if beta.get("effects"):
        effect_df = pd.DataFrame(beta["effects"]).sort_values("lift", ascending=False)
        top = effect_df.iloc[0]
        cards.append(InsightCard(
            agent="ab_test_agent",
            finding_type=_label(lang, "内测提升", "User-test lift"),
            title=_label(lang, f"内测提升最明显：{top['outcome']}", f"Strongest user-test lift: {top['outcome']}"),
            insight=_label(lang, f"实验组均值 {top['treatment_mean']:.2f}，对照组 {top['control_mean']:.2f}，净提升 {top['lift']:.2f}。", f"Treatment mean {top['treatment_mean']:.2f}, control {top['control_mean']:.2f}, lift {top['lift']:.2f}."),
            action=_label(lang, "把该指标作为下一轮内测主指标，扩大样本后再决定发布。", "Use this as the primary metric in the next user-test wave before release."),
            priority="high" if top["lift"] > 0 else "medium",
            confidence=_confidence(int(len(beta_tests)), _safe_float(top["lift"]), lang=lang),
            metric=str(top["outcome"]),
            effect=_safe_float(top["lift"]),
            sample_size=int(len(beta_tests)),
        ))
    return {"cards": [asdict(c) for c in cards], "ab_results": ab_result, "beta_effects": beta}


def run_agent_suite(
    contents: pd.DataFrame,
    revenues: pd.DataFrame,
    campaigns: pd.DataFrame,
    ab_tests: pd.DataFrame,
    products: pd.DataFrame,
    feedback: pd.DataFrame,
    beta_tests: pd.DataFrame,
    lang: str = "中文",
) -> dict[str, Any]:
    modules = {
        "data_parser_agent": data_parser_agent(contents, products, feedback, revenues, campaigns, ab_tests, beta_tests, lang),
        "revenue_analysis_agent": revenue_analysis_agent(contents, revenues, campaigns, lang),
        "strategy_simulation_agent": strategy_simulation_agent(contents, revenues, products, lang),
        "causal_estimator_agent": causal_estimator_agent(contents, products, beta_tests, lang),
        "ab_test_agent": ab_test_agent(ab_tests, beta_tests, lang),
        "feedback_analysis_agent": feedback_analysis_agent(feedback, products, contents, lang),
    }
    cards = []
    for name, result in modules.items():
        for card in result.get("cards", []):
            card["agent_label"] = (MODULE_NAMES_ZH if lang == "中文" else MODULE_NAMES_EN).get(name, name)
            cards.append(card)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    cards = sorted(cards, key=lambda c: priority_order.get(c.get("priority", "medium"), 1))

    overlap = detect_content_overlap(contents) if not contents.empty else pd.DataFrame()
    variants = detect_product_variants(products) if not products.empty else pd.DataFrame()
    content_series = content_series_performance(contents) if not contents.empty else pd.DataFrame()
    product_series = product_series_performance(products) if not products.empty else pd.DataFrame()
    cannibal = cannibalization_check(contents, "series_id", "publish_time", "revenue") if not contents.empty and "series_id" in contents.columns else pd.DataFrame()

    report = generate_unified_report(cards, modules, content_series, product_series, cannibal, lang)
    return {
        "cards": cards,
        "modules": modules,
        "overlap": overlap,
        "product_variants": variants,
        "content_series": content_series,
        "product_series": product_series,
        "cannibalization": cannibal,
        "report": report,
    }


def generate_unified_report(cards: list[dict[str, Any]], modules: dict[str, Any], content_series: pd.DataFrame, product_series: pd.DataFrame, cannibal: pd.DataFrame, lang: str = "中文") -> str:
    if lang != "中文":
        lines = ["# SoloDeck Operating Report", "", "## Next Actions"]
        for card in cards[:6]:
            lines.append(f"- **{card['title']}**: {card['action']}")
        lines.extend(["", "## Causal and Experiment Notes"])
        for item in modules.get("causal_estimator_agent", {}).get("ate", [])[:4]:
            lines.append(f"- {item.get('question')}: effect {item.get('effect_estimate', item.get('mean_difference', 0))}")
        return "\n".join(lines)

    lines = [
        "# SoloDeck 经营分析报告",
        "",
        "## 先做什么",
    ]
    for card in cards[:6]:
        lines.append(f"- **{card['title']}**：{card['action']}")
    lines.extend(["", "## 因果与实验结论"])
    for item in modules.get("causal_estimator_agent", {}).get("ate", [])[:5]:
        effect = item.get("effect_estimate", item.get("mean_difference", item.get("effect", 0)))
        question = str(item.get("question", "")).replace("title_style -> views", "标题风格对播放量").replace("same_content_cross_platform -> revenue", "同内容跨平台收入差异").replace("advanced_methods_available", "专业估计能力")
        lines.append(f"- {question}：估计增量 {effect:.2f}。这是统计估计，需要用小实验继续验证。")
    ab = modules.get("ab_test_agent", {}).get("ab_results", pd.DataFrame())
    if isinstance(ab, pd.DataFrame) and not ab.empty:
        best = ab.sort_values("relative_lift", ascending=False).iloc[0]
        lines.append(f"- 最强实验：{best['experiment_id']}，{best['outcome_metric']} 相对提升 {best['relative_lift']:.1%}。")
    lines.extend(["", "## 系列与产品风险"])
    if isinstance(content_series, pd.DataFrame) and not content_series.empty:
        row = content_series.iloc[0]
        lines.append(f"- 内容系列 {row['series_id']}：总收入 ¥{row['total_revenue']:.0f}，疲劳风险 {row['fatigue_warning']}。")
    if isinstance(product_series, pd.DataFrame) and not product_series.empty:
        row = product_series.iloc[0]
        lines.append(f"- 产品系列 {row['series_id']}：最强款式 {row['strongest_variant']}，退款率 {row['refund_rate']:.1%}。")
    if isinstance(cannibal, pd.DataFrame) and not cannibal.empty:
        high = cannibal[cannibal["cannibalization_risk"].isin(["high", "medium"])].head(3)
        for _, row in high.iterrows():
            lines.append(f"- {row['series_id']}：{row['explanation']}")
    lines.extend(["", "## 说明", "SoloDeck 区分相关性发现、估计因果和实验结果。建议先用低成本实验验证，再放大投入。"])
    return "\n".join(lines)


def analysis_to_json(result: dict[str, Any]) -> str:
    def convert(value: Any):
        if isinstance(value, pd.DataFrame):
            return value.to_dict("records")
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    return json.dumps(convert(result), ensure_ascii=False, indent=2, default=str)
