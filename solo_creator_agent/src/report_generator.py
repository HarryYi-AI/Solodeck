import pandas as pd

from .auto_insights import generate_auto_insights, insight_markdown
from .business_collab import campaign_risk_alerts
from .causal_interface import run_causal_closed_loop
from .causal_estimator import stratified_regression_effect
from .causal_experiment import analyze_ab_test, design_ab_test
from .causal_refute import refute_suite
from .experiment_planner import weekly_experiment_plan as weekly_experiment_candidates
from .recommendation_engine import revenue_recommendations
from .revenue_analysis import classify_revenue_summary, content_commercial_value
from .strategy_analysis import platform_strategy_analysis, publish_time_analysis, title_style_analysis, topic_strategy_analysis, weekly_topic_plan


def _md_table(df: pd.DataFrame, max_rows: int = 8, language: str = "中文") -> str:
    if df is None or df.empty:
        return "暂无数据" if language == "中文" else "No data"
    out = df.head(max_rows).copy()
    if language == "中文":
        labels = {
            "title": "标题",
            "platform": "平台",
            "topic": "主题",
            "views": "播放/阅读",
            "favorites": "收藏",
            "new_followers": "新增粉丝",
            "consultations": "咨询",
            "conversions": "成交",
            "revenue": "收入",
            "commercial_score": "商业评分",
            "insight": "说明",
            "revenue_type": "收入类型",
            "amount": "金额",
            "share": "占比",
            "content_count": "内容数",
            "favorite_rate": "收藏率",
            "follow_rate": "转粉率",
            "revenue_per_hour": "单位时间收益",
            "weekday": "星期",
            "hour": "小时",
            "sample_size": "样本量",
            "avg_views": "平均播放",
            "confidence": "可信度",
        }
        out = out.rename(columns={k: v for k, v in labels.items() if k in out.columns})
    return out.to_markdown(index=False)


def generate_weekly_business_report(contents: pd.DataFrame, revenues: pd.DataFrame, campaigns: pd.DataFrame, language: str = "中文") -> str:
    zh = language == "中文"
    value_df = content_commercial_value(contents, revenues, language=language)
    revenue_df = classify_revenue_summary(revenues)
    auto_insights = generate_auto_insights(contents, language=language, max_items=5)
    recs = revenue_recommendations(contents, revenues, language=language)
    risks = campaign_risk_alerts(campaigns, language=language)
    plan = weekly_topic_plan(contents, revenues, n=5, language=language)
    week_contents = contents.sort_values("publish_time", ascending=False).head(7)

    rec_lines = "\n".join([f"- **{r['title']}**：{r['action']}｜{r['reason']}" if zh else f"- **{r['title']}**: {r['action']} | {r['reason']}" for r in recs[:5]])
    risk_lines = "\n".join([f"- [{r['level']}] {r['title']}：{r['detail']}" if zh else f"- [{r['level']}] {r['title']}: {r['detail']}" for r in risks[:6]]) or ("- 暂无高风险商务事项。" if zh else "- No high-risk business items.")
    plan_lines = "\n".join([f"- {p['suggested_platform']}｜{p['objective']}｜{p['sample_title']}" for p in plan])

    if not zh:
        return f"""# SoloDeck Weekly Business Report

## Content Performance This Week
{_md_table(week_contents[['title', 'platform', 'topic', 'views', 'favorites', 'new_followers', 'consultations', 'conversions', 'revenue']], language=language)}

## Automatic Data Understanding
{insight_markdown(auto_insights, language=language)}

## Revenue Mix
{_md_table(revenue_df, language=language)}

## High-Value Content
{_md_table(value_df[['title', 'platform', 'topic', 'commercial_score', 'revenue', 'insight']], 5, language=language)}

## Next Week Topic Plan
{plan_lines}

## Sponsorship Risks
{risk_lines}

## Top Actions
{rec_lines}
"""

    return f"""# 自媒体经营周报

## 本周内容表现
{_md_table(week_contents[['title', 'platform', 'topic', 'views', 'favorites', 'new_followers', 'consultations', 'conversions', 'revenue']])}

## 系统看到的重点
{insight_markdown(auto_insights, language=language)}

## 收入结构
{_md_table(revenue_df)}

## 高价值内容
{_md_table(value_df[['title', 'platform', 'topic', 'commercial_score', 'revenue', 'insight']], 5)}

## 下周选题计划
{plan_lines}

## 商务合作风险
{risk_lines}

## 本周最该做
{rec_lines}
"""


def generate_strategy_report(contents: pd.DataFrame, ab_tests: pd.DataFrame, language: str = "中文") -> str:
    zh = language == "中文"
    title_result = title_style_analysis(contents)
    topic_result = topic_strategy_analysis(contents)
    platform_result = platform_strategy_analysis(contents, pd.DataFrame(columns=["platform", "amount"]))
    time_df = publish_time_analysis(contents)
    ab_df = analyze_ab_test(ab_tests, language=language)
    first_topic = topic_result["table"].iloc[0]["topic"] if not topic_result["table"].empty else "核心主题"
    first_platform = platform_result["table"].iloc[0]["platform"] if not platform_result["table"].empty else "xiaohongshu"
    experiment = design_ab_test("提升转化" if zh else "Improve conversion", first_platform, first_topic, "pain_point 标题" if zh else "pain point title", "tutorial 标题" if zh else "tutorial title", "conversion_rate", language=language)
    treatment = "title_style" if "title_style" in contents.columns else "platform"
    outcome = "views" if "views" in contents.columns else "revenue"
    covariates = [c for c in ["platform", "topic", "followers_before", "ad_spend"] if c in contents.columns and c != treatment]
    causal = stratified_regression_effect(contents, treatment, outcome, covariates, n_boot=60)
    refute = refute_suite(contents, treatment, outcome, covariates)
    causal_loop = run_causal_closed_loop(contents, "验证内容策略是否带来增量" if zh else "Validate strategy lift", treatment, outcome, covariates, language=language)
    auto_experiments = weekly_experiment_candidates(contents, language=language)
    auto_lines = "\n".join([
        f"- {p['hypothesis']}｜估计增量 {p['estimated_effect']:.2f}｜建议样本 {p['suggested_min_total_sample']}"
        if zh else f"- {p['hypothesis']} | effect {p['estimated_effect']:.2f} | suggested sample {p['suggested_min_total_sample']}"
        for p in auto_experiments[:3]
    ])

    if not zh:
        return f"""# SoloDeck Strategy Report

## Title Strategy
Confidence: {title_result['confidence']}. Validate the winning pattern before scaling.
{_md_table(title_result['table'], language=language)}

## Topic Strategy
{_md_table(topic_result['table'][['topic', 'content_count', 'views', 'favorite_rate', 'follow_rate', 'consultations', 'conversions', 'revenue', 'revenue_per_hour']], language=language)}

## Platform Strategy
Platform roles: {platform_result['platform_roles']}
{_md_table(platform_result['table'], language=language)}

## Publishing Time Strategy
Low-sample slots should be validated before scaling.
{_md_table(time_df[['platform', 'weekday', 'hour', 'sample_size', 'avg_views', 'confidence']], 10, language=language)}

## AB Test / Lift Analysis
Use these results to choose the next small test.
{_md_table(ab_df, language=language)}

Estimated effect: `{treatment}` -> `{outcome}` = {causal['effect']:.2f}, interval [{causal['ci_low']:.2f}, {causal['ci_high']:.2f}], confidence {causal['confidence']}.

Refutation checks passed: {refute['summary']['passed']}/{refute['summary']['total']}.

Closed-loop method: identify variables → estimate effect → run refutation → generate experiment plan. Conclusion type: {causal_loop['conclusion_type']}.

## Next Experiment Plan
- Hypothesis: {experiment['hypothesis']}
- Treatment: {experiment['treatment_group']}
- Control: {experiment['control_group']}
- Primary metric: {experiment['primary_metric']}
- Execution suggestion: {experiment['duration_suggestion']}, {experiment['minimum_sample_suggestion']}

## Auto-Generated Experiment Candidates
{auto_lines or "- No candidate available."}
"""

    return f"""# 内容策略分析报告

## 标题策略
可信度：{title_result['confidence']}。先验证高表现模式，再放大。
{_md_table(title_result['table'])}

## 主题策略
{_md_table(topic_result['table'][['topic', 'content_count', 'views', 'favorite_rate', 'follow_rate', 'consultations', 'conversions', 'revenue', 'revenue_per_hour']])}

## 平台策略
平台定位：{platform_result['platform_roles']}
{_md_table(platform_result['table'])}

## 发布时间策略
低样本时间段建议先小范围验证。
{_md_table(time_df[['platform', 'weekday', 'hour', 'sample_size', 'avg_views', 'confidence']], 10)}

## 实验与增量分析
用这些结果决定下一轮小实验。
{_md_table(ab_df)}

估计结果：{treatment} 对 {outcome} 的增量约为 {causal['effect']:.2f}，区间 [{causal['ci_low']:.2f}, {causal['ci_high']:.2f}]，可信度 {causal['confidence']}。

反驳检验通过：{refute['summary']['passed']}/{refute['summary']['total']}。

分析流程：先识别影响因素，再估计增量，再检查稳定性，最后生成可执行实验计划。

## 下周实验计划
- 假设：{experiment['hypothesis']}
- 实验组：{experiment['treatment_group']}
- 对照组：{experiment['control_group']}
- 主指标：{experiment['primary_metric']}
- 执行建议：{experiment['duration_suggestion']}，{experiment['minimum_sample_suggestion']}

## 自动生成的实验候选
{auto_lines or "- 暂无可用候选。"}
"""
