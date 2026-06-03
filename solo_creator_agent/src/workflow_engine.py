from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclass
class WorkflowStep:
    name: str
    status: str
    summary: str
    record_count: int = 0
    output_count: int = 0


def langgraph_available() -> bool:
    try:
        import langgraph  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def build_workflow_trace(
    contents: pd.DataFrame,
    revenues: pd.DataFrame,
    campaigns: pd.DataFrame,
    ab_tests: pd.DataFrame,
    products: pd.DataFrame,
    feedback: pd.DataFrame,
    beta_tests: pd.DataFrame,
    agent_result: dict[str, Any],
    lang: str = "中文",
) -> dict[str, Any]:
    zh = lang == "中文"
    cards = agent_result.get("cards", [])
    modules = agent_result.get("modules", {})
    steps = [
        WorkflowStep(
            "资料整理" if zh else "Data intake",
            "完成" if zh else "done",
            "已读取用户上传或演示资料，并整理为经营数据。" if zh else "Materials were organized into operating data.",
            record_count=sum(len(df) for df in [contents, revenues, campaigns, ab_tests, products, feedback, beta_tests]),
            output_count=1,
        ),
        WorkflowStep(
            "收益判断" if zh else "Revenue analysis",
            "完成" if zh else "done",
            "识别收入主阵地、待收款和商务风险。" if zh else "Identified revenue anchors, receivables and business risks.",
            record_count=len(revenues) + len(campaigns),
            output_count=len(modules.get("revenue_analysis_agent", {}).get("cards", [])),
        ),
        WorkflowStep(
            "策略判断" if zh else "Strategy simulation",
            "完成" if zh else "done",
            "比较平台、主题、标题和产品功能的可验证机会。" if zh else "Compared platforms, topics, titles and product feature opportunities.",
            record_count=len(contents) + len(products),
            output_count=len(modules.get("strategy_simulation_agent", {}).get("cards", [])),
        ),
        WorkflowStep(
            "增量判断" if zh else "Lift estimation",
            "完成" if zh else "done",
            "控制账号、平台、主题等背景因素，估计策略是否可能带来真实增量。" if zh else "Controlled background factors to estimate whether strategies may add lift.",
            record_count=len(contents) + len(beta_tests),
            output_count=len(modules.get("causal_estimator_agent", {}).get("ate", [])),
        ),
        WorkflowStep(
            "实验设计" if zh else "Experiment planning",
            "完成" if zh else "done",
            "把结果转成可执行的小实验和观察指标。" if zh else "Converted findings into small tests and metrics.",
            record_count=len(ab_tests) + len(beta_tests),
            output_count=len(modules.get("ab_test_agent", {}).get("cards", [])),
        ),
        WorkflowStep(
            "行动生成" if zh else "Action generation",
            "完成" if zh else "done",
            "只保留最值得本周执行的建议。" if zh else "Kept only the most useful actions for this week.",
            record_count=0,
            output_count=len(cards),
        ),
    ]
    return {
        "engine": "LangGraph 可接入" if zh and langgraph_available() else "轻量工作流" if zh else "LangGraph-ready" if langgraph_available() else "lightweight workflow",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": [asdict(step) for step in steps],
    }


def workflow_markdown(trace: dict[str, Any], lang: str = "中文") -> str:
    zh = lang == "中文"
    lines = ["### 系统如何得出建议" if zh else "### How SoloDeck Reached These Actions"]
    for index, step in enumerate(trace.get("steps", []), start=1):
        if zh:
            lines.append(f"{index}. **{step['name']}**：{step['summary']}（输出 {step['output_count']} 项）")
        else:
            lines.append(f"{index}. **{step['name']}**: {step['summary']} ({step['output_count']} outputs)")
    return "\n".join(lines)
