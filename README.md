# SoloDeck

SoloDeck is an AI operating advisor for creators, solo businesses, and small product teams. It turns content, revenue, feedback, campaign, and experiment data into causal-aware next actions.

中文简介：SoloDeck 是面向内容创作者、一人公司和小型产品团队的 AI 经营分析 Agent，把内容、收入、反馈、商务合作和实验数据转化为可验证的下一步经营动作。

<img width="1495" height="798" alt="image" src="https://github.com/user-attachments/assets/1092a0cb-7e6a-4f41-a9a3-9560e0cc4729" />
<img width="1529" height="920" alt="image" src="https://github.com/user-attachments/assets/ba179ed1-0c22-4571-a1d8-6dab9796b157" />
<img width="1493" height="534" alt="image" src="https://github.com/user-attachments/assets/8f6c8ac1-d855-4424-93b6-e050ee360511" />

## Why SoloDeck

Most analytics tools answer:

> What happened in the data?

SoloDeck focuses on:

> What should I do next, where should I do it, and how can I verify it?

Creators and solo businesses often have useful data scattered across platform dashboards, spreadsheets, payment screenshots, feedback notes, and campaign records. SoloDeck helps them move from fragmented data to concrete operating decisions.

## What It Solves

- Content creators know which posts performed well, but not whether the title, platform, topic, timing, or account size caused the difference.
- Solo businesses often miss receivables, invoices, campaign reports, and follow-up tasks.
- Small product teams need to know whether a new feature, product variant, or beta-test result is worth scaling.
- Users want actions, not a wall of dashboards.

SoloDeck turns uploaded materials into:

- short-term and long-term task lists
- weekly validation plans
- revenue and campaign risk alerts
- content and platform strategy suggestions
- product and feedback priorities
- downloadable operating reports

## Core Features

### 1. Multi-Source Data Intake

SoloDeck supports:

- CSV files
- screenshots
- manual text input
- content performance data
- revenue records
- campaign records
- product data
- user feedback
- beta-test records
- experiment records

It does not require WeChat APIs or real platform APIs, so it is easy to demo and practical for real-world use.

### 2. Action-First Workspace

The default screen is intentionally simple:

1. Choose platforms
2. Add materials
3. Read the next actions

Metrics, charts, and detailed analysis are folded by default. Users first see the most important actions instead of long tables.

### 3. Creator and Content Strategy

SoloDeck analyzes:

- title styles
- topics
- platforms
- publishing time
- content series
- content fatigue
- duplication risk
- commercial value per content

It suggests what to publish next and how to validate the strategy.

### 4. Revenue and Business Workflow

SoloDeck identifies:

- revenue mix
- platform revenue
- pending payments
- high-value clients
- sponsorship risks
- missing reports
- invoice/payment issues

It also supports pricing suggestions and brand report generation.

### 5. Product and Feedback Analysis

SoloDeck also works beyond self-media scenarios. It supports small e-commerce teams, robot products, knowledge products, and productized services.

It analyzes:

- product variants
- feature tags
- new vs old versions
- refunds
- ratings
- beta-test results
- feedback themes
- roadmap priorities

### 6. Causal-Aware Analysis

SoloDeck does not treat correlation as guaranteed causality.

It separates:

- correlation findings
- controlled lift estimates
- paired comparisons
- experiment results
- validation plans

The system can control for factors such as:

- account ID
- platform
- topic
- follower base
- production hours
- ad spend
- content type

Supported statistical ideas include:

- group mean comparison
- ATE estimation from treatment/control groups
- paired differences for same-content cross-platform analysis
- fixed-effect style controls
- propensity-score matching fallback
- inverse-probability weighting fallback
- bootstrap confidence intervals
- placebo-style refutation checks
- subsample stability checks

The output is cautious: if evidence is weak, SoloDeck recommends a small validation experiment instead of directly scaling the strategy.

### 7. Workflow Trace, Knowledge Base, and Feedback Learning

SoloDeck includes a lightweight workflow layer:

```text
Data intake -> Revenue analysis -> Strategy analysis -> Lift estimation -> Experiment planning -> Action generation
```

It also includes:

- a small operating knowledge base for content, e-commerce, product tests, and campaign follow-up
- TF-IDF retrieval to explain why a recommendation is relevant
- preference feedback so users can mark recommendations as useful or not useful
- a lightweight bandit-style ranking adjustment for future suggestions

## Demo Upload Pack

A ready-to-use demo pack is included:

```text
solo_creator_agent/demo_upload_pack/
```

It contains virtual screenshots and CSV files that can be uploaded during a live demo:

- operating dashboard screenshot
- pending payment screenshot
- feedback notes screenshot
- campaign tracker screenshot
- content CSV
- revenue CSV
- campaign CSV
- product CSV
- feedback CSV
- experiment CSV
- beta-test CSV

Zip file:

```text
solo_creator_agent/solodeck_demo_upload_pack.zip
```

Suggested demo flow:

1. Open SoloDeck.
2. Upload screenshots first to show that the product can work without platform APIs.
3. Paste a manual note, for example:

```text
Tomorrow at 9 AM I need to review robot campaign data with the client, but the report is not ready.
```

4. Upload CSV files to show full analysis.
5. Show "Next Actions", "What to Validate This Week", "How SoloDeck Reached These Actions", and the downloadable report.

## Project Structure

```text
.
├── solo_creator_agent/
│   ├── app.py
│   ├── requirements.txt
│   ├── src/
│   │   ├── agent_orchestrator.py
│   │   ├── auto_insights.py
│   │   ├── business_collab.py
│   │   ├── causal_estimator.py
│   │   ├── causal_experiment.py
│   │   ├── causal_refute.py
│   │   ├── data_loader.py
│   │   ├── knowledge_base.py
│   │   ├── llm_agent.py
│   │   ├── product_feedback.py
│   │   ├── recommendation_learning.py
│   │   ├── revenue_analysis.py
│   │   ├── strategy_analysis.py
│   │   ├── text_structured.py
│   │   ├── user_storage.py
│   │   └── workflow_engine.py
│   ├── data/
│   ├── demo_upload_pack/
│   ├── scripts/
│   ├── deploy/
│   ├── Dockerfile
│   └── docker-compose.yml
├── package.json
└── README.md
```

## Quick Start

```bash
cd solo_creator_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open:

```text
http://localhost:8501
```

For a remote server:

```bash
ssh -L 8501:localhost:8501 username@server_ip
```

Then open:

```text
http://localhost:8501
```

## Environment Variables

Create `.env` in the repository root or configure environment variables directly:

```env
OPENAI_BASE_URL=https://aiping.cn/api/v1
OPENAI_API_KEY_BASIC=your-basic-model-key
OPENAI_MODEL_BASIC=Qwen3.5-Plus
OPENAI_API_KEY_ADVANCED=your-advanced-model-key
OPENAI_MODEL_ADVANCED=GLM-5-Turbo

SOLODECK_ACCESS_CODE=your-demo-code
SOLODECK_REQUIRE_LOGIN=false
CREATOR_ALIPAY_ACCOUNT=your-payment-account
```

The app can run with mock/demo data without an LLM key. LLM keys enable screenshot/text extraction and more polished natural-language advice.

Security note:

- Do not commit `.env` to GitHub.
- Keep real API keys, access codes, and payment accounts in environment variables.
- This repository only includes placeholders and `.env.example` style configuration.
- If a key is accidentally committed, revoke it immediately and generate a new one.

## Generate Demo Data

```bash
cd solo_creator_agent
python scripts/generate_demo_upload_pack.py
```

This regenerates:

```text
demo_upload_pack/
solodeck_demo_upload_pack.zip
```

## Public Demo Script

```bash
cd solo_creator_agent
bash scripts/run_public_demo.sh
```

For production-like deployment:

```bash
cd solo_creator_agent
bash scripts/deploy_docker.sh
```

Nginx and systemd examples are in:

```text
solo_creator_agent/deploy/
```

## Data Privacy and User Storage

SoloDeck includes a local account and workspace system:

- user uploads are stored by user ID
- imported records are stored per user
- task status is stored per user
- recommendation feedback is stored per user
- passwords are hashed with PBKDF2

The current version uses SQLite for easy demo and development. For commercial deployment, migrate storage to PostgreSQL or another managed database.

## Current Limitations

- Causal estimates are exploratory and should not be treated as definitive causal proof.
- Screenshot extraction depends on the configured vision-capable model.
- LangGraph is not required at runtime yet. The current workflow layer is LangGraph-ready but implemented as a lightweight local trace.
- RAG is currently a lightweight TF-IDF knowledge matching module, not a full vector database pipeline.
- Recommendation learning is a transparent bandit-style ranking adjustment, not a full reinforcement-learning system.

## Suggested GitHub Topics

```text
ai-agent
creator-economy
solo-business
causal-inference
streamlit
data-analysis
business-intelligence
ab-testing
productivity
```

## License

This project is prepared for hackathon and demo use. Add a license before public commercial distribution.
