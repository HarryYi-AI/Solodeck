# SoloDeck

SoloDeck 面向内容创作者、知识付费创作者和一人公司，目标是把内容数据、收入数据、商务合作数据和实验数据转化为可执行经营任务。

当前实现覆盖四层架构中的第二、第三、第四层：

- 第二层：收益与商业价值层  
  收入结构、平台收入、待收款、内容商业价值评分、主题商业汇总、RPM、收益建议。
- 第三层：策略分析层  
  自动数据理解、文本变量结构化、标题风格、主题策略、平台定位、发布时间、变量语义理解、PSM/IPTW/分层回归、反驳检验、下周选题计划、AB Test / 轻量准实验分析。
- 第四层：商务合作流程层  
  商单看板、报价建议、风险提醒、甲方复盘报告。
- 扩展层：相似内容 / 产品系列 / 用户反馈 / 内测实验  
  相似内容识别、产品变体分析、系列边际增量、内部蚕食风险、用户反馈 Roadmap、内测实验设计、新功能推广建议。

基础层的数据接入由 CSV / 手动上传 / mock 数据提供，不依赖微信接口，也不依赖真实平台 API。

LLM 只用于变量语义理解、截图/文字抽取、洞察解释和实验计划撰写；因果估计、置信区间和反驳检验由统计方法完成。

产品默认是极简工作流：选择平台 → 添加资料 → 查看下一步行动。指标、图表和专业因果模块默认折叠，避免新用户被大量文字和表格淹没。

SoloDeck 现在也支持一人公司、小商家、独立产品团队判断“相似内容、相似产品、新功能、新款式、新版本”的增量效果。核心问题包括：

- 是否继续做同一系列内容，还是已经出现重复疲劳。
- 是否换平台、换标题、换角度。
- 新产品变体是否真正带来新增收益，还是内部蚕食。
- 新功能是否值得正式推广。
- 哪些用户反馈应该进入下一版 Roadmap。

## 运行方式

```bash
cd /workspace/ylj/harry_main/heikesong/solo_creator_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.mock_data
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

浏览器访问：

```text
http://服务器IP:8501/
```

如果是远程服务器且无法直接访问端口，使用 SSH 转发：

```bash
ssh -L 8501:localhost:8501 用户名@服务器公网IP
```

然后本地浏览器打开：

```text
http://localhost:8501/
```

## 发给别人体验

最短路径：

```bash
cd /workspace/ylj/harry_main/heikesong/solo_creator_agent
bash scripts/run_public_demo.sh
```

然后把服务器公网地址发给别人：

```text
http://服务器公网IP:8501/
```

建议在父目录 `.env` 中设置体验码，避免别人刷 API：

```env
SOLODECK_ACCESS_CODE=your-demo-code
```

账号系统已经内置，默认不强制登录，适合低摩擦试用。需要强制登录时设置：

```env
SOLODECK_REQUIRE_LOGIN=true
SOLODECK_AUTH_DB=/workspace/ylj/harry_main/heikesong/solo_creator_agent/data/solodeck_auth.db
```

当前账号系统使用本地 SQLite，密码使用 PBKDF2 哈希存储，不保存明文密码。正式商用时建议迁移到 PostgreSQL，并把用户数据按 user_id 隔离。

Docker 部署：

```bash
cd /workspace/ylj/harry_main/heikesong/solo_creator_agent
bash scripts/deploy_docker.sh
```

Docker Compose 会读取父目录 `../.env`，并把应用暴露到 `8501` 端口。

正式一点的公开链接建议使用：

```text
域名 + Nginx 反向代理 + HTTPS + Docker Compose
```

Nginx 示例在：

```text
deploy/nginx-solodeck.conf
```

如果不使用 Docker，也可以参考：

```text
deploy/solodeck.service
```

把它安装成 systemd 服务后，可长期保活运行。

## 数据字段

`data/mock_contents.csv`

- content_id, title, platform, topic, content_type, publish_time, title_style
- body, tags, language
- cover_style, duration_sec, production_hours, followers_before
- impressions, completion_rate, ad_spend, is_sponsored
- views, likes, favorites, comments, shares
- new_followers, consultations, conversions, revenue
- cost
- series_id, parent_content_id, content_similarity_group
- knowledge_domain, difficulty_level, novelty_score, duplication_risk, user_fatigue_risk

`data/mock_revenues.csv`

- revenue_id, date, amount, revenue_type, platform
- content_id, client_name, status, note

`data/mock_campaigns.csv`

- campaign_id, brand_name, campaign_name, platform, deliverables, price, deadline
- status, payment_status, invoice_status, revision_count, report_status, related_content_id

`data/mock_ab_tests.csv`

- experiment_id, date, platform, topic
- treatment_name, treatment_value, control_value
- outcome_metric, group, content_id, outcome_value, covariates_json

`data/mock_products.csv`

- product_id, product_name, category, series_id, launch_date
- price, cost, material, color, style, size, weight
- feature_tags, target_user, platform
- views, clicks, consultations, conversions, revenue
- refund_count, review_count, avg_rating
- is_new_version, parent_product_id

`data/mock_feedback.csv`

- feedback_id, user_id, source_type
- related_content_id, related_product_id, user_segment, country, platform
- feedback_text, rating, sentiment, issue_type, severity
- created_at, converted_after_feedback, retained_after_feedback

`data/mock_beta_tests.csv`

- beta_test_id, product_id, feature_name, test_group, user_id, user_segment
- invited_at, experienced_at, feedback_submitted, rating
- activated, retained_7d, converted, revenue, notes

## 当前因果分析边界

当前版本是轻量探索版：

- AB Test 使用 treatment/control 均值差异、相对提升、bootstrap 置信区间。
- 分层回归、PSM、IPTW 用于估计策略变量对结果指标的增量影响。
- 反驳检验包括 placebo shuffle、子样本稳定性和协变量平衡性。
- 自动洞察借鉴 QuickInsights / InsightPilot 的思路，先做多维聚合、极端样本、文本变量和相关性线索，再交给实验计划验证。
- 文本结构化借鉴 CAST 的思路，当前使用本地 fallback 提取语言、标题长度、数字/问题句/动作动词、关键词、情感和主题簇。
- 新功能和新产品推广建议使用固定效应、配对差异、PSM/IPTW fallback 和内测分组结果，默认输出为“探索性准实验估计”。
- 用户反馈属于证据，不是最终市场结论；如果内测用户不是随机邀请，会提示选择偏差。
- 样本量不足会明确标记低置信度。
- 所有策略结论会区分“相关性发现 / 估计因果 / 实验建议”，不声称完整因果识别。

## 双模型协作

在父目录 `.env` 中配置：

```env
OPENAI_API_KEY_BASIC=...
OPENAI_MODEL_BASIC=Qwen3.5-Plus
OPENAI_API_KEY_ADVANCED=...
OPENAI_MODEL_ADVANCED=GLM-5-Turbo
```

当前按成本和任务复杂度分层：

- Qwen3.5-Plus：更低成本、更长上下文，用于普通经营建议、长文本/截图抽取和批量处理。
- GLM-5-Turbo：更高成本，用于关键结论复核、因果解释和高价值报告润色。

## 经营流程、知识依据与反馈学习

当前版本新增三类底层能力，前台只展示为简洁的经营动作：

- `workflow_engine.py`：按“资料整理 → 收益判断 → 策略判断 → 增量判断 → 实验设计 → 行动生成”记录分析链路。当前使用轻量状态机，后续可替换为 LangGraph durable workflow。
- `knowledge_base.py`：内置内容、电商、产品内测、商务回款等经营知识，用 TF-IDF 检索匹配当前建议，为用户提供“为什么这样做”的业务依据。它只解释和补充，不替代统计/因果计算。
- `recommendation_learning.py`：记录用户对建议的“适合我 / 先不看 / 完成”反馈，用轻量权重调整后续建议排序。当前是可解释的 bandit-style 排序，不使用黑箱强化学习。

## 后续可扩展

- LangGraph：用于企业版流程持久化、任务恢复、人工确认节点和多 Agent 调度。
- RAG：接入行业案例库、报价模板、产品内测方法库和用户个人经营记忆。
- Contextual Bandit：当真实执行反馈足够后，用于优化建议排序和实验优先级。
- CausalPy：用于 DID、ITS、Synthetic Control 等准实验。
- PyMC-Marketing：用于 MMM、CLV、预算优化、渠道贡献分析。
- 接入评论文本聚类：为甲方复盘报告生成评论摘要和用户需求洞察。
- DoWhy / EconML / CausalPy：用于更完整的因果识别、DID、ITS、Synthetic Control。
