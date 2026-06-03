import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  ArrowDownToLine,
  BarChart3,
  CheckCircle2,
  CircleDollarSign,
  ClipboardList,
  FileSpreadsheet,
  Lightbulb,
  Loader2,
  Plus,
  ReceiptText,
  Search,
  Upload,
  WalletCards
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import * as XLSX from "xlsx";
import "./styles.css";

const seedEntries = [
  {
    date: "2026-05-18",
    direction: "income",
    amount: 12800,
    counterparty: "南枝品牌",
    category: "品牌顾问",
    channel: "银行转账",
    project: "品牌定位月包",
    invoiceStatus: "已开票",
    note: "月包首款",
    confidence: 0.96
  },
  {
    date: "2026-05-20",
    direction: "expense",
    amount: 1380,
    counterparty: "广告投放平台",
    category: "广告投放",
    channel: "支付宝",
    project: "品牌定位月包",
    invoiceStatus: "待补票",
    note: "获客测试预算",
    confidence: 0.9
  },
  {
    date: "2026-05-21",
    direction: "receivable",
    amount: 5200,
    counterparty: "北岸咨询",
    category: "咨询尾款",
    channel: "银行转账",
    project: "增长咨询月包",
    invoiceStatus: "未开票",
    note: "预计 5 月底回款",
    confidence: 0.84
  }
];

const seedActions = [
  {
    level: "high",
    title: "北岸咨询尾款进入催收窗口",
    detail: "5200 元应收款尚未到账，建议今天发送付款提醒，并同步开票信息。"
  },
  {
    level: "medium",
    title: "广告投放费用需要补票",
    detail: "1380 元广告支出缺少发票，建议本周内补齐，避免月底集中整理。"
  },
  {
    level: "low",
    title: "品牌定位月包客户贡献突出",
    detail: "南枝品牌贡献本月主要收入，可优先维护交付节奏和复购机会。"
  }
];

const directionText = {
  income: "收入",
  expense: "支出",
  receivable: "应收"
};

const levelText = {
  high: "紧急",
  medium: "关注",
  low: "建议"
};

function money(value) {
  return new Intl.NumberFormat("zh-CN", {
    style: "currency",
    currency: "CNY",
    maximumFractionDigits: 0
  }).format(value || 0);
}

function buildSummary(entries) {
  const income = entries.filter((item) => item.direction === "income").reduce((sum, item) => sum + item.amount, 0);
  const expense = entries.filter((item) => item.direction === "expense").reduce((sum, item) => sum + item.amount, 0);
  const receivable = entries.filter((item) => item.direction === "receivable").reduce((sum, item) => sum + item.amount, 0);
  const missingInvoices = entries.filter((item) => ["未开票", "待补票"].includes(item.invoiceStatus)).length;
  const clientMap = new Map();
  const categoryMap = new Map();

  entries.forEach((item) => {
    if (item.direction === "income" || item.direction === "receivable") {
      clientMap.set(item.counterparty, (clientMap.get(item.counterparty) || 0) + item.amount);
    }
    if (item.direction === "expense") {
      categoryMap.set(item.category, (categoryMap.get(item.category) || 0) + item.amount);
    }
  });

  const topClient = [...clientMap.entries()].sort((a, b) => b[1] - a[1])[0];
  const topExpense = [...categoryMap.entries()].sort((a, b) => b[1] - a[1])[0];

  return {
    income,
    expense,
    net: income - expense,
    receivable,
    missingInvoices,
    topClient: topClient ? `${topClient[0]} ${money(topClient[1])}` : "待识别",
    topExpense: topExpense ? `${topExpense[0]} ${money(topExpense[1])}` : "暂无支出"
  };
}

function App() {
  const inputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [entries, setEntries] = useState(seedEntries);
  const [actions, setActions] = useState(seedActions);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState("当前为示例数据。上传截图或表格后可生成新的台账。");
  const [query, setQuery] = useState("");

  const filteredEntries = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    if (!keyword) return entries;
    return entries.filter((item) => Object.values(item).join(" ").toLowerCase().includes(keyword));
  }, [entries, query]);

  const summary = useMemo(() => buildSummary(entries), [entries]);

  const categoryData = useMemo(() => {
    const map = new Map();
    entries.filter((item) => item.direction === "expense").forEach((item) => {
      map.set(item.category, (map.get(item.category) || 0) + item.amount);
    });
    return [...map.entries()].map(([name, value]) => ({ name, value }));
  }, [entries]);

  const flowData = useMemo(() => [
    { name: "收入", amount: summary.income },
    { name: "支出", amount: summary.expense },
    { name: "应收", amount: summary.receivable }
  ], [summary]);

  async function analyzeFiles(selectedFiles) {
    if (!selectedFiles.length) return;

    setLoading(true);
    setNotice("正在分析上传资料，识别金额、客户、票据状态和行动建议。");

    const formData = new FormData();
    selectedFiles.forEach((file) => formData.append("files", file));

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });
      const payload = await response.json();
      const data = payload.fallback || payload;

      if (!response.ok && !payload.fallback) {
        throw new Error(payload.detail || payload.error || "分析失败");
      }

      setEntries(data.entries || []);
      setActions(data.actions || []);
      setNotice(data.message || (data.source === "llm" ? "已通过大模型完成识别与经营建议生成。" : "已生成演示识别结果。"));
    } catch (error) {
      setNotice(`分析失败：${error.message}`);
    } finally {
      setLoading(false);
    }
  }

  function onFilesChange(event) {
    const selected = [...event.target.files];
    setFiles(selected);
    analyzeFiles(selected);
  }

  function exportWorkbook() {
    const rows = entries.map((item) => ({
      日期: item.date,
      类型: directionText[item.direction],
      金额: item.amount,
      交易对象: item.counterparty,
      分类: item.category,
      渠道: item.channel,
      关联项目: item.project,
      票据状态: item.invoiceStatus,
      备注: item.note,
      置信度: item.confidence
    }));
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, XLSX.utils.json_to_sheet(rows), "经营台账");
    XLSX.writeFile(workbook, "SoloLedger-经营台账.xlsx");
  }

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <div className="brand-row">
            <WalletCards size={28} />
            <span>SoloLedger</span>
          </div>
          <h1>一人公司经营台账 Agent</h1>
          <p>把支付截图、订单、发票和表格变成收支台账、经营看板与今日行动建议。</p>
        </div>
        <div className="topbar-actions">
          <button className="ghost-button" type="button" onClick={() => inputRef.current?.click()}>
            <Plus size={18} />
            上传资料
          </button>
          <button className="primary-button" type="button" onClick={exportWorkbook}>
            <ArrowDownToLine size={18} />
            导出 Excel
          </button>
        </div>
      </section>

      <section className="workspace">
        <aside className="upload-panel">
          <div className="panel-title">
            <Upload size={20} />
            <span>资料入口</span>
          </div>
          <button className="dropzone" type="button" onClick={() => inputRef.current?.click()}>
            {loading ? <Loader2 className="spin" size={34} /> : <FileSpreadsheet size={34} />}
            <strong>{loading ? "AI 正在整理台账" : "上传截图 / 发票 / Excel"}</strong>
            <span>支持微信、支付宝、订单截图、发票图片、收支表。未配置 key 时自动使用演示数据。</span>
          </button>
          <input
            ref={inputRef}
            className="file-input"
            type="file"
            multiple
            accept="image/*,.pdf,.xlsx,.xls,.csv"
            onChange={onFilesChange}
          />
          <div className="file-list">
            {files.length ? files.map((file) => (
              <div className="file-chip" key={`${file.name}-${file.size}`}>
                <ReceiptText size={16} />
                <span>{file.name}</span>
              </div>
            )) : (
              <div className="empty-file">等待上传经营碎片资料</div>
            )}
          </div>
          <div className="notice">
            <CheckCircle2 size={18} />
            <span>{notice}</span>
          </div>
        </aside>

        <section className="main-panel">
          <div className="metric-grid">
            <Metric icon={CircleDollarSign} label="本月收入" value={money(summary.income)} tone="green" />
            <Metric icon={WalletCards} label="本月支出" value={money(summary.expense)} tone="coral" />
            <Metric icon={BarChart3} label="净收入" value={money(summary.net)} tone="ink" />
            <Metric icon={AlertTriangle} label="应收未收" value={money(summary.receivable)} tone="amber" />
            <Metric icon={ReceiptText} label="票据缺口" value={`${summary.missingInvoices} 笔`} tone="coral" />
            <Metric icon={ClipboardList} label="高价值客户" value={summary.topClient} tone="green" />
          </div>

          <div className="insight-grid">
            <section className="chart-panel">
              <div className="section-head">
                <h2>经营概览</h2>
                <span>{summary.topExpense}</span>
              </div>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={flowData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(value) => money(value)} />
                  <Bar dataKey="amount" radius={[8, 8, 0, 0]}>
                    {flowData.map((item) => (
                      <Cell key={item.name} fill={item.name === "收入" ? "#2c9468" : item.name === "支出" ? "#d85b48" : "#e3a72f"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </section>

            <section className="chart-panel">
              <div className="section-head">
                <h2>支出结构</h2>
                <span>自动分类</span>
              </div>
              <ResponsiveContainer width="100%" height={240}>
                <PieChart>
                  <Pie data={categoryData} dataKey="value" nameKey="name" innerRadius={58} outerRadius={88} paddingAngle={4}>
                    {categoryData.map((item, index) => (
                      <Cell key={item.name} fill={["#d85b48", "#2c9468", "#e3a72f", "#44546a", "#4f8c8d"][index % 5]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => money(value)} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </section>
          </div>

          <section className="table-section">
            <div className="section-head">
              <h2>经营台账</h2>
              <label className="search-box">
                <Search size={16} />
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索客户、项目、分类" />
              </label>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>日期</th>
                    <th>类型</th>
                    <th>金额</th>
                    <th>交易对象</th>
                    <th>分类</th>
                    <th>项目</th>
                    <th>票据</th>
                    <th>置信度</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEntries.map((item, index) => (
                    <tr key={`${item.date}-${item.counterparty}-${index}`}>
                      <td>{item.date}</td>
                      <td><span className={`tag ${item.direction}`}>{directionText[item.direction]}</span></td>
                      <td>{money(item.amount)}</td>
                      <td>{item.counterparty}</td>
                      <td>{item.category}</td>
                      <td>{item.project}</td>
                      <td>{item.invoiceStatus}</td>
                      <td>{Math.round((item.confidence || 0) * 100)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </section>

        <aside className="action-panel">
          <div className="panel-title">
            <Lightbulb size={20} />
            <span>今日行动 Agent</span>
          </div>
          <div className="action-list">
            {actions.map((action, index) => (
              <article className={`action-item ${action.level}`} key={`${action.title}-${index}`}>
                <div className="action-level">{levelText[action.level] || "建议"}</div>
                <h3>{action.title}</h3>
                <p>{action.detail}</p>
              </article>
            ))}
          </div>
        </aside>
      </section>
    </main>
  );
}

function Metric({ icon: Icon, label, value, tone }) {
  return (
    <article className={`metric ${tone}`}>
      <Icon size={20} />
      <span>{label}</span>
      <strong title={value}>{value}</strong>
    </article>
  );
}

createRoot(document.getElementById("root")).render(<App />);
