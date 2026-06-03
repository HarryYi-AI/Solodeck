import "dotenv/config";
import express from "express";
import multer from "multer";
import OpenAI from "openai";
import { z } from "zod";

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 12 * 1024 * 1024 } });
const port = Number(process.env.PORT || 8787);

app.use(express.json({ limit: "2mb" }));

const ledgerSchema = z.object({
  entries: z.array(z.object({
    date: z.string(),
    direction: z.enum(["income", "expense", "receivable"]),
    amount: z.number(),
    counterparty: z.string(),
    category: z.string(),
    channel: z.string(),
    project: z.string(),
    invoiceStatus: z.enum(["已开票", "未开票", "无需发票", "待补票"]),
    note: z.string(),
    confidence: z.number().min(0).max(1)
  })),
  actions: z.array(z.object({
    level: z.enum(["high", "medium", "low"]),
    title: z.string(),
    detail: z.string()
  }))
});

function mockAnalysis(files) {
  const today = new Date().toISOString().slice(0, 10);
  const names = files.map((file) => file.originalname).join("、") || "示例截图";

  return {
    source: "mock",
    message: "当前返回可演示的样例识别结果。",
    entries: [
      {
        date: today,
        direction: "income",
        amount: 6800,
        counterparty: "澄星设计工作室",
        category: "官网项目款",
        channel: "微信支付",
        project: "品牌官网改版",
        invoiceStatus: "未开票",
        note: `从 ${names} 识别：首付款到账`,
        confidence: 0.93
      },
      {
        date: today,
        direction: "expense",
        amount: 568,
        counterparty: "云服务平台",
        category: "云服务",
        channel: "支付宝",
        project: "品牌官网改版",
        invoiceStatus: "待补票",
        note: "服务器与对象存储费用",
        confidence: 0.88
      },
      {
        date: today,
        direction: "receivable",
        amount: 2400,
        counterparty: "北岸咨询",
        category: "咨询尾款",
        channel: "银行转账",
        project: "增长咨询月包",
        invoiceStatus: "已开票",
        note: "尾款逾期 8 天，建议催收",
        confidence: 0.86
      },
      {
        date: today,
        direction: "expense",
        amount: 199,
        counterparty: "AI 工具订阅",
        category: "软件订阅",
        channel: "信用卡",
        project: "通用经营",
        invoiceStatus: "无需发票",
        note: "月度订阅",
        confidence: 0.91
      }
    ],
    actions: [
      {
        level: "high",
        title: "北岸咨询尾款需要今天跟进",
        detail: "2400 元应收款已逾期 8 天，建议发送催收提醒并同步已开票信息。"
      },
      {
        level: "medium",
        title: "云服务支出缺少票据",
        detail: "568 元云服务费用处于待补票状态，建议本周内补齐发票，避免月底集中整理。"
      },
      {
        level: "low",
        title: "官网项目毛利健康",
        detail: "品牌官网改版当前收入 6800 元，已记录支出 568 元，可优先推进尾款节点。"
      }
    ]
  };
}

function normalizeLevel(value) {
  const text = String(value || "").toLowerCase();
  if (["high", "紧急", "高", "高优先级", "严重"].includes(text)) return "high";
  if (["medium", "关注", "中", "中优先级", "一般"].includes(text)) return "medium";
  if (["low", "建议", "低", "低优先级"].includes(text)) return "low";
  return "medium";
}

function normalizeModelOutput(parsed) {
  const entries = Array.isArray(parsed.entries) ? parsed.entries : Array.isArray(parsed.ledger) ? parsed.ledger : [];
  const actions = Array.isArray(parsed.actions)
    ? parsed.actions
    : Array.isArray(parsed.suggestions)
      ? parsed.suggestions
      : Array.isArray(parsed.advice)
        ? parsed.advice
        : [];

  return {
    entries: entries.map((item) => ({
      date: String(item.date || item.日期 || "待确认"),
      direction: ["income", "expense", "receivable"].includes(item.direction)
        ? item.direction
        : String(item.direction || item.类型 || item.type || "").includes("支")
          ? "expense"
          : String(item.direction || item.类型 || item.type || "").includes("应")
            ? "receivable"
            : "income",
      amount: Number(item.amount ?? item.金额 ?? 0),
      counterparty: String(item.counterparty || item.交易对象 || item.customer || item.client || "待确认"),
      category: String(item.category || item.分类 || "待分类"),
      channel: String(item.channel || item.渠道 || "待确认"),
      project: String(item.project || item.项目 || "待关联"),
      invoiceStatus: ["已开票", "未开票", "无需发票", "待补票"].includes(item.invoiceStatus || item.票据状态)
        ? item.invoiceStatus || item.票据状态
        : "待补票",
      note: String(item.note || item.备注 || ""),
      confidence: Math.max(0, Math.min(1, Number(item.confidence ?? item.置信度 ?? 0.75)))
    })),
    actions: actions.map((item) => ({
      level: normalizeLevel(item.level || item.优先级 || item.priority || item.risk),
      title: String(item.title || item.标题 || item.action || item.建议 || "经营建议"),
      detail: String(item.detail || item.详情 || item.content || item.内容 || item.reason || item.说明 || "")
    }))
  };
}

function fileToContentPart(file) {
  if (file.mimetype.startsWith("image/")) {
    return {
      type: "image_url",
      image_url: {
        url: `data:${file.mimetype};base64,${file.buffer.toString("base64")}`
      }
    };
  }

  return {
    type: "text",
    text: `文件名：${file.originalname}\nMIME：${file.mimetype}\n大小：${file.size} bytes。请根据文件名和可见上下文推断可能的台账字段；如果无法确定，降低 confidence。`
  };
}

async function analyzeWithModel(files) {
  const client = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL || undefined
  });

  const request = {
    model: process.env.OPENAI_MODEL || "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: [
          "你是一人公司经营台账 Agent，负责从支付截图、订单截图、发票截图、收支表中抽取经营数据。",
          "只返回 JSON，不要 Markdown。",
          "字段必须包含 entries 和 actions。",
          "direction 只能是 income、expense、receivable。",
          "invoiceStatus 只能是 已开票、未开票、无需发票、待补票。",
          "无法确认的信息要合理留空或写“待确认”，并降低 confidence。",
          "行动建议要围绕催收、补票、控成本、维护高价值客户、项目利润风险。"
        ].join("\n")
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "请分析这些上传文件，生成结构化经营台账和可执行行动建议。返回 JSON 格式：{ entries: [...], actions: [...] }。"
          },
          ...files.slice(0, 8).map(fileToContentPart)
        ]
      }
    ]
  };

  if ((process.env.OPENAI_RESPONSE_FORMAT || "json_object") !== "none") {
    request.response_format = { type: process.env.OPENAI_RESPONSE_FORMAT || "json_object" };
  }

  if (process.env.OPENAI_BASE_URL?.includes("aiping.cn")) {
    request.extra_body = {
      enable_thinking: process.env.OPENAI_ENABLE_THINKING === "true",
      provider: {
        only: [],
        order: [],
        sort: null,
        input_price_range: [],
        output_price_range: [],
        input_length_range: [],
        output_length_range: [],
        throughput_range: [],
        latency_range: []
      }
    };
  }

  const completion = await client.chat.completions.create(request);

  const raw = completion.choices[0]?.message?.content || "{}";
  const parsed = JSON.parse(raw);
  return { source: "llm", ...ledgerSchema.parse(normalizeModelOutput(parsed)) };
}

app.post("/api/analyze", upload.array("files", 12), async (req, res) => {
  const files = req.files || [];

  try {
    if (!process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY.includes("your-api-key")) {
      return res.json(mockAnalysis(files));
    }

    const result = await analyzeWithModel(files);
    return res.json(result);
  } catch (error) {
    console.error(error);
    return res.status(500).json({
      error: "分析失败",
      detail: error instanceof Error ? error.message : "Unknown error",
      fallback: {
        ...mockAnalysis(files),
        message: "大模型返回格式暂未完全匹配，已切换到演示数据。"
      }
    });
  }
});

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    modelConfigured: Boolean(process.env.OPENAI_API_KEY && !process.env.OPENAI_API_KEY.includes("your-api-key"))
  });
});

app.listen(port, () => {
  console.log(`SoloLedger API running at http://localhost:${port}`);
});
