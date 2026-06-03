# 🎒 Paimon Companion — 原神智能游戏伴侣

> 基于 **意图路由 + 混合 RAG（BM25+向量+RRF）+ Reranker 精排 + 实时搜索** 的多模态智能 Agent，为原神玩家提供角色 Lore 问答、抽卡资产管理、最新资讯搜索和自由闲聊。

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.12-purple)](https://llamaindex.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)
[![tests](https://img.shields.io/badge/tests-43%20passed-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 项目亮点

| 能力 | 实现 |
|------|------|
| 🧠 **意图路由** | 专用轻量模型 `qwen-mt-flash`（temperature=0）做 4 分类，92% 准确率 (50 eval) |
| 📚 **混合 RAG** | BM25 关键词 + 向量语义 → RRF 融合 → `qwen3-rerank` 精排，616 nodes 中文知识库 |
| 🌐 **实时搜索** | DuckDuckGo 中文优化（自动补全"原神"+ cn-zh 区域+中文过滤） |
| 📊 **资产管理** | SQLite WAL 持久化，原子 UPDATE 防竞态，Gradio 面板 / FastAPI 双接口 |
| 💬 **流式对话** | FastAPI SSE 流式端点 + Gradio 前端 + 线程桥接异步 token 输出 |
| 🐳 **部署** | Docker / docker-compose 一键启动 |
| 🧠 **对话记忆** | LLM 自动摘要压缩历史对话，跨轮次保留关键上下文 |
| 🔧 **优雅降级** | 路由失败→chat，RAG 超时→chat，Reranker 连续失败→自动关闭 |
| 🧪 **43 个测试** | 27 mock（1.8s，无 API key）+ 16 集成测试，GitHub Actions CI |
| 📊 **质量评估** | intent eval (50 条+混淆矩阵) + RAG recall eval (8 条关键词) |

---

## 🏗️ 架构

```
用户消息
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Step 1/3  意图分类 (qwen-mt-flash, ~0.6s, 92% acc)      │
│  ┌──────────────────────────────────────┐                │
│  │ asset │ lore │ web │ chat            │                │
│  └───┬───────┬───────┬───────┬──────────┘                │
└──────┼───────┼───────┼───────┼───────────────────────────┘
       │       │       │       │
       ▼       ▼       ▼       ▼
  ┌────────┐┌─────┐┌─────┐┌──────────┐
  │ SQLite ││混合 ││DDGS ││ 直接 LLM │
  │ 资产查询││RAG  ││搜索 ││ 聊天     │
  └───┬────┘└──┬──┘└──┬──┘└────┬─────┘
      │        │      │        │
      │   ┌────▼────┐ │        │
      │   │ BM25+向量│ │        │
      │   │ → RRF   │ │        │
      │   │ → Rerank│ │        │
      │   └────┬────┘ │        │
      └────────┴──────┴────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│  Step 3/3  LLM 流式合成 (qwen-max)                       │
│  → 注入工具结果 + 对话历史 + 记忆摘要 + 派蒙人设         │
│  → 线程桥接异步逐 token 流式输出 + 📖 来源引用           │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### Docker（推荐）

```bash
# 1. 设置 API Key
echo "DASHSCOPE_API_KEY=你的密钥" > .env

# 2. 启动
docker compose up -d
# → 浏览器打开 http://localhost:7860 (Gradio) 或 http://localhost:8000 (FastAPI)
```

### 本地运行

```bash
# 1. 安装依赖
pip install -e .

# 2. 设置 API Key
export DASHSCOPE_API_KEY=你的阿里云DashScope密钥

# 3. 初始化 git submodule + 构建中文向量索引
git submodule update --init
python data_indexer_cn.py

# 4. 启动服务（二选一）
python simple_start.py   # Gradio UI → http://127.0.0.1:7860
python server.py          # FastAPI + SSE → http://127.0.0.1:8000
```

### CLI 命令

```bash
paimon           # 启动 Gradio UI
paimon-serve     # 启动 FastAPI 服务
paimon-index-cn  # 重建中文向量索引
paimon-eval      # 运行意图路由评估 (50 条)
paimon-eval-rag  # 运行 RAG 召回评估 (8 条)
```

---

## 📁 项目结构

```
game_companion/
├── agent.py                 # 核心 Agent — 意图路由 + 工具调度 + 线程桥接流式合成
├── game_core.py             # GameCompanion — HybridRetriever (BM25+向量+RRF+Rerank)
├── llm_manager.py           # DashScope LLM/Embedding + NO_PROXY 注入
├── config.py                # AppConfig 集中配置 (env var 覆盖) + UserProfile (SQLite)
├── tools.py                 # DuckDuckGo 中文优化搜索 + Jina 网页读取
├── data_indexer.py          # 英文向量索引构建器 (characters.txt + world_lore.txt)
├── data_indexer_cn.py       # 中文向量索引构建器 (genshin-data 123 角色)
├── server.py                # FastAPI + SSE 流式服务 + 内嵌前端
├── simple_start.py          # Gradio UI (流式聊天 + 资产管理面板)
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .github/workflows/test.yml   # CI: Python 3.10/11/12 矩阵
│
├── data/
│   ├── rag_texts/           # 英文 RAG 源文本
│   └── genshin-data/        # git submodule: 简体中文角色 JSON
│
├── tests/
│   ├── conftest.py          # Mock fixtures (FakeResponse, mock_agent, etc.)
│   ├── test_pipeline_mock.py # 19 mock 测试 (无 API key, 1.8s)
│   ├── test_agent.py        # Agent + LLMManager 集成测试 (6)
│   ├── test_config.py       # UserProfile SQLite 原子操作 (8)
│   └── test_rag.py          # RAG 文档加载 + 检索 (10)
│
├── eval_intent.py           # 意图路由评估 (50 条标注 + 混淆矩阵)
├── eval_rag.py              # RAG 召回评估 (8 条关键词)
├── storage/                 # 中文向量索引 (616 nodes, gitignored)
├── storage_en/              # 英文向量索引 (140 nodes, gitignored)
└── pyproject.toml
```

---

## 🛠️ 技术栈

| 层 | 选型 | 说明 |
|----|------|------|
| **LLM (对话)** | DashScope `qwen-max` | 高质量中文对话合成 |
| **LLM (路由)** | DashScope `qwen-mt-flash` | temperature=0, max_tokens=10, 快速分类 |
| **Embedding** | DashScope `text-embedding-v4` | 中英文语义向量化 |
| **RAG 框架** | LlamaIndex 0.12 | VectorStoreIndex + HybridRetriever |
| **关键词检索** | BM25Okapi (rank-bm25) | 中文字符级分词，精确人名/术语匹配 |
| **精排** | DashScope `qwen3-rerank` | RRF 候选 → Reranker 精排 top-k |
| **文本分割** | SentenceSplitter | chunk_size=512, chunk_overlap=128 |
| **存储** | SQLite WAL + LlamaIndex VectorStore | 原子操作, 并发安全 |
| **前端** | Gradio 6.x + FastAPI + SSE | 双前端方案 |
| **搜索** | DuckDuckGo (cn-zh) + Jina Reader | 中文优化 |
| **部署** | Docker / docker-compose | 一键启动 |
| **测试** | pytest 9.x + GitHub Actions | 43 用例, mock+集成分离 |

---

## 📊 评估指标

| 指标 | 数值 |
|------|------|
| 意图路由准确率 | **92%** (50 条标注，qwen-mt-flash) |
| RAG 节点数 | **616** (123 角色 × ~5 片段) |
| 检索方式 | BM25 + 向量 → RRF → qwen3-rerank |
| Mock 测试 | 27 个，1.8s，无 API key |
| 总测试 | 43 个，CI 自动运行 |

---

## 🔄 最近更新 (v0.4)

- **混合检索** — BM25 + 向量 RRF 融合 + qwen3-rerank 精排
- **FastAPI + SSE** — 流式 API + 内嵌前端，替代 Gradio 用于生产
- **Docker** — Dockerfile + docker-compose 一键部署
- **对话记忆** — LLM 自动摘要压缩，跨轮次保留上下文
- **Mock 测试 + CI** — 27 个 mock 测试 (1.8s) + GitHub Actions
- **Router 优化** — few-shot prompt 重构，60% → 92% (+32pp)
- **Web Search 中文优化** — 自动补全 + cn-zh 区域 + 中文过滤
- **低分过滤** — RAG_MIN_SCORE 0.05 → 0.30，空节点降级保护
- **Gradio 6.x 适配** — 移除废弃 API，新增资产管理面板

<details>
<summary>v0.3 及更早</summary>

- **v0.3** — text-embedding-v4 + 中文索引 (genshin-data) + 流式输出 + 调试日志
- **v0.2** — 意图路由 + 英文 RAG + 完整数据管线
- **v0.1** — 轻量 RAG 游戏伴侣原型

</details>

---

*Built with LlamaIndex, DashScope Qwen, FastAPI, and Gradio. For Genshin Impact players who want a knowledgeable companion.*
