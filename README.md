# 🎒 Paimon Companion — 原神智能游戏伴侣

> 基于 **意图路由 + RAG 检索增强生成 + 实时网络搜索** 的多模态智能 Agent，为原神玩家提供角色 Lore 问答、抽卡资产管理、最新资讯搜索和自由闲聊。

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.12-purple)](https://llamaindex.ai)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 项目亮点

| 能力 | 实现 |
|------|------|
| 🧠 **意图路由** | 专用轻量级模型 `qwen-mt-flash`（temperature=0）做 4 分类，50 条 eval case |
| 📚 **中文 RAG** | 90 角色 + 50 世界观条目 → LlamaIndex 向量索引，`QueryFusionRetriever` 检索 |
| 🌐 **实时搜索** | DuckDuckGo 异步搜索 + Jina Reader 网页正文提取 |
| 📊 **资产管理** | SQLite 持久化抽卡数据（原石/纠缠之缘/保底），原子 UPDATE 防竞态 |
| 💬 **流式对话** | Gradio 前端逐 token 流式输出 + 控制台分步调试日志 |
| 🔧 **优雅降级** | 缺少 API Key → 警告不崩，RAG 超时/失败 → 回退闲聊，工具无结果 → 降级 chat |
| 🧪 **22 个测试** | pytest 覆盖 Agent 初始化、SQLite 原子操作、RAG 文档加载与检索 |
| 📊 **质量评估** | intent eval（50 条标注 + 混淆矩阵）+ RAG recall eval（8 条关键词命中率） |
| 🔄 **完整数据管线** | 4 源 ETL（Fandom EN/CN + Bilibili + Dimbreath GitLab）→ SQLite → RAG texts |

---

## 🏗️ 架构

```
用户消息
  │
  ▼
┌──────────────────────────────────────────────┐
│  Step 1/3  意图分类 (qwen-mt-flash, 0.3s)    │
│  ┌──────────────────────────────────────┐    │
│  │ asset │ lore │ web │ chat            │    │
│  └───┬───────┬───────┬───────┬──────────┘    │
└──────┼───────┼───────┼───────┼───────────────┘
       │       │       │       │
       ▼       ▼       ▼       ▼
  ┌────────┐┌─────┐┌─────┐┌──────────┐
  │ SQLite ││RAG  ││DDGS ││ 直接 LLM │
  │ 资产查询││检索 ││搜索 ││ 聊天     │
  └───┬────┘└──┬──┘└──┬──┘└────┬─────┘
      │        │      │        │
      └────────┴──────┴────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│  Step 3/3  LLM 流式合成 (qwen-max)           │
│  → 注入工具结果 + 对话历史 + 派蒙人设        │
│  → Gradio 逐 token 流式输出 + 📖 来源引用    │
└──────────────────────────────────────────────┘
```

### 数据管线

```
Fandom Wiki API (EN) ──┐
Fandom Wiki API (CN) ──┤
Bilibili Wiki (CN) ────┼→ merger.py → SQLite DB
Dimbreath GitLab ──────┘       │
                               ├→ characters.txt (90 角色)
                               └→ world_lore.txt (50 条目)
                                       │
                              data_indexer.py
                              (text-embedding-v4)
                                       │
                                       ▼
                                  ./storage/
                              (向量索引, 140 nodes)
```

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key
set DASHSCOPE_API_KEY=你的阿里云DashScope密钥

# 3. 构建向量索引（首次运行或数据更新后）
python data_indexer.py

# 4. 启动 UI
python simple_start.py
# → 浏览器打开 http://127.0.0.1:7860
```

---

## 📁 项目结构

```
game_companion/
├── agent.py                # 核心 Agent — 意图路由 + 工具调度 + LLM 流式合成
├── game_core.py            # GameCompanion — RAG 索引加载 + Agent 初始化
├── llm_manager.py          # DashScope LLM/Embedding 管理 + Settings 全局注入
├── config.py               # AppConfig 集中配置 + UserProfile (SQLite)
├── tools.py                # DuckDuckGo 搜索 + Jina 网页读取 (async)
├── data_indexer.py         # 向量索引构建器 (characters.txt + world_lore.txt)
├── simple_start.py         # Gradio ChatInterface 流式 UI
│
├── data/
│   ├── rag_texts/          # RAG 源文本
│   │   ├── characters.txt  # 90 角色详情
│   │   └── world_lore.txt  # 50 世界观条目
│   ├── processed/          # 合并后的结构化 JSON + SQLite
│   └── scraper/            # 数据采集管线 (10 个模块)
│       ├── pipeline.py     # 编排器
│       ├── wiki_fetcher.py # Fandom EN 爬虫
│       ├── fandom_cn_fetcher.py
│       ├── bilibili_fetcher.py
│       ├── dimbreath_fetcher.py
│       ├── merger.py       # 多源合并 → SQLite + RAG texts
│       └── *_builder.py    # 多种索引构建器
│
├── tests/
│   ├── test_agent.py       # Agent 初始化 + LLM Manager (6 tests)
│   ├── test_config.py      # UserProfile SQLite 原子操作 (8 tests)
│   └── test_rag.py         # RAG 文档加载 + 检索 (8 tests)
│
├── eval_intent.py          # 意图路由准确率评估 (50 条标注)
├── eval_rag.py             # RAG 召回率评估 (8 条关键词)
├── storage/                # 向量索引持久化目录
└── requirements.txt
```

---

## 🛠️ 技术栈

| 层 | 选型 | 说明 |
|----|------|------|
| **LLM (对话)** | DashScope `qwen-max` | 高质量中文对话合成 |
| **LLM (路由)** | DashScope `qwen-mt-flash` | temperature=0, max_tokens=10, 快速分类 |
| **Embedding** | DashScope `text-embedding-v4` | 中文语义向量化 |
| **RAG 框架** | LlamaIndex 0.12 | VectorStoreIndex + QueryFusionRetriever |
| **文本分割** | SentenceSplitter | chunk_size=512, chunk_overlap=128 |
| **存储** | SQLite + LlamaIndex VectorStore | WAL 模式, 原子操作 |
| **前端** | Gradio 4.x ChatInterface | 流式逐 token 输出 |
| **搜索** | DuckDuckGo + Jina Reader | 异步, 非阻塞 |
| **测试** | pytest 9.x | 22 个用例, tmp_path 隔离 |

---

## 📊 评估指标

### 意图路由 (50 条)

```bash
python eval_intent.py
```

- 模型: `qwen-mt-flash`
- 准确率目标: >90%
- 输出混淆矩阵 (asset/lore/web/chat 四分类)

### RAG 召回 (8 条)

```bash
python eval_rag.py
```

- 衡量 RAG 回答中期望关键词的命中率
- 示例: "七神分别是谁" → 期望含 [风神, 岩神, 雷神, 草神]

---

## 🔄 最近更新

- **v0.3** — 嵌入模型升级至 `text-embedding-v4`，索引简化至角色+世界观 140 nodes
- **v0.3** — Gradio 流式输出 + 控制台分步调试日志
- **v0.3** — 修复所有硬编码嵌入模型名，统一用 `AppConfig`
- **v0.3** — 测试覆盖更新，22/22 通过
- **v0.2** — 意图路由 + 中文 RAG + 完整数据管线
- **v0.1** — 轻量 RAG 游戏伴侣原型

---

*Built with LlamaIndex, DashScope Qwen, and Gradio. For Genshin Impact players who want a knowledgeable companion.*
