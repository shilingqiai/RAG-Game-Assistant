# 🎒 Paimon Companion — 原神智能游戏伴侣

基于 **LlamaIndex FunctionAgent** 的智能游戏伴侣，支持真实工具调用、RAG 知识库检索、流式对话输出。

---

## 核心特性

- ✅ **FunctionAgent 工具调用** — LLM 可以真正调用工具，查询用户资产、搜索网络
- ✅ **RAG 知识库** — 基于原神 lore 数据构建向量索引，查询世界观知识
- ✅ **流式输出** — Gradio 界面实时显示 token 级生成
- ✅ **用户资产持久化** — JSON 本地存储，支持原石/纠缠之缘/保底水位管理
- ✅ **多轮对话** — 携带历史上下文
- ✅ **FastMCP 微服务（可选）** — 工具能力解耦为独立 MCP 服务

---

## 架构

```
┌────────────────────────────────────────────────────────┐
│              Gradio UI (simple_start.py)                │
│              async chat_fn(msg, history)                │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│          game_core.py (GameCompanion)                   │
│  - 初始化 LLM / Agent / RAG 索引                       │
│  - chat_stream(query, history) → async generator       │
└───────────────────────┬────────────────────────────────┘
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  agent.py    │ │ llm_manager  │ │  config.py   │
│ FunctionAgent│ │ DashScope    │ │ UserProfile  │
│ + 6 tools    │ │ (qwen-plus)  │ │ JSON 持久化  │
└──────┬───────┘ └──────────────┘ └──────────────┘
       │
       ├─ get_user_profile     → 读取用户资产
       ├─ update_user_assets   → 更新资产
       ├─ save_user_preference → 保存偏好
       ├─ search_web           → DuckDuckGo 搜索
       ├─ read_webpage         → Jina API 读网页
       └─ query_lore (RAG)     → 本地向量知识库

(可选) genshin_mcp_server.py → FastMCP 独立微服务
```

---

## Agent 工具列表

| 工具 | 来源 | 功能 |
|------|------|------|
| `get_user_profile` | config.py | 查询用户原石、纠缠之缘、保底状态 |
| `update_user_assets` | config.py | 更新资产（抽卡消耗/获取） |
| `save_user_preference` | config.py | 保存用户偏好和笔记 |
| `search_web` | tools.py | DuckDuckGo 搜索最新游戏资讯 |
| `read_webpage` | tools.py | 读取网页正文内容 |
| `query_lore` (RAG) | game_core.py | 查询本地原神世界观知识库 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置 API Key

```bash
# Windows PowerShell
set DASHSCOPE_API_KEY=你的阿里云DashScope API Key

# Linux/Mac
export DASHSCOPE_API_KEY="你的阿里云DashScope API Key"
```

### 3. (可选) 构建 RAG 知识库

```bash
python data_indexer.py
```

### 4. 启动

```bash
# Gradio UI（推荐）
python simple_start.py
# 浏览器打开 http://127.0.0.1:7860

# 或终端对话模式
python game_core.py

# 或启动 MCP 微服务（供其他 MCP Client 使用）
python genshin_mcp_server.py
```

---

## 文件结构

```
game_companion/
├── agent.py                # FunctionAgent 核心 — 工具定义 + 流式对话
├── game_core.py            # 核心整合 — LLM/Agent/RAG 初始化
├── llm_manager.py          # DashScope LLM 和 Embedding 管理
├── config.py               # UserProfile — JSON 持久化用户资产
├── tools.py                # 网络搜索 (DuckDuckGo) + 网页读取 (Jina)
├── data_indexer.py         # RAG 索引构建工具
├── genshin_mcp_server.py   # FastMCP 微服务（可选独立部署）
├── simple_start.py         # Gradio UI 启动脚本
├── user_profile.json       # 用户资产数据（自动生成）
├── storage/                # 向量索引持久化目录
└── requirements.txt        # 依赖清单
```

---

## 技术栈

- **LLM**: 阿里云 DashScope (qwen-plus)
- **Agent 框架**: LlamaIndex FunctionAgent (workflow-based)
- **Embedding**: DashScope text-embedding-v3
- **UI**: Gradio 6.x (async generator streaming)
- **向量存储**: LlamaIndex VectorStoreIndex (本地持久化)
- **搜索**: DuckDuckGo + Jina API
- **MCP**: FastMCP (可选微服务)

---

## 依赖

```bash
# 核心
llama-index-core>=0.10.57
llama-index-llms-dashscope>=0.5.0
llama-index-embeddings-dashscope>=0.5.0

# UI
gradio>=4.0.0

# 搜索
duckduckgo-search>=8.0.0
requests>=2.28.0

# DashScope
dashscope>=1.20.0

# 可选
fastmcp>=3.0.0
llama-index-tools-mcp>=0.4.0
```
