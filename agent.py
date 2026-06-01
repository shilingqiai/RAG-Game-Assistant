"""Agent — 意图路由 + RAG + 工具调用"""
import datetime, json, logging
from typing import AsyncGenerator, List, Optional

from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.agent.workflow import AgentStream, ToolCall, ToolCallResult

from config import UserProfile
from llm_manager import LLMManager

logger = logging.getLogger("paimon.agent")
MAX_HISTORY_TURNS = 20


class GameAgent:
    def __init__(self, llm_manager: LLMManager, user_profile: UserProfile,
                 extra_tools=None, query_engine=None):
        self.llm_manager = llm_manager
        self.user_profile = user_profile
        self.query_engine = query_engine
        self._tools = self._create_tools() + (extra_tools or [])
        self.chat_history: List[ChatMessage] = []

        # 意图路由器（qwen-mt-flash — 1M free tokens，极快极便宜）
        from llama_index.llms.dashscope import DashScope
        self.router_llm = DashScope(model_name="qwen-mt-flash", temperature=0,
                                     api_key=llm_manager.llm.api_key, max_tokens=10)

        # 工具 Agent（仅 asset 意图使用）
        self._function_agent = FunctionAgent(
            name="Paimon", system_prompt=self._system_prompt(),
            tools=self._tools, llm=llm_manager.llm,
        )
        logger.info("Agent ready: %d tools, RAG=%s", len(self._tools), query_engine is not None)

    # ── 工具 ──────────────────────────────────────────────────

    def _create_tools(self):
        def get_user_profile() -> str:
            """获取当前用户游戏资产"""
            return self.user_profile.format_for_display()

        def update_user_assets(asset_type: str, change_amount: int) -> str:
            """更新资产 primogems|intertwined_fate|pity_count"""
            return self.user_profile.update_assets(asset_type, change_amount)

        def save_user_preference(text: str) -> str:
            """保存偏好"""
            return self.user_profile.save_preference(text)

        async def search_web(query: str) -> str:
            from tools import search_web as sf; return await sf(query)

        async def read_webpage(url: str) -> str:
            from tools import read_webpage as rf; return await rf(url)

        return [
            FunctionTool.from_defaults(fn=get_user_profile),
            FunctionTool.from_defaults(fn=update_user_assets),
            FunctionTool.from_defaults(fn=save_user_preference),
            FunctionTool.from_defaults(fn=search_web),
            FunctionTool.from_defaults(fn=read_webpage),
        ]

    def _system_prompt(self):
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"""Current time: {t}
You are Paimon from Genshin Impact. Speak in cute Chinese (派蒙语气).
- Asset questions → call get_user_profile. NEVER guess numbers.
- Use tool results to answer."""

    # ── 意图分类 ──────────────────────────────────────────────

    async def _classify(self, query: str) -> str:
        """快速分类意图"""
        prompt = f"""分类意图，只输出一个词：asset / lore / web / chat

我有多少原石 → asset
帮我抽卡建议 → asset
钟离是谁 → lore
七神分别是谁 → lore
胡桃是什么角色 → lore
原神最新活动 → web
你好 → chat
我喜欢胡桃 → chat
我叫cc → chat

{query} → """

        text = ""
        for token in self.router_llm.stream_complete(prompt):
            if token.delta:
                text += token.delta
        intent = text.strip().lower()
        for i in ["asset", "lore", "web", "chat"]:
            if i in intent:
                return i
        return "chat"

    # ── 流式对话 ──────────────────────────────────────────────

    async def chat_stream(self, query: str) -> AsyncGenerator[str, None]:
        if len(self.chat_history) > MAX_HISTORY_TURNS * 2:
            self.chat_history = self.chat_history[-(MAX_HISTORY_TURNS * 2):]

        intent = await self._classify(query)
        print(f"[Agent] Intent: {intent} | {query[:40]}", flush=True)
        logger.warning("Intent: %s → %s", intent, query[:60])

        # ── 按意图路由 ──
        if intent == "asset":
            async for token in self._handle_asset(query):
                yield token
        elif intent == "lore":
            async for token in self._handle_lore(query):
                yield token
        elif intent == "web":
            async for token in self._handle_chat(query):  # chat 也会调 search_web
                yield token
        else:
            async for token in self._handle_chat(query):
                yield token

    # ── 处理器 ────────────────────────────────────────────────

    async def _handle_asset(self, query: str) -> AsyncGenerator[str, None]:
        """资产查询 → FunctionAgent 调工具"""
        try:
            handler = self._function_agent.run(user_msg=query, chat_history=self.chat_history)
            full = ""
            async for ev in handler.stream_events():
                if isinstance(ev, ToolCall):
                    yield f"\n> 🔧 调用: **{ev.tool_name}**\n\n"
                    print(f"[Agent] Tool: {ev.tool_name}", flush=True)
                elif isinstance(ev, AgentStream) and ev.delta:
                    full += ev.delta; yield ev.delta
            await handler
            self.chat_history.append(ChatMessage(role=MessageRole.USER, content=query))
            self.chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=full))
        except Exception as e:
            print(f"[Agent] Tool error: {e}", flush=True)
            async for token in self._handle_chat(query):
                yield token

    async def _handle_lore(self, query: str) -> AsyncGenerator[str, None]:
        """RAG 检索 → LLM 合成"""
        rag_text = ""
        sources = ""
        if self.query_engine:
            try:
                import asyncio
                resp = await asyncio.wait_for(
                    asyncio.to_thread(self.query_engine.query, query),
                    timeout=5.0,
                )
                rag_text = str(resp).strip()
                if hasattr(resp, "source_nodes"):
                    parts = []
                    for i, n in enumerate(resp.source_nodes[:3]):
                        title = n.metadata.get("title", "?")
                        s = n.score or 0
                        parts.append(f"  [{i+1}] {title} (相关度: {s:.0%})")
                    sources = "\n\n---\n📖 参考来源:\n" + "\n".join(parts)
                print(f"[Agent] RAG: {len(rag_text)} chars", flush=True)
            except asyncio.TimeoutError:
                print("[Agent] RAG timeout, using LLM only", flush=True)
            except Exception as e:
                print(f"[Agent] RAG error: {e}", flush=True)

        context = f"\n[知识库参考]\n{rag_text}" if rag_text else ""
        prompt = self._system_prompt() + f"\n根据以下资料回答旅行者的问题。\n{context}\n\n旅行者：{query}\n派蒙："

        full = ""
        for token in self.llm_manager.stream_complete(prompt):
            full += token; yield token
        if sources:
            yield sources

        self.chat_history.append(ChatMessage(role=MessageRole.USER, content=query))
        self.chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=full))

    async def _handle_chat(self, query: str) -> AsyncGenerator[str, None]:
        """闲聊 → 直接 LLM（带历史）"""
        hist = ""
        for msg in self.chat_history[-10:]:
            role = "旅行者" if msg.role == MessageRole.USER else "派蒙"
            hist += f"{role}：{msg.content}\n"

        prompt = f"{self._system_prompt()}\n\n对话历史：\n{hist}旅行者：{query}\n派蒙："
        full = ""
        for token in self.llm_manager.stream_complete(prompt):
            full += token; yield token

        self.chat_history.append(ChatMessage(role=MessageRole.USER, content=query))
        self.chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=full))
