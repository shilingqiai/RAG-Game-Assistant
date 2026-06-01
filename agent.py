"""Agent 模块 — 基于 LlamaIndex FunctionAgent 的游戏伴侣"""

import datetime
import logging
from typing import AsyncGenerator, List, Optional

from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.agent.workflow import AgentStream, AgentOutput, ToolCall, ToolCallResult

from config import UserProfile
from llm_manager import LLMManager

logger = logging.getLogger("paimon.agent")

# 保留最近 N 轮对话
MAX_HISTORY_TURNS = 20


class GameAgent:
    """基于 FunctionAgent 的游戏伴侣 — 真实工具调用 + 多轮对话 + 流式输出"""

    def __init__(
        self,
        llm_manager: LLMManager,
        user_profile: UserProfile,
        extra_tools: Optional[List[FunctionTool]] = None,
    ):
        self.llm_manager = llm_manager
        self.user_profile = user_profile
        self._tools = self._create_tools() + (extra_tools or [])
        self._system_prompt = self._build_system_prompt()
        self.chat_history: List[ChatMessage] = []

        self._function_agent = FunctionAgent(
            name="Paimon",
            description="原神旅行者忠诚、可爱、热情的向导派蒙",
            system_prompt=self._system_prompt,
            tools=self._tools,
            llm=llm_manager.llm,
        )
        logger.info("FunctionAgent initialized with %d tools", len(self._tools))

    # ── 工具定义 ──────────────────────────────────────────────

    def _create_tools(self) -> List[FunctionTool]:
        """创建工具 — 同步工具 + 异步 IO 工具混用"""

        def get_user_profile() -> str:
            """获取当前用户的游戏资产信息：原石数量、纠缠之缘、保底水位等"""
            return self.user_profile.format_for_display()

        def update_user_assets(asset_type: str, change_amount: int) -> str:
            """更新用户游戏资产。asset_type: 'primogems'|'intertwined_fate'|'pity_count'，change_amount: 变化量"""
            return self.user_profile.update_assets(asset_type, change_amount)

        def save_user_preference(topic: str) -> str:
            """Save a topic the traveller is interested in (e.g., a character name, game mechanic). Use short keywords only."""
            return self.user_profile.save_preference(topic)

        async def search_web(query: str) -> str:
            """搜索网络获取原神相关的最新信息与攻略"""
            from tools import search_web as search_func
            return await search_func(query)

        async def read_webpage(url: str) -> str:
            """读取指定网页的正文内容"""
            from tools import read_webpage as read_func
            return await read_func(url)

        return [
            FunctionTool.from_defaults(fn=get_user_profile),
            FunctionTool.from_defaults(fn=update_user_assets),
            FunctionTool.from_defaults(fn=save_user_preference),
            FunctionTool.from_defaults(fn=search_web),
            FunctionTool.from_defaults(fn=read_webpage),
        ]

    # ── 系统提示词 ────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"""Current time: {current_time}

You are Paimon, the loyal and cheerful guide from Genshin Impact. You MUST call tools to get real data before answering.

Rules:
- When asked about user's primogems, fates, or pity count → call get_user_profile FIRST. NEVER make up numbers.
- When asked about game lore, characters, or world setting → call search_game_knowledge FIRST.
- Use search_web for latest news, banners, and event information.
- Use the tool results to form your answer. Do NOT ignore tool output.
- Speak in a cute, enthusiastic Paimon style. Address the user as "Traveller".
- When calling tools, always provide valid JSON arguments. Escape special characters in strings."""

    # ── 流式对话 ──────────────────────────────────────────────

    async def chat_stream(self, query: str) -> AsyncGenerator[str, None]:
        """异步流式对话 — 多轮记忆 + 工具调用可见 + token 级输出"""
        logger.info("Processing: %s", query[:80])

        # 控制上下文窗口
        if len(self.chat_history) > MAX_HISTORY_TURNS * 2:
            self.chat_history = self.chat_history[-(MAX_HISTORY_TURNS * 2):]

        try:
            handler = self._function_agent.run(
                user_msg=query,
                chat_history=self.chat_history,
            )

            full_response = ""
            async for event in handler.stream_events():
                if isinstance(event, ToolCall):
                    # P0-2: 让用户看到工具调用
                    logger.info("Tool call: %s", event.tool_name)
                    yield f"\n> 🔧 *{event.tool_name}*\n\n"

                elif isinstance(event, ToolCallResult):
                    logger.info("Tool result: %s → %s",
                                event.tool_name,
                                str(event.tool_output)[:200])

                elif isinstance(event, AgentStream) and event.delta:
                    full_response += event.delta
                    yield event.delta

            await handler

            # P0-1: 保存本轮对话
            self.chat_history.append(ChatMessage(role=MessageRole.USER, content=query))
            self.chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=full_response))

        except Exception as e:
            logger.warning("Workflow error, falling back to direct LLM: %s", e)
            yield "\n> ⚠️ *工具调用遇到问题，正在尝试直接回答...*\n\n"

            try:
                # 构建含历史上下文的降级 prompt
                history_text = ""
                for msg in self.chat_history[-10:]:  # 最近 5 轮
                    role = "旅行者" if msg.role == MessageRole.USER else "派蒙"
                    history_text += f"{role}：{msg.content}\n"

                fallback_prompt = (
                    f"{self._system_prompt}\n\n"
                    "你暂时无法调用工具，请基于对话历史直接回答。\n\n"
                    f"---对话历史---\n{history_text}"
                    f"---当前问题---\n旅行者：{query}\n派蒙："
                )
                fallback_text = ""
                for token in self.llm_manager.stream_complete(fallback_prompt):
                    fallback_text += token
                    yield token

                # 降级回复也保存到历史
                self.chat_history.append(ChatMessage(role=MessageRole.USER, content=query))
                self.chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=fallback_text))

            except Exception as fallback_err:
                logger.error("Fallback also failed: %s", fallback_err)
                yield f"\n抱歉旅行者，派蒙出了点问题：{e}"
