"""Agent 管线 mock 测试 — 不依赖真实 API key，CI 可运行"""
import asyncio
import pytest
from unittest.mock import MagicMock, patch


class TestIntentClassification:
    """意图路由 — mock 路由模型"""

    def test_classify_asset(self, mock_agent):
        """asset 意图识别"""
        from tests.conftest import _FakeResponse
        mock_agent.router.complete.return_value = _FakeResponse("asset")
        result = asyncio.run(mock_agent._classify("我有多少原石"))
        assert result == "asset"

    def test_classify_lore(self, mock_agent):
        """lore 意图识别"""
        from tests.conftest import _FakeResponse
        mock_agent.router.complete.return_value = _FakeResponse("lore")
        result = asyncio.run(mock_agent._classify("钟离是谁"))
        assert result == "lore"

    def test_classify_web(self, mock_agent):
        """web 意图识别"""
        from tests.conftest import _FakeResponse
        mock_agent.router.complete.return_value = _FakeResponse("web")
        result = asyncio.run(mock_agent._classify("原神最新卡池"))
        assert result == "web"

    def test_classify_chat(self, mock_agent):
        """chat 意图识别（使用 mock_router 默认值 "chat"）"""
        result = asyncio.run(mock_agent._classify("你好"))
        assert result == "chat"

    def test_classify_exact_match_with_period(self, mock_agent):
        """模型输出带标点 → 前缀匹配"""
        from tests.conftest import _FakeResponse
        mock_agent.router.complete.return_value = _FakeResponse("lore.")
        result = asyncio.run(mock_agent._classify("雷电将军是谁"))
        assert result == "lore"

    def test_classify_fallback_on_unexpected(self, mock_agent):
        """模型输出异常文本 → fallback chat"""
        from tests.conftest import _FakeResponse
        mock_agent.router.complete.return_value = _FakeResponse(
            "i think this is about game lore"
        )
        result = asyncio.run(mock_agent._classify("钟离"))
        assert result == "chat"

    def test_classify_fallback_on_error(self, mock_agent):
        """路由模型异常 → fallback chat"""
        mock_agent.router.complete.side_effect = RuntimeError("API error")
        result = asyncio.run(mock_agent._classify("钟离"))
        assert result == "chat"


class TestToolExecution:
    """工具调度 — mock 工具返回"""

    def test_execute_asset(self, mock_agent, mock_profile):
        """asset 意图 → 返回用户资产"""
        result, sources = asyncio.run(
            mock_agent._execute_tool("asset", "我有多少原石")
        )
        assert "原石" in result
        assert sources == ""

    def test_execute_lore(self, mock_agent):
        """lore 意图 → 返回 RAG 结果"""
        result, sources = asyncio.run(
            mock_agent._execute_tool("lore", "钟离是谁")
        )
        assert len(result) > 0
        # sources 应包含标题
        assert "钟离" in sources

    def test_execute_chat(self, mock_agent):
        """chat 意图 → 返回空（由 LLM 直接回答）"""
        result, sources = asyncio.run(
            mock_agent._execute_tool("chat", "你好")
        )
        assert result == ""
        assert sources == ""


class TestPromptBuilding:
    """Prompt 构建 — 不依赖外部服务"""

    def test_build_chat_prompt(self, mock_agent):
        """闲聊 prompt 包含对话历史"""
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        mock_agent.chat_history = [
            ChatMessage(role=MessageRole.USER, content="你好"),
            ChatMessage(role=MessageRole.ASSISTANT, content="旅行者你好！"),
        ]
        mock_agent._memory_summary = ""

        prompt = mock_agent._build_prompt("我叫cc", "", "chat")
        assert "旅行者" in prompt
        assert "派蒙" in prompt
        assert "你好" in prompt
        assert "我叫cc" in prompt

    def test_build_lore_prompt(self, mock_agent):
        """lore prompt 包含知识库资料"""
        mock_agent.chat_history = []
        mock_agent._memory_summary = ""

        tool_result = "钟离是璃月的岩神，被称为岩王帝君。"
        prompt = mock_agent._build_prompt("钟离是谁", tool_result, "lore")
        assert "知识库资料" in prompt
        assert "岩王帝君" in prompt

    def test_build_prompt_with_memory_summary(self, mock_agent):
        """包含历史摘要的 prompt"""
        mock_agent.chat_history = []
        mock_agent._memory_summary = "用户之前表示喜欢胡桃，冒险等级55。"

        prompt = mock_agent._build_prompt("推荐一个角色", "", "chat")
        assert "喜欢胡桃" in prompt
        assert "冒险等级55" in prompt


class TestChatStream:
    """流式对话 — mock 完整管线"""

    async def _collect(self, gen):
        tokens = []
        async for token in gen:
            tokens.append(token)
        return "".join(tokens)

    def test_chat_stream_chat_intent(self, mock_agent):
        """闲聊意图 → 无工具调用，直接流式回复"""
        mock_agent.router.complete.return_value = MagicMock(text="chat")

        result = asyncio.run(
            self._collect(mock_agent.chat_stream("你好"))
        )
        assert "思考中" in result or "模拟" in result or "派蒙" in result
        assert len(mock_agent.chat_history) == 2  # 用户 + 助手

    def test_chat_stream_lore_intent(self, mock_agent, mock_query_engine):
        """lore 意图 → RAG 检索 + 流式回复"""
        mock_agent.router.complete.return_value = MagicMock(text="lore")

        # 确保 query_engine 返回有效结果
        result = asyncio.run(
            self._collect(mock_agent.chat_stream("钟离是谁"))
        )
        assert len(result) > 0

    def test_chat_stream_preserves_history(self, mock_agent):
        """多次对话后历史正确累积"""
        mock_agent.router.complete.return_value = MagicMock(text="chat")

        asyncio.run(self._collect(mock_agent.chat_stream("你好")))
        asyncio.run(self._collect(mock_agent.chat_stream("再见")))

        assert len(mock_agent.chat_history) >= 4  # 2轮 × 2条

    def test_chat_stream_empty_result_fallback(self, mock_agent):
        """工具结果为空 → 降级为 chat"""
        mock_agent.router.complete.return_value = MagicMock(text="lore")
        # query_engine 返回空
        mock_agent.query_engine.query.return_value.source_nodes = []
        mock_agent.query_engine.query.return_value.response = ""

        result = asyncio.run(
            self._collect(mock_agent.chat_stream("不存在的角色"))
        )
        # 应能正常完成（降级为 chat）
        assert len(result) > 0


class TestMemorySummarization:
    """对话记忆摘要"""

    def test_summarize_compresses_history(self, mock_agent):
        """摘要方法正确调用 LLM 并返回非空摘要"""
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        messages = [
            ChatMessage(role=MessageRole.USER, content="我喜欢胡桃"),
            ChatMessage(role=MessageRole.ASSISTANT, content="胡桃很棒！"),
            ChatMessage(role=MessageRole.USER, content="我冒险等级55"),
            ChatMessage(role=MessageRole.ASSISTANT, content="厉害！"),
        ]
        # mock_llm.complete 已在 conftest 中预设返回 FakeResponse

        summary = asyncio.run(mock_agent._summarize_history(messages))
        assert len(summary) > 0
        assert isinstance(summary, str)

    def test_summarize_handles_failure(self, mock_agent):
        """摘要失败 → 返回空字符串，不抛异常"""
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        mock_agent.llm.complete.side_effect = RuntimeError("LLM error")
        messages = [
            ChatMessage(role=MessageRole.USER, content="test"),
        ]

        summary = asyncio.run(mock_agent._summarize_history(messages))
        assert summary == ""
