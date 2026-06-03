"""pytest 共享 fixtures — mock 层，使测试无需真实 API key 即可运行"""
import os
import sys

# 确保所有测试有 fake API key（在 import 被测模块之前）
os.environ.setdefault("DASHSCOPE_API_KEY", "test-fake-key-for-ci")

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ── 工具类 ─────────────────────────────────────────────────


class _FakeResponse:
    """模拟 LLM complete() 返回对象 — 支持 str() 和 .text 属性"""
    def __init__(self, text: str):
        self.text = text
    def __str__(self):
        return self.text


# ── Agent 级别的 mock（不依赖真实 DashScope）─────────────────


@pytest.fixture
def mock_profile():
    """Mock UserProfile — 返回固定资产数据"""
    profile = MagicMock()
    profile.format_for_display.return_value = (
        "📊 旅行者当前抽卡资产\n"
        "├─ 原石: 12000 (约 75 抽)\n"
        "├─ 纠缠之缘: 10\n"
        "└─ 总计可抽: 85 抽"
    )
    profile.get_asset.return_value = 12000
    profile.update_assets.return_value = "已更新 primogems: -160，当前值: 11840"
    profile.save_preference.return_value = "已保存偏好: 喜欢胡桃"
    return profile


@pytest.fixture
def mock_router():
    """Mock Router LLM — 返回分类结果"""
    router = MagicMock()
    router.complete.return_value = _FakeResponse("chat")
    return router


@pytest.fixture
def mock_llm():
    """Mock 主 LLM — 流式输出"""
    llm = MagicMock()
    llm.api_key = "fake-test-key"  # 必须是真实字符串，Pydantic 校验

    def fake_stream(prompt):
        for ch in "这是派蒙的模拟回复！":
            token = MagicMock()
            token.delta = ch
            yield token

    llm.stream_complete.side_effect = fake_stream
    llm.complete.return_value = _FakeResponse("模拟回复：用户喜欢胡桃，冒险等级55。")
    return llm


@pytest.fixture
def mock_query_engine():
    """Mock RAG query_engine — 返回知识库检索结果"""
    from llama_index.core.schema import NodeWithScore, TextNode

    qe = MagicMock()
    node = TextNode(
        text="钟离是璃月的岩神，被称为岩王帝君。",
        metadata={"title": "钟离", "element": "岩"},
    )
    source_node = NodeWithScore(node=node, score=0.85)
    resp = MagicMock()
    resp.response = "钟离是璃月的岩神，被称为岩王帝君。"
    resp.source_nodes = [source_node]
    resp.__str__ = lambda s: s.response
    qe.query.return_value = resp
    return qe


@pytest.fixture
def mock_agent(mock_profile, mock_router, mock_llm, mock_query_engine):
    """完整 mock GameAgent — 所有外部依赖已替换"""
    from agent import GameAgent
    from llm_manager import LLMManager

    # Mock LLMManager — 只给 llm 属性
    llm_mgr = MagicMock(spec=LLMManager)
    llm_mgr.llm = mock_llm

    agent = GameAgent(
        llm_manager=llm_mgr,
        user_profile=mock_profile,
        query_engine=mock_query_engine,
    )
    # 替换 router 为 mock
    agent.router = mock_router
    return agent
