"""Agent 创建和初始化测试"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from config import UserProfile
from llm_manager import LLMManager


@pytest.fixture
def user_profile(tmp_path):
    db_path = str(tmp_path / "test_agent.db")
    return UserProfile(db_path=db_path)


@pytest.fixture
def llm_manager():
    return LLMManager()


class TestAgentCreation:

    def test_agent_import(self):
        """Agent 模块可导入"""
        from agent import GameAgent
        assert GameAgent is not None

    def test_agent_init(self, llm_manager, user_profile):
        """Agent 正常初始化（手搓管道模式：router + llm + profile）"""
        from agent import GameAgent
        agent = GameAgent(llm_manager=llm_manager, user_profile=user_profile)
        assert agent.llm is not None
        assert agent.user_profile is not None
        assert agent.router is not None
        assert agent.chat_history == []

    def test_agent_init_with_query_engine(self, llm_manager, user_profile):
        """Agent 带可选 query_engine 初始化"""
        from agent import GameAgent
        agent = GameAgent(llm_manager=llm_manager, user_profile=user_profile, query_engine="mock_engine")
        assert agent.query_engine == "mock_engine"

    def test_chat_history_management(self, llm_manager, user_profile):
        """对话历史管理"""
        from agent import GameAgent
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        agent = GameAgent(llm_manager=llm_manager, user_profile=user_profile)
        agent.chat_history.append(ChatMessage(role=MessageRole.USER, content="test"))
        agent.chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content="response"))
        assert len(agent.chat_history) == 2

    def test_agent_methods_exist(self, llm_manager, user_profile):
        """Agent 核心方法存在"""
        from agent import GameAgent
        agent = GameAgent(llm_manager=llm_manager, user_profile=user_profile)
        assert callable(agent._classify)
        assert callable(agent._execute_tool)
        assert callable(agent._build_prompt)
        assert callable(agent._get_profile)
        assert callable(agent.chat_stream)


class TestLLMManager:

    def test_llm_manager_init(self):
        """LLMManager 正常初始化"""
        from llama_index.core import Settings
        lm = LLMManager()
        assert lm.llm is not None
        assert lm.embed_model is not None
        # Settings 应已被注入
        assert Settings.embed_model is not None
