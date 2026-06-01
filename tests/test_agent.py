"""Agent 工具创建和初始化测试"""
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
        """Agent 正常初始化"""
        from agent import GameAgent
        agent = GameAgent(llm_manager=llm_manager, user_profile=user_profile)
        assert len(agent._tools) >= 5
        assert agent.chat_history == []
        assert agent._function_agent is not None

    def test_tools_have_names(self, llm_manager, user_profile):
        """所有工具有名称"""
        from agent import GameAgent
        agent = GameAgent(llm_manager=llm_manager, user_profile=user_profile)
        for tool in agent._tools:
            assert tool.metadata.name, f"Tool {tool} has no name"
            assert tool.metadata.description, f"Tool {tool.metadata.name} has no description"

    def test_chat_history_management(self, llm_manager, user_profile):
        """对话历史管理"""
        from agent import GameAgent
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        agent = GameAgent(llm_manager=llm_manager, user_profile=user_profile)
        agent.chat_history.append(ChatMessage(role=MessageRole.USER, content="test"))
        agent.chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content="response"))
        assert len(agent.chat_history) == 2


class TestLLMManager:

    def test_llm_manager_init(self):
        """LLMManager 正常初始化"""
        from llama_index.core import Settings
        lm = LLMManager()
        assert lm.llm is not None
        assert lm.embed_model is not None
        # Settings 应已被注入
        assert Settings.embed_model is not None
