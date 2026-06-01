"""RAG 检索质量测试"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from data_indexer import load_game_data, create_documents, clean_text


@pytest.fixture
def game_data():
    """加载测试数据"""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "genshin_impact_lore.json",
    )
    if os.path.exists(data_path):
        return load_game_data(data_path)
    return []


class TestDataLoading:

    def test_data_file_exists(self):
        """数据文件应存在"""
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "genshin_impact_lore.json",
        )
        assert os.path.exists(data_path), f"Data file not found: {data_path}"

    def test_load_game_data(self, game_data):
        """加载游戏数据"""
        assert len(game_data) > 0, "Game data should not be empty"
        for item in game_data:
            assert "title" in item
            assert "content" in item

    def test_create_documents(self, game_data):
        """创建文档对象"""
        if not game_data:
            pytest.skip("No game data available")
        docs = create_documents(game_data)
        assert len(docs) == len(game_data)
        for doc in docs:
            assert doc.text
            assert doc.metadata["title"]

    def test_clean_text(self):
        """文本清理"""
        assert clean_text("  hello   world  ") == "hello world"
        assert clean_text("line1\n\nline2") == "line1 line2"


class TestRAGRetrieval:
    """需要已构建索引的集成测试"""

    @pytest.fixture
    def companion(self):
        from game_core import GameCompanion
        return GameCompanion()

    def test_query_lore_basic(self, companion):
        """基本 RAG 查询"""
        if not companion.query_engine:
            pytest.skip("RAG index not available")
        result = companion.query_lore("七神")
        assert len(result) > 0
        assert "Error" not in result

    def test_query_lore_short(self, companion):
        """短查询正常返回"""
        if not companion.query_engine:
            pytest.skip("RAG index not available")
        result = companion.query_lore("原神")
        assert isinstance(result, str)
        assert len(result) > 0
