"""RAG 检索质量测试"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from data_indexer import _split_entries, _load_characters, _load_world_lore


@pytest.fixture
def chars_path():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "rag_texts", "characters.txt",
    )
    if os.path.exists(path):
        return path
    return None


@pytest.fixture
def lore_path():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "rag_texts", "world_lore.txt",
    )
    if os.path.exists(path):
        return path
    return None


class TestDataLoading:

    def test_characters_file_exists(self, chars_path):
        """角色文本文件应存在"""
        assert chars_path is not None, "characters.txt not found"
        assert os.path.exists(chars_path)

    def test_world_lore_file_exists(self, lore_path):
        """世界观文本文件应存在"""
        assert lore_path is not None, "world_lore.txt not found"
        assert os.path.exists(lore_path)

    def test_load_characters(self, chars_path):
        """加载角色文档"""
        if not chars_path:
            pytest.skip("characters.txt not available")
        docs = _load_characters(chars_path)
        assert len(docs) > 0, "Should have character documents"
        for doc in docs:
            assert doc.text
            assert "title" in doc.metadata
            assert doc.metadata["category"] == "character"

    def test_load_world_lore(self, lore_path):
        """加载世界观文档"""
        if not lore_path:
            pytest.skip("world_lore.txt not available")
        docs = _load_world_lore(lore_path)
        assert len(docs) > 0, "Should have lore documents"
        for doc in docs:
            assert doc.text
            assert "title" in doc.metadata
            assert doc.metadata["category"] == "lore"

    def test_split_entries(self):
        """条目分隔器"""
        text = "标题：A\n\nContent A\n\n---\n\n标题：B\n\nContent B"
        entries = _split_entries(text)
        assert len(entries) == 2
        assert "标题：A" in entries[0]
        assert "标题：B" in entries[1]

    def test_split_entries_single(self):
        """单一条目"""
        text = "角色：Test\n\n简介：Just a test"
        entries = _split_entries(text)
        assert len(entries) == 1


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
