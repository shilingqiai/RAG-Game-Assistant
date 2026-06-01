"""UserProfile SQLite 后端测试"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UserProfile


@pytest.fixture
def profile(tmp_path):
    """创建临时数据库的 UserProfile"""
    db_path = str(tmp_path / "test_user.db")
    return UserProfile(db_path=db_path)


class TestUserProfile:

    def test_init_creates_db(self, profile):
        """初始化应创建数据库文件"""
        assert os.path.exists(profile.db_path)

    def test_default_assets(self, profile):
        """应有默认资产值"""
        assert profile.get_asset("primogems") == 12000
        assert profile.get_asset("intertwined_fate") == 10
        assert profile.get_asset("pity_count") == 0

    def test_update_asset_atomic(self, profile):
        """原子更新资产"""
        profile.update_asset("primogems", -1600)
        assert profile.get_asset("primogems") == 10400

        profile.update_asset("pity_count", 10)
        assert profile.get_asset("pity_count") == 10

    def test_add_preference(self, profile):
        """添加偏好"""
        profile.add_preference("喜欢胡桃")
        profile.add_preference("想抽钟离")

        data = profile.load()
        assert "喜欢胡桃" in data["preferences"]
        assert "想抽钟离" in data["preferences"]

    def test_format_for_display(self, profile):
        """格式化显示不崩溃"""
        result = profile.format_for_display()
        assert "12000" in result
        assert "原石" in result

    def test_load_returns_dict(self, profile):
        """load() 兼容旧接口"""
        data = profile.load()
        assert isinstance(data, dict)
        assert "assets" in data
        assert "preferences" in data
        assert data["assets"]["primogems"] == 12000

    def test_save_and_load_roundtrip(self, profile):
        """save/load 往返一致性"""
        new_data = {
            "assets": {"primogems": 5000, "intertwined_fate": 20, "pity_count": 50, "is_guaranteed": True},
            "preferences": ["test pref"],
        }
        profile.save(new_data)
        loaded = profile.load()
        assert loaded["assets"]["primogems"] == 5000
        assert loaded["preferences"] == ["test pref"]

    def test_update_assets_invalid_type(self, profile):
        """无效资产类型应报错"""
        result = profile.update_assets("invalid_type", 100)
        assert "不支持" in result
