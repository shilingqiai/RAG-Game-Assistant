"""用户配置和记忆管理模块 — 基于 SQLite 持久化 + 项目全局配置"""
import sqlite3
import logging

logger = logging.getLogger("paimon.config")

# ── 项目全局配置 ──────────────────────────────────────────────

class AppConfig:
    """统一管理模型名、超时、阈值，避免硬编码散落各处"""

    # LLM
    MAIN_MODEL = "qwen-max"
    ROUTER_MODEL = "qwen-mt-flash"
    EMBED_MODEL = "text-embedding-v4"

    # 超时（秒）
    RAG_TIMEOUT = 5.0
    SEARCH_TIMEOUT = 10.0

    # 阈值
    RAG_MIN_SCORE = 0.05       # RAG 注入最低相关度
    MAX_HISTORY_TURNS = 20     # 最大对话轮数

    # 代理（需要时设环境变量，不硬编码）
    HTTP_PROXY = None           # "http://127.0.0.1:12450"


class UserProfile:
    """用户配置和记忆管理器 — SQLite 后端，支持并发安全与原子操作"""

    def __init__(self, db_path: str = "./user_profile.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # 读写并发优化
            conn.execute("""
                CREATE TABLE IF NOT EXISTS assets (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    text       TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 默认资产（仅首次创建）
            defaults = {
                "primogems": "12000",
                "intertwined_fate": "10",
                "pity_count": "0",
                "is_guaranteed": "false",
            }
            for k, v in defaults.items():
                conn.execute(
                    "INSERT OR IGNORE INTO assets(key, value) VALUES(?, ?)", (k, v)
                )
        logger.info("User profile ready (SQLite)")

    # ── 兼容旧 JSON 接口（供 agent.py 工具函数使用）──────────

    def load(self) -> dict:
        """加载用户配置（兼容旧接口，返回 dict）"""
        with sqlite3.connect(self.db_path) as conn:
            assets = {}
            for row in conn.execute("SELECT key, value FROM assets"):
                v = row[1]
                # 尝试类型转换
                if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                    assets[row[0]] = int(v)
                elif v in ("true", "false"):
                    assets[row[0]] = v == "true"
                else:
                    assets[row[0]] = v

            prefs = [
                row[0]
                for row in conn.execute(
                    "SELECT text FROM preferences ORDER BY created_at DESC"
                )
            ]

        return {"assets": assets, "preferences": prefs}

    def save(self, profile: dict):
        """保存用户配置（兼容旧接口）"""
        with sqlite3.connect(self.db_path) as conn:
            # 全量更新 assets
            assets = profile.get("assets", {})
            conn.execute("DELETE FROM assets")
            for k, v in assets.items():
                conn.execute(
                    "INSERT OR REPLACE INTO assets(key, value) VALUES(?, ?)",
                    (k, str(v).lower() if isinstance(v, bool) else str(v)),
                )

            # 追加新 preferences
            prefs = profile.get("preferences", [])
            current = {
                row[0]
                for row in conn.execute("SELECT text FROM preferences")
            }
            for p in prefs:
                if p not in current:
                    conn.execute("INSERT INTO preferences(text) VALUES(?)", (p,))

    # ── 新增原子操作接口 ──────────────────────────────────────

    def get_asset(self, key: str) -> int:
        """获取单个资产值（数值型），布尔型返回 0/1"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM assets WHERE key=?", (key,)
            ).fetchone()
            if not row:
                return 0
            v = row[0]
            # 布尔值转换
            if v in ("true", "false"):
                return 1 if v == "true" else 0
            try:
                return int(v)
            except ValueError:
                return 0

    def update_asset(self, key: str, delta: int):
        """原子更新资产（避免 read-modify-write 竞态）"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE assets SET value = CAST(value AS INTEGER) + ? WHERE key=?",
                (delta, key),
            )

    def add_preference(self, text: str):
        """添加偏好记录"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO preferences(text) VALUES(?)", (text,))

    # ── 兼容旧函数签名的方法 ──────────────────────────────────

    def update_assets(self, asset_type: str, change_amount: int) -> str:
        """更新用户资产（兼容 agent.py 工具函数签名）"""
        valid_types = {"primogems", "intertwined_fate", "pity_count"}
        if asset_type not in valid_types:
            return f"不支持的资产类型: {asset_type}"

        old = self.get_asset(asset_type)
        self.update_asset(asset_type, change_amount)
        new = self.get_asset(asset_type)
        result = f"已更新 {asset_type}: {change_amount:+d}，当前值: {new}"
        logger.info("Asset update: %s %+d (%d → %d)", asset_type, change_amount, old, new)
        return result

    def save_preference(self, preference_text: str) -> str:
        """保存用户偏好（兼容 agent.py 工具函数签名）"""
        self.add_preference(preference_text)
        result = f"已保存偏好: {preference_text}"
        logger.info("Preference saved: %s", preference_text)
        return result

    # ── 展示 ──────────────────────────────────────────────────

    def format_for_display(self) -> str:
        """格式化用户配置为显示字符串"""
        primogems = self.get_asset("primogems")
        fate = self.get_asset("intertwined_fate")
        pity = self.get_asset("pity_count")
        guaranteed = (
            self.get_asset("is_guaranteed") != 0
        )  # is_guaranteed 存储为 0/1

        total_pulls = primogems // 160 + fate
        guarantee_needed = max(0, 180 - pity)

        with sqlite3.connect(self.db_path) as conn:
            prefs = conn.execute(
                "SELECT text FROM preferences ORDER BY created_at DESC LIMIT 5"
            ).fetchall()
        pref_text = (
            "\n".join(f"  - {p[0]}" for p in prefs)
            if prefs
            else "  (暂无偏好记录)"
        )

        return f"""
📊 旅行者当前抽卡资产
├─ 原石: {primogems} (约 {primogems // 160} 抽)
├─ 纠缠之缘: {fate}
├─ 总计可抽: {total_pulls} 抽
├─ 当前垫水位: {pity}/90
├─ 是否大保底: {'是' if guaranteed else '否'}
└─ 距离下一个大保底还需要: {guarantee_needed} 抽

💡 旅行者的偏好记录
{pref_text}
""".strip()
