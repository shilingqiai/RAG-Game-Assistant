"""数据合并引擎 — Wiki 文本 + Dimbreath 数值 → SQLite → RAG 文本导出"""
import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("paimon.scraper.merger")

RAW_DIR = Path(__file__).parent.parent / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "processed"
RAG_DIR = Path(__file__).parent.parent / "rag_texts"


# ── Schema ──────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS characters (
    id              TEXT PRIMARY KEY,
    name_en         TEXT NOT NULL,
    name_cn         TEXT,
    rarity          INTEGER,
    element         TEXT,
    weapon_type     TEXT,
    region          TEXT,
    affiliation     TEXT,
    birthday        TEXT,
    description     TEXT,
    lore_text       TEXT,
    hp_base         REAL,
    atk_base        REAL,
    def_base        REAL,
    source          TEXT DEFAULT 'wiki+dimbreath'
);

CREATE TABLE IF NOT EXISTS weapons (
    id              TEXT PRIMARY KEY,
    name_en         TEXT NOT NULL,
    name_cn         TEXT,
    rarity          INTEGER,
    weapon_type     TEXT,
    base_atk        REAL,
    substat_type    TEXT,
    substat_value   REAL,
    description     TEXT,
    lore_text       TEXT,
    source          TEXT DEFAULT 'wiki+dimbreath'
);

CREATE TABLE IF NOT EXISTS world_lore (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT NOT NULL,
    title_cn        TEXT,
    category        TEXT,
    content         TEXT,
    related_chars   TEXT,  -- JSON array of character names
    source          TEXT DEFAULT 'wiki'
);

CREATE TABLE IF NOT EXISTS text_map (
    hash_id         INTEGER PRIMARY KEY,
    text_en         TEXT,
    text_chs        TEXT
);

CREATE INDEX IF NOT EXISTS idx_chars_element ON characters(element);
CREATE INDEX IF NOT EXISTS idx_chars_rarity ON characters(rarity);
CREATE INDEX IF NOT EXISTS idx_chars_region ON characters(region);
CREATE INDEX IF NOT EXISTS idx_weapons_type ON weapons(weapon_type);
CREATE INDEX IF NOT EXISTS idx_lore_category ON world_lore(category);
"""


# ── 加载原始数据 ────────────────────────────────────────────


def load_raw_json(filename: str) -> dict | list:
    path = RAW_DIR / filename
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_dimbreath_json(name: str) -> list:
    path = RAW_DIR / "dimbreath" / f"{name}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


# ── TextMap 构建 ────────────────────────────────────────────


def build_text_map() -> dict[int, tuple[str, str]]:
    """构建哈希 ID → (英文, 中文) 映射表"""
    chs = {int(k): v for k, v in load_dimbreath_json("textmap_chs").items()}
    en = {int(k): v for k, v in load_dimbreath_json("textmap_en").items()}

    merged = {}
    for hid in set(chs) | set(en):
        merged[hid] = (en.get(hid, ""), chs.get(hid, ""))
    return merged


# ── 角色合并 ────────────────────────────────────────────────


def _element_map(raw: str) -> str:
    mapping = {
        "Geo": "岩", "Anemo": "风", "Electro": "雷",
        "Dendro": "草", "Hydro": "水", "Pyro": "火", "Cryo": "冰",
    }
    return mapping.get(raw, raw)


def _weapon_map(raw: str) -> str:
    mapping = {
        "Sword": "单手剑", "Claymore": "双手剑", "Polearm": "长柄武器",
        "Bow": "弓", "Catalyst": "法器",
    }
    return mapping.get(raw, raw)


def merge_characters(
    wiki_chars: list[dict],
    dimbreath_avatars: list[dict],
    text_map: dict[int, tuple[str, str]],
) -> list[dict]:
    """合并 wiki 文本和 dimbreath 数值"""
    # 构建 TextMap 查找
    name_lookup: dict[int, str] = {}
    for hid, (en_text, _) in text_map.items():
        name_lookup[hid] = en_text

    # 为每个 dimbreath 角色找到可读名称（通过 TextMap 的某个已知 key 匹配）
    # 注意：Dimbreath 数据中的 key 是哈希值，需要 TextMap 翻译
    # 这里我们主要依赖 wiki 数据，dimbreath 提供数值补充

    results = []
    for wc in wiki_chars:
        char = {
            "id": wc["name"].lower().replace(" ", "_"),
            "name_en": wc["name"],
            "name_cn": wc.get("name_cn") or wc["name"],
            "rarity": wc.get("rarity", 0),
            "element": wc.get("element", ""),
            "weapon_type": wc.get("weapon_type", ""),
            "region": wc.get("region", ""),
            "affiliation": wc.get("affiliation", ""),
            "birthday": wc.get("birthday", ""),
            "description": wc.get("description", ""),
            "lore_text": wc.get("lore_text", ""),
            "hp_base": 0,
            "atk_base": 0,
            "def_base": 0,
            "source": "wiki",
        }

        # 尝试从 dimbreath 补充数值（按名称模糊匹配）
        if dimbreath_avatars:
            matched = _find_dimbreath_avatar(wc["name"], dimbreath_avatars, text_map)
            if matched:
                char["hp_base"] = round(matched.get("hpBase", 0) or 0, 0)
                char["atk_base"] = round(matched.get("attackBase", 0) or 0, 0)
                char["def_base"] = round(matched.get("defenseBase", 0) or 0, 0)
                char["source"] = "wiki+dimbreath"

        results.append(char)

    return results


# 角色名映射（Dimbreath iconName → Wiki 名）
_NAME_ALIASES = {
    "playergirl": "Lumine",
    "playerboy": "Aether",
    "qin": "Jean",
    "feiyan": "Yanfei",
    "hutao": "Hu Tao",
    "kazuha": "Kaedehara Kazuha",
    "yaemiko": "Yae Miko",
    "shougun": "Raiden Shogun",
    "kokomi": "Sangonomiya Kokomi",
    "sara": "Kujou Sara",
    "itto": "Arataki Itto",
    "heizo": "Shikanoin Heizou",
    "ayato": "Kamisato Ayato",
    "tighnari": "Tighnari",
    "alhaitham": "Alhaitham",
    "yaoyao": "Yaoyao",
    "kaveh": "Kaveh",
    "lyney": "Lyney",
}

_QUALITY_MAP = {
    "QUALITY_ORANGE": 5,
    "QUALITY_PURPLE": 4,
    "QUALITY_BLUE": 3,
}


def _find_dimbreath_avatar(
    wiki_name: str, avatars: list[dict], text_map: dict[int, tuple[str, str]]
) -> dict | None:
    """在 Dimbreath 数据中查找匹配的角色（通过 iconName 匹配）"""
    wiki_lower = wiki_name.lower().replace(" ", "").replace("_", "")

    for av in avatars:
        icon = av.get("iconName", "")
        if not icon.startswith("UI_AvatarIcon_"):
            continue
        av_name = icon[len("UI_AvatarIcon_"):].lower()

        # 直接匹配或别名匹配
        if av_name == wiki_lower or _NAME_ALIASES.get(av_name, "") == wiki_name:
            return {
                "hpBase": av.get("hpBase", 0) or 0,
                "attackBase": av.get("attackBase", 0) or 0,
                "defenseBase": av.get("defenseBase", 0) or 0,
                "quality": _QUALITY_MAP.get(av.get("qualityType", ""), 0),
            }

    return None


# ── SQLite 写入 ─────────────────────────────────────────────


def create_db(characters: list[dict], lore: list[dict], db_path: Path):
    """创建 SQLite 数据库并写入数据"""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 删除旧库重建
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)

    # 写入角色
    for c in characters:
        conn.execute(
            """INSERT INTO characters
               (id, name_en, name_cn, rarity, element, weapon_type,
                region, affiliation, birthday, description, lore_text,
                hp_base, atk_base, def_base, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                c["id"], c["name_en"], c["name_cn"], c["rarity"],
                c["element"], c["weapon_type"], c["region"],
                c["affiliation"], c["birthday"], c["description"],
                c["lore_text"], c["hp_base"], c["atk_base"],
                c["def_base"], c["source"],
            ),
        )

    # 写入世界观
    for l in lore:
        conn.execute(
            """INSERT INTO world_lore (title, content, category, source)
               VALUES (?,?,?,?)""",
            (l["title"], l.get("content", ""), l.get("category", "lore"), "wiki"),
        )

    conn.commit()
    logger.info(
        "SQLite created: %d characters, %d lore entries → %s",
        len(characters), len(lore), db_path,
    )
    return conn


# ── RAG 文本导出 ────────────────────────────────────────────


def export_rag_texts(characters: list[dict], lore: list[dict]):
    """将结构化数据导出为纯文本文件，供 RAG 索引"""
    RAG_DIR.mkdir(parents=True, exist_ok=True)

    # 角色文本
    char_lines = []
    for c in characters:
        parts = [
            f"角色：{c['name_cn']}（{c['name_en']}）",
            f"稀有度：{'⭐' * c['rarity']}" if c['rarity'] else "",
            f"元素：{_element_map(c['element'])}" if c['element'] else "",
            f"武器：{_weapon_map(c['weapon_type'])}" if c['weapon_type'] else "",
            f"所属：{c['region']}" if c['region'] else "",
            f"简介：{c['description']}" if c.get('description') else "",
            f"背景：{c['lore_text']}" if c.get('lore_text') else "",
        ]
        text = "\n".join(p for p in parts if p)
        char_lines.append(text)

    char_path = RAG_DIR / "characters.txt"
    char_path.write_text("\n\n---\n\n".join(char_lines), encoding="utf-8")
    logger.info("RAG characters: %d entries → %s", len(char_lines), char_path)

    # 世界观文本
    lore_lines = []
    for l in lore:
        text = f"标题：{l['title']}\n\n{l.get('content', '')}"
        lore_lines.append(text)

    lore_path = RAG_DIR / "world_lore.txt"
    lore_path.write_text("\n\n---\n\n".join(lore_lines), encoding="utf-8")
    logger.info("RAG lore: %d entries → %s", len(lore_lines), lore_path)


# ── 保存结构化 JSON（备查）──────────────────────────────────


def export_json(characters: list[dict], lore: list[dict]):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    (PROCESSED_DIR / "characters.json").write_text(
        json.dumps(characters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (PROCESSED_DIR / "world_lore.json").write_text(
        json.dumps(lore, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Exported JSON to %s", PROCESSED_DIR)


# ── 主流程 ──────────────────────────────────────────────────


def run_merge():
    """执行完整合并流程"""
    logger.info("=" * 50)
    logger.info("Loading raw data...")

    wiki_chars = load_raw_json("wiki_characters.json")
    wiki_lore = load_raw_json("wiki_lore.json")
    dimbreath_avatars = load_dimbreath_json("avatars")
    text_map = build_text_map()

    logger.info(
        "Loaded: %d wiki chars, %d wiki lore, %d dimbreath avatars, %d textmap entries",
        len(wiki_chars) if isinstance(wiki_chars, list) else 0,
        len(wiki_lore) if isinstance(wiki_lore, list) else 0,
        len(dimbreath_avatars),
        len(text_map),
    )

    if isinstance(wiki_chars, list):
        characters = merge_characters(wiki_chars, dimbreath_avatars, text_map)
    else:
        characters = []

    if not isinstance(wiki_lore, list):
        wiki_lore = []

    # 输出
    db_path = PROCESSED_DIR / "genshin_data.db"
    create_db(characters, wiki_lore, db_path)
    export_rag_texts(characters, wiki_lore)
    export_json(characters, wiki_lore)

    # 统计
    conn = sqlite3.connect(str(db_path))
    counts = {}
    for table in ["characters", "weapons", "world_lore"]:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = row[0]
    conn.close()

    logger.info("=" * 50)
    logger.info("Merge complete: %s", counts)

    return db_path
