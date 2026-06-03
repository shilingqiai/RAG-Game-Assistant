"""中文向量索引构建器 — 基于 genshin-data 简体中文角色数据"""
import json
import os
import re
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import llm_manager  # noqa: 注入 Settings.embed_model
from config import AppConfig

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding

DATA_DIR = "data/genshin-data/src/data/chinese-simplified/characters"

# 用不到的数值字段
SKIP_FIELDS = {"ascension", "talent_materials", "outfits", "substat",
               "_id", "id", "release", "birthday", "cv", "domain", "gender"}

# 元素 / 武器 / 地区 中文名映射（genshin-data 有些字段是英文值）
ELEMENT_MAP = {"Pyro": "火", "Hydro": "水", "Anemo": "风", "Electro": "雷",
               "Dendro": "草", "Cryo": "冰", "Geo": "岩"}
WEAPON_MAP = {"Sword": "单手剑", "Claymore": "双手剑", "Polearm": "长柄武器",
              "Bow": "弓", "Catalyst": "法器"}
REGION_MAP = {"Mondstadt": "蒙德", "Liyue": "璃月", "Inazuma": "稻妻",
              "Sumeru": "须弥", "Fontaine": "枫丹", "Natlan": "纳塔", "Snezhnaya": "至冬"}


def _clean(text: str) -> str:
    """去掉 HTML 标签和多余空白"""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)   # 去 HTML
    text = re.sub(r"\{[^}]+\}", "", text)  # 去模板标记
    text = re.sub(r"\s+", " ", text)       # 合并空白
    return text.strip()


def _val(obj, key: str) -> str:
    """安全取字段：dict 取 name，字符串直接返回"""
    v = obj.get(key, "")
    if isinstance(v, dict):
        return str(v.get("name", v.get("id", "")))
    return str(v) if v else ""


def _make_document(char: dict) -> Document | None:
    """将一个角色 JSON 转为中文 Document"""
    name = _val(char, "name")
    if not name:
        return None

    parts = [f"角色：{name}"]

    # 称号
    title = _val(char, "title")
    if title and title != name:
        parts.append(f"称号：{title}")

    # 稀有度
    rarity = char.get("rarity", 0)
    if rarity and isinstance(rarity, (int, float)):
        parts.append(f"稀有度：{'⭐' * int(rarity)}")

    # 元素
    element = _val(char, "element")
    if element:
        cn = ELEMENT_MAP.get(element, "")
        parts.append(f"神之眼：{cn}" if cn else f"神之眼：{element}")

    # 武器
    weapon = _val(char, "weapon_type")
    if weapon:
        cn = WEAPON_MAP.get(weapon, "")
        parts.append(f"武器：{cn}" if cn else f"武器：{weapon}")

    # 地区
    region = _val(char, "region")
    if region:
        cn = REGION_MAP.get(region, "")
        parts.append(f"所属：{cn}" if cn else f"所属：{region}")

    #  affiliation
    affiliation = char.get("affiliation", "")
    if isinstance(affiliation, dict):
        affiliation = affiliation.get("name", affiliation.get("id", ""))
    if affiliation:
        parts.append(f" affiliation：{affiliation}")

    constellation = char.get("constellation", "")
    if isinstance(constellation, dict):
        constellation = constellation.get("name", constellation.get("id", ""))
    if constellation:
        parts.append(f"命之座：{constellation}")

    # 角色描述
    desc = _clean(char.get("description", ""))
    if desc:
        parts.append(f"\n角色简介：{desc}")

    # 技能描述（只取描述，不要数值）
    skills = char.get("skills", [])
    if skills:
        parts.append("\n【技能】")
        for sk in skills:
            sk_name = _clean(sk.get("name", ""))
            sk_desc = _clean(sk.get("description", ""))
            if sk_name and sk_desc:
                parts.append(f"  {sk_name}：{sk_desc}")

    # 固有天赋
    passives = char.get("passives", [])
    if passives:
        parts.append("\n【固有天赋】")
        for ps in passives:
            ps_name = _clean(ps.get("name", ""))
            ps_desc = _clean(ps.get("description", ""))
            if ps_name and ps_desc:
                parts.append(f"  {ps_name}：{ps_desc}")

    # 命之座（每层都有故事性描述）
    constellations = char.get("constellations", [])
    if constellations:
        parts.append("\n【命之座】")
        for cn in constellations:
            cn_name = _clean(cn.get("name", ""))
            cn_desc = _clean(cn.get("description", ""))
            if cn_name and cn_desc:
                parts.append(f"  {cn_name}：{cn_desc}")

    text = "\n".join(parts)
    return Document(
        text=text,
        metadata={
            "title": name,
            "category": "character",
            "language": "zh",
            "source": "genshin-data",
            "element": element,
            "weapon": weapon,
            "region": region,
        },
    )


def build_documents() -> list[Document]:
    """加载所有角色"""
    docs = []
    if not os.path.isdir(DATA_DIR):
        print(f"❌ 数据目录不存在: {DATA_DIR}")
        print("   请确保已初始化 git submodule: git submodule update --init")
        return docs

    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(DATA_DIR, fn), encoding="utf-8") as f:
            char = json.load(f)
        doc = _make_document(char)
        if doc:
            docs.append(doc)

    # 统计
    total_chars = sum(1 for d in docs if d.text)
    print(f"✅ 加载 {len(docs)} 个中文角色文档 ({total_chars} 有效)")
    return docs


def build_index(documents: list[Document], storage_dir: str = "./storage"):
    """构建向量索引"""
    api_key = os.getenv("DASHSCOPE_API_KEY")

    embed_model = DashScopeEmbedding(
        model_name=AppConfig.EMBED_MODEL,
        api_key=api_key,
        embed_batch_size=10,
    )

    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)

    print(f"🔧 构建中文索引 ({AppConfig.EMBED_MODEL})，{len(documents)} 个文档...")
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        transformations=[text_splitter],
        show_progress=True,
    )
    os.makedirs(storage_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=storage_dir)
    print(f"✅ 中文索引已保存: {storage_dir} ({len(index.docstore.docs)} nodes)")
    return index


def main():
    """CLI: paimon-index-cn"""
    documents = build_documents()
    if not documents:
        return
    build_index(documents, "./storage")


if __name__ == "__main__":
    main()
