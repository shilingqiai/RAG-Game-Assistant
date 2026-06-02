"""混合索引构建器 — 中文 Wiki + 英文 Wiki → 统一 RAG 索引"""
import json
import os
import shutil
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import llm_manager  # noqa
from config import AppConfig

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding


ELEMENT_MAP = {"Pyro":"火","Hydro":"水","Anemo":"风","Electro":"雷","Dendro":"草","Cryo":"冰","Geo":"岩"}
WEAPON_MAP = {"Sword":"单手剑","Claymore":"双手剑","Polearm":"长柄武器","Bow":"弓","Catalyst":"法器"}
REGION_MAP = {"Mondstadt":"蒙德","Liyue":"璃月","Inazuma":"稻妻","Sumeru":"须弥","Fontaine":"枫丹","Natlan":"纳塔","Snezhnaya":"至冬"}


def build_documents() -> list[Document]:
    """构建混合文档：中文角色 + 英文角色 + 世界观"""
    docs = []

    # ── 1. 中文角色（Fandom ZH）──
    cn_path = "data/raw/fandom_cn_characters.json"
    if os.path.exists(cn_path):
        with open(cn_path, encoding="utf-8") as f:
            cn_chars = json.load(f)
        for c in cn_chars:
            # 过滤武器页面
            if "天空之" in c.get("name_cn", ""):
                continue
            parts = [f"角色：{c.get('name_cn', c.get('name_en', ''))}"]
            if c.get("name_en"): parts.append(f"英文名：{c['name_en']}")
            if c.get("rarity"): parts.append(f"稀有度：{'⭐' * c['rarity']}")
            if c.get("element"): parts.append(f"元素：{c['element']}")
            if c.get("weapon_type"): parts.append(f"武器：{c['weapon_type']}")
            if c.get("region"): parts.append(f"地区：{c['region']}")
            if c.get("lore_text"): parts.append(f"背景：{c['lore_text'][:2000]}")
            docs.append(Document(
                text="\n".join(parts),
                metadata={"title": c.get("name_cn", ""), "category": "character", "language": "zh", "source": "fandom_cn"},
            ))
        print(f"CN characters: {len(docs)}")

    # ── 2. 英文角色（Fandom EN + 中文名标签）──
    en_path = "data/processed/characters.json"
    cn_name_map = {}
    if os.path.exists(cn_path):
        with open(cn_path, encoding="utf-8") as f:
            for c in json.load(f):
                if c.get("name_cn"):
                    cn_name_map[c["name_en"].lower()] = c["name_cn"]

    if os.path.exists(en_path):
        with open(en_path, encoding="utf-8") as f:
            en_chars = json.load(f)

        en_count = 0
        for c in en_chars:
            name_en = c["name_en"]
            name_cn = cn_name_map.get(name_en.lower(), "")
            cn_label = f"（{name_cn}）" if name_cn else ""

            parts = [f"角色：{name_en}{cn_label}"]
            if name_cn: parts.append(f"中文名：{name_cn}")
            if c.get("rarity"): parts.append(f"稀有度：{'⭐' * c['rarity']} Rarity: {c['rarity']}★")
            if c.get("element"):
                cn_el = ELEMENT_MAP.get(c["element"], "")
                parts.append(f"元素：{cn_el}（{c['element']}）" if cn_el else f"元素：{c['element']}")
            if c.get("weapon_type"):
                cn_wp = WEAPON_MAP.get(c["weapon_type"], "")
                parts.append(f"武器：{cn_wp}（{c['weapon_type']}）" if cn_wp else f"武器：{c['weapon_type']}")
            if c.get("region"):
                cn_rg = REGION_MAP.get(c["region"], "")
                parts.append(f"地区：{cn_rg}（{c['region']}）" if cn_rg else f"地区：{c['region']}")
            if c.get("description"): parts.append(f"简介：{c['description']}")
            if c.get("lore_text"): parts.append(f"背景：{c['lore_text'][:2000]}")

            docs.append(Document(
                text="\n".join(parts),
                metadata={"title": f"{name_cn}（{name_en}）" if name_cn else name_en, "name_en": name_en, "name_cn": name_cn, "category": "character", "language": "bilingual", "source": "fandom_en"},
            ))
            en_count += 1
        print(f"EN characters: {en_count}")

    # ── 3. 世界观（英文 + 中文翻译）──
    lore_paths = ["data/raw/wiki_lore.json", "data/raw/bilibili_lore.json"]
    lore_count = 0
    for lp in lore_paths:
        if not os.path.exists(lp): continue
        with open(lp, encoding="utf-8") as f:
            lore = json.load(f)
        for l in lore:
            docs.append(Document(
                text=f"标题：{l.get('title', '')}\n\n{l.get('content', '')[:3000]}",
                metadata={"title": l.get("title", ""), "category": "lore", "language": l.get("language", "en"), "source": l.get("source", "wiki")},
            ))
            lore_count += 1
    print(f"Lore: {lore_count}")

    return docs


def build_hybrid_index(documents: list[Document]):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    embed_model = DashScopeEmbedding(model_name=AppConfig.EMBED_MODEL, api_key=api_key, embed_batch_size=10)

    pipeline = IngestionPipeline(transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=128),
        # KeywordExtractor 省略（305 nodes × LLM 调用太慢，metadata 已足够）
        embed_model,
    ])

    print(f"\nBuilding hybrid index from {len(documents)} documents...")
    nodes = pipeline.run(documents=documents, show_progress=True)
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir="./storage")

    zh = sum(1 for d in documents if d.metadata.get("language") == "zh")
    en = sum(1 for d in documents if d.metadata.get("language") == "en")
    bi = sum(1 for d in documents if d.metadata.get("language") == "bilingual")
    print(f"Hybrid Index: {len(nodes)} nodes ({zh} zh + {en} en + {bi} bilingual)")
    return index


def main():
    documents = build_documents()
    if not documents:
        print("No data! Run fetchers first.")
        return

    if os.path.exists("./storage"):
        shutil.rmtree("./storage")

    build_hybrid_index(documents)
    print("Done! Run: python eval_rag.py")


if __name__ == "__main__":
    main()
