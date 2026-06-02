"""中文 RAG 索引构建器 — Bilibili Wiki 数据"""
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


def build_cn_documents(chars_path: str, lore_path: str) -> list[Document]:
    """构建中文文档"""
    docs = []

    # ── 角色文档 ──
    if os.path.exists(chars_path):
        with open(chars_path, encoding="utf-8") as f:
            characters = json.load(f)

        for c in characters:
            parts = [f"角色：{c.get('name_cn', c.get('name_en', ''))}"]
            if c.get("name_en"):
                parts.append(f"英文名：{c['name_en']}")
            if c.get("title"):
                parts.append(f"称号：{c['title']}")
            if c.get("rarity"):
                parts.append(f"稀有度：{'⭐' * c['rarity']}")
            if c.get("element"):
                parts.append(f"元素：{c['element']}")
            if c.get("weapon_type"):
                parts.append(f"武器：{c['weapon_type']}")
            if c.get("region"):
                parts.append(f"地区：{c['region']}")
            if c.get("affiliation"):
                parts.append(f"所属：{c['affiliation']}")
            if c.get("constellation"):
                parts.append(f"命之座：{c['constellation']}")
            if c.get("description"):
                parts.append(f"简介：{c['description']}")
            if c.get("lore_text"):
                parts.append(f"背景故事：{c['lore_text']}")

            docs.append(Document(
                text="\n".join(parts),
                metadata={
                    "title": c.get("name_cn", ""),
                    "name_en": c.get("name_en", ""),
                    "category": "character",
                    "rarity": c.get("rarity", 0),
                    "element": c.get("element", ""),
                    "region": c.get("region", ""),
                    "language": "zh",
                    "source": "bilibili_wiki",
                },
            ))

    # ── 世界观文档 ──
    if os.path.exists(lore_path):
        with open(lore_path, encoding="utf-8") as f:
            lore_entries = json.load(f)

        for l in lore_entries:
            docs.append(Document(
                text=f"标题：{l.get('title', '')}\n\n{l.get('content', '')}",
                metadata={
                    "title": l.get("title", ""),
                    "category": "lore",
                    "language": "zh",
                    "source": "bilibili_wiki",
                },
            ))

    return docs


def build_cn_index(documents: list[Document], storage_dir: str = "./storage_cn"):
    """构建中文向量索引"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    embed_model = DashScopeEmbedding(
        model_name=AppConfig.EMBED_MODEL, api_key=api_key, embed_batch_size=10,
    )

    pipeline = IngestionPipeline(transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=128),
        KeywordExtractor(keywords=5),
        embed_model,
    ])

    print(f"Building CN index from {len(documents)} documents...")
    nodes = pipeline.run(documents=documents, show_progress=True)

    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir=storage_dir)

    cn_chars = sum(1 for d in documents if d.metadata.get("category") == "character")
    cn_lore = sum(1 for d in documents if d.metadata.get("category") == "lore")
    print(f"CN Index: {len(nodes)} nodes ({cn_chars} characters, {cn_lore} lore)")
    return index


def main():
    chars_path = "data/raw/bilibili_characters.json"
    lore_path = "data/raw/bilibili_lore.json"

    if not os.path.exists(chars_path):
        print("No CN data found. Run: python data/scraper/bilibili_fetcher.py")
        return

    documents = build_cn_documents(chars_path, lore_path)
    print(f"CN documents: {len(documents)}")

    if os.path.exists("./storage_cn"):
        shutil.rmtree("./storage_cn")

    build_cn_index(documents)
    print("Done! Chinese RAG index ready at ./storage_cn")


if __name__ == "__main__":
    main()
