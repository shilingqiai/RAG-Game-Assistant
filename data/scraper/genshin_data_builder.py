"""中文 RAG 索引构建器 — genshin-data + 本地 BGE embedding（免费离线）"""
import json, os, shutil, sys

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import llm_manager  # noqa: 注入 Settings
from config import AppConfig

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding

DATA = "data/genshin-data/src/data/chinese-simplified"


def _v(obj, key, default=""):
    v = obj.get(key, default)
    if isinstance(v, dict): return v.get("name", v.get("id", str(v)))
    if isinstance(v, list): return ", ".join(str(x) for x in v)
    return str(v) if v else default


def load_category(category: str, label: str, fields: list[tuple[str, str]]) -> list[Document]:
    dirpath = os.path.join(DATA, category)
    if not os.path.isdir(dirpath):
        return []
    docs = []
    for fn in sorted(os.listdir(dirpath)):
        if not fn.endswith(".json"): continue
        with open(os.path.join(dirpath, fn), encoding="utf-8") as f:
            obj = json.load(f)
        parts = [f"{label}：{obj.get('name', '')}"]
        for key, prefix in fields:
            val = obj.get(key)
            if val:
                if isinstance(val, str) and len(val) > 10:
                    parts.append(f"{prefix}：{val}")
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            n, d = item.get("name"), item.get("description")
                            if n and d: parts.append(f"{n}：{d}")
        docs.append(Document(
            text="\n".join(parts),
            metadata={"title": obj.get("name", ""), "category": category, "language": "zh", "source": "genshin-data"},
        ))
    print(f"  {label}: {len(docs)}")
    return docs


def build_documents():
    print("Loading categories:")
    docs = []
    docs += load_category("characters", "角色", [("title","称号"),("description","简介"),("affiliation","所属")])
    # docs += load_category("weapons", "武器", [("description","描述")])
    # docs += load_category("artifacts", "圣遗物", [("description","描述")])
    docs += load_category("geography", "地区", [("description","描述")])
    print(f"  Total: {len(docs)}")
    return docs


def build_index(documents, storage_dir="./storage"):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    embed_model = DashScopeEmbedding(model_name=AppConfig.EMBED_MODEL, api_key=api_key, embed_batch_size=10)

    pipeline = IngestionPipeline(transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=128),
        embed_model,
    ])

    print(f"Building index from {len(documents)} documents...")
    nodes = pipeline.run(documents=documents, show_progress=True)
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir=storage_dir)
    print(f"Done: {len(nodes)} nodes → {storage_dir}")
    return index


def main():
    docs = build_documents()
    if os.path.exists("./storage"): shutil.rmtree("./storage")
    build_index(docs)


if __name__ == "__main__":
    main()
