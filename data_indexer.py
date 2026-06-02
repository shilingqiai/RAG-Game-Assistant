"""数据索引构建器 — 从 characters.txt + world_lore.txt 构建向量索引"""
import os, re, sys

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding
from config import AppConfig


def _split_entries(text: str) -> list[str]:
    """按 --- 分隔符拆分条目（允许前后有空白行）"""
    return [block.strip() for block in re.split(r'\n{0,2}---\n{0,2}', text) if block.strip()]


def _load_characters(path: str) -> list[Document]:
    """加载角色文本，每个角色一个 Document"""
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    docs = []
    for block in _split_entries(raw):
        # 提取角色名作为 title
        title_match = re.search(r'^角色[：:]\s*(.+)', block, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "Unknown"

        docs.append(Document(
            text=block,
            metadata={"title": title, "category": "character", "source": "rag_texts"},
        ))
    print(f"  Characters: {len(docs)} documents")
    return docs


def _load_world_lore(path: str) -> list[Document]:
    """加载世界观文本，每个条目一个 Document"""
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    docs = []
    for block in _split_entries(raw):
        title_match = re.search(r'^标题[：:]\s*(.+)', block, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "Unknown"

        docs.append(Document(
            text=block,
            metadata={"title": title, "category": "lore", "source": "rag_texts"},
        ))
    print(f"  World lore: {len(docs)} documents")
    return docs


def build_index(documents: list[Document], storage_dir: str = "./storage") -> VectorStoreIndex:
    """构建或加载向量索引"""
    api_key = os.getenv("DASHSCOPE_API_KEY")

    if os.path.exists(storage_dir) and os.listdir(storage_dir):
        print(f"Loading existing index: {storage_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        print(f"Loaded index: {len(index.docstore.docs)} nodes")
        return index

    embed_model = DashScopeEmbedding(
        model_name=AppConfig.EMBED_MODEL,
        api_key=api_key,
        embed_batch_size=10,
    )

    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)

    print(f"Building new index ({AppConfig.EMBED_MODEL}) from {len(documents)} documents...")
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        transformations=[text_splitter],
        show_progress=True,
    )
    index.storage_context.persist(persist_dir=storage_dir)
    print(f"Index saved: {storage_dir}")
    return index


def main():
    data_dir = "./data/rag_texts"
    storage_dir = "./storage"

    if not os.path.exists(data_dir):
        print(f"Data dir not found: {data_dir}")
        return

    chars_path = os.path.join(data_dir, "characters.txt")
    lore_path = os.path.join(data_dir, "world_lore.txt")

    documents = []
    if os.path.exists(chars_path):
        documents.extend(_load_characters(chars_path))
    else:
        print(f"Missing: {chars_path}")

    if os.path.exists(lore_path):
        documents.extend(_load_world_lore(lore_path))
    else:
        print(f"Missing: {lore_path}")

    if not documents:
        print("No documents to index!")
        return

    print(f"Total documents: {len(documents)}")
    index = build_index(documents, storage_dir)
    print(f"Done! Nodes in index: {len(index.docstore.docs)}")


if __name__ == "__main__":
    main()
