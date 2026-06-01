import json
import os
import re
import sys

# Fix Windows encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dashscope
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)


def load_game_data(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    text = text.strip()
    text = " ".join(text.split())
    return text


def split_by_sentences(text: str) -> list[str]:
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def create_documents(game_data: list) -> list[Document]:
    documents = []
    for item in game_data:
        title = clean_text(item.get("title", ""))
        content = clean_text(item.get("content", ""))
        
        text = f"{title}\n\n{content}"
        
        doc = Document(
            text=text,
            metadata={
                "title": title,
                "source": "genshin_impact_lore.json",
                "category": item.get("category", "general"),
                "sentence_count": len(split_by_sentences(content)),
            }
        )
        documents.append(doc)
    return documents


def build_index(
    documents: list[Document], storage_dir: str = "./storage"
) -> VectorStoreIndex:
    if os.path.exists(storage_dir) and os.listdir(storage_dir):
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        print(f"✅ 加载已存在的索引: {storage_dir}")
    else:
        from llama_index.embeddings.dashscope import DashScopeEmbedding
        
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        
        embed_model = DashScopeEmbedding(
            model_name="text-embedding-v3",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            embed_batch_size=10,
        )
        
        print("🔧 初始化智能文本切分器 (chunk_size=512, chunk_overlap=128, min_sentences=1)...")
        
        from llama_index.core.text_splitter import SentenceSplitter
        
        text_splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=128,
        )
        
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            transformations=[text_splitter],
            show_progress=True,
        )
        index.storage_context.persist(persist_dir=storage_dir)
        print(f"✅ 创建并保存新索引到: {storage_dir}")
    return index


def main():
    data_path = "./data/genshin_impact_lore.json"
    storage_dir = "./storage"

    os.makedirs(storage_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据文件不存在: {data_path}")
        return

    print("📥 加载游戏数据...")
    game_data = load_game_data(data_path)
    
    print("📝 创建文档对象...")
    documents = create_documents(game_data)
    
    print(f"🔄 处理 {len(documents)} 个文档...")
    index = build_index(documents, storage_dir)

    print(f"\n🎉 成功处理 {len(documents)} 个文档")


if __name__ == "__main__":
    main()