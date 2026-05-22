import json
import os
import dashscope
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter


def load_game_data(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_documents(game_data: list) -> list[Document]:
    documents = []
    for item in game_data:
        text = f"Title: {item['title']}\n\nContent: {item['content']}"
        doc = Document(text=text)
        documents.append(doc)
    return documents


def build_index(
    documents: list[Document], storage_dir: str = "./storage"
) -> VectorStoreIndex:
    if os.path.exists(storage_dir) and os.listdir(storage_dir):
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        print(f"Loaded existing index from {storage_dir}")
    else:
        from llama_index.embeddings.dashscope import DashScopeEmbedding
        
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        
        embed_model = DashScopeEmbedding(
            model_name="text-embedding-v3",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            embed_batch_size=10,
        )
        
        node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            node_parser=node_parser,
            show_progress=True,
        )
        index.storage_context.persist(persist_dir=storage_dir)
        print(f"Created and saved new index to {storage_dir}")
    return index


def main():
    data_path = "./data/elden_ring_lore.json"
    storage_dir = "./storage"

    os.makedirs(storage_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    game_data = load_game_data(data_path)
    documents = create_documents(game_data)
    
    index = build_index(documents, storage_dir)

    print(f"Successfully processed {len(documents)} documents")


if __name__ == "__main__":
    main()