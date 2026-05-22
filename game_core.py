import os
import logging
from typing import Optional
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.dashscope import DashScope
from llama_index.core import Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GameCompanion:
    def __init__(self, storage_dir: str = "./storage"):
        self.storage_dir = storage_dir
        self.index = None
        self.llm = None
        self._initialize()

    def _initialize(self):
        Settings.llm = DashScope(
            model_name="qwen-plus",
            temperature=0.1,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        self.llm = Settings.llm
        
        Settings.embed_model = DashScopeEmbedding(
            model_name="text-embedding-v3",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            embed_batch_size=10,
        )

        if os.path.exists(self.storage_dir) and os.listdir(self.storage_dir):
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            self.index = load_index_from_storage(storage_context)
            print(f"\n✅ 已从 {self.storage_dir} 加载索引")
        else:
            raise ValueError(f"No index found in {self.storage_dir}. Run data_indexer.py first.")

    def query_lore(self, query: str, top_k: int = 3) -> str:
        if not self.index:
            return "Error: Index not initialized."

        print("\n" + "="*80)
        print("🔍 [RAG检索开始]")
        print(f"📝 用户查询: {query}")
        print("="*80)

        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact",
        )
        response = query_engine.query(query)

        print("\n📚 [检索到的原文片段]:")
        print("-"*60)
        for idx, node in enumerate(response.source_nodes, 1):
            print(f"\n📍 来源 #{idx} (相关性: {node.score:.4f})")
            print(f"标题: {node.node.metadata.get('title', '未知')}")
            print(f"内容:\n{node.node.text[:500]}..." if len(node.node.text) > 500 else f"内容:\n{node.node.text}")
            print("-"*60)

        print("\n🎯 [检索结果摘要]:")
        print("-"*60)
        print(str(response))
        print("-"*60)

        print("\n✅ [RAG检索完成]")
        print("="*80 + "\n")

        return str(response)

    def chat(self, user_message: str) -> str:
        print(f"\n\n🚀 开始处理用户消息: {user_message}")
        
        context = self.query_lore(user_message)
        
        print("\n🤖 正在调用大模型生成回答...")
        
        system_prompt = """
You are a helpful Elden Ring game companion. 
Use the following context to answer the user's question:

{context}

If the context doesn't contain the answer, say you don't have that information.
Always provide clear, concise answers in a friendly tone.
""".format(context=context)

        response = self.llm.complete(
            f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        )
        
        print(f"\n🎬 大模型回答完成")
        
        return str(response)


if __name__ == "__main__":
    companion = GameCompanion()
    while True:
        user_input = input("\n💬 请输入关于 Elden Ring 的问题 (输入 'quit' 退出): ")
        if user_input.lower() == "quit":
            print("👋 会话结束")
            break
        response = companion.chat(user_input)
        print(f"\n🌟 Companion 回答:\n{response}\n")