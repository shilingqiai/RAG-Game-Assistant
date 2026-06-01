"""游戏伴侣核心模块 — 整合 LLM / Agent / RAG 知识库"""
import os
from typing import AsyncGenerator, Optional, List

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.tools import FunctionTool

from config import UserProfile
from llm_manager import LLMManager
# GameAgent 在 _initialize() 中延迟导入，避免触发 Settings.embed_model 过早解析


class GameCompanion:
    """游戏伴侣主类 — 统一入口"""

    def __init__(self, storage_dir: str = "./storage"):
        self.storage_dir = storage_dir
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None
        self.user_profile: Optional[UserProfile] = None
        self.llm_manager: Optional[LLMManager] = None
        self.agent: Optional[GameAgent] = None
        self._initialize()

    def _initialize(self):
        """初始化所有组件"""
        self.llm_manager = LLMManager()
        self.user_profile = UserProfile()

        # 加载 / 构建 RAG 知识库
        if os.path.exists(self.storage_dir) and os.listdir(self.storage_dir):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
                self.index = load_index_from_storage(storage_context)
                self.query_engine = self.index.as_query_engine(
                    similarity_top_k=3,
                    response_mode="compact",
                )
                print("[Core] Loaded RAG index from storage")
            except Exception as e:
                print(f"[Core] Failed to load index: {e}")
        else:
            print("[Core] No index found — RAG disabled")

        # 构建 RAG 工具（如果有索引）
        rag_tools = [self._create_rag_tool()] if self.query_engine else []

        # 延迟导入 Agent（此时 Settings.embed_model 已设置）
        from agent import GameAgent

        # 初始化 Agent
        self.agent = GameAgent(
            llm_manager=self.llm_manager,
            user_profile=self.user_profile,
            extra_tools=rag_tools,
        )
        print("[Core] Game companion initialized")

    # ── RAG 工具 ──────────────────────────────────────────────

    def _create_rag_tool(self) -> FunctionTool:
        """将本地知识库查询包装为 FunctionTool"""

        def search_game_knowledge(query: str) -> str:
            """Search the Genshin Impact knowledge base for lore, characters, story, and world setting. Provide a specific search query."""
            if not self.query_engine:
                return "知识库未初始化"
            logger = __import__("logging").getLogger("paimon.rag")
            logger.info("RAG query: %s", query[:100])
            response = self.query_engine.query(query)
            result = str(response)
            logger.info("RAG result: %s", result[:200])
            return result

        return FunctionTool.from_defaults(fn=search_game_knowledge)

    # ── 对话接口 ──────────────────────────────────────────────

    async def chat_stream(self, query: str) -> AsyncGenerator[str, None]:
        """流式对话 — Agent 工具调用 + RAG + 多轮记忆 + token 级输出"""
        async for token in self.agent.chat_stream(query):
            yield token

    def query_lore(self, query: str, top_k: int = 3) -> str:
        """直接查询本地知识库（不经过 Agent）"""
        if not self.query_engine:
            return "Error: Index not initialized."

        response = self.query_engine.query(query)
        return str(response)


if __name__ == "__main__":
    import asyncio

    async def main():
        companion = GameCompanion()
        while True:
            user_input = input("\nPaimon is ready! (type 'quit' to exit): ")
            if user_input.lower() == "quit":
                print("Session ended")
                break
            full = ""
            async for token in companion.chat_stream(user_input):
                full += token
                print(token, end="", flush=True)
            print()

    asyncio.run(main())
