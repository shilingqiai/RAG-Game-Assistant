"""游戏伴侣核心 — RAG + Agent"""
import logging, os
from typing import AsyncGenerator

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever

from config import UserProfile
from llm_manager import LLMManager

logger = logging.getLogger("paimon.core")


class GameCompanion:
    def __init__(self, storage_dir="./storage"):
        self.query_engine = None
        self.agent = None
        self._init(storage_dir)

    def _init(self, storage_dir):
        llm = LLMManager()
        profile = UserProfile()

        if os.path.exists(storage_dir) and os.listdir(storage_dir):
            try:
                ctx = StorageContext.from_defaults(persist_dir=storage_dir)
                index = load_index_from_storage(ctx)
                retriever = QueryFusionRetriever(
                    retrievers=[index.as_retriever(similarity_top_k=10)],
                    similarity_top_k=10, num_queries=3, mode="reciprocal_rerank",
                )
                self.query_engine = RetrieverQueryEngine(retriever=retriever)
                logger.info("RAG: %d nodes", len(index.docstore.docs))
            except Exception as e:
                logger.warning("RAG failed: %s", e)

        from agent import GameAgent
        self.agent = GameAgent(llm_manager=llm, user_profile=profile, query_engine=self.query_engine)
        logger.info("Companion ready")

    async def chat_stream(self, query: str) -> AsyncGenerator[str, None]:
        async for token in self.agent.chat_stream(query):
            yield token

    def query_lore(self, query: str) -> str:
        if not self.query_engine: return "RAG not initialized."
        resp = self.query_engine.query(query)
        result = str(resp)
        if hasattr(resp, "source_nodes"):
            parts = []
            for i, n in enumerate(resp.source_nodes[:3]):
                title = n.metadata.get("title", "?")
                s = n.score or 0
                parts.append(f"  [{i+1}] {title} (相关度: {s:.0%})" if s else f"  [{i+1}] {title}")
            if parts: result += "\n\n---\n📖 参考来源:\n" + "\n".join(parts)
        return result
