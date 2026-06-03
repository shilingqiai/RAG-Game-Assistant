"""游戏伴侣核心 — RAG + Agent"""
import hashlib, logging, os, time
from typing import AsyncGenerator

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine

from config import UserProfile, AppConfig
from llm_manager import LLMManager

logger = logging.getLogger("paimon.core")


class GameCompanion:
    def __init__(self, storage_dir="./storage"):
        self.query_engine = None
        self.agent = None
        self._rag_cache: dict[str, tuple[float, str, str]] = {}  # key → (timestamp, text, sources)
        self._init(storage_dir)

    def _init(self, storage_dir):
        llm = LLMManager()
        profile = UserProfile()

        if os.path.exists(storage_dir) and os.listdir(storage_dir):
            try:
                ctx = StorageContext.from_defaults(persist_dir=storage_dir)
                index = load_index_from_storage(ctx)
                # 直接检索器 — 140 节点不需要 QueryFusion（省掉 3 次 LLM 查询改写 + 3 次检索）
                retriever = index.as_retriever(similarity_top_k=AppConfig.RETRIEVAL_TOP_K)
                self.query_engine = RetrieverQueryEngine(retriever=retriever)
                logger.info("RAG: %d nodes (direct retriever, top_k=%d)",
                            len(index.docstore.docs), AppConfig.RETRIEVAL_TOP_K)
            except Exception as e:
                logger.warning("RAG failed: %s", e)

        from agent import GameAgent
        self.agent = GameAgent(llm_manager=llm, user_profile=profile, query_engine=self.query_engine)
        logger.info("Companion ready")

    async def chat_stream(self, query: str) -> AsyncGenerator[str, None]:
        async for token in self.agent.chat_stream(query):
            yield token

    def query_lore(self, query: str) -> str:
        """RAG 查询（带缓存），供 eval_rag.py 使用"""
        if not self.query_engine:
            return "RAG not initialized."

        # 缓存检查
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if AppConfig.RAG_CACHE_TTL > 0 and cache_key in self._rag_cache:
            ts, text, sources = self._rag_cache[cache_key]
            if time.time() - ts < AppConfig.RAG_CACHE_TTL:
                logger.info("RAG cache hit: %s", query[:40])
                return text + sources

        # 检索
        resp = self.query_engine.query(query)
        result = str(resp)

        # 来源引用
        sources = ""
        if hasattr(resp, "source_nodes"):
            parts = []
            for i, n in enumerate(resp.source_nodes[:3]):
                title = n.metadata.get("title", "?")
                s = n.score or 0
                parts.append(f"  [{i+1}] {title} (相关度: {s:.0%})" if s else f"  [{i+1}] {title}")
            if parts:
                sources = "\n\n---\n📖 参考来源:\n" + "\n".join(parts)

        # 写入缓存
        if AppConfig.RAG_CACHE_TTL > 0:
            self._rag_cache[cache_key] = (time.time(), result, sources)
            # 清理过期缓存
            if len(self._rag_cache) > 200:
                now = time.time()
                self._rag_cache = {
                    k: v for k, v in self._rag_cache.items()
                    if now - v[0] < AppConfig.RAG_CACHE_TTL
                }

        return result + sources
