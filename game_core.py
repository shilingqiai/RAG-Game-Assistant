"""游戏伴侣核心 — RAG + Agent（混合检索：向量 + BM25 → RRF 融合）"""
import hashlib, logging, os, re, time
from typing import AsyncGenerator

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle

from rank_bm25 import BM25Okapi

from config import UserProfile, AppConfig
from llm_manager import LLMManager

logger = logging.getLogger("paimon.core")


# ── BM25 分词 ----------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """中文字符单字 + 英文单词分词，供 BM25 使用"""
    tokens = []
    for ch in re.findall(r"[一-鿿]", text):
        tokens.append(ch)
    for w in re.findall(r"[a-zA-Z0-9]+", text):
        tokens.append(w.lower())
    return tokens


# ── 混合检索器 ----------------------------------------------------

class HybridRetriever:
    """向量检索 + BM25 关键词检索 → RRF 融合

    - 向量检索捕捉语义相似
    - BM25 捕捉精确关键词（人名、地名、术语）
    - RRF (Reciprocal Rank Fusion) 合并排序，无需调权重
    - 向量分数低于 min_score 的节点不进入融合（减少噪音）
    """

    def __init__(self, vector_retriever, nodes, top_k=5, min_score=0.3):
        self._vector = vector_retriever
        self._top_k = top_k
        self._min_score = min_score

        # 构建 BM25 索引
        self._nodes = list(nodes)
        self._corpus = [_tokenize(n.get_content()) for n in self._nodes]
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def retrieve(self, query) -> list[NodeWithScore]:
        """统一检索入口（兼容 str 和 QueryBundle）"""
        if isinstance(query, QueryBundle):
            query_str = query.query_str
        else:
            query_str = str(query)

        # 1. 向量检索
        vec_results = self._vector.retrieve(query)

        # 2. BM25 检索
        bm25_pairs: list[tuple] = []  # (node, bm25_score)
        if self._bm25 and self._nodes:
            tokenized = _tokenize(query_str)
            if tokenized:
                scores = self._bm25.get_scores(tokenized)
                indexed = [
                    (self._nodes[i], scores[i])
                    for i in range(len(self._nodes))
                    if scores[i] > 0
                ]
                indexed.sort(key=lambda x: x[1], reverse=True)
                bm25_pairs = indexed[: self._top_k * 2]

        # 3. RRF 融合 (k=60 是经典参数)
        K = 60
        rrf: dict[str, float] = {}
        node_map: dict[str, tuple] = {}  # node_id → (node, vec_score)

        for rank, nws in enumerate(vec_results):
            node = nws.node
            score = nws.score or 0
            if score < self._min_score:
                continue  # 低相关度过滤
            node_map[node.node_id] = (node, score)
            rrf[node.node_id] = 1.0 / (K + rank + 1)

        for rank, (node, _bm25_score) in enumerate(bm25_pairs):
            if node.node_id in rrf:
                rrf[node.node_id] += 1.0 / (K + rank + 1)
            else:
                rrf[node.node_id] = 1.0 / (K + rank + 1)
                node_map[node.node_id] = (node, 0.0)  # 仅 BM25 命中

        # 4. 按 RRF 分数降序
        sorted_ids = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[
            : self._top_k
        ]

        results = []
        for nid, rrf_score in sorted_ids:
            node, vec_score = node_map[nid]
            nws = NodeWithScore(node=node, score=vec_score)  # 展示用向量分数
            results.append(nws)

        return results


# ── 游戏伴侣 ----------------------------------------------------


class GameCompanion:
    def __init__(self, storage_dir="./storage"):
        self.query_engine = None
        self.agent = None
        self._rag_cache: dict[str, tuple[float, str, str]] = {}
        self._init(storage_dir)

    def _init(self, storage_dir):
        llm = LLMManager()
        profile = UserProfile()

        if os.path.exists(storage_dir) and os.listdir(storage_dir):
            try:
                ctx = StorageContext.from_defaults(persist_dir=storage_dir)
                index = load_index_from_storage(ctx)

                # 向量检索器
                vec_retriever = index.as_retriever(
                    similarity_top_k=AppConfig.RETRIEVAL_TOP_K
                )
                # 混合检索器（向量 + BM25 → RRF 融合）
                nodes = index.docstore.docs.values()
                hybrid = HybridRetriever(
                    vector_retriever=vec_retriever,
                    nodes=nodes,
                    top_k=AppConfig.RETRIEVAL_TOP_K,
                    min_score=AppConfig.RAG_MIN_SCORE,
                )
                self.query_engine = RetrieverQueryEngine(retriever=hybrid)

                logger.info(
                    "RAG: %d nodes | hybrid (vec+BM25) | top_k=%d | min_score=%.2f",
                    len(index.docstore.docs),
                    AppConfig.RETRIEVAL_TOP_K,
                    AppConfig.RAG_MIN_SCORE,
                )
            except Exception as e:
                logger.warning("RAG failed: %s", e)

        from agent import GameAgent

        self.agent = GameAgent(
            llm_manager=llm, user_profile=profile, query_engine=self.query_engine
        )
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
                if s > 0:
                    parts.append(f"  [{i+1}] {title} (相关度: {s:.0%})")
                else:
                    parts.append(f"  [{i+1}] {title}")
            if parts:
                sources = "\n\n---\n📖 参考来源:\n" + "\n".join(parts)

        # 写入缓存
        if AppConfig.RAG_CACHE_TTL > 0:
            self._rag_cache[cache_key] = (time.time(), result, sources)
            if len(self._rag_cache) > 200:
                now = time.time()
                self._rag_cache = {
                    k: v
                    for k, v in self._rag_cache.items()
                    if now - v[0] < AppConfig.RAG_CACHE_TTL
                }

        return result + sources
