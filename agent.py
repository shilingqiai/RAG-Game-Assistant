"""Agent — 3-stage pipeline: intent routing -> tool execution -> LLM synthesis"""
import asyncio, datetime, logging, sys, time
from typing import AsyncGenerator, List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from config import UserProfile, AppConfig
from llm_manager import LLMManager

logger = logging.getLogger("paimon.agent")


def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


class GameAgent:
    def __init__(self, llm_manager: LLMManager, user_profile: UserProfile,
                 extra_tools=None, query_engine=None):
        self.llm = llm_manager.llm
        self.user_profile = user_profile
        self.query_engine = query_engine
        self.chat_history: List[ChatMessage] = []
        self._history_lock = asyncio.Lock()
        self._memory_summary: str = ""  # 历史对话摘要（旧轮次压缩结果）

        from llama_index.llms.dashscope import DashScope
        self.router = DashScope(model_name=AppConfig.ROUTER_MODEL, temperature=0,
                                api_key=llm_manager.llm.api_key, max_tokens=10)
        rag_status = "OK" if query_engine else "OFF(chat only)"
        print(f"[{_ts()}] Agent ready | RAG={rag_status}", flush=True)

    # ---- tools -------------------------------------------------

    def _get_profile(self) -> str:
        result = self.user_profile.format_for_display()
        print(f"[{_ts()}]    [asset] query done", flush=True)
        return result

    def _update_assets(self, asset_type: str, amount: int) -> str:
        return self.user_profile.update_assets(asset_type, amount)

    def _save_pref(self, text: str) -> str:
        return self.user_profile.save_preference(text)

    async def _search(self, query: str) -> str:
        from tools import search_web
        t0 = time.time()
        print(f"[{_ts()}]    [web] searching ...", flush=True)
        result = await search_web(query)
        print(f"[{_ts()}]    [web] done ({time.time() - t0:.1f}s)", flush=True)
        return result

    async def _rag_query(self, query: str) -> tuple[str, str]:
        if not self.query_engine:
            return "", ""
        import asyncio
        t0 = time.time()
        print(f"[{_ts()}]    [rag] retrieving ...", flush=True)
        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(self.query_engine.query, query),
                timeout=AppConfig.RAG_TIMEOUT,
            )
            text = str(resp).strip()
            node_count = len(resp.source_nodes) if hasattr(resp, "source_nodes") else 0
            elapsed = time.time() - t0

            # 无相关节点 → 不编造答案
            if node_count == 0:
                print(f"[{_ts()}]    [rag] done ({elapsed:.1f}s, 0 nodes -> fallback)", flush=True)
                return "", ""

            print(f"[{_ts()}]    [rag] done ({elapsed:.1f}s, {node_count} nodes)", flush=True)

            sources = ""
            if hasattr(resp, "source_nodes"):
                parts = []
                for i, n in enumerate(resp.source_nodes[:3]):
                    title = n.metadata.get("title", "?")
                    s = n.score or 0
                    if s > 0:
                        parts.append(f"  [{i+1}] {title} (相关度: {s:.0%})")
                    else:
                        parts.append(f"  [{i+1}] {title} (关键词匹配)")
                if parts:
                    sources = "\n\n---\nSources:\n" + "\n".join(parts)
            return text, sources
        except asyncio.TimeoutError:
            print(f"[{_ts()}]    [warn] RAG timeout ({AppConfig.RAG_TIMEOUT}s)", flush=True)
            return "", ""
        except Exception as e:
            print(f"[{_ts()}]    [err] RAG error: {e}", flush=True)
            return "", ""

    # ---- intent classify ---------------------------------------

    async def _classify(self, query: str) -> str:
        prompt = f"""你是一个意图分类器。分析用户消息，只输出一个词：asset / lore / web / chat

意图说明：
- asset：用户查询自己的库存（原石、纠缠之缘、抽卡次数、垫水位），不是问角色
- lore：询问角色设定、世界观、剧情、背景故事（即使问"怎么获得"也是问设定）
- web：需要最新网络信息（卡池、活动、兑换码、版本更新）
- chat：闲聊、表达感受、无明确查询目标、问AI能力

示例：
我有多少原石 → asset
垫了多少抽 → asset
钟离是谁 → lore
胡桃是什么角色 → lore
神之眼怎么获得 → lore
原神有哪些国家 → lore
原神最新卡池 → web
最新兑换码 → web
你好 → chat
你会什么 → chat
我喜欢胡桃 → chat

{query} → """
        try:
            import asyncio as _asyncio
            resp = await _asyncio.to_thread(self.router.complete, prompt)
            text = str(resp).strip().lower()
            # 精确匹配（先完全匹配，再前缀匹配）
            for i in ["asset", "lore", "web", "chat"]:
                if text == i or text.startswith(i):
                    return i
            print(f"[{_ts()}]    [warn] router unexpected: {text!r} -> fallback chat", flush=True)
            return "chat"
        except Exception as e:
            print(f"[{_ts()}]    [warn] router failed: {e} -> fallback chat", flush=True)
            return "chat"

    # ---- step 2: execute tool ---------------------------------

    async def _execute_tool(self, intent: str, query: str) -> tuple[str, str]:
        if intent == "asset":
            return self._get_profile(), ""
        elif intent == "web":
            try:
                result = await self._search(query)
                return result, ""
            except Exception as e:
                print(f"[{_ts()}]    [err] search failed: {e}", flush=True)
                return f"search failed: {e}", ""
        elif intent == "lore":
            return await self._rag_query(query)
        return "", ""

    # ---- step 3: build prompt --------------------------------

    def _build_prompt(self, query: str, tool_result: str, intent: str) -> str:
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        base = f"""当前时间：{t}
你是派蒙，原神中旅行者忠诚可爱的向导。用活泼热情的语气回答。"""

        # 历史摘要（压缩的旧对话）
        summary_block = ""
        if self._memory_summary:
            summary_block = f"\n[更早的对话摘要]\n{self._memory_summary}\n"

        hist = ""
        for msg in self.chat_history[-20:]:
            role = "旅行者" if msg.role == MessageRole.USER else "派蒙"
            hist += f"{role}：{msg.content}\n"

        if intent == "asset" and tool_result:
            return f"{base}\n{summary_block}\n对话历史：\n{hist}\n[用户资产数据]\n{tool_result}\n\n旅行者问：{query}\n请根据真实数据回答，不要编造数字。\n派蒙："
        elif intent == "web" and tool_result:
            return f"{base}\n{summary_block}\n对话历史：\n{hist}\n[搜索结果]\n{tool_result}\n\n旅行者问：{query}\n请根据搜索结果回答。\n派蒙："
        elif intent == "lore" and tool_result:
            return f"{base}\n{summary_block}\n对话历史：\n{hist}\n[知识库资料]\n{tool_result}\n\n旅行者问：{query}\n请根据资料回答。\n派蒙："
        else:
            return f"{base}\n{summary_block}\n对话历史：\n{hist}\n旅行者：{query}\n派蒙："

    # ---- memory management -----------------------------------

    async def _summarize_history(self, messages: list[ChatMessage]) -> str:
        """用 LLM 将旧对话压缩为一句话摘要，保留关键信息"""
        dialog = ""
        for msg in messages:
            role = "旅行者" if msg.role == MessageRole.USER else "派蒙"
            dialog += f"{role}：{msg.content}\n"

        prompt = f"""用一段话总结以下对话的关键信息（用户偏好、提到的事实、重要上下文）。只输出摘要，不加前缀：

{dialog}

摘要："""
        try:
            import asyncio as _asyncio
            resp = await _asyncio.to_thread(self.llm.complete, prompt)
            summary = str(resp).strip()
            if summary:
                print(f"[{_ts()}]    [mem] summarized {len(messages)} msgs -> {len(summary)} chars", flush=True)
                return summary
        except Exception as e:
            print(f"[{_ts()}]    [warn] summarization failed: {e}", flush=True)
        return ""

    # ---- main flow -------------------------------------------

    async def chat_stream(self, query: str) -> AsyncGenerator[str, None]:
        t_start = time.time()
        print(f"\n{'='*50}", flush=True)
        print(f"[{_ts()}] Q: {query[:80]}{'...' if len(query) > 80 else ''}", flush=True)

        async with self._history_lock:
            max_msgs = AppConfig.MAX_HISTORY_TURNS * 2
            if len(self.chat_history) > max_msgs:
                # 把最早的 N 条压缩为摘要，保留最近的轮次
                overflow = len(self.chat_history) - max_msgs + 4  # 多取 2 轮一起压缩
                old = self.chat_history[:overflow]
                recent = self.chat_history[overflow:]
                summary = await self._summarize_history(old)
                if summary:
                    # 合并：新摘要追加到旧摘要后面
                    if self._memory_summary:
                        self._memory_summary = f"{self._memory_summary}\n{summary}"
                    else:
                        self._memory_summary = summary
                self.chat_history = recent

        yield "> 思考中...\n\n"

        # Step 1: intent
        t1 = time.time()
        intent = await self._classify(query)
        print(f"[{_ts()}] Step1 intent -> {intent} ({time.time() - t1:.1f}s)", flush=True)

        # Step 2: execute tool
        tool_result = ""
        sources = ""
        if intent != "chat":
            t2 = time.time()
            print(f"[{_ts()}] Step2 exec tool ({intent}) ...", flush=True)
            if intent == "web":
                yield "> 正在搜索网络...\n\n"
            elif intent == "lore":
                yield "> 正在检索知识库...\n\n"

            try:
                tool_result, sources = await self._execute_tool(intent, query)
                if not tool_result:
                    print(f"[{_ts()}]    [warn] empty tool result -> fallback chat", flush=True)
                    intent = "chat"
                else:
                    print(f"[{_ts()}]    tool done ({time.time() - t2:.1f}s, {len(tool_result)} chars)",
                          flush=True)
            except Exception as e:
                print(f"[{_ts()}]    [err] tool exception: {e}", flush=True)
                intent = "chat"
        else:
            print(f"[{_ts()}] Step2 skip (chat intent)", flush=True)

        # Step 3: LLM synthesis
        prompt = self._build_prompt(query, tool_result, intent)
        full = ""
        t3 = time.time()
        print(f"[{_ts()}] Step3 LLM streaming ...", flush=True)

        token_count = 0
        for token in self.llm.stream_complete(prompt):
            if token.delta:
                full += token.delta
                token_count += 1
                yield token.delta

        if sources:
            yield sources

        elapsed_total = time.time() - t_start
        print(f"[{_ts()}] done -> {intent} | {token_count} tokens | {elapsed_total:.1f}s total",
              flush=True)

        async with self._history_lock:
            self.chat_history.append(ChatMessage(role=MessageRole.USER, content=query))
            self.chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=full))
