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
            print(f"[{_ts()}]    [rag] done ({elapsed:.1f}s, {node_count} nodes)", flush=True)

            sources = ""
            if hasattr(resp, "source_nodes"):
                parts = []
                for i, n in enumerate(resp.source_nodes[:3]):
                    title = n.metadata.get("title", "?")
                    s = n.score or 0
                    parts.append(f"  [{i+1}] {title} (relevance: {s:.0%})")
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
        prompt = f"""分类意图，只输出一个词：asset / lore / web / chat

我有多少原石 → asset
帮我抽卡建议 → asset
钟离是谁 → lore
七神分别是谁 → lore
原神最新卡池 → web
原神最新活动 → web
你好 → chat
我喜欢胡桃 → chat
我叫cc → chat

{query} → """
        try:
            text = ""
            for token in self.router.stream_complete(prompt):
                if token.delta:
                    text += token.delta
            intent = text.strip().lower()
            for i in ["asset", "lore", "web", "chat"]:
                if i in intent:
                    return i
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

        hist = ""
        for msg in self.chat_history[-20:]:
            role = "旅行者" if msg.role == MessageRole.USER else "派蒙"
            hist += f"{role}：{msg.content}\n"

        if intent == "asset" and tool_result:
            return f"{base}\n\n对话历史：\n{hist}\n[用户资产数据]\n{tool_result}\n\n旅行者问：{query}\n请根据真实数据回答，不要编造数字。\n派蒙："
        elif intent == "web" and tool_result:
            return f"{base}\n\n对话历史：\n{hist}\n[搜索结果]\n{tool_result}\n\n旅行者问：{query}\n请根据搜索结果回答。\n派蒙："
        elif intent == "lore" and tool_result:
            return f"{base}\n\n对话历史：\n{hist}\n[知识库资料]\n{tool_result}\n\n旅行者问：{query}\n请根据资料回答。\n派蒙："
        else:
            return f"{base}\n\n对话历史：\n{hist}\n旅行者：{query}\n派蒙："

    # ---- main flow -------------------------------------------

    async def chat_stream(self, query: str) -> AsyncGenerator[str, None]:
        t_start = time.time()
        print(f"\n{'='*50}", flush=True)
        print(f"[{_ts()}] Q: {query[:80]}{'...' if len(query) > 80 else ''}", flush=True)

        async with self._history_lock:
            if len(self.chat_history) > AppConfig.MAX_HISTORY_TURNS * 2:
                self.chat_history = self.chat_history[-(AppConfig.MAX_HISTORY_TURNS * 2):]

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
