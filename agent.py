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
        prompt = f"""classify intent, output ONE word: asset / lore / web / chat

how many primogems do I have -> asset
gacha pull advice -> asset
who is Zhongli -> lore
who are The Seven -> lore
latest Genshin banner -> web
latest event -> web
hello -> chat
I like Hu Tao -> chat
my name is cc -> chat

{query} -> """
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
        base = (f"current time: {t}\n"
                f"You are Paimon, the loyal and adorable guide from Genshin Impact. "
                f"Answer in a lively, enthusiastic tone.")

        hist = ""
        for msg in self.chat_history[-20:]:
            role = "Traveler" if msg.role == MessageRole.USER else "Paimon"
            hist += f"{role}: {msg.content}\n"

        if intent == "asset" and tool_result:
            return (f"{base}\n\nchat history:\n{hist}\n"
                    f"[user assets]\n{tool_result}\n\n"
                    f"Traveler asks: {query}\n"
                    f"Answer based on real data, do not make up numbers.\nPaimon:")
        elif intent == "web" and tool_result:
            return (f"{base}\n\nchat history:\n{hist}\n"
                    f"[search results]\n{tool_result}\n\n"
                    f"Traveler asks: {query}\n"
                    f"Answer based on search results.\nPaimon:")
        elif intent == "lore" and tool_result:
            return (f"{base}\n\nchat history:\n{hist}\n"
                    f"[knowledge base]\n{tool_result}\n\n"
                    f"Traveler asks: {query}\n"
                    f"Answer based on the provided data.\nPaimon:")
        else:
            return (f"{base}\n\nchat history:\n{hist}\n"
                    f"Traveler: {query}\nPaimon:")

    # ---- main flow -------------------------------------------

    async def chat_stream(self, query: str) -> AsyncGenerator[str, None]:
        t_start = time.time()
        print(f"\n{'='*50}", flush=True)
        print(f"[{_ts()}] Q: {query[:80]}{'...' if len(query) > 80 else ''}", flush=True)

        async with self._history_lock:
            if len(self.chat_history) > AppConfig.MAX_HISTORY_TURNS * 2:
                self.chat_history = self.chat_history[-(AppConfig.MAX_HISTORY_TURNS * 2):]

        yield "> thinking...\n\n"

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
                yield "> searching the web...\n\n"
            elif intent == "lore":
                yield "> searching knowledge base...\n\n"

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
