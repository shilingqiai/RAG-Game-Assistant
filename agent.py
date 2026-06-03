"""Agent — 手搓三步：意图路由 → 执行工具 → LLM 合成（无 FunctionAgent）"""
import asyncio, datetime, logging, sys, time
from typing import AsyncGenerator, List, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from config import UserProfile, AppConfig
from llm_manager import LLMManager

logger = logging.getLogger("paimon.agent")


def _ts() -> str:
    """当前时间戳字符串，用于控制台输出"""
    return datetime.datetime.now().strftime("%H:%M:%S")


class GameAgent:
    def __init__(self, llm_manager: LLMManager, user_profile: UserProfile,
                 extra_tools=None, query_engine=None):
        self.llm = llm_manager.llm
        self.user_profile = user_profile
        self.query_engine = query_engine
        self.chat_history: List[ChatMessage] = []
        self._history_lock = asyncio.Lock()  # 防止并发请求损坏对话历史

        # 意图路由器
        from llama_index.llms.dashscope import DashScope
        self.router = DashScope(model_name=AppConfig.ROUTER_MODEL, temperature=0,
                                api_key=llm_manager.llm.api_key, max_tokens=10)
        print(f"[{_ts()}] 🤖 Agent 就绪 | RAG={'✅ 已加载' if query_engine else '⚠️ 未加载（仅聊天模式）'}",
              flush=True)

    # ── 工具函数 ──────────────────────────────────────────────

    def _get_profile(self) -> str:
        result = self.user_profile.format_for_display()
        print(f"[{_ts()}]    📊 资产查询完成", flush=True)
        return result

    def _update_assets(self, asset_type: str, amount: int) -> str:
        return self.user_profile.update_assets(asset_type, amount)

    def _save_pref(self, text: str) -> str:
        return self.user_profile.save_preference(text)

    async def _search(self, query: str) -> str:
        from tools import search_web
        t0 = time.time()
        print(f"[{_ts()}]    🌐 网络搜索中 ...", flush=True)
        result = await search_web(query)
        result_count = result.count("[") // 2 if result else 0  # 粗略估计结果数
        print(f"[{_ts()}]    🌐 搜索完成 ({time.time() - t0:.1f}s)", flush=True)
        return result

    async def _rag_query(self, query: str) -> tuple[str, str]:
        """RAG 检索，返回 (文本, 来源引用)"""
        if not self.query_engine:
            return "", ""
        import asyncio
        t0 = time.time()
        print(f"[{_ts()}]    📚 检索知识库 ...", flush=True)
        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(self.query_engine.query, query),
                timeout=AppConfig.RAG_TIMEOUT,
            )
            text = str(resp).strip()
            node_count = len(resp.source_nodes) if hasattr(resp, "source_nodes") else 0
            elapsed = time.time() - t0
            print(f"[{_ts()}]    📚 检索完成 ({elapsed:.1f}s, {node_count} 条相关)", flush=True)

            sources = ""
            if hasattr(resp, "source_nodes"):
                parts = []
                for i, n in enumerate(resp.source_nodes[:3]):
                    title = n.metadata.get("title", "?")
                    s = n.score or 0
                    parts.append(f"  [{i+1}] {title} (相关度: {s:.0%})")
                if parts:
                    sources = "\n\n---\n📖 参考来源:\n" + "\n".join(parts)
            return text, sources
        except asyncio.TimeoutError:
            print(f"[{_ts()}]    ⚠️ 检索超时 ({AppConfig.RAG_TIMEOUT}s)", flush=True)
            return "", ""
        except Exception as e:
            print(f"[{_ts()}]    ❌ 检索出错: {e}", flush=True)
            return "", ""

    # ── 意图分类 ──────────────────────────────────────────────

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
                if token.delta: text += token.delta
            intent = text.strip().lower()
            for i in ["asset", "lore", "web", "chat"]:
                if i in intent: return i
            return "chat"
        except Exception as e:
            print(f"[{_ts()}]    ⚠️ 路由模型调用失败: {e} → 降级为 chat", flush=True)
            return "chat"

    # ── Step 2: 执行工具 ──────────────────────────────────────

    async def _execute_tool(self, intent: str, query: str) -> tuple[str, str]:
        """执行工具，返回 (工具结果文本, 来源引用)"""

        if intent == "asset":
            return self._get_profile(), ""

        elif intent == "web":
            try:
                result = await self._search(query)
                return result, ""
            except Exception as e:
                print(f"[{_ts()}]    ❌ 搜索失败: {e}", flush=True)
                return f"搜索失败: {e}", ""

        elif intent == "lore":
            return await self._rag_query(query)

        return "", ""

    # ── Step 3: LLM 合成 ──────────────────────────────────────

    def _build_prompt(self, query: str, tool_result: str, intent: str) -> str:
        """构建合成 prompt"""
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        base = f"""当前时间：{t}
你是派蒙，原神中旅行者忠诚可爱的向导。用活泼热情的语气回答。"""

        # 历史
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

    # ── 主流程 ────────────────────────────────────────────────

    async def chat_stream(self, query: str) -> AsyncGenerator[str, None]:
        t_start = time.time()
        # ── 控制台：请求分隔线 ──
        print(f"\n{'─'*50}", flush=True)
        print(f"[{_ts()}] 💬 旅行者: {query[:80]}{'...' if len(query) > 80 else ''}", flush=True)

        async with self._history_lock:
            if len(self.chat_history) > AppConfig.MAX_HISTORY_TURNS * 2:
                self.chat_history = self.chat_history[-(AppConfig.MAX_HISTORY_TURNS * 2):]

        # 立即给反馈，避免"卡住"的感觉
        yield "> 💭 *思考中...*\n\n"

        # ── Step 1: 意图分类 ──
        t1 = time.time()
        intent = await self._classify(query)
        intent_emoji = {"asset": "📊", "lore": "📚", "web": "🌐", "chat": "💬"}.get(intent, "❓")
        print(f"[{_ts()}] 🧠 Step 1/3 意图 → {intent_emoji} {intent} ({time.time() - t1:.1f}s)", flush=True)

        # ── Step 2: 执行工具 ──
        tool_result = ""
        sources = ""
        if intent != "chat":
            t2 = time.time()
            print(f"[{_ts()}] 🔧 Step 2/3 执行工具 ({intent}) ...", flush=True)
            # UI 反馈
            if intent == "web":
                yield "> 🌐 正在搜索...\n\n"
            elif intent == "lore":
                yield "> 📚 正在检索知识库...\n\n"

            try:
                tool_result, sources = await self._execute_tool(intent, query)
                if not tool_result:
                    print(f"[{_ts()}]    ⚠️ 工具无结果 → 降级为 chat", flush=True)
                    intent = "chat"
                else:
                    result_len = len(tool_result)
                    print(f"[{_ts()}]    ✅ 工具完成 ({time.time() - t2:.1f}s, {result_len} 字符)", flush=True)
            except Exception as e:
                print(f"[{_ts()}]    ❌ 工具异常: {e}", flush=True)
                intent = "chat"
        else:
            print(f"[{_ts()}] ⏭️  Step 2/3 跳过（聊天意图）", flush=True)

        # ── Step 3: LLM 流式合成 ──
        prompt = self._build_prompt(query, tool_result, intent)
        full = ""
        t3 = time.time()
        print(f"[{_ts()}] 🤖 Step 3/3 LLM 流式合成中 ...", flush=True)

        token_count = 0
        for token in self.llm.stream_complete(prompt):
            if token.delta:
                full += token.delta
                token_count += 1
                yield token.delta

        # 追加来源
        if sources:
            yield sources

        elapsed_total = time.time() - t_start
        print(f"[{_ts()}] ✅ 完成 → {intent} | {token_count} tokens | {elapsed_total:.1f}s 总计", flush=True)

        # 保存历史（加锁防并发）
        async with self._history_lock:
            self.chat_history.append(ChatMessage(role=MessageRole.USER, content=query))
            self.chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=full))
