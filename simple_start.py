"""Paimon Companion — Gradio UI (流式输出)"""
import os, sys, time
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import asyncio
import threading
import queue


def create_app():
    """创建 Gradio 应用（供 paimon start 和 python simple_start.py 共用）"""
    import gradio as gr
    from game_core import GameCompanion

    print("=" * 60)
    print("🎒 Paimon Companion 启动中...")
    print("=" * 60)

    t0 = time.time()
    print("[UI] 初始化 GameCompanion ...")
    companion = GameCompanion()
    print(f"[UI] ✅ GameCompanion 就绪 ({time.time() - t0:.1f}s)")

    if companion.agent.query_engine:
        print("[UI] ✅ RAG 向量索引已加载")
    else:
        print("[UI] ⚠️  RAG 向量索引未加载（仅聊天模式）")

    print("[UI] ✅ 流式输出已启用")

    def respond(message, history):
        """
        流式 generator — 将异步 token 流桥接到同步 yield。
        Gradio ChatInterface 识别到 generator 后会自动逐 token 更新 UI。
        """
        q: queue.Queue = queue.Queue()

        async def _produce():
            try:
                async for token in companion.chat_stream(message):
                    q.put(("token", token))
            except Exception as e:
                q.put(("token", f"\n\n❌ 出错了: {e}"))
            q.put(("done", None))

        def _run_async():
            asyncio.run(_produce())

        t = threading.Thread(target=_run_async, daemon=True)
        t.start()

        response_text = ""
        while True:
            msg_type, token = q.get()
            if msg_type == "done":
                break
            response_text += token
            yield response_text

    demo = gr.ChatInterface(
        fn=respond,
        title="🎒 Paimon Companion",
        description="你的原神向导派蒙！资产查询、知识检索、攻略搜索。",
        examples=["你好！", "我有多少原石？", "钟离是谁", "雷电将军是谁", "原神有哪些国家"],
    )
    return demo, companion


def main():
    """CLI 入口: paimon start 或 python -m simple_start"""
    demo, _companion = create_app()
    print()
    print("=" * 60)
    print("🌐 UI 地址: http://127.0.0.1:7860")
    print("🟢 流式输出已启用 — 回复会逐字显示")
    print("=" * 60)
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
