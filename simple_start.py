"""Paimon Companion — Gradio UI (流式输出 + 资产管理面板)"""
import os, sys, time

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import asyncio
import threading
import queue


def create_app():
    """创建 Gradio 应用"""
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
        print("[UI] ✅ RAG 向量索引已加载（混合检索）")
    else:
        print("[UI] ⚠️  RAG 向量索引未加载（仅聊天模式）")

    print("[UI] ✅ 流式输出已启用")

    # ── 资产面板逻辑 ──────────────────────────────────────────

    def get_asset_display() -> str:
        """获取当前资产展示文本"""
        p = companion.agent.user_profile
        primos = p.get_asset("primogems")
        fate = p.get_asset("intertwined_fate")
        pity = p.get_asset("pity_count")
        guaranteed = p.get_asset("is_guaranteed") != 0
        total_pulls = primos // 160 + fate
        return f"""```
📊 旅行者当前资产
══════════════════════
  原　　石: {primos:,}  (≈{primos // 160} 抽)
  纠缠之缘: {fate}
  总计可抽: {total_pulls} 抽
  垫 水 位: {pity} / 90
  大 保 底: {'是' if guaranteed else '否'}
══════════════════════
```"""

    def update_primos(delta_str: str):
        """更新原石（正数增加，负数减少）"""
        try:
            delta = int(delta_str)
        except ValueError:
            return get_asset_display()
        companion.agent.user_profile.update_asset("primogems", delta)
        return get_asset_display()

    def update_fate(delta_str: str):
        """更新纠缠之缘"""
        try:
            delta = int(delta_str)
        except ValueError:
            return get_asset_display()
        companion.agent.user_profile.update_asset("intertwined_fate", delta)
        return get_asset_display()

    def set_pity(value_str: str):
        """设置垫水位"""
        try:
            value = int(value_str)
        except ValueError:
            return get_asset_display()
        current = companion.agent.user_profile.get_asset("pity_count")
        delta = value - current
        companion.agent.user_profile.update_asset("pity_count", delta)
        return get_asset_display()

    def toggle_guarantee():
        """切换大保底状态"""
        current = companion.agent.user_profile.get_asset("is_guaranteed")
        new_val = 0 if current else 1
        companion.agent.user_profile.update_asset("is_guaranteed", new_val - current)
        return get_asset_display()

    # ── 聊天逻辑 ──────────────────────────────────────────────

    def respond(message: str, history: list):
        """
        流式 generator — 将异步 token 流桥接到同步 yield。
        返回 Chatbot 格式的历史记录。
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

        # 添加用户消息
        history.append({"role": "user", "content": message})
        # 初始化助手回复
        history.append({"role": "assistant", "content": ""})

        bot_text = ""
        while True:
            msg_type, token = q.get()
            if msg_type == "done":
                break
            bot_text += token
            # 更新最后一条助手消息
            history[-1]["content"] = bot_text
            yield history

    # ── UI 布局 ───────────────────────────────────────────────

    with gr.Blocks(title="🎒 Paimon Companion") as demo:
        gr.Markdown("# 🎒 Paimon Companion — 你的原神向导派蒙")

        with gr.Row(equal_height=False):
            # 左侧：聊天区
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=500,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="问派蒙任何原神相关问题...",
                        label="",
                        scale=9,
                        container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

            # 右侧：资产面板
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### 📊 旅行者资产")
                asset_display = gr.Markdown(get_asset_display(), every=1)

                gr.Markdown("**原石变动** (+160 一抽, -160 消耗)")
                with gr.Row():
                    primo_input = gr.Textbox(
                        value="160",
                        label="数量",
                        container=False,
                        scale=2,
                    )
                    primo_add = gr.Button("+ 增加", size="sm", scale=1)
                    primo_sub = gr.Button("- 消耗", size="sm", scale=1)

                gr.Markdown("**纠缠之缘变动**")
                with gr.Row():
                    fate_input = gr.Textbox(
                        value="1", label="数量", container=False, scale=2
                    )
                    fate_add = gr.Button("+ 增加", size="sm", scale=1)
                    fate_sub = gr.Button("- 消耗", size="sm", scale=1)

                gr.Markdown("**垫水位**")
                with gr.Row():
                    pity_input = gr.Textbox(
                        value="0", label="当前值", container=False, scale=2
                    )
                    pity_set = gr.Button("设置", size="sm", scale=1)

                guarantee_btn = gr.Button("🔄 切换大保底状态", size="sm")

                gr.Markdown("---")
                gr.Markdown(
                    """
                **示例提问**
                - 我有多少原石
                - 钟离是谁
                - 雷电将军的背景故事
                - 原神有哪些国家
                - 最新兑换码
                """,
                    elem_classes=["examples-box"],
                )

        # ── 事件绑定 ──────────────────────────────────────────

        # 聊天
        def on_send(message, history):
            if not message.strip():
                return "", history
            # yield from the streaming generator
            for h in respond(message, history):
                yield "", h

        send_btn.click(
            on_send,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )
        msg_input.submit(
            on_send,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )

        # 资产操作
        primo_add.click(
            lambda v: update_primos(v),
            inputs=[primo_input],
            outputs=[asset_display],
        )
        primo_sub.click(
            lambda v: update_primos(f"-{v}"),
            inputs=[primo_input],
            outputs=[asset_display],
        )
        fate_add.click(
            lambda v: update_fate(v),
            inputs=[fate_input],
            outputs=[asset_display],
        )
        fate_sub.click(
            lambda v: update_fate(f"-{v}"),
            inputs=[fate_input],
            outputs=[asset_display],
        )
        pity_set.click(
            set_pity,
            inputs=[pity_input],
            outputs=[asset_display],
        )
        guarantee_btn.click(
            toggle_guarantee,
            outputs=[asset_display],
        )

    return demo, companion


def main():
    """CLI 入口: paimon 或 python simple_start.py"""
    import gradio as gr
    demo, _companion = create_app()
    print()
    print("=" * 60)
    print("🌐 UI 地址: http://127.0.0.1:7860")
    print("🟢 流式输出已启用 — 回复会逐字显示")
    print("📊 资产面板已就绪 — 右侧可直接管理")
    print("=" * 60)
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
