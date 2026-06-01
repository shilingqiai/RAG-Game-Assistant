"""Paimon Companion — Gradio UI"""
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"

print("=" * 60)
print("Paimon Companion 启动中...")
print("=" * 60)

import gradio as gr
from game_core import GameCompanion

companion = GameCompanion()
print("加载完成！")


def respond(message, history):
    """非流式 — 收集完整回复后返回"""
    import asyncio
    response = ""

    async def collect():
        nonlocal response
        async for token in companion.chat_stream(message):
            response += token
    asyncio.run(collect())
    return response


demo = gr.ChatInterface(
    fn=respond,
    title="🎒 Paimon Companion",
    description="你的原神向导派蒙！资产查询、攻略搜索、提瓦特知识。",
    examples=["你好！", "我有多少原石？", "钟离是谁", "雷电将军是谁"],
)

print("\n" + "=" * 60)
print("UI: http://127.0.0.1:7860")
print("=" * 60)
demo.launch(server_name="127.0.0.1", server_port=7860)
