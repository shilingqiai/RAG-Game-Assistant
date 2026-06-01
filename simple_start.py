"""Paimon Companion — Gradio UI 启动脚本"""

import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"

print("=" * 60)
print("正在启动 Paimon Companion...")
print("=" * 60)


def main():
    import gradio as gr
    from game_core import GameCompanion

    print("加载 GameCompanion...")
    companion = GameCompanion()
    print("加载完成！")

    async def chat_fn(message: str, history: list):
        """流式对话回调 — Agent 内部维护多轮记忆"""
        response = ""
        async for token in companion.chat_stream(message):
            response += token
            yield response

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="Paimon Companion — 原神智能伴侣",
        description="你的旅行向导派蒙！可以查询资产、搜索攻略、探索提瓦特知识。",
        examples=[
            "你好！",
            "我有多少原石？",
            "帮我查一下原神最新卡池信息",
            "原神中七神分别是谁？",
        ],
    )

    print("\n" + "=" * 60)
    print("启动 UI 界面...")
    print("地址: http://127.0.0.1:7860")
    print("=" * 60)
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
        input("按 Enter 退出...")
