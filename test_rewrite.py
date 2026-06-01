"""测试重构后的系统 — 验证 Agent 工具调用"""
import os
import sys
import asyncio

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 60)
print("Paimon Companion — 集成测试")
print("=" * 60)

from game_core import GameCompanion


async def main():
    print("\n初始化 GameCompanion...")
    companion = GameCompanion()
    print("OK: 初始化成功！")

    tests = [
        "你好派蒙！",
        "我有多少原石？",
    ]

    for query in tests:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        print(f"{'─' * 60}")
        full = ""
        async for token in companion.chat_stream(query):
            full += token
            print(token, end="")
        print()

    print(f"\n{'=' * 60}")
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
