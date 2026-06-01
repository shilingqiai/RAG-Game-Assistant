"""测试 FunctionAgent 流式对话"""
import os
import sys
import asyncio

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 60)
print("Full Chat Test")
print("=" * 60)

from game_core import GameCompanion


async def main():
    companion = GameCompanion()

    query = "我有多少原石？"
    print(f"\nQuery: {query}\n")
    print("Response:")

    count = 0
    async for token in companion.chat_stream(query):
        print(token, end="")
        count += 1
        if count > 100:
            print("...")
            break

    print("\n\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
