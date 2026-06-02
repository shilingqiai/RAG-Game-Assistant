"""意图路由评估 — 50 条标注查询 + 准确率报告"""
import asyncio, os, sys, time
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from game_core import GameCompanion

# 50 条人工标注 (query, expected_intent)
TEST_CASES = [
    # === asset (10) ===
    ("我有多少原石", "asset"),
    ("原石数量", "asset"),
    ("帮我查一下纠缠之缘", "asset"),
    ("垫水位多少", "asset"),
    ("抽卡记录", "asset"),
    ("更新资产 原石-160", "asset"),
    ("大保底还有多远", "asset"),
    ("我能抽多少发", "asset"),
    ("资产查询", "asset"),
    ("我够不够抽钟离", "asset"),

    # === lore (15) ===
    ("钟离是谁", "lore"),
    ("七神分别是谁", "lore"),
    ("雷电将军的背景故事", "lore"),
    ("胡桃是什么角色", "lore"),
    ("温迪是哪里的神", "lore"),
    ("纳西妲介绍", "lore"),
    ("枫丹在哪里", "lore"),
    ("原神世界观", "lore"),
    ("提瓦特大陆有哪些国家", "lore"),
    ("愚人众执行官", "lore"),
    ("深渊是什么", "lore"),
    ("神之眼怎么获得", "lore"),
    ("璃月港的历史", "lore"),
    ("坎瑞亚灭国原因", "lore"),
    ("什么是命之座", "lore"),

    # === web (10) ===
    ("原神最新卡池", "web"),
    ("3.7版本更新内容", "web"),
    ("原神最新活动", "web"),
    ("今天有什么福利", "web"),
    ("原神兑换码", "web"),
    ("下期UP角色", "web"),
    ("原神联动消息", "web"),
    ("原石礼包怎么买", "web"),
    ("官方公告", "web"),
    ("最新角色强度排行", "web"),

    # === chat (15) ===
    ("你好", "chat"),
    ("我叫cc", "chat"),
    ("我喜欢胡桃", "chat"),
    ("谢谢派蒙", "chat"),
    ("今天天气不错", "chat"),
    ("你会什么", "chat"),
    ("派蒙可爱吗", "chat"),
    ("讲个笑话", "chat"),
    ("再见", "chat"),
    ("我想抽卡但是很纠结", "chat"),
    ("原神好玩吗", "chat"),
    ("你有妹妹吗", "chat"),
    ("我叫什么名字", "chat"),
    ("刚刚问了什么", "chat"),
    ("我很生气", "chat"),
]


async def evaluate(companion: GameCompanion) -> dict:
    results = []
    correct = 0
    t0 = time.time()

    for query, expected in TEST_CASES:
        predicted = await companion.agent._classify(query)
        ok = predicted == expected
        if ok: correct += 1
        results.append({"query": query, "expected": expected, "predicted": predicted, "ok": ok})

    elapsed = time.time() - t0
    accuracy = correct / len(TEST_CASES)

    # 混淆矩阵
    matrix = {"asset": {}, "lore": {}, "web": {}, "chat": {}}
    for r in results:
        matrix[r["expected"]][r["predicted"]] = matrix[r["expected"]].get(r["predicted"], 0) + 1

    return {"total": len(TEST_CASES), "correct": correct, "accuracy": accuracy,
            "elapsed_s": elapsed, "matrix": matrix, "details": results}


def print_report(report: dict):
    from config import AppConfig
    print(f"\n{'='*60}")
    print(f"  意图路由评估报告")
    print(f"{'='*60}")
    print(f"  总数: {report['total']}  正确: {report['correct']}  准确率: {report['accuracy']:.0%}")
    print(f"  耗时: {report['elapsed_s']:.1f}s  (~{report['elapsed_s']/report['total']*1000:.0f}ms/条)")
    print(f"  模型: {AppConfig.ROUTER_MODEL}")

    print(f"\n  混淆矩阵:")
    print(f"  {'':>8s} {'asset':>8s} {'lore':>8s} {'web':>8s} {'chat':>8s}")
    for intent in ["asset", "lore", "web", "chat"]:
        row = report["matrix"][intent]
        vals = " ".join(f"{row.get(i, 0):>8d}" for i in ["asset", "lore", "web", "chat"])
        print(f"  {intent:>8s} {vals}")

    # 错误案例
    errors = [r for r in report["details"] if not r["ok"]]
    if errors:
        print(f"\n  错误 ({len(errors)} 条):")
        for r in errors[:10]:
            print(f"    {r['expected']} → {r['predicted']} | {r['query']}")


def main():
    from config import AppConfig

    companion = GameCompanion()
    if not companion.agent:
        print("Agent not initialized")
        return

    print("Running intent evaluation...")
    report = asyncio.run(evaluate(companion))
    print_report(report)


if __name__ == "__main__":
    main()
