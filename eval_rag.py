"""RAG 检索质量评估脚本"""
import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from game_core import GameCompanion

# 测试用例：query → 期望检索到的关键词
TEST_CASES = [
    {
        "query": "七神分别是谁",
        "expect_contains": ["风神", "岩神", "雷神", "草神"],
    },
    {
        "query": "原神有哪些国家",
        "expect_contains": ["蒙德", "璃月", "稻妻"],
    },
    {
        "query": "钟离是什么神",
        "expect_contains": ["岩神", "摩拉克斯", "契约"],
    },
    {
        "query": "温迪是谁",
        "expect_contains": ["风神", "巴巴托斯", "蒙德"],
    },
    {
        "query": "雷电将军",
        "expect_contains": ["雷神", "稻妻", "雷电"],
    },
    {
        "query": "纳西妲",
        "expect_contains": ["草神", "须弥", "智慧"],
    },
    {
        "query": "胡桃是什么角色",
        "expect_contains": ["胡桃", "往生堂"],
    },
    {
        "query": "原神中的元素反应",
        "expect_contains": ["元素"],
    },
]


def evaluate(companion: GameCompanion) -> dict:
    """评估 RAG 检索质量"""
    if not companion.query_engine:
        print("RAG index not available — skipping evaluation")
        return {"total": 0, "scores": []}

    results = []
    for tc in TEST_CASES:
        # 使用 query() 获取完整响应文本（包含 LLM 合成的答案）
        response = companion.query_lore(tc["query"])
        texts = response

        hits = sum(1 for kw in tc["expect_contains"] if kw in texts)
        score = hits / len(tc["expect_contains"]) if tc["expect_contains"] else 0
        results.append(
            {
                "query": tc["query"],
                "score": score,
                "hits": hits,
                "total_keywords": len(tc["expect_contains"]),
            }
        )

        print(f"  [{score:.0%}] {tc['query']}")

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    return {"total": len(results), "avg_score": avg_score, "scores": results}


def main():
    print("=" * 60)
    print("RAG Evaluation")
    print("=" * 60)
    print()

    companion = GameCompanion()

    if not companion.query_engine:
        print("RAG index not available. Run: python data_indexer.py")
        return

    print(f"Test cases: {len(TEST_CASES)}")
    print()

    result = evaluate(companion)

    print()
    print(f"Average recall: {result['avg_score']:.1%}")
    print(f"Test cases: {result['total']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
