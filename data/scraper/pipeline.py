#!/usr/bin/env python
"""一键数据下载 + 合并流水线

Usage:
    python data/scraper/pipeline.py              # 完整流程
    python data/scraper/pipeline.py --wiki-only  # 只下载 wiki
    python data/scraper/pipeline.py --merge-only # 只做合并（已有原始数据）
    python data/scraper/pipeline.py --force      # 强制重新下载所有
    python data/scraper/pipeline.py --chars 30   # 限制角色数量（测试用）

Output:
    data/raw/             原始下载数据
    data/processed/       结构化 JSON + SQLite 数据库
    data/rag_texts/       RAG 索引用纯文本
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paimon.pipeline")


async def main():
    parser = argparse.ArgumentParser(description="Genshin Data Pipeline")
    parser.add_argument("--wiki-only", action="store_true", help="只下载 wiki 数据")
    parser.add_argument("--merge-only", action="store_true", help="只做合并（跳过下载）")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--chars", type=int, default=90, help="角色数量限制（默认 90）")
    parser.add_argument("--lore", type=int, default=80, help="世界观页面数量限制（默认 80）")
    args = parser.parse_args()

    raw_dir = Path(__file__).parent.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.merge_only:
        # ── Step 1: Wiki 数据 ──
        wiki_path = raw_dir / "wiki_characters.json"
        if args.force or not wiki_path.exists():
            logger.info(">>> Phase 1: Wiki fetch")
            from data.scraper.wiki_fetcher import run_wiki_fetch

            await run_wiki_fetch(char_limit=args.chars, lore_limit=args.lore)
        else:
            logger.info(">>> Phase 1: Wiki data already exists, skipping (use --force to re-download)")

        if args.wiki_only:
            logger.info(">>> --wiki-only: stopping after wiki fetch")
            return

        # ── Step 2: Dimbreath 数据 ──
        dimbreath_dir = raw_dir / "dimbreath"
        if args.force or not dimbreath_dir.exists() or not any(dimbreath_dir.iterdir()):
            logger.info(">>> Phase 2: Dimbreath fetch")
            from data.scraper.dimbreath_fetcher import run_dimbreath_fetch

            await run_dimbreath_fetch()
        else:
            logger.info(">>> Phase 2: Dimbreath data already exists, skipping (use --force to re-download)")

    # ── Step 3: 合并 ──
    logger.info(">>> Phase 3: Merge data → SQLite + RAG texts")
    from data.scraper.merger import run_merge

    db_path = run_merge()

    # ── 总结 ──
    logger.info("=" * 50)
    logger.info("Pipeline complete!")
    logger.info("  SQLite DB : %s", db_path)
    logger.info("  RAG texts : %s", Path(__file__).parent.parent / "rag_texts")
    logger.info("  JSON      : %s", Path(__file__).parent.parent / "processed")
    logger.info("  Run: python data_indexer.py  to rebuild RAG index")
    logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
