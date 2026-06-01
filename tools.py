"""工具函数模块 — 网络搜索与网页读取"""
import asyncio
import logging

import httpx
from ddgs import DDGS

logger = logging.getLogger("paimon.tools")


async def search_web(query: str) -> str:
    """异步搜索网页获取最新信息"""
    try:
        logger.info("Searching: %s", query[:80])
        # ddgs 是同步库，用 asyncio.to_thread 避免阻塞 event loop
        results = await asyncio.to_thread(_sync_search, query)
        formatted = []
        for idx, r in enumerate(results, 1):
            formatted.append(
                f"[{idx}] {r['title']}\n链接: {r['href']}\n摘要: {r['body']}"
            )
        logger.info("Search complete: %d results", len(results))
        return "\n\n".join(formatted) if formatted else "未找到相关结果"
    except Exception as e:
        logger.warning("Search failed: %s", e)
        return f"搜索失败: {e}"


def _sync_search(query: str) -> list:
    """同步搜索实现（在 asyncio.to_thread 中运行）"""
    return list(DDGS().text(query, max_results=3))


async def read_webpage(url: str) -> str:
    """异步读取网页正文内容（使用 Jina Reader API）"""
    try:
        logger.info("Reading webpage: %s", url[:80])
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(f"https://r.jina.ai/{url}")
            response.raise_for_status()
            content = response.text[:6000]
            logger.info("Webpage read: %d chars", len(content))
            return content
    except Exception as e:
        logger.warning("Webpage read failed: %s", e)
        return f"网页读取失败: {e}"
