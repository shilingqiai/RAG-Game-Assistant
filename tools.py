"""工具函数模块 — 网络搜索与网页读取"""
import asyncio
import logging
import re

import httpx
from ddgs import DDGS

logger = logging.getLogger("paimon.tools")

# 原神相关关键词，用于自动补全搜索词
_GENSHIN_KEYWORDS = [
    "原神", "genshin", "派蒙", "提瓦特", "钟离", "胡桃", "雷电将军",
    "纳西妲", "温迪", "枫丹", "璃月", "蒙德", "稻妻", "须弥", "纳塔",
    "卡池", "角色", "武器", "圣遗物", "深渊", "七神", "神之眼", "命之座",
]


def _enhance_query(query: str) -> str:
    """自动补全搜索词：如果跟原神无关，加上'原神'前缀"""
    lower = query.lower()
    for kw in _GENSHIN_KEYWORDS:
        if kw.lower() in lower:
            return query  # 已经包含原神相关词
    return f"原神 {query}"


def _filter_chinese_results(results: list, query: str) -> list:
    """过滤和排序搜索结果：优先中文 + 相关度"""
    scored = []
    query_chars = set(re.findall(r"[一-鿿]+", query))
    for r in results:
        score = 0
        body = r.get("body", "")
        title = r.get("title", "")

        # 中文内容加分
        cn_chars = len(re.findall(r"[一-鿿]", body))
        score += min(cn_chars / 50, 3)  # 最多 +3

        # 标题包含查询词加分
        for ch in query_chars:
            if ch in title:
                score += 0.5

        # 过滤纯英文结果（中文搜索场景）
        if cn_chars < 5:
            score -= 2

        scored.append((r, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, s in scored if s > -1]  # 只保留不太差的


async def search_web(query: str) -> str:
    """异步搜索网页获取最新信息（中文优化）"""
    try:
        enhanced = _enhance_query(query)
        logger.info("Searching: %s → %s", query[:50], enhanced[:60])

        results = await asyncio.to_thread(_sync_search, enhanced)

        # 过滤排序
        if results:
            results = _filter_chinese_results(results, enhanced)

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
    """同步搜索实现（在 asyncio.to_thread 中运行），中文区域优先"""
    try:
        # region="cn-zh" 优先中文结果
        return list(DDGS().text(query, max_results=5, region="cn-zh"))
    except Exception:
        # 降级：不设 region
        return list(DDGS().text(query, max_results=5))


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
