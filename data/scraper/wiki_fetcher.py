"""Fandom Wiki 数据爬取 — 角色列表 + 页面正文"""
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx

logger = logging.getLogger("paimon.scraper.wiki")

WIKI_API = "https://genshin-impact.fandom.com/api.php"
USER_AGENT = "PaimonCompanion/1.0 (Educational Project; contact@example.com)"
REQUEST_DELAY = 0.6  # 请求间隔（秒），尊重 wiki 服务器
RAW_DIR = Path(__file__).parent.parent / "raw"


def _wiki_params(**kwargs) -> dict:
    return {"format": "json", **kwargs}


async def fetch_category_members(
    client: httpx.AsyncClient, category: str, limit: int = 500
) -> list[dict]:
    """获取指定分类下的所有页面"""
    members = []
    params = _wiki_params(
        action="query",
        list="categorymembers",
        cmtitle=f"Category:{category}",
        cmlimit=min(limit, 500),
    )

    while True:
        resp = await client.get(WIKI_API, params=params)
        resp.raise_for_status()
        data = resp.json()

        members.extend(data.get("query", {}).get("categorymembers", []))

        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
            await asyncio.sleep(0.3)
        else:
            break

    return members


async def fetch_page_wikitext(
    client: httpx.AsyncClient, page_title: str, section: int = 0
) -> str | None:
    """获取页面 wikitext 源码"""
    try:
        resp = await client.get(
            WIKI_API,
            params=_wiki_params(
                action="parse",
                page=page_title,
                prop="wikitext",
                section=section,
            ),
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("parse", {}).get("wikitext", {}).get("*", "")
    except Exception as e:
        logger.warning("Failed to fetch page '%s': %s", page_title, e)
        return None


# ── Wikitext 解析 ──────────────────────────────────────────


def clean_wikitext(text: str) -> str:
    """清洗 wikitext 标记，提取纯文本"""
    if not text:
        return ""
    # 移除模板 {{...}}（嵌套处理）
    text = _remove_templates(text)
    # 移除 wiki 链接 [[...]]
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+?)\]\]", r"\1", text)
    # 移除 HTML 注释
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # 移除 ref 标签
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*?/>", "", text)
    # 移除其他 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 移除 wiki 表格
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    # 移除多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_templates(text: str) -> str:
    """移除嵌套 wiki 模板 {{...}}"""
    result = []
    depth = 0
    i = 0
    while i < len(text):
        if text[i : i + 2] == "{{":
            depth += 1
            i += 2
            continue
        elif text[i : i + 2] == "}}" and depth > 0:
            depth -= 1
            i += 2
            continue
        if depth == 0:
            result.append(text[i])
        i += 1
    return "".join(result)


def parse_infobox(wikitext: str) -> dict:
    """从 infobox 模板提取结构化字段"""
    info = {}
    patterns = {
        "element": r"\|\s*element\s*=\s*(\w+)",
        "weapon": r"\|\s*weapon\s*=\s*(\w+)",
        "rarity": r"\|\s*rarity\s*=\s*(\d+)",
        "region": r"\|\s*region\s*=\s*(\w+)",
        "affiliation": r"\|\s*affiliation\s*=\s*(.+?)(?:\n\||\n\})",
        "title": r"\|\s*title\s*=\s*(.+?)(?:\n\||\n\})",
        "birthday": r"\|\s*birthday\s*=\s*(.+?)(?:\n\||\n\})",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, wikitext, re.IGNORECASE)
        if match:
            info[key] = match.group(1).strip()
    return info


# ── 主流程 ─────────────────────────────────────────────────


async def fetch_all_characters(
    client: httpx.AsyncClient, max_chars: int = 90
) -> list[dict]:
    """下载所有角色数据"""
    logger.info("Fetching character list from wiki...")
    members = await fetch_category_members(client, "Playable_Characters")

    # 过滤掉非角色页面（如 Category、Template 等）
    char_names = [
        m["title"]
        for m in members
        if not m["title"].startswith(("Category:", "Template:", "List of"))
    ]
    logger.info("Found %d playable characters", len(char_names))

    # 增量保存路径
    char_path = RAW_DIR / "wiki_characters.json"
    characters = []

    for i, name in enumerate(char_names[:max_chars]):
        logger.info("[%d/%d] Fetching: %s", i + 1, min(len(char_names), max_chars), name)

        wikitext = await fetch_page_wikitext(client, name, section=0)
        if not wikitext:
            logger.warning("Skipping %s (no content)", name)
            continue

        info = parse_infobox(wikitext)
        clean_text = clean_wikitext(wikitext)

        characters.append(
            {
                "name": name,
                "name_cn": info.get("title", ""),
                "rarity": int(info.get("rarity", 0)),
                "element": info.get("element", ""),
                "weapon_type": info.get("weapon", ""),
                "region": info.get("region", ""),
                "affiliation": info.get("affiliation", ""),
                "birthday": info.get("birthday", ""),
                "raw_wikitext": wikitext,
                "description": _extract_first_paragraph(clean_text),
                "lore_text": clean_text[:3000],
            }
        )

        # 每 10 个角色增量保存一次（支持断点续传）
        if (i + 1) % 10 == 0:
            char_path.write_text(
                json.dumps(characters, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("  Progress saved: %d/%d characters", len(characters), max_chars)

        await asyncio.sleep(REQUEST_DELAY)

    # 最终保存
    char_path.write_text(
        json.dumps(characters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return characters


async def fetch_lore_pages(
    client: httpx.AsyncClient, categories: list[str], max_pages: int = 80
) -> list[dict]:
    """下载世界观相关页面"""
    all_pages = set()
    for cat in categories:
        members = await fetch_category_members(client, cat, limit=100)
        for m in members:
            if not m["title"].startswith(("Category:", "Template:")):
                all_pages.add(m["title"])

    pages = list(all_pages)[:max_pages]
    logger.info("Found %d lore pages across categories: %s", len(pages), categories)

    results = []
    for i, title in enumerate(pages):
        logger.info("[%d/%d] Fetching lore: %s", i + 1, len(pages), title)
        wikitext = await fetch_page_wikitext(client, title, section=0)
        if wikitext:
            results.append(
                {
                    "title": title,
                    "category": "lore",
                    "content": clean_wikitext(wikitext)[:5000],
                }
            )
        await asyncio.sleep(REQUEST_DELAY)

    return results


def _extract_first_paragraph(text: str) -> str:
    """提取 wiki 文本的第一段有意义内容"""
    paras = text.split("\n\n")
    for p in paras:
        p = p.strip()
        # 跳过太短或明显不是正文的行
        if len(p) > 50 and not p.startswith(("=", "*", "{{", "[[")):
            return p
    return paras[0] if paras else ""


# ── 保存 / 加载 ────────────────────────────────────────────


async def run_wiki_fetch(char_limit: int = 90, lore_limit: int = 80):
    """主入口：下载 wiki 数据并保存"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(
        timeout=30, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:

        # 角色
        logger.info("=" * 50)
        logger.info("Phase 1: Fetching characters")
        logger.info("=" * 50)
        characters = await fetch_all_characters(client, max_chars=char_limit)
        char_path = RAW_DIR / "wiki_characters.json"
        char_path.write_text(
            json.dumps(characters, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Saved %d characters to %s", len(characters), char_path)

        # 世界观
        logger.info("=" * 50)
        logger.info("Phase 2: Fetching lore pages")
        logger.info("=" * 50)
        lore = await fetch_lore_pages(
            client,
            categories=["Lore", "Regions", "History", "Organizations"],
            max_pages=lore_limit,
        )
        lore_path = RAW_DIR / "wiki_lore.json"
        lore_path.write_text(
            json.dumps(lore, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Saved %d lore pages to %s", len(lore), lore_path)

    return characters, lore
