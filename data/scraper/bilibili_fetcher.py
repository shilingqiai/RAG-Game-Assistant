"""Bilibili Wiki 原神中文数据爬取 — 角色 + 世界观"""
import asyncio
import json
import logging
import re
from pathlib import Path

import httpx

logger = logging.getLogger("paimon.scraper.bilibili")

WIKI_API = "https://wiki.biligame.com/ys/api.php"
USER_AGENT = "Mozilla/5.0 PaimonCompanion/1.0 (Educational Project)"
REQUEST_DELAY = 0.5
RAW_DIR = Path(__file__).parent.parent / "raw"


async def fetch_category_members(
    client: httpx.AsyncClient, category: str, limit: int = 200
) -> list[dict]:
    """获取分类下所有页面"""
    members = []
    params = {
        "action": "query", "list": "categorymembers",
        "cmtitle": f"Category:{category}", "cmlimit": min(limit, 500),
        "format": "json",
    }
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


async def fetch_page_wikitext(client: httpx.AsyncClient, page_title: str) -> str | None:
    """获取页面 wikitext"""
    try:
        resp = await client.get(WIKI_API, params={
            "action": "parse", "page": page_title,
            "prop": "wikitext", "format": "json",
        })
        resp.raise_for_status()
        return resp.json().get("parse", {}).get("wikitext", {}).get("*", "")
    except Exception as e:
        logger.warning("Failed to fetch '%s': %s", page_title, e)
        return None


# ── Wikitext 解析（中文 infobox）────────────────────────────

def parse_cn_infobox(wikitext: str) -> dict:
    """从中文 wiki infobox 提取结构化字段"""
    info = {}
    patterns = {
        "name_cn": r"\|\s*名字\s*=\s*(.+?)(?:\n\||\n\})",
        "title": r"\|\s*称号\s*=\s*(.+?)(?:\n\||\n\})",
        "full_name": r"\|\s*全名\s*=\s*(.+?)(?:\n\||\n\})",
        "name_en": r"\|\s*英语名\s*=\s*(.+?)(?:\n\||\n\})",
        "rarity_text": r"\|\s*稀有度\s*=\s*(\d+)星",
        "element_cn": r"\|\s*元素\s*=\s*(.+?)(?:\n\||\n\})",
        "weapon_cn": r"\|\s*武器\s*=\s*(.+?)(?:\n\||\n\})",
        "region_cn": r"\|\s*地区\s*=\s*(.+?)(?:\n\||\n\})",
        "affiliation_cn": r"\|\s*所属\s*=\s*(.+?)(?:\n\||\n\})",
        "birthday": r"\|\s*生日\s*=\s*(.+?)(?:\n\||\n\})",
        "gender": r"\|\s*性别\s*=\s*(.+?)(?:\n\||\n\})",
        "constellation": r"\|\s*命之座\s*=\s*(.+?)(?:\n\||\n\})",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, wikitext)
        if match:
            info[key] = match.group(1).strip()
    return info


def clean_wikitext(text: str) -> str:
    """清洗中文 wikitext"""
    if not text:
        return ""
    # 移除模板 {{...}}
    depth = 0
    result = []
    i = 0
    while i < len(text):
        if text[i:i+2] == "{{":
            depth += 1; i += 2; continue
        elif text[i:i+2] == "}}" and depth > 0:
            depth -= 1; i += 2; continue
        if depth == 0:
            result.append(text[i])
        i += 1
    text = "".join(result)

    # 移除 wiki 链接 [[...]]
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+?)\]\]", r"\1", text)
    # 移除 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 移除 ref
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    # 移除表格
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    # 移除多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_lore_sections(wikitext: str) -> dict[str, str]:
    """提取角色故事章节"""
    sections = {}
    current_section = "intro"
    current_text = []

    for line in wikitext.split("\n"):
        header_match = re.match(r"==+\s*(.+?)\s*==+", line)
        if header_match:
            if current_text:
                sections[current_section] = "\n".join(current_text)
            current_section = header_match.group(1).strip()
            current_text = []
        else:
            current_text.append(line)

    if current_text:
        sections[current_section] = "\n".join(current_text)

    return sections


# ── 映射 ────────────────────────────────────────────────────

ELEMENT_MAP = {"岩": "Geo", "风": "Anemo", "雷": "Electro", "草": "Dendro",
               "水": "Hydro", "火": "Pyro", "冰": "Cryo"}
WEAPON_MAP = {"单手剑": "Sword", "双手剑": "Claymore", "长柄武器": "Polearm",
              "弓": "Bow", "法器": "Catalyst"}
REGION_MAP = {"蒙德": "Mondstadt", "璃月": "Liyue", "稻妻": "Inazuma",
              "须弥": "Sumeru", "枫丹": "Fontaine", "纳塔": "Natlan",
              "至冬": "Snezhnaya"}


def _cn_to_en(element_cn: str, weapon_cn: str, region_cn: str) -> tuple:
    return (
        ELEMENT_MAP.get(element_cn, element_cn),
        WEAPON_MAP.get(weapon_cn, weapon_cn),
        REGION_MAP.get(region_cn, region_cn),
    )


# ── 主流程 ──────────────────────────────────────────────────


async def fetch_characters(client: httpx.AsyncClient, max_chars: int = 90) -> list[dict]:
    """下载中文角色数据"""
    logger.info("Fetching character list from Bilibili Wiki...")
    members = await fetch_category_members(client, "角色")

    # 过滤非角色页面
    char_names = [
        m["title"] for m in members
        if not m["title"].startswith(("Category:", "Template:", "File:", "Widget:"))
        and not any(skip in m["title"] for skip in ["/", "剧情", "台词", "资料", "筛选"])
    ]
    logger.info("Found %d character pages", len(char_names))

    characters = []
    for i, name in enumerate(char_names[:max_chars]):
        logger.info("[%d/%d] %s", i + 1, min(len(char_names), max_chars), name)

        wikitext = await fetch_page_wikitext(client, name)
        if not wikitext:
            continue

        info = parse_cn_infobox(wikitext)
        clean = clean_wikitext(wikitext)
        sections = extract_lore_sections(wikitext)

        # 映射中文字段到英文标准字段
        element, weapon, region = _cn_to_en(
            info.get("element_cn", ""),
            info.get("weapon_cn", ""),
            info.get("region_cn", ""),
        )

        # 提取故事文本
        story_parts = []
        for sec_name in ["角色故事", "角色详细", "故事", "背景"]:
            for key, text in sections.items():
                if sec_name in key and len(text) > 50:
                    story_parts.append(text)

        lore_text = "\n\n".join(story_parts) if story_parts else clean[:3000]

        characters.append({
            "name_en": info.get("name_en", "") or name,
            "name_cn": info.get("name_cn") or name,
            "title": info.get("title", ""),
            "rarity": int(info.get("rarity_text", "0")[0]) if info.get("rarity_text") else 0,
            "element": element,
            "weapon_type": weapon,
            "region": region,
            "affiliation": info.get("affiliation_cn", ""),
            "birthday": info.get("birthday", ""),
            "constellation": info.get("constellation", ""),
            "description": _extract_paragraph(clean),
            "lore_text": lore_text[:3000],
            "source": "bilibili_wiki",
            "language": "zh",
        })

        # 增量保存
        if (i + 1) % 10 == 0:
            _save(characters, "bilibili_characters.json")
            logger.info("  Progress: %d characters saved", len(characters))

        await asyncio.sleep(REQUEST_DELAY)

    _save(characters, "bilibili_characters.json")
    return characters


async def fetch_world_lore(client: httpx.AsyncClient, max_pages: int = 50) -> list[dict]:
    """下载世界观页面"""
    categories = ["世界观", "地区", "组织", "任务", "活动"]
    all_pages = []
    for cat in categories:
        try:
            members = await fetch_category_members(client, cat, limit=50)
            all_pages.extend(m for m in members if not m["title"].startswith(("Category:", "Template:")))
        except Exception:
            pass

    pages = all_pages[:max_pages]
    logger.info("Found %d lore pages", len(pages))

    results = []
    for i, page in enumerate(pages):
        title = page["title"]
        logger.info("[%d/%d] %s", i + 1, len(pages), title)
        wikitext = await fetch_page_wikitext(client, title)
        if wikitext:
            results.append({
                "title": title,
                "title_cn": title,
                "category": "lore",
                "content": clean_wikitext(wikitext)[:5000],
                "source": "bilibili_wiki",
                "language": "zh",
            })
        await asyncio.sleep(REQUEST_DELAY)

    _save(results, "bilibili_lore.json")
    return results


def _extract_paragraph(text: str) -> str:
    """提取第一段有意义文本"""
    paras = text.split("\n\n")
    for p in paras:
        p = p.strip()
        if len(p) > 30 and not p.startswith(("=", "*", "{{", "[[")):
            return p
    return paras[0] if paras else ""


def _save(data: list, filename: str):
    path = RAW_DIR / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


async def run():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        logger.info("=== Phase 1: Chinese Characters ===")
        chars = await fetch_characters(client, max_chars=90)
        logger.info("=== Phase 2: Chinese Lore ===")
        lore = await fetch_world_lore(client, max_pages=50)
    return chars, lore


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    asyncio.run(run())
