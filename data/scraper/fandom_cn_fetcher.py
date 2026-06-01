"""Fandom 中文 Wiki 爬取 — allpages API + 角色名匹配"""
import asyncio
import json
import logging
import re
from pathlib import Path

import httpx

logger = logging.getLogger("paimon.scraper.fandomcn")

WIKI_API = "https://genshin-impact.fandom.com/zh/api.php"
USER_AGENT = "Mozilla/5.0 PaimonCompanion/1.0 (Educational Project)"
REQUEST_DELAY = 0.5
RAW_DIR = Path(__file__).parent.parent / "raw"

# 已知角色中文名 → 用于从 allpages 中筛选
TARGET_CHARACTERS = [
    "鍾離", "溫迪", "雷電將軍", "納西妲", "胡桃", "魈", "甘雨", "刻晴",
    "迪盧克", "琴", "七七", "莫娜", "可莉", "達達利亞", "阿貝多", "優菈",
    "楓原萬葉", "神里綾華", "宵宮", "早柚", "珊瑚宮心海", "九條裟羅",
    "荒瀧一斗", "五郎", "托馬", "雲菫", "申鶴", "八重神子", "夜蘭",
    "神里綾人", "久岐忍", "鹿野院平藏", "柯萊", "提納里", "多莉", "賽諾",
    "妮露", "萊依拉", "琺露珊", "艾爾海森", "瑤瑤", "迪希雅", "米卡",
    "白朮", "卡維", "綺良良", "林尼", "琳妮特", "菲米尼", "那維萊特",
    "萊歐斯利", "芙寧娜", "夏洛蒂", "娜維婭", "夏沃蕾", "閒雲", "嘉明",
    "千織", "阿蕾奇諾", "克洛琳德", "希格雯", "艾梅莉埃", "卡齊娜",
    "基尼奇", "瑪拉妮", "希諾寧", "茜特菈莉", "恰斯卡", "瑪薇卡",
    "班尼特", "菲謝爾", "香菱", "行秋", "凝光", "北斗", "重雲",
    "諾艾爾", "芭芭拉", "麗莎", "凱亞", "安柏", "雷澤", "砂糖",
    "迪奧娜", "羅莎莉亞", "煙緋", "空", "熒", "派蒙",
]

ELEMENT_MAP = {"岩": "Geo", "風": "Anemo", "雷": "Electro", "草": "Dendro",
               "水": "Hydro", "火": "Pyro", "冰": "Cryo"}
WEAPON_MAP = {"單手劍": "Sword", "雙手劍": "Claymore", "長柄武器": "Polearm",
              "長槍": "Polearm", "弓": "Bow", "法器": "Catalyst"}
REGION_MAP = {"蒙德": "Mondstadt", "璃月": "Liyue", "稻妻": "Inazuma",
              "須彌": "Sumeru", "楓丹": "Fontaine", "納塔": "Natlan", "至冬": "Snezhnaya"}


async def fetch_all_pages(client: httpx.AsyncClient, limit: int = 500) -> list[str]:
    """获取所有页面标题"""
    titles = []
    params = {
        "action": "query", "list": "allpages",
        "apnamespace": 0, "aplimit": min(limit, 500), "format": "json",
    }
    while True:
        resp = await client.get(WIKI_API, params=params)
        resp.raise_for_status()
        data = resp.json()
        for p in data.get("query", {}).get("allpages", []):
            titles.append(p["title"])
        if "continue" in data and len(titles) < limit:
            params["apcontinue"] = data["continue"]["apcontinue"]
            await asyncio.sleep(0.3)
        else:
            break
    return titles


async def fetch_page_wikitext(client: httpx.AsyncClient, title: str) -> str | None:
    """获取页面 wikitext"""
    try:
        resp = await client.get(WIKI_API, params={
            "action": "parse", "page": title,
            "prop": "wikitext", "format": "json",
        })
        resp.raise_for_status()
        return resp.json().get("parse", {}).get("wikitext", {}).get("*", "")
    except Exception as e:
        logger.warning("Failed: %s — %s", title, e)
        return None


def parse_infobox(wikitext: str) -> dict:
    """解析中文 infobox"""
    info = {}
    patterns = {
        "name_cn": r"\|\s*名字\s*=\s*(.+?)(?:\n\||\n\})",
        "title": r"\|\s*稱號\s*=\s*(.+?)(?:\n\||\n\})",
        "name_en": r"\|\s*英語名\s*=\s*(.+?)(?:\n\||\n\})",
        "rarity_text": r"\|\s*稀有度\s*=\s*(\d+)星",
        "element_cn": r"\|\s*元素\s*=\s*(.+?)(?:\n\||\n\})",
        "weapon_cn": r"\|\s*武器\s*=\s*(.+?)(?:\n\||\n\})",
        "region_cn": r"\|\s*地區\s*=\s*(.+?)(?:\n\||\n\})",
        "affiliation_cn": r"\|\s*所屬\s*=\s*(.+?)(?:\n\||\n\})",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, wikitext)
        if m:
            info[key] = m.group(1).strip()
    return info


def clean_wikitext(text: str) -> str:
    """清洗 wikitext"""
    if not text: return ""
    # 移除模板
    depth = 0; result = []; i = 0
    while i < len(text):
        if text[i:i+2] == "{{": depth += 1; i += 2; continue
        elif text[i:i+2] == "}}" and depth > 0: depth -= 1; i += 2; continue
        if depth == 0: result.append(text[i])
        i += 1
    text = "".join(result)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+?)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_paragraph(text: str) -> str:
    for p in text.split("\n\n"):
        p = p.strip()
        if len(p) > 30 and not p.startswith(("=", "*", "{{", "[[")):
            return p
    return ""


async def run(max_chars: int = 80):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        # Step 1: Get all page titles
        logger.info("Fetching all pages from Fandom ZH...")
        all_titles = await fetch_all_pages(client, limit=500)
        logger.info("Found %d total pages", len(all_titles))

        # Step 2: Filter for target characters (try both Traditional and Simplified)
        char_titles = []
        for title in all_titles:
            for target in TARGET_CHARACTERS:
                if target in title and not any(skip in title for skip in ["/", "劇情", "台詞", "語音"]):
                    char_titles.append(title)
                    break

        # Deduplicate and limit
        char_titles = list(dict.fromkeys(char_titles))[:max_chars]
        logger.info("Matched %d character pages", len(char_titles))

        # Step 3: Fetch each character page
        characters = []
        for i, title in enumerate(char_titles):
            logger.info("[%d/%d] %s", i + 1, len(char_titles), title)
            wikitext = await fetch_page_wikitext(client, title)
            if not wikitext or "REDIRECT" in wikitext[:50]:
                continue

            info = parse_infobox(wikitext)
            clean = clean_wikitext(wikitext)

            element, weapon, region = (
                ELEMENT_MAP.get(info.get("element_cn", ""), info.get("element_cn", "")),
                WEAPON_MAP.get(info.get("weapon_cn", ""), info.get("weapon_cn", "")),
                REGION_MAP.get(info.get("region_cn", ""), info.get("region_cn", "")),
            )

            characters.append({
                "name_en": info.get("name_en", "") or title,
                "name_cn": info.get("name_cn") or title,
                "title": info.get("title", ""),
                "rarity": int(info.get("rarity_text", "0")[0]) if info.get("rarity_text") else 0,
                "element": element,
                "weapon_type": weapon,
                "region": region,
                "affiliation": info.get("affiliation_cn", ""),
                "description": _extract_paragraph(clean),
                "lore_text": clean[:3000],
                "source": "fandom_cn",
                "language": "zh",
            })

            if (i + 1) % 10 == 0:
                path = RAW_DIR / "fandom_cn_characters.json"
                path.write_text(json.dumps(characters, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("  Progress: %d saved", len(characters))

            await asyncio.sleep(REQUEST_DELAY)

        # Final save
        path = RAW_DIR / "fandom_cn_characters.json"
        path.write_text(json.dumps(characters, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Done! %d characters saved to %s", len(characters), path)

    return characters


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    asyncio.run(run(max_chars=80))
