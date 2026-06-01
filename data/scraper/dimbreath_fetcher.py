"""Dimbreath AnimeGameData 下载 — 角色数值 / 武器 / TextMap"""
import asyncio
import json
import logging
from pathlib import Path

import httpx

logger = logging.getLogger("paimon.scraper.dimbreath")

BASE_URL = "https://gitlab.com/Dimbreath/AnimeGameData/-/raw/master"
RAW_DIR = Path(__file__).parent.parent / "raw" / "dimbreath"

# 要下载的文件列表
FILES = {
    "avatars": f"{BASE_URL}/ExcelBinOutput/AvatarExcelConfigData.json",
    "weapons": f"{BASE_URL}/ExcelBinOutput/WeaponExcelConfigData.json",
    "materials": f"{BASE_URL}/ExcelBinOutput/MaterialExcelConfigData.json",
    "textmap_chs": f"{BASE_URL}/TextMap/TextMapCHS.json",
    "textmap_en": f"{BASE_URL}/TextMap/TextMapEN.json",
}


async def download_file(
    client: httpx.AsyncClient, name: str, url: str
) -> dict | list | None:
    """下载单个 JSON 文件"""
    try:
        logger.info("Downloading %s...", name)
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("Failed to download %s: %s", name, e)
        return None


async def run_dimbreath_fetch() -> dict[str, Path]:
    """下载所有 Dimbreath 数据文件"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    saved = {}
    async with httpx.AsyncClient(
        timeout=120, follow_redirects=True, headers={"User-Agent": "PaimonCompanion/1.0"}
    ) as client:

        for name, url in FILES.items():
            data = await download_file(client, name, url)
            if data is None:
                continue

            path = RAW_DIR / f"{name}.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            saved[name] = path
            logger.info(
                "Saved %s: %d entries (%s bytes)", name, len(data), path.stat().st_size
            )

    return saved
