"""双语索引构建器 — 英文 wiki 数据 + 中文名称映射 → 跨语言 RAG"""
import json
import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding

# ── 中文名称映射（英文 wiki 名 → 简体中文名）─────────────────

CN_NAME_MAP = {
    "aether": "空", "lumine": "荧", "zhongli": "钟离", "venti": "温迪",
    "raiden shogun": "雷电将军", "nahida": "纳西妲", "hu tao": "胡桃",
    "xiao": "魈", "ganyu": "甘雨", "keqing": "刻晴", "diluc": "迪卢克",
    "jean": "琴", "qiqi": "七七", "mona": "莫娜", "klee": "可莉",
    "tartaglia": "达达利亚", "albedo": "阿贝多", "eula": "优菈",
    "kaedehara kazuha": "枫原万叶", "kamisato ayaka": "神里绫华",
    "yoimiya": "宵宫", "sayu": "早柚", "sangonomiya kokomi": "珊瑚宫心海",
    "kujou sara": "九条裟罗", "arataki itto": "荒泷一斗", "gorou": "五郎",
    "thoma": "托马", "yun jin": "云堇", "shenhe": "申鹤",
    "yae miko": "八重神子", "yelan": "夜兰", "kamisato ayato": "神里绫人",
    "kuki shinobu": "久岐忍", "shikanoin heizou": "鹿野院平藏",
    "collei": "柯莱", "tighnari": "提纳里", "dori": "多莉", "cyno": "赛诺",
    "nilou": "妮露", "layla": "莱依拉", "faruzan": "珐露珊",
    "alhaitham": "艾尔海森", "yaoyao": "瑶瑶", "dehya": "迪希雅",
    "mika": "米卡", "baizhu": "白术", "kaveh": "卡维", "kirara": "绮良良",
    "lyney": "林尼", "lynette": "琳妮特", "freminet": "菲米尼",
    "neuvillette": "那维莱特", "wriothesley": "莱欧斯利",
    "furina": "芙宁娜", "charlotte": "夏洛蒂", "navia": "娜维娅",
    "chevreuse": "夏沃蕾", "xianyun": "闲云", "gaming": "嘉明",
    "chiori": "千织", "arlecchino": "阿蕾奇诺", "clorinde": "克洛琳德",
    "sigewinne": "希格雯", "emilie": "艾梅莉埃", "kachina": "卡齐娜",
    "kinich": "基尼奇", "mualani": "玛拉妮", "xilonen": "希诺宁",
    "citlali": "茜特菈莉", "chasca": "恰斯卡", "mavuika": "玛薇卡",
    "iansan": "伊安珊", "ororon": "欧洛伦", "bennett": "班尼特",
    "fischl": "菲谢尔", "xiangling": "香菱", "xingqiu": "行秋",
    "ningguang": "凝光", "beidou": "北斗", "chongyun": "重云",
    "noelle": "诺艾尔", "barbara": "芭芭拉", "lisa": "丽莎",
    "kaeya": "凯亚", "amber": "安柏", "razor": "雷泽",
    "sucrose": "砂糖", "diona": "迪奥娜", "rosaria": "罗莎莉亚",
    "yanfei": "烟绯", "aloy": "埃洛伊",
}

ELEMENT_MAP = {
    "Pyro": "火", "Hydro": "水", "Anemo": "风", "Electro": "雷",
    "Dendro": "草", "Cryo": "冰", "Geo": "岩",
}
WEAPON_MAP = {
    "Sword": "单手剑", "Claymore": "双手剑", "Polearm": "长柄武器",
    "Bow": "弓", "Catalyst": "法器",
}
REGION_MAP = {
    "Mondstadt": "蒙德", "Liyue": "璃月", "Inazuma": "稻妻",
    "Sumeru": "须弥", "Fontaine": "枫丹", "Natlan": "纳塔",
    "Snezhnaya": "至冬",
}


def build_bilingual_documents(characters_path: str, lore_path: str) -> list[Document]:
    """构建双语文档"""
    documents = []

    # ── 角色文档（双语）──
    with open(characters_path, encoding="utf-8") as f:
        characters = json.load(f)

    for c in characters:
        name_en = c["name_en"]
        name_cn = CN_NAME_MAP.get(name_en.lower(), c.get("name_cn", ""))

        # 中文优先的标题
        title = f"{name_cn}（{name_en}）" if name_cn else name_en

        parts = [title]
        if c.get("rarity"):
            parts.append(f"稀有度：{'⭐' * c['rarity']}  Rarity: {c['rarity']}★")
        if c.get("element"):
            cn_el = ELEMENT_MAP.get(c["element"], c["element"])
            parts.append(f"元素：{cn_el}（{c['element']}）")
        if c.get("weapon_type"):
            cn_wp = WEAPON_MAP.get(c["weapon_type"], c["weapon_type"])
            parts.append(f"武器：{cn_wp}（{c['weapon_type']}）")
        if c.get("region"):
            cn_rg = REGION_MAP.get(c["region"], c["region"])
            parts.append(f"地区：{cn_rg}（{c['region']}）")
        if c.get("affiliation"):
            parts.append(f"所属：{c['affiliation']}")
        if c.get("description"):
            parts.append(f"简介：{c['description']}")
        if c.get("lore_text"):
            lore = c["lore_text"]
            if len(lore) > 2000:
                lore = lore[:2000]
            parts.append(f"背景故事：{lore}")

        text = "\n".join(parts)

        documents.append(
            Document(
                text=text,
                metadata={
                    "title": title,
                    "name_en": name_en,
                    "name_cn": name_cn,
                    "category": "character",
                    "rarity": c.get("rarity", 0),
                    "element": c.get("element", ""),
                    "region": c.get("region", ""),
                    "language": "bilingual",
                },
            )
        )

    # ── 世界观文档（英文 + 中文关键字）──
    if os.path.exists(lore_path):
        with open(lore_path, encoding="utf-8") as f:
            lore_entries = json.load(f)

        for l in lore_entries:
            title = l.get("title", "")
            content = l.get("content", "")

            # 为世界观标题添加常见中文翻译
            cn_hints = _get_lore_cn_hints(title)
            display_title = f"{cn_hints}（{title}）" if cn_hints else title

            text = f"{display_title}\n\n{content[:3000]}"
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "title": title,
                        "category": "lore",
                        "language": "bilingual" if cn_hints else "en",
                    },
                )
            )

    return documents


def _get_lore_cn_hints(title: str) -> str:
    """为世界观标题提供中文提示"""
    hints = {
        "The Seven Archons": "七神",
        "Archon": "执政官/神",
        "Vision": "神之眼",
        "Gnosis": "神之心",
        "Abyss": "深渊",
        "Celestia": "天空岛",
        "Khaenri'ah": "坎瑞亚",
        "Fatui": "愚人众",
        "Hilichurl": "丘丘人",
        "Adeptus": "仙人",
        "Fontaine": "枫丹",
        "Inazuma": "稻妻",
        "Liyue": "璃月",
        "Mondstadt": "蒙德",
        "Sumeru": "须弥",
        "Natlan": "纳塔",
        "Snezhnaya": "至冬",
        "Descender": "降临者",
        "Teyvat": "提瓦特",
        "Morax": "摩拉克斯",
        "Barbatos": "巴巴托斯",
        "Raiden": "雷电",
    }
    matches = [cn for en, cn in hints.items() if en.lower() in title.lower()]
    return " ".join(matches) if matches else ""


def build_bilingual_index(documents: list[Document], storage_dir: str = "./storage"):
    """构建双语索引"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import llm_manager  # noqa: 触发 Settings 注入

    api_key = os.getenv("DASHSCOPE_API_KEY")
    embed_model = DashScopeEmbedding(
        model_name="text-embedding-v3",
        api_key=api_key,
        embed_batch_size=10,
    )

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=128),
            KeywordExtractor(keywords=5),
            embed_model,
        ],
    )

    print(f"Running pipeline on {len(documents)} bilingual documents...")
    nodes = pipeline.run(documents=documents, show_progress=True)

    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir=storage_dir)

    print(f"Bilingual index: {len(nodes)} nodes saved to {storage_dir}")

    # 统计
    cn_chars = sum(1 for d in documents if d.metadata.get("name_cn"))
    print(f"Characters with CN names: {cn_chars}/{len([d for d in documents if d.metadata.get('category')=='character'])}")
    return index


def main():
    characters_path = "data/processed/characters.json"
    lore_path = "data/raw/wiki_lore.json"

    print("Building bilingual documents...")
    documents = build_bilingual_documents(characters_path, lore_path)
    print(f"Total documents: {len(documents)}")

    # 清理旧索引
    import shutil
    if os.path.exists("./storage"):
        shutil.rmtree("./storage")

    build_bilingual_index(documents)
    print("Done! Run: python eval_rag.py")


if __name__ == "__main__":
    main()
