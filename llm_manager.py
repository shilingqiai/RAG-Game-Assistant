"""LLM 调用封装模块"""
import logging
import os
import sys

# DashScope 是国内阿里云服务，直连即可。如果系统设了全局代理（Clash/V2Ray），
# 代理会把 dashscope 流量也劫持，绕境外一圈回来反而超时。
# 必须在 import 任何 HTTP 库之前把 ali 域名加入 NO_PROXY。
# urllib3 用后缀匹配，所以 "aliyuncs.com" 就能覆盖所有子域名。
for _key in ("NO_PROXY", "no_proxy"):
    _existing = os.environ.get(_key, "")
    if "aliyuncs.com" not in _existing:
        os.environ[_key] = f"{_existing},aliyuncs.com" if _existing else "aliyuncs.com"

# 统一日志配置（在所有模块之前执行）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("paimon.log", encoding="utf-8"),
    ],
)

from llama_index.core import Settings
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding
from config import AppConfig

# 在模块导入时即设置嵌入模型，防止 LlamaIndex 回退到 OpenAI 默认值
_SETTINGS_PATCHED = False


def _patch_settings():
    global _SETTINGS_PATCHED
    if not _SETTINGS_PATCHED:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        try:
            # 在模块导入时即注入 LLM 和嵌入模型，防止 LlamaIndex 回退到 OpenAI
            Settings.embed_model = DashScopeEmbedding(
                model_name=AppConfig.EMBED_MODEL,
                api_key=api_key,
                embed_batch_size=10,
            )
            Settings.llm = DashScope(
                model_name=AppConfig.MAIN_MODEL,
                temperature=0.1,
                api_key=api_key,
                max_tokens=2000,
            )
            _SETTINGS_PATCHED = True
        except Exception:
            pass  # 如果 API key 未设置，稍后在 LLMManager 中处理


_patch_settings()


class LLMManager:
    """LLM 管理器 — DashScope (通义千问)"""

    def __init__(self):
        api_key = os.getenv("DASHSCOPE_API_KEY")

        if not api_key:
            print("[LLM] 警告: DASHSCOPE_API_KEY 环境变量未设置！")
            print("[LLM] 请设置: export DASHSCOPE_API_KEY='你的阿里云DashScope API Key'")

        # 主 LLM（qwen-plus-2025-12-01）
        self.llm = DashScope(
            model_name=AppConfig.MAIN_MODEL,
            temperature=0.1,
            api_key=api_key,
            max_tokens=2000,
        )

        # 嵌入模型 — 优先复用全局 Settings 中已设置的实例
        self.embed_model = (
            Settings.embed_model
            if Settings.embed_model
            else DashScopeEmbedding(
                model_name=AppConfig.EMBED_MODEL,
                api_key=api_key,
                embed_batch_size=10,
            )
        )

    def complete(self, prompt: str) -> str:
        """非流式调用 LLM"""
        response = self.llm.complete(prompt)
        return str(response)

    def stream_complete(self, prompt: str):
        """流式调用 LLM"""
        stream_response = self.llm.stream_complete(prompt)
        for chunk in stream_response:
            if chunk.delta:
                yield chunk.delta
