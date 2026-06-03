# Paimon Companion — 原神智能游戏伴侣
# 构建: docker build -t paimon .
# 运行: docker run -p 7860:7860 --env-file .env paimon

FROM python:3.12-slim

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用 Docker 缓存层）
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir \
        llama-index-core>=0.10.57 \
        llama-index-llms-dashscope>=0.5.0 \
        llama-index-embeddings-dashscope>=0.5.0 \
        gradio>=4.0.0 \
        duckduckgo-search>=8.0.0 \
        httpx>=0.25.0 \
        dashscope>=1.20.0 \
        rank-bm25>=0.2.0

# 复制源码
COPY *.py ./
COPY tests/ ./tests/
COPY data/ ./data/

# 初始化 git submodule（genshin-data）
RUN if [ -f .gitmodules ]; then \
        git submodule update --init data/genshin-data; \
    fi

# 构建中文向量索引（如果 storage/ 不存在）
RUN python -c "import os; os.makedirs('storage', exist_ok=True)" && \
    ( [ -f storage/index_store.json ] || python data_indexer_cn.py )

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["python", "simple_start.py"]
