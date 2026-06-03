"""Paimon Companion — FastAPI + SSE 服务端

启动: uvicorn server:app --host 0.0.0.0 --port 8000
访问: http://localhost:8000
"""

import asyncio
import json
import os
import sys
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel


# ── 生命周期 ────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global companion
    print("=" * 60)
    print("🎒 Paimon Companion — FastAPI 启动中...")
    print("=" * 60)
    t0 = time.time()
    from game_core import GameCompanion

    companion = GameCompanion()
    print(f"[Server] ✅ GameCompanion 就绪 ({time.time() - t0:.1f}s)")
    if companion.agent.query_engine:
        print("[Server] ✅ RAG 混合检索已加载")
    print(f"[Server] 🌐 http://0.0.0.0:8000")
    yield
    print("[Server] 关闭...")


app = FastAPI(title="Paimon Companion API", version="0.4.0", lifespan=lifespan)
companion = None  # 在 lifespan 中初始化


# ── 请求模型 ────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str


class AssetUpdateRequest(BaseModel):
    key: str  # primogems, intertwined_fate, pity_count, is_guaranteed
    value: int


# ── SSE 流式聊天 ─────────────────────────────────────────────


@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    """SSE 流式聊天端点"""
    if not companion:
        return JSONResponse({"error": "Server not ready"}, status_code=503)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            async for token in companion.chat_stream(req.message):
                payload = json.dumps({"token": token, "type": "token"}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except Exception as e:
            payload = json.dumps({"token": f"出错了: {e}", "type": "error"}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
        finally:
            yield f"data: {json.dumps({'token': '', 'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


# ── 资产管理 API ────────────────────────────────────────────


@app.get("/api/assets")
async def get_assets():
    """获取用户资产"""
    if not companion:
        return JSONResponse({"error": "Server not ready"}, status_code=503)

    p = companion.agent.user_profile
    return {
        "primogems": p.get_asset("primogems"),
        "intertwined_fate": p.get_asset("intertwined_fate"),
        "pity_count": p.get_asset("pity_count"),
        "is_guaranteed": bool(p.get_asset("is_guaranteed")),
        "total_pulls": p.get_asset("primogems") // 160 + p.get_asset("intertwined_fate"),
    }


@app.post("/api/assets/update")
async def update_asset(req: AssetUpdateRequest):
    """更新单个资产值（设置绝对值）"""
    if not companion:
        return JSONResponse({"error": "Server not ready"}, status_code=503)

    valid_keys = {"primogems", "intertwined_fate", "pity_count", "is_guaranteed"}
    if req.key not in valid_keys:
        return JSONResponse({"error": f"Invalid key: {req.key}"}, status_code=400)

    p = companion.agent.user_profile
    current = p.get_asset(req.key)
    delta = req.value - current
    p.update_asset(req.key, delta)

    return {"status": "ok", "key": req.key, "old": current, "new": req.value}


@app.get("/api/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "rag": companion.agent.query_engine is not None if companion else False,
    }


# ── 前端页面 ────────────────────────────────────────────────


@app.get("/")
async def index():
    """简单聊天前端"""
    return HTMLResponse(content=HTML_PAGE)


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Paimon Companion</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #1a1a2e; color: #eee; height: 100vh; display: flex; }
.sidebar { width: 260px; background: #16213e; padding: 16px; overflow-y: auto; border-right: 1px solid #0f3460; }
.sidebar h2 { font-size: 16px; margin-bottom: 12px; color: #e94560; }
.asset-item { background: #0f3460; border-radius: 8px; padding: 10px; margin-bottom: 8px; }
.asset-item .label { font-size: 12px; color: #aaa; }
.asset-item .value { font-size: 20px; font-weight: bold; color: #f0c040; }
.asset-item input { width: 100%; margin-top: 4px; padding: 4px 8px; border-radius: 4px; border: 1px solid #0f3460; background: #1a1a2e; color: #eee; font-size: 14px; }
.asset-item button { margin-top: 4px; padding: 4px 12px; border: none; border-radius: 4px; background: #e94560; color: #fff; cursor: pointer; font-size: 12px; }
.main { flex: 1; display: flex; flex-direction: column; max-width: calc(100vw - 260px); }
.header { padding: 12px 20px; background: #16213e; border-bottom: 1px solid #0f3460; font-size: 18px; font-weight: bold; }
.header span { color: #e94560; }
.chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
.msg { max-width: 80%; padding: 10px 14px; border-radius: 12px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
.msg.user { align-self: flex-end; background: #0f3460; }
.msg.bot { align-self: flex-start; background: #1a1a3e; border: 1px solid #0f3460; }
.msg.status { align-self: flex-start; color: #e94560; font-size: 13px; background: none; border: none; padding: 4px 0; }
.input-area { display: flex; padding: 12px 20px; background: #16213e; border-top: 1px solid #0f3460; }
.input-area input { flex: 1; padding: 10px 16px; border-radius: 20px; border: 1px solid #0f3460; background: #1a1a2e; color: #eee; font-size: 15px; outline: none; }
.input-area input:focus { border-color: #e94560; }
.input-area button { margin-left: 10px; padding: 10px 24px; border-radius: 20px; border: none; background: #e94560; color: #fff; font-size: 15px; cursor: pointer; font-weight: bold; }
.input-area button:hover { background: #c1314d; }
.input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
</head>
<body>

<div class="sidebar">
  <h2>📊 旅行者资产</h2>
  <div id="assets"></div>
</div>

<div class="main">
  <div class="header">🎒 <span>Paimon Companion</span> — 你的原神向导派蒙</div>
  <div class="chat" id="chat">
    <div class="msg bot">旅行者你好！我是派蒙，有什么想知道的尽管问我！<br>可以问我角色故事、世界设定，或者帮你管理抽卡资产~</div>
  </div>
  <div class="input-area">
    <input id="input" placeholder="问派蒙任何原神相关问题..." autofocus onkeydown="if(event.key==='Enter')send()">
    <button id="sendBtn" onclick="send()">发送</button>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');

function addMsg(text, role) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function send() {
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  sendBtn.disabled = true;
  input.disabled = true;

  addMsg(msg, 'user');
  let botDiv = addMsg('', 'bot');
  let statusDiv = addMsg('> 思考中...', 'status');

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = JSON.parse(line.slice(6));
        if (data.type === 'done') break;
        if (data.type === 'error') {
          botDiv.textContent += data.token;
          break;
        }
        if (data.type === 'token') {
          if (data.token.startsWith('> ')) {
            statusDiv.textContent = data.token;
          } else {
            if (statusDiv) { statusDiv.remove(); statusDiv = null; }
            botDiv.textContent += data.token;
            chat.scrollTop = chat.scrollHeight;
          }
        }
      }
    }
    if (statusDiv) statusDiv.remove();
  } catch (e) {
    botDiv.textContent = '连接失败: ' + e.message;
    if (statusDiv) statusDiv.remove();
  } finally {
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

// 加载资产
async function loadAssets() {
  try {
    const resp = await fetch('/api/assets');
    const data = await resp.json();
    document.getElementById('assets').innerHTML = `
      <div class="asset-item">
        <div class="label">原石</div>
        <div class="value" id="val-primos">${data.primogems.toLocaleString()}</div>
        <input id="inp-primos" type="number" value="${data.primogems}">
        <button onclick="updateAsset('primogems')">更新</button>
      </div>
      <div class="asset-item">
        <div class="label">纠缠之缘</div>
        <div class="value" id="val-fate">${data.intertwined_fate}</div>
        <input id="inp-fate" type="number" value="${data.intertwined_fate}">
        <button onclick="updateAsset('intertwined_fate')">更新</button>
      </div>
      <div class="asset-item">
        <div class="label">垫水位</div>
        <div class="value" id="val-pity">${data.pity_count} / 90</div>
        <input id="inp-pity" type="number" value="${data.pity_count}" min="0" max="90">
        <button onclick="updateAsset('pity_count')">更新</button>
      </div>
      <div class="asset-item" style="text-align:center">
        <div class="label">总计可抽</div>
        <div class="value">${data.total_pulls} 抽</div>
        <button onclick="toggleGuarantee()" style="margin-top:6px;width:100%">
          🔄 大保底: ${data.is_guaranteed ? '是' : '否'}
        </button>
      </div>
    `;
  } catch (e) {
    console.error('Failed to load assets:', e);
  }
}

async function updateAsset(key) {
  const inp = document.getElementById('inp-' + (key === 'intertwined_fate' ? 'fate' : key === 'pity_count' ? 'pity' : 'primos'));
  const value = parseInt(inp.value);
  if (isNaN(value)) return;
  await fetch('/api/assets/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, value }),
  });
  loadAssets();
}

async function toggleGuarantee() {
  const resp = await fetch('/api/assets');
  const data = await resp.json();
  await fetch('/api/assets/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key: 'is_guaranteed', value: data.is_guaranteed ? 0 : 1 }),
  });
  loadAssets();
}

loadAssets();
</script>
</body>
</html>"""

# ── CLI 入口 ────────────────────────────────────────────────


def main():
    import uvicorn

    host = os.getenv("PAIMON_HOST", "0.0.0.0")
    port = int(os.getenv("PAIMON_PORT", "8000"))
    print(f"Starting Paimon Companion on {host}:{port}")
    uvicorn.run("server:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
