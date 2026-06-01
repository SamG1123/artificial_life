"""FastAPI demo server with WebSocket for live chat.

Run with: `uvicorn demo_server_fastapi:app --host 0.0.0.0 --port 8000`
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from agent import AgentController  # type: ignore
except Exception:
    # Fallback lightweight stub when heavy deps (LLM, groq) are unavailable
    class AgentController:
        def __init__(self, headless: bool = True):
            self.headless = headless
            self.running = True
            class Brain:
                def _say(self, text: str):
                    return None
            self.brain = Brain()

        def start(self):
            return None

        def stop(self):
            self.running = False

        def send_message(self, text: str):
            # echo back for demo
            def echo():
                try:
                    self.brain._say(f"Echo: {text}")
                except Exception:
                    pass
            import threading
            threading.Thread(target=echo, daemon=True).start()

        def status(self) -> dict:
            return {
                "running": self.running,
                "mode": "headless",
                "brain_state": "IDLE",
                "memory_events": 0,
                "active_projects": 0,
            }

app = FastAPI(title="Artificial Life Demo")

HTML_PAGE = open("demo_server.py").read().split('HTML_PAGE = ', 1)[1]


class Runtime:
    def __init__(self):
        self.agent = AgentController(headless=True)
        self._messages: List[dict] = []
        self._next_id = 1
        self._lock = asyncio.Lock()
        self._clients: List[WebSocket] = []
        self._patch()
        try:
            self.agent.start()
        except AttributeError:
            # Some AgentController start() implementations use platform-specific
            # signals (e.g., SIGBREAK on Windows). Ignore when not available
            # so the demo can run in Linux containers.
            pass

    def _patch(self):
        original_say = self.agent.brain._say

        def hooked_say(text: str):
            if text:
                asyncio.get_event_loop().create_task(self.record_message("agent", text))
            return original_say(text)

        self.agent.brain._say = hooked_say

    async def record_message(self, role: str, text: str) -> dict:
        async with self._lock:
            message = {
                "id": self._next_id,
                "role": role,
                "text": text,
                "ts": time.time(),
            }
            self._next_id += 1
            self._messages.append(message)
        await self._broadcast(message)
        return dict(message)

    async def _broadcast(self, message: dict) -> None:
        data = json.dumps({"type": "message", "message": message})
        to_remove: List[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(data)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            if ws in self._clients:
                self._clients.remove(ws)

    async def get_messages(self, after: int = 0, limit: int = 100) -> List[dict]:
        async with self._lock:
            return [m for m in self._messages if m["id"] > after][:limit]

    def status(self) -> dict:
        return self.agent.status()

    async def chat(self, text: str) -> dict:
        msg = await self.record_message("user", text)
        self.agent.send_message(text)
        return {"ok": True, "message": msg}

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        if ws in self._clients:
            self._clients.remove(ws)


runtime: Optional[Runtime] = None


@app.on_event("startup")
async def on_startup():
    global runtime
    runtime = Runtime()


@app.on_event("shutdown")
async def on_shutdown():
    global runtime
    if runtime:
        try:
            runtime.agent.stop()
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(content=HTML_PAGE)


def _require_api_key(x_api_key: str | None = Header(None)) -> bool:
    key = os.getenv("DEMO_API_KEY")
    if key:
        if x_api_key != key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return True


@app.get("/status")
async def status():
    if not runtime:
        raise HTTPException(status_code=503, detail="Runtime not ready")
    return JSONResponse(content=runtime.status())


@app.get("/messages")
async def messages(after: int = 0, limit: int = 100):
    if not runtime:
        raise HTTPException(status_code=503, detail="Runtime not ready")
    msgs = await runtime.get_messages(after=after, limit=limit)
    return JSONResponse(content={"messages": msgs})


@app.post("/chat")
async def chat(request: Request, _auth: bool = Depends(_require_api_key)):
    payload = await request.json()
    text = (payload.get("text") or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "missing text"})
    if not runtime:
        return JSONResponse(status_code=503, content={"error": "runtime not ready"})
    res = await runtime.chat(text)
    return JSONResponse(status_code=201, content=res)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    key = os.getenv("DEMO_API_KEY")
    if key:
        token = ws.query_params.get("api_key")
        if token != key:
            await ws.close(code=1008)
            return
    if not runtime:
        await ws.close(code=1011)
        return
    await runtime.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            # expect simple JSON {"text":"..."}
            try:
                payload = json.loads(data)
            except Exception:
                payload = {"text": data}
            text = (payload.get("text") or "").strip()
            if text:
                await runtime.chat(text)
    except WebSocketDisconnect:
        await runtime.disconnect(ws)
