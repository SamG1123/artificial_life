"""Browser demo server for the hosted AI agent.

Run with:
    C:/Users/myste/OneDrive/Desktop/personal/artificial_life/venv/Scripts/python.exe demo_server.py
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from agent import AgentController


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Artificial Life Demo</title>
  <style>
    :root{--bg:#0b1020;--panel:#0f1724;--muted:#9aa4b2;--accent:#60dbff}
    body{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;background:linear-gradient(180deg,#071025,#0b1020);color:#e6eef8}
    .app{max-width:960px;margin:28px auto;padding:18px;display:flex;gap:16px}
    .col{flex:1}
    .panel{background:linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01));border:1px solid rgba(255,255,255,0.04);border-radius:12px;overflow:hidden}
    header{padding:14px 16px;border-bottom:1px solid rgba(255,255,255,0.02)}
    h1{margin:0;font-size:18px}
    .messages{height:64vh;padding:12px;overflow:auto;display:flex;flex-direction:column;gap:8px}
    .msg{max-width:78%;padding:10px 12px;border-radius:10px;white-space:pre-wrap}
    .msg.user{align-self:flex-end;background:linear-gradient(90deg,#073642,#0b4b5c);}
    .msg.agent{align-self:flex-start;background:linear-gradient(90deg,#0b2a3a,#103146);}
    .composer{display:flex;gap:10px;padding:12px;border-top:1px solid rgba(255,255,255,0.02)}
    textarea{flex:1;height:56px;border-radius:8px;padding:10px;border:1px solid rgba(255,255,255,0.03);background:#071220;color:inherit}
    button{background:var(--accent);border:none;padding:10px 16px;border-radius:8px;color:#04202a;font-weight:700}
    .status{padding:12px;display:flex;flex-direction:column;gap:8px}
    .stat{font-size:13px;color:var(--muted)}
    .small{font-size:12px;color:var(--muted)}
    @media(max-width:900px){.app{flex-direction:column}.messages{height:50vh}}
  </style>
</head>
<body>
  <div class="app">
    <div class="col panel">
      <header><h1>Artificial Life — Demo</h1></header>
      <div id="messages" class="messages" aria-live="polite"></div>
      <div class="composer">
        <textarea id="input" placeholder="Say something..." aria-label="Message"></textarea>
        <button id="send">Send</button>
      </div>
    </div>
    <div style="width:260px" class="panel">
      <header><h1>Status</h1></header>
      <div id="status" class="status small">Loading...</div>
      <div style="padding:12px;border-top:1px solid rgba(255,255,255,0.02)"><div class="small">Tips: Use the WebSocket (fast) or HTTP endpoints. Set DEMO_API_KEY for access control.</div></div>
    </div>
  </div>

  <script>
    const messagesEl=document.getElementById('messages');const statusEl=document.getElementById('status');const inputEl=document.getElementById('input');const sendEl=document.getElementById('send');let lastId=0;let ws=null;const apiKey=new URLSearchParams(location.search).get('api_key')||'';

    function addMessage(role,text,ts){const d=document.createElement('div');d.className='msg '+role;d.textContent=text;const wrap=document.createElement('div');wrap.appendChild(d);messagesEl.appendChild(d);messagesEl.scrollTop=messagesEl.scrollHeight}

    async function fetchStatus(){try{const r=await fetch('/status');if(!r.ok)throw r;const j=await r.json();statusEl.innerHTML='';for(const k of ['mode','brain_state','mood','current_goal','memory_events']){const v=j[k]===undefined?'-':j[k];const el=document.createElement('div');el.className='stat';el.textContent=k.replace('_',' ')+': '+String(v);statusEl.appendChild(el)}}catch(e){statusEl.textContent='Status unavailable'}}

    async function fetchMessages(){try{const r=await fetch('/messages?after='+lastId);if(!r.ok)throw r;const j=await r.json();for(const m of j.messages){addMessage(m.role,m.text,m.ts);lastId=Math.max(lastId,m.id)}}catch(e){}
    }

    function connectWS(){try{const proto=(location.protocol==='https:'?'wss':'ws');const url=proto+'://'+location.host+'/ws'+(apiKey?('?api_key='+encodeURIComponent(apiKey)): '');ws=new WebSocket(url);ws.onmessage=(ev)=>{try{const d=JSON.parse(ev.data);if(d.type==='message'){addMessage(d.message.role,d.message.text,d.message.ts);lastId=Math.max(lastId,d.message.id)}}catch(e){console.warn(e)}};ws.onopen=()=>{console.log('ws open')};ws.onclose=()=>{console.log('ws close');setTimeout(connectWS,3000)}}catch(e){console.warn('ws failed',e)} }

    async function send(){const text=inputEl.value.trim();if(!text) return;inputEl.value='';if(ws && ws.readyState===WebSocket.OPEN){ws.send(JSON.stringify({text}));addMessage('user',text,Date.now()/1000);}else{await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});await fetchMessages();}}

    sendEl.addEventListener('click',send);inputEl.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
    connectWS();fetchStatus();fetchMessages();setInterval(fetchStatus,3000);setInterval(fetchMessages,1500);
  </script>
</body>
</html>
"""


class DemoRuntime:
    def __init__(self):
        self.agent = AgentController(headless=True)
        self._lock = threading.Lock()
        self._messages: list[dict] = []
        self._next_id = 1
        self._patch_speech_output()
        self.agent.start()

    def _patch_speech_output(self) -> None:
        original_say = self.agent.brain._say

        def hooked_say(text: str):
            if text:
                self.record_message("agent", text)
            return original_say(text)

        self.agent.brain._say = hooked_say

    def record_message(self, role: str, text: str) -> dict:
        with self._lock:
            message = {
                "id": self._next_id,
                "role": role,
                "text": text,
                "ts": time.time(),
            }
            self._next_id += 1
            self._messages.append(message)
            return dict(message)

    def get_messages(self, after: int = 0, limit: int = 100) -> list[dict]:
        with self._lock:
            return [m for m in self._messages if m["id"] > after][:limit]

    def status(self) -> dict:
        return self.agent.status()

    def chat(self, text: str) -> dict:
        user_message = self.record_message("user", text)
        self.agent.send_message(text)
        return {"ok": True, "message": user_message}

    def stop(self) -> None:
        self.agent.stop()


class DemoRequestHandler(BaseHTTPRequestHandler):
    server_version = "ArtificialLifeDemo/1.0"

    @property
    def runtime(self) -> DemoRuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return
        if parsed.path == "/status":
            self._send_json(self.runtime.status())
            return
        if parsed.path == "/messages":
            after = int(parse_qs(parsed.query).get("after", ["0"])[0] or 0)
            messages = self.runtime.get_messages(after=after)
            self._send_json({"messages": messages})
            return
        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/chat":
            self.send_error(404, "Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        text = ""
        if body:
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                payload = parse_qs(body.decode("utf-8"))
            if isinstance(payload, dict):
                text = payload.get("text") or ""
                if isinstance(text, list):
                    text = text[0] if text else ""
        text = text.strip()
        if not text:
            self.send_error(400, "Missing text")
            return
        result = self.runtime.chat(text)
        self._send_json(result, status=201)

    def log_message(self, format, *args):
        return

    def _send_json(self, payload: dict, status: int = 200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, content: str, status: int = 200):
        data = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Artificial Life demo server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    runtime = DemoRuntime()
    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    server.runtime = runtime  # type: ignore[attr-defined]

    try:
        print(f"Demo server running at http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        runtime.stop()


if __name__ == "__main__":
    main()