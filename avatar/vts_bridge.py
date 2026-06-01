"""VTube Studio bridge worker.

This is intentionally resilient and optional. If websocket support is not
available or VTube Studio is not running, it stays dormant and retries.
"""

from __future__ import annotations

import json
import os
import threading
import time
from queue import Empty, Queue
from uuid import uuid4

from logging_config import get_logger

log = get_logger("avatar.vts")


class VTubeStudioBridge:
    def __init__(self, host: str, port: int,
                 plugin_name: str,
                 plugin_developer: str,
                 expression_map: dict[str, str] | None = None,
                 store_dir: str = "memory_store"):
        self.host = host
        self.port = port
        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer
        self._updates: Queue[dict] = Queue(maxsize=64)
        self._connected = False
        self._token_path = os.path.join(store_dir, "vts_auth_token.json")
        self._expression_map = dict(expression_map or {})
        self._last_expression = ""
        self._last_expression_ts = 0.0
        os.makedirs(store_dir, exist_ok=True)

    def submit_state(self, *, speaking: bool, valence: float,
                     arousal: float, viseme: str,
                     emotion: str,
                     expression_strength: float) -> None:
        payload = {
            "speaking": bool(speaking),
            "valence": float(valence),
            "arousal": float(arousal),
            "viseme": str(viseme),
            "emotion": str(emotion or "neutral"),
            "expression_strength": float(expression_strength),
        }
        if self._updates.full():
            try:
                self._updates.get_nowait()
            except Empty:
                pass
        try:
            self._updates.put_nowait(payload)
        except Exception:
            pass

    def run(self, stop_event: threading.Event) -> None:
        ws = None
        backoff = 2.0

        while not stop_event.is_set():
            try:
                if ws is None:
                    ws = self._connect()
                    self._connected = ws is not None
                    if self._connected:
                        if not self._authenticate(ws):
                            raise RuntimeError("VTube Studio authentication failed")
                        backoff = 2.0
                        log.info("Connected and authenticated with VTube Studio")

                payload = self._updates.get(timeout=0.2)
                if ws is not None:
                    ws.send(json.dumps(self._build_param_inject_request(payload)))
                    expr_req = self._build_expression_request(payload)
                    if expr_req is not None:
                        ws.send(json.dumps(expr_req))
            except Empty:
                continue
            except Exception as e:
                if self._connected:
                    log.warning("VTS bridge connection dropped: %s", e)
                self._connected = False
                ws = None
                stop_event.wait(backoff)
                backoff = min(20.0, backoff * 1.5)

    def _connect(self):
        try:
            from websocket import create_connection
        except Exception:
            log.warning("websocket-client not installed; VTS bridge disabled")
            return None

        url = f"ws://{self.host}:{self.port}"
        try:
            ws = create_connection(url, timeout=2.0)
            ws.settimeout(2.0)
            return ws
        except Exception:
            return None

    def _authenticate(self, ws) -> bool:
        token = self._load_token()
        if not token:
            token = self._request_auth_token(ws)
            if not token:
                return False
            self._save_token(token)

        if self._request_auth(ws, token):
            return True

        # Token may be stale; request a new one once.
        token = self._request_auth_token(ws)
        if not token:
            return False
        self._save_token(token)
        return self._request_auth(ws, token)

    def _request_auth_token(self, ws) -> str:
        req = self._make_request(
            message_type="AuthenticationTokenRequest",
            data={
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer,
            },
        )
        ws.send(json.dumps(req))
        try:
            resp = json.loads(ws.recv())
        except Exception:
            return ""
        return str((resp.get("data") or {}).get("authenticationToken") or "")

    def _request_auth(self, ws, token: str) -> bool:
        req = self._make_request(
            message_type="AuthenticationRequest",
            data={
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer,
                "authenticationToken": token,
            },
        )
        ws.send(json.dumps(req))
        try:
            resp = json.loads(ws.recv())
        except Exception:
            return False
        return bool((resp.get("data") or {}).get("authenticated", False))

    def _build_param_inject_request(self, state_payload: dict) -> dict:
        speaking = bool(state_payload.get("speaking", False))
        valence = float(state_payload.get("valence", 0.0))
        arousal = float(state_payload.get("arousal", 0.2))
        viseme = str(state_payload.get("viseme", "rest"))
        expression_strength = float(state_payload.get("expression_strength", 0.0))

        mouth_open = 0.0 if not speaking else self._viseme_open_amount(viseme)
        smile = max(0.0, min(1.0, valence * (0.6 + 0.4 * expression_strength)))
        angle_z = max(-15.0, min(15.0, (arousal - 0.5) * 15.0))
        mouth_form = self._viseme_form_amount(viseme)
        eye_open = max(0.2, min(1.0, 1.0 - 0.5 * expression_strength if speaking else 1.0))

        return self._make_request(
            message_type="InjectParameterDataRequest",
            data={
                "faceFound": True,
                "mode": "set",
                "parameterValues": [
                    {"id": "ParamMouthOpenY", "value": mouth_open},
                    {"id": "ParamMouthForm", "value": mouth_form},
                    {"id": "ParamSmile", "value": smile},
                    {"id": "ParamEyeOpen", "value": eye_open},
                    {"id": "ParamAngleZ", "value": angle_z},
                ],
            },
        )

    def _build_expression_request(self, state_payload: dict) -> dict | None:
        emotion = str(state_payload.get("emotion", "neutral")).lower()
        strength = float(state_payload.get("expression_strength", 0.0))

        if strength < 0.25:
            return None

        expression_file = self._expression_map.get(emotion, "")
        if not expression_file:
            return None

        now = time.time()
        if expression_file == self._last_expression and (now - self._last_expression_ts) < 0.8:
            return None

        self._last_expression = expression_file
        self._last_expression_ts = now

        return self._make_request(
            message_type="ExpressionActivationRequest",
            data={
                "expressionFile": expression_file,
                "active": True,
            },
        )

    @staticmethod
    def _viseme_open_amount(viseme: str) -> float:
        open_map = {
            "rest": 0.02,
            "mbp": 0.05,
            "fv": 0.14,
            "td": 0.16,
            "sz": 0.24,
            "l": 0.22,
            "r": 0.24,
            "wq": 0.33,
            "ee": 0.40,
            "ih": 0.43,
            "uh": 0.52,
            "oh": 0.58,
            "aa": 0.78,
            "kg": 0.26,
        }
        return float(open_map.get(viseme, 0.2))

    @staticmethod
    def _viseme_form_amount(viseme: str) -> float:
        form_map = {
            "rest": 0.0,
            "mbp": -0.25,
            "fv": -0.15,
            "td": -0.05,
            "sz": 0.05,
            "l": 0.1,
            "r": 0.12,
            "wq": 0.28,
            "ee": 0.35,
            "ih": 0.2,
            "uh": -0.08,
            "oh": -0.22,
            "aa": -0.12,
            "kg": -0.02,
        }
        return float(form_map.get(viseme, 0.0))

    def _load_token(self) -> str:
        if not os.path.exists(self._token_path):
            return ""
        try:
            with open(self._token_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return str(raw.get("token", ""))
        except Exception:
            return ""

    def _save_token(self, token: str) -> None:
        try:
            with open(self._token_path, "w", encoding="utf-8") as f:
                json.dump({"token": token}, f)
        except Exception:
            pass

    @staticmethod
    def _make_request(*, message_type: str, data: dict) -> dict:
        return {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": str(uuid4()),
            "messageType": message_type,
            "data": data,
        }
