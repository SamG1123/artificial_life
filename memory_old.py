"""
Memory system for the autonomous AI.

Three layers:
1. Short-term (working memory) — rolling buffer of recent events, thoughts, observations
2. Episodic (experience log) — past goals, outcomes, conversations
3. Long-term (persistent knowledge) — learned facts, user preferences, self-knowledge

Persisted to disk as JSON so the AI remembers across restarts.
"""

import json
import os
import time
from collections import deque
from threading import Lock

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_store")
LONG_TERM_FILE = os.path.join(MEMORY_DIR, "long_term.json")
EPISODIC_FILE = os.path.join(MEMORY_DIR, "episodes.json")

SHORT_TERM_CAPACITY = 50    # max items in working memory
EPISODIC_CAPACITY = 200     # max episodes stored


class MemorySystem:
    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self._lock = Lock()

        # ── Short-term / working memory ──────────────────────────────
        # Rolling buffer of recent events: observations, thoughts, user speech, actions
        self.short_term: deque = deque(maxlen=SHORT_TERM_CAPACITY)

        # ── Episodic memory ──────────────────────────────────────────
        # Past goal executions with outcomes
        self.episodes: list = self._load_json(EPISODIC_FILE, default=[])
        if len(self.episodes) > EPISODIC_CAPACITY:
            self.episodes = self.episodes[-EPISODIC_CAPACITY:]

        # ── Long-term memory ─────────────────────────────────────────
        # Persistent facts, preferences, self-knowledge
        self.long_term: dict = self._load_json(LONG_TERM_FILE, default={
            "user_name": None,
            "user_preferences": [],
            "learned_facts": [],
            "self_notes": [],        # AI's own observations about itself
            "personality_traits": [
                "curious", "helpful", "calm", "slightly playful"
            ],
        })

    # ── Short-term operations ────────────────────────────────────────

    def add_event(self, event_type: str, content: str):
        """Add an event to working memory.
        
        event_type: 'observation', 'thought', 'user_speech', 'action',
                    'action_result', 'system'
        """
        with self._lock:
            self.short_term.append({
                "type": event_type,
                "content": content,
                "timestamp": time.time(),
                "time_str": time.strftime("%H:%M:%S"),
            })

    def get_recent_events(self, n: int = 15, event_type: str = None) -> list:
        """Get the most recent n events, optionally filtered by type."""
        with self._lock:
            items = list(self.short_term)
            if event_type:
                items = [e for e in items if e["type"] == event_type]
            return items[-n:]

    def get_context_summary(self, max_events: int = 20) -> str:
        """Build a text summary of recent working memory for LLM context."""
        events = self.get_recent_events(max_events)
        if not events:
            return "(no recent events)"
        lines = []
        for e in events:
            prefix = {
                "observation": "👁 SAW",
                "thought": "💭 THOUGHT",
                "user_speech": "🗣 USER",
                "action": "⚡ DID",
                "action_result": "📋 RESULT",
                "system": "⚙ SYS",
            }.get(e["type"], e["type"].upper())
            lines.append(f"[{e['time_str']}] {prefix}: {e['content']}")
        return "\n".join(lines)

    # ── Episodic operations ──────────────────────────────────────────

    def log_episode(self, goal: str, outcome: str, steps: int, success: bool):
        """Record a completed goal execution as an episode."""
        episode = {
            "goal": goal,
            "outcome": outcome,
            "steps": steps,
            "success": success,
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M"),
        }
        with self._lock:
            self.episodes.append(episode)
            if len(self.episodes) > EPISODIC_CAPACITY:
                self.episodes = self.episodes[-EPISODIC_CAPACITY:]
        self._save_json(EPISODIC_FILE, self.episodes)

    def get_recent_episodes(self, n: int = 5) -> list:
        """Get the n most recent episodes."""
        with self._lock:
            return self.episodes[-n:]

    def get_episodes_summary(self, n: int = 5) -> str:
        """Text summary of recent episodes for LLM context."""
        episodes = self.get_recent_episodes(n)
        if not episodes:
            return "(no past experiences)"
        lines = []
        for ep in episodes:
            status = "✓" if ep["success"] else "✗"
            lines.append(f"[{ep['time_str']}] {status} \"{ep['goal']}\" → {ep['outcome']} ({ep['steps']} steps)")
        return "\n".join(lines)

    # ── Long-term operations ─────────────────────────────────────────

    def learn_fact(self, fact: str):
        """Store a learned fact in long-term memory."""
        with self._lock:
            if fact not in self.long_term["learned_facts"]:
                self.long_term["learned_facts"].append(fact)
                # Keep bounded
                if len(self.long_term["learned_facts"]) > 100:
                    self.long_term["learned_facts"] = self.long_term["learned_facts"][-100:]
        self._save_long_term()

    def learn_preference(self, preference: str):
        """Store a user preference."""
        with self._lock:
            if preference not in self.long_term["user_preferences"]:
                self.long_term["user_preferences"].append(preference)
        self._save_long_term()

    def add_self_note(self, note: str):
        """AI's own observation about itself or its behavior."""
        with self._lock:
            self.long_term["self_notes"].append(note)
            if len(self.long_term["self_notes"]) > 50:
                self.long_term["self_notes"] = self.long_term["self_notes"][-50:]
        self._save_long_term()

    def set_user_name(self, name: str):
        with self._lock:
            self.long_term["user_name"] = name
        self._save_long_term()

    def get_long_term_summary(self) -> str:
        """Text summary of long-term memory for LLM context."""
        lt = self.long_term
        parts = []
        if lt.get("user_name"):
            parts.append(f"User's name: {lt['user_name']}")
        if lt.get("user_preferences"):
            parts.append(f"User preferences: {', '.join(lt['user_preferences'][-10:])}")
        if lt.get("learned_facts"):
            parts.append(f"Known facts: {'; '.join(lt['learned_facts'][-10:])}")
        if lt.get("self_notes"):
            parts.append(f"Self-notes: {'; '.join(lt['self_notes'][-5:])}")
        return "\n".join(parts) if parts else "(no long-term memories yet)"

    # ── Persistence ──────────────────────────────────────────────────

    def _save_long_term(self):
        self._save_json(LONG_TERM_FILE, self.long_term)

    @staticmethod
    def _save_json(path: str, data):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Memory] Failed to save {path}: {e}")

    @staticmethod
    def _load_json(path: str, default=None):
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Memory] Failed to load {path}: {e}")
        return default if default is not None else {}
