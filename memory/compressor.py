"""
Memory Compressor — LLM-powered compression of short-term and episodic
memories into dense long-term summaries.

Without compression, old memories are simply discarded (short-term buffer
overwrites at 50 items, episodic compresses to goal+success only).
This module preserves the *meaning* of fading memories by periodically:

    1. Gathering a batch of recent short-term events
    2. Using an LLM to distill them into 1-3 key insights
    3. Storing those insights as semantic facts / self-notes

    4. Gathering recent episodic outcomes
    5. Summarizing patterns (what works, what fails, what to remember)
    6. Storing strategy notes in semantic memory

The result: the agent's long-term memory retains the *essence* of its
experiences even after the raw events are gone.

Usage:
    compressor = MemoryCompressor(memory_system)
    compressor.compress()  # run once (called periodically by brain)
"""

import os
import json
import time
from threading import Lock

from groq import Groq
from dotenv import load_dotenv

from .compressed_storage import save_json, load_json, migrate_if_needed

load_dotenv()


class MemoryCompressor:
    """Compresses fading memories into dense long-term knowledge."""

    def __init__(self, memory, store_dir: str = "memory_store"):
        """
        Args:
            memory: MemorySystem instance (has .short_term, .episodic, .semantic)
            store_dir: path for persistence files
        """
        self.memory = memory
        self._store_dir = store_dir
        self._lock = Lock()

        # Track what's already been compressed to avoid re-processing
        self._state_file = os.path.join(store_dir, "compressor_state.json")
        self._state = self._load_state()

        self._groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Minimum batch sizes before compressing
        self.MIN_EVENTS_BATCH = 20
        self.MIN_EPISODES_BATCH = 5

    # ── Public API ───────────────────────────────────────────────

    def compress(self) -> dict:
        """Run a compression cycle. Returns summary of what was compressed.

        Safe to call frequently — skips if there isn't enough new data.
        """
        results = {"events_compressed": 0, "episodes_compressed": 0,
                    "insights_stored": 0}

        try:
            ev = self._compress_events()
            results["events_compressed"] = ev["processed"]
            results["insights_stored"] += ev["insights"]
        except Exception as e:
            print(f"[Compressor] Event compression failed: {e}")

        try:
            ep = self._compress_episodes()
            results["episodes_compressed"] = ep["processed"]
            results["insights_stored"] += ep["insights"]
        except Exception as e:
            print(f"[Compressor] Episode compression failed: {e}")

        return results

    def compress_conversation(self, messages: list[dict]) -> str | None:
        """Compress a conversation into a concise summary.

        Args:
            messages: list of {"role": "user"|"agent", "content": str}

        Returns:
            Compressed summary string, or None on failure.
        """
        if len(messages) < 3:
            return None

        text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Agent'}: {m['content']}"
            for m in messages[-20:]
        )

        prompt = f"""Compress this conversation into 2-3 sentences capturing:
- What the user wanted
- What happened
- Any important details to remember

Conversation:
{text}

Output ONLY the compressed summary, nothing else."""

        try:
            resp = self._groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150, temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Compressor] Conversation compression failed: {e}")
            return None

    # ── Event compression ────────────────────────────────────────

    def _compress_events(self) -> dict:
        """Compress short-term events into long-term insights."""
        last_ts = self._state.get("last_event_ts", 0)
        events = self.memory.short_term.get_recent(50)

        # Filter only events newer than last compression
        new_events = [e for e in events if e.get("timestamp", 0) > last_ts]

        if len(new_events) < self.MIN_EVENTS_BATCH:
            return {"processed": 0, "insights": 0}

        # Format events for the LLM
        event_text = "\n".join(
            f"[{e.get('time_str', '?')}] ({e['type']}) {e['content']}"
            for e in new_events
        )

        prompt = f"""You are analyzing an AI agent's recent activity log.
Extract the most important insights to remember long-term.

Events:
{event_text}

Output a JSON object:
{{
  "facts": ["key factual observations worth remembering (0-3 items)"],
  "self_notes": ["insights about the agent's own behavior or patterns (0-2 items)"],
  "user_insights": ["things learned about the user (0-2 items)"]
}}

Rules:
- Only include genuinely useful insights, not trivial observations
- Each item should be a single concise sentence
- If nothing is worth remembering, use empty lists
- Output ONLY valid JSON"""

        try:
            resp = self._groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300, temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
        except Exception as e:
            print(f"[Compressor] LLM event compression failed: {e}")
            return {"processed": len(new_events), "insights": 0}

        # Store insights into semantic memory
        insights_count = 0
        for fact in result.get("facts", []):
            if fact.strip():
                self.memory.semantic.learn_fact(fact.strip())
                insights_count += 1
        for note in result.get("self_notes", []):
            if note.strip():
                self.memory.semantic.add_self_note(note.strip())
                insights_count += 1
        for pref in result.get("user_insights", []):
            if pref.strip():
                self.memory.semantic.learn_preference(pref.strip())
                insights_count += 1

        # Update watermark
        if new_events:
            self._state["last_event_ts"] = max(
                e.get("timestamp", 0) for e in new_events
            )
            self._save_state()

        if insights_count > 0:
            print(f"[Compressor] Extracted {insights_count} insights from {len(new_events)} events")

        return {"processed": len(new_events), "insights": insights_count}

    # ── Episode compression ──────────────────────────────────────

    def _compress_episodes(self) -> dict:
        """Analyze recent episodes for patterns and store strategies."""
        last_count = self._state.get("last_episode_count", 0)
        current_count = self.memory.episodic.total_episodes

        new_count = current_count - last_count
        if new_count < self.MIN_EPISODES_BATCH:
            return {"processed": 0, "insights": 0}

        # Get recent episodes for analysis
        episodes = self.memory.episodic.get_recent(min(new_count, 20))
        if not episodes:
            return {"processed": 0, "insights": 0}

        ep_text = "\n".join(
            f"- {'✓' if ep['success'] else '✗'} \"{ep['goal']}\" → {ep['outcome']} ({ep['steps']} steps)"
            for ep in episodes
        )

        success_rate = self.memory.episodic.get_success_rate(20)

        prompt = f"""You are analyzing an AI agent's recent task execution history.
Current success rate: {success_rate:.0%}

Recent tasks:
{ep_text}

Extract patterns and strategies. Output a JSON object:
{{
  "patterns": ["observed patterns in successes/failures (0-2 items)"],
  "strategies": ["strategies to improve performance (0-2 items)"],
  "facts": ["factual observations worth remembering (0-2 items)"]
}}

Rules:
- Focus on actionable insights
- Each should be a concise sentence
- If success rate is high and no issues, use empty lists
- Output ONLY valid JSON"""

        try:
            resp = self._groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300, temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
        except Exception as e:
            print(f"[Compressor] LLM episode compression failed: {e}")
            return {"processed": new_count, "insights": 0}

        insights_count = 0
        for item in result.get("patterns", []) + result.get("strategies", []):
            if item.strip():
                self.memory.semantic.add_self_note(item.strip())
                insights_count += 1
        for fact in result.get("facts", []):
            if fact.strip():
                self.memory.semantic.learn_fact(fact.strip())
                insights_count += 1

        self._state["last_episode_count"] = current_count
        self._save_state()

        if insights_count > 0:
            print(f"[Compressor] Extracted {insights_count} strategies from {new_count} episodes")

        return {"processed": new_count, "insights": insights_count}

    # ── Persistence ──────────────────────────────────────────────

    def _load_state(self) -> dict:
        migrate_if_needed(self._state_file)
        result = load_json(self._state_file)
        return result if result is not None else {}

    def _save_state(self):
        try:
            save_json(self._state_file, self._state)
        except Exception as e:
            print(f"[Compressor] Failed to save state: {e}")
