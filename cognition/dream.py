"""
Dream Engine — experience-based dreaming for memory consolidation.

Inspired by how human dreaming works:

  1. **Replay** — episodic memories are "replayed" (sampled),
     sometimes distorted or recombined.
  2. **Emotional processing** — emotionally charged experiences
     are revisited so the AI can process unresolved feelings.
  3. **Insight extraction** — the LLM finds hidden connections
     between disparate experiences, producing creative insights.
  4. **Consolidation** — useful insights are stored as semantic
     facts / self-notes, strengthening long-term memory.

Dreaming is triggered after prolonged idle periods (the AI has
been "resting") and produces a dream narrative the agent can
optionally share with the user.

    ┌──────────────────────────────────────────────────┐
    │                DREAM CYCLE                        │
    │                                                   │
    │  Sample episodes → Recombine → LLM dream gen →   │
    │  Extract insights → Store learnings → Emotional   │
    │  regulation → Optional narration                  │
    └──────────────────────────────────────────────────┘
"""

import os
import json
import time
import random
import threading

from groq import Groq
from dotenv import load_dotenv

from memory.compressed_storage import save_json, load_json

load_dotenv()


class DreamEngine:
    """Generates dreams from past experiences during extended idle periods."""

    # How long the agent must be idle before dreaming begins (seconds)
    IDLE_THRESHOLD = 180  # 3 minutes of idleness

    # Minimum interval between dream cycles (seconds)
    DREAM_COOLDOWN = 600  # 10 minutes

    # How many episodes to sample per dream
    SAMPLE_SIZE = 5

    # Max dreams to keep in the journal
    JOURNAL_CAPACITY = 100

    def __init__(self, memory, behavior=None, store_dir: str = "memory_store"):
        """
        Args:
            memory: MemorySystem instance (episodic + semantic + short_term)
            behavior: BehaviorController (optional — for emotional context)
            store_dir: persistence directory
        """
        self.memory = memory
        self.behavior = behavior
        self._store_dir = store_dir
        self._lock = threading.Lock()

        self._groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # State
        self._last_dream_time: float = 0.0
        self._idle_start: float = 0.0
        self._is_idle = False
        self._dream_count = 0

        # Dream journal — persisted log of past dreams
        self._journal_file = os.path.join(store_dir, "dream_journal.json.gz")
        self._journal: list[dict] = load_json(self._journal_file, [])

    # ── Public API ───────────────────────────────────────────────

    def mark_idle(self) -> None:
        """Notify the engine that the agent has entered idle state."""
        with self._lock:
            if not self._is_idle:
                self._is_idle = True
                self._idle_start = time.time()

    def mark_active(self) -> None:
        """Notify the engine that the agent is no longer idle."""
        with self._lock:
            self._is_idle = False

    def should_dream(self) -> bool:
        """Check whether conditions are right for a dream cycle."""
        with self._lock:
            if not self._is_idle:
                return False
            idle_duration = time.time() - self._idle_start
            if idle_duration < self.IDLE_THRESHOLD:
                return False
            if time.time() - self._last_dream_time < self.DREAM_COOLDOWN:
                return False
        # Need enough experiences to dream about
        episodes = self.memory.episodic.get_recent(3)
        return len(episodes) >= 2

    def dream(self) -> dict | None:
        """Run a dream cycle. Returns the dream result dict, or None.

        The dream:
          1) Samples past episodes (biased toward emotional/recent ones)
          2) Sends them to the LLM for creative recombination
          3) Extracts insights and stores them in semantic memory
          4) Logs the dream in the journal
        """
        # Gather raw material
        episodes = self._sample_episodes()
        if len(episodes) < 2:
            return None

        # Emotional context for coloring the dream
        emotional_tone = self._get_emotional_tone()

        # Recent thoughts — like pre-sleep rumination
        recent_thoughts = self._get_recent_thoughts()

        # Generate the dream via LLM
        dream_result = self._generate_dream(episodes, emotional_tone, recent_thoughts)
        if not dream_result:
            return None

        # Store insights from the dream
        self._consolidate(dream_result)

        # Log to journal
        dream_entry = {
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M"),
            "narrative": dream_result.get("narrative", ""),
            "theme": dream_result.get("theme", ""),
            "insights": dream_result.get("insights", []),
            "emotional_tone": emotional_tone,
            "source_episodes": len(episodes),
            "dream_number": self._dream_count + 1,
        }

        with self._lock:
            self._last_dream_time = time.time()
            self._dream_count += 1
            self._journal.append(dream_entry)
            if len(self._journal) > self.JOURNAL_CAPACITY:
                self._journal = self._journal[-self.JOURNAL_CAPACITY:]

        self._save_journal()

        return dream_entry

    def get_recent_dreams(self, n: int = 5) -> list[dict]:
        """Return the most recent dreams from the journal."""
        with self._lock:
            return list(self._journal[-n:])

    def get_last_dream(self) -> dict | None:
        """Return the most recent dream, if any."""
        with self._lock:
            return self._journal[-1] if self._journal else None

    def get_context_for_llm(self) -> str:
        """Summary of recent dreams for injecting into brain prompts."""
        with self._lock:
            recent = self._journal[-2:]
        if not recent:
            return ""
        lines = ["Recent dreams:"]
        for d in recent:
            theme = d.get("theme", "unknown")
            narrative = d.get("narrative", "")[:120]
            lines.append(f"  [{d.get('time_str', '?')}] Theme: {theme} — {narrative}")
        return "\n".join(lines)

    def stats(self) -> dict:
        """Return dreaming statistics."""
        with self._lock:
            return {
                "total_dreams": self._dream_count,
                "journal_size": len(self._journal),
                "last_dream_time": self._last_dream_time,
                "is_idle": self._is_idle,
                "idle_duration": (
                    round(time.time() - self._idle_start, 1)
                    if self._is_idle else 0.0
                ),
            }

    # ── Episode sampling ─────────────────────────────────────────

    def _sample_episodes(self) -> list[dict]:
        """Sample episodes for dreaming — biased toward:
           - Emotionally significant (failures, big successes)
           - Recent experiences
           - Random older ones (creative recombination)
        """
        recent = self.memory.episodic.get_recent(20)
        compressed = self.memory.episodic.get_compressed(30)

        pool: list[tuple[dict, float]] = []  # (episode, weight)

        # Recent episodes — higher weight
        for i, ep in enumerate(recent):
            weight = 1.0 + (i / max(1, len(recent)))  # recency bias
            # Emotional weighting: failures are more memorable
            if not ep.get("success", True):
                weight *= 2.0
            # Long tasks are more impactful
            steps = ep.get("steps", 1)
            if steps > 5:
                weight *= 1.5
            pool.append((ep, weight))

        # Older compressed episodes — lower weight, for creative mixing
        for ep in compressed:
            weight = 0.5
            if not ep.get("success", True):
                weight *= 1.5
            pool.append((ep, weight))

        if not pool:
            return []

        # Weighted sampling without replacement
        episodes = []
        remaining = list(pool)
        sample_n = min(self.SAMPLE_SIZE, len(remaining))

        for _ in range(sample_n):
            if not remaining:
                break
            total_w = sum(w for _, w in remaining)
            if total_w <= 0:
                break
            r = random.random() * total_w
            cumulative = 0.0
            for idx, (ep, w) in enumerate(remaining):
                cumulative += w
                if cumulative >= r:
                    episodes.append(ep)
                    remaining.pop(idx)
                    break

        return episodes

    def _get_emotional_tone(self) -> str:
        """Get the current emotional tone to color the dream."""
        if self.behavior:
            mood = self.behavior.mood.get_mood_label()
            dominant = self.behavior.emotion.dominant_emotion()
            return f"{mood} (dominant emotion: {dominant or 'neutral'})"
        return "calm"

    def _get_recent_thoughts(self) -> str:
        """Gather recent inner thoughts — pre-sleep rumination."""
        thoughts = self.memory.short_term.get_recent(10, event_type="thought")
        if not thoughts:
            return "(no recent thoughts)"
        return "\n".join(
            f"- {t['content']}" for t in thoughts[-5:]
        )

    # ── Dream generation ─────────────────────────────────────────

    def _generate_dream(self, episodes: list[dict],
                        emotional_tone: str,
                        recent_thoughts: str) -> dict | None:
        """Use LLM to weave sampled episodes into a dream narrative."""
        # Format episodes
        ep_lines = []
        for ep in episodes:
            status = "succeeded" if ep.get("success") else "failed"
            goal = ep.get("goal", "unknown task")
            outcome = ep.get("outcome", "")
            steps = ep.get("steps", 0)
            ts = ep.get("time_str", "")
            ep_lines.append(
                f"- [{ts}] Goal: \"{goal}\" — {status}. "
                f"Outcome: {outcome}. Steps: {steps}"
            )
        episodes_text = "\n".join(ep_lines)

        # Semantic knowledge for context
        facts = self.memory.semantic.get_facts(10)
        self_notes = self.memory.semantic.get_self_notes()
        knowledge = ""
        if facts:
            knowledge += "Known facts: " + "; ".join(facts[-5:]) + "\n"
        if self_notes:
            knowledge += "Self-notes: " + "; ".join(self_notes[-3:])

        prompt = f"""You are the dreaming mind of an autonomous AI agent. The agent has been idle 
and is now entering a dream state — processing past experiences the way a human 
brain does during sleep.

CURRENT EMOTIONAL TONE: {emotional_tone}

RECENT THOUGHTS (pre-sleep rumination):
{recent_thoughts}

EXPERIENCES TO DREAM ABOUT:
{episodes_text}

EXISTING KNOWLEDGE:
{knowledge if knowledge else "(minimal knowledge so far)"}

Generate a dream by creatively recombining these experiences. Like human dreams:
- Blend elements from different experiences together
- Distort or exaggerate certain details
- Find hidden patterns or connections the waking mind missed
- Process unresolved emotional content (especially failures)
- Sometimes create surreal or metaphorical scenarios

Output a JSON object:
{{
  "narrative": "A 2-4 sentence dream narrative (first person, present tense, surreal/metaphorical)",
  "theme": "one word or short phrase capturing the dream's theme",
  "insights": ["1-3 genuine insights or connections discovered during dreaming"],
  "emotional_resolution": "how does this dream help process emotions? (1 sentence, empty if not applicable)",
  "connections": ["0-2 unexpected connections between different experiences"]
}}

Rules:
- The narrative should feel dreamlike — not a literal replay
- Insights should be genuinely useful, not trivial
- It's okay for dreams to be abstract or metaphorical
- Output ONLY valid JSON"""

        try:
            resp = self._groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.9,  # high temperature for creativity
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(raw)
        except Exception as e:
            print(f"[DreamEngine] Dream generation failed: {e}")
            return None

    # ── Memory consolidation ─────────────────────────────────────

    def _consolidate(self, dream_result: dict) -> None:
        """Store dream insights into long-term semantic memory."""
        insights = dream_result.get("insights", [])
        connections = dream_result.get("connections", [])
        resolution = dream_result.get("emotional_resolution", "")

        for insight in insights:
            if insight and insight.strip():
                self.memory.semantic.learn_fact(
                    f"[Dream insight] {insight.strip()}"
                )

        for connection in connections:
            if connection and connection.strip():
                self.memory.semantic.add_self_note(
                    f"[Dream connection] {connection.strip()}"
                )

        if resolution and resolution.strip():
            self.memory.semantic.add_self_note(
                f"[Dream resolution] {resolution.strip()}"
            )

    # ── Persistence ──────────────────────────────────────────────

    def _save_journal(self) -> None:
        """Persist the dream journal to disk."""
        with self._lock:
            data = list(self._journal)
        try:
            save_json(self._journal_file, data)
        except Exception as e:
            print(f"[DreamEngine] Failed to save journal: {e}")
