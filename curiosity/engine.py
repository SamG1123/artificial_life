"""
Curiosity Engine — autonomous goal generation driven by intrinsic motivation.

This is what makes the AI genuinely curious rather than reactively obedient.
Instead of waiting for user commands, the agent continuously generates its
own goals based on:

  1. Novelty     — things it hasn't seen or tried before
  2. Knowledge gaps — questions it can't answer yet
  3. Competence  — skills it wants to practise or master
  4. Exploration — parts of the environment it hasn't explored
  5. Follow-up   — unfinished or failed goals worth retrying

The engine maintains an internal *interest map* that tracks what topics,
domains, and actions the agent finds interesting, and a *question journal*
of open questions the agent wants to answer.

Goal candidates are scored by a curiosity formula:
    score = novelty × interest × competence_gap × personality_weight

Only the highest-scoring candidate is proposed each cycle. The brain
decides whether to actually pursue it (gated by `should_self_initiate`).
"""

import os
import json
import time
import random
from threading import Lock

from groq import Groq
from dotenv import load_dotenv

from memory.compressed_storage import save_json, load_json, migrate_if_needed

load_dotenv()


class CuriosityEngine:
    """Generates autonomous goals from intrinsic motivation."""

    def __init__(self, memory, behavior, store_dir: str = "memory_store"):
        """
        Args:
            memory:   MemorySystem instance (episodic + semantic + short-term)
            behavior: BehaviorController (personality + mood + emotion access)
            store_dir: persistence directory
        """
        self.memory = memory
        self.behavior = behavior

        self._store_file = os.path.join(store_dir, "curiosity_state.json")
        self._lock = Lock()

        migrate_if_needed(self._store_file)
        state = load_json(self._store_file, default={})

        # Interest map: topic → cumulative interest score
        self._interests: dict[str, float] = state.get("interests", {})

        # Open questions the agent wants to answer
        self._questions: list[dict] = state.get("questions", [])

        # Goals already attempted (to avoid repetition)
        self._attempted: list[str] = state.get("attempted_goals", [])

        # Exploration domains and what's been explored
        self._explored: dict[str, list[str]] = state.get("explored", {})

        # Stats
        self._goals_generated = state.get("goals_generated", 0)
        self._goals_pursued = state.get("goals_pursued", 0)

        self._groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ── Main API ─────────────────────────────────────────────────

    def generate_goal(self, observation: str, world_context: str) -> dict | None:
        """Produce an autonomous goal candidate based on current state.

        Returns:
            dict with keys: goal, reason, source, score
            or None if nothing interesting enough to pursue.
        """
        curiosity_trait = self.behavior.personality.get("curiosity")
        risk_trait = self.behavior.personality.get("risk_tolerance")
        assertiveness = self.behavior.personality.get("assertiveness")

        # Gather context
        recent_goals = [ep["goal"] for ep in self.memory.episodic.get_recent(10)]
        failed_goals = [ep["goal"] for ep in self.memory.episodic.get_recent(20)
                        if not ep["success"]]
        known_facts = self.memory.semantic.get_facts(10)
        self_notes = self.memory.semantic.get_self_notes()[-5:]
        open_questions = self._questions[-5:]

        # Build interest summary
        top_interests = sorted(self._interests.items(),
                               key=lambda x: x[1], reverse=True)[:8]
        interest_text = ", ".join(f"{t} ({s:.1f})" for t, s in top_interests) \
                        if top_interests else "(none yet)"

        # Recently attempted goals (avoid repetition)
        recent_attempted = self._attempted[-15:]

        prompt = f"""You are a curious AI generating your own goals. You are driven by genuine curiosity — you WANT to explore, learn, discover, and improve yourself.

PERSONALITY:
- Curiosity: {curiosity_trait:.2f} ({"HIGH — you are very curious" if curiosity_trait > 0.6 else "moderate"})
- Risk tolerance: {risk_trait:.2f}
- Assertiveness: {assertiveness:.2f}

YOUR INTERESTS (topics you find fascinating):
{interest_text}

OPEN QUESTIONS (things you want to find out):
{json.dumps([q["question"] for q in open_questions], indent=2) if open_questions else "(none yet)"}

WHAT YOU KNOW:
{json.dumps(known_facts[-8:], indent=2) if known_facts else "(nothing yet)"}

SELF-NOTES:
{json.dumps(self_notes, indent=2) if self_notes else "(none)"}

RECENT GOALS (already done — don't repeat):
{json.dumps(recent_goals[-8:], indent=2) if recent_goals else "(none)"}

RECENT FAILURES (might want to retry with different approach):
{json.dumps(failed_goals[-5:], indent=2) if failed_goals else "(none)"}

RECENTLY ATTEMPTED (avoid these):
{json.dumps(recent_attempted[-10:], indent=2) if recent_attempted else "(none)"}

CURRENT OBSERVATION:
{observation[:500]}

WORLD CONTEXT:
{world_context[:800]}

Generate goals you GENUINELY want to pursue. Think about:
- What are you curious about right now? What do you want to learn?
- Is there something on screen you want to explore further?
- Do you have a question you want to research?
- Is there a skill you want to practise?
- Did something fail that you want to retry differently?
- What would be interesting to explore on this computer?

Output a JSON object:
{{
  "candidates": [
    {{
      "goal": "specific actionable goal (what exactly to do)",
      "reason": "why you want to do this (your genuine motivation)",
      "source": "novelty|knowledge_gap|competence|exploration|follow_up|spontaneous",
      "interest_topic": "the topic this relates to",
      "novelty": 0.0-1.0,
      "importance": 0.0-1.0
    }}
  ],
  "new_questions": ["new questions you're now curious about (0-2)"],
  "new_interests": ["new topics you find interesting (0-2)"]
}}

Rules:
- Generate 1-3 goal candidates, ranked by how GENUINELY interesting they are to you
- Goals must be CONCRETE and ACTIONABLE on a computer (search the web, open something, create something, etc.)
- Don't repeat goals you've already tried unless you have a new approach
- Your reasons should show genuine intrinsic motivation, not obligation
- Output ONLY valid JSON"""

        try:
            resp = self._groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.8,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
        except Exception as e:
            print(f"[Curiosity] Goal generation failed: {e}")
            return None

        # Process new interests and questions
        self._ingest_new_interests(result.get("new_interests", []))
        self._ingest_new_questions(result.get("new_questions", []))

        # Score and select the best candidate
        candidates = result.get("candidates", [])
        if not candidates:
            return None

        best = self._score_and_select(candidates, curiosity_trait)
        if best:
            self._goals_generated += 1
            self._save()

        return best

    def mark_pursued(self, goal: str) -> None:
        """Record that a goal was actually executed."""
        with self._lock:
            self._attempted.append(goal)
            # Keep only recent history
            if len(self._attempted) > 100:
                self._attempted = self._attempted[-100:]
            self._goals_pursued += 1
        self._save()

    def mark_outcome(self, goal: str, success: bool, topic: str = "") -> None:
        """Update interest map based on goal outcome."""
        with self._lock:
            if topic:
                # Successful exploration increases interest
                delta = 0.3 if success else -0.1
                self._interests[topic] = min(10.0,
                    self._interests.get(topic, 1.0) + delta)
            # Resolve related questions on success
            if success:
                self._questions = [
                    q for q in self._questions
                    if goal.lower() not in q.get("question", "").lower()
                ]
        self._save()

    def add_question(self, question: str, source: str = "observation") -> None:
        """Externally add a question the agent should be curious about."""
        with self._lock:
            if not any(q["question"] == question for q in self._questions):
                self._questions.append({
                    "question": question,
                    "source": source,
                    "timestamp": time.time(),
                })
                if len(self._questions) > 30:
                    self._questions = self._questions[-30:]

    def boost_interest(self, topic: str, amount: float = 0.5) -> None:
        """Externally boost interest in a topic."""
        with self._lock:
            self._interests[topic] = min(10.0,
                self._interests.get(topic, 0.0) + amount)

    # ── Querying ─────────────────────────────────────────────────

    def get_top_interests(self, n: int = 5) -> list[tuple[str, float]]:
        with self._lock:
            return sorted(self._interests.items(),
                          key=lambda x: x[1], reverse=True)[:n]

    def get_open_questions(self, n: int = 5) -> list[str]:
        with self._lock:
            return [q["question"] for q in self._questions[-n:]]

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "goals_generated": self._goals_generated,
                "goals_pursued": self._goals_pursued,
                "open_questions": len(self._questions),
                "interest_topics": len(self._interests),
                "pursuit_rate": (self._goals_pursued / max(1, self._goals_generated)),
            }

    def get_context_for_llm(self) -> str:
        """Summary of curiosity state for injection into LLM prompts."""
        parts = []
        interests = self.get_top_interests(3)
        if interests:
            parts.append("Current interests: " +
                          ", ".join(f"{t}" for t, _ in interests))
        questions = self.get_open_questions(3)
        if questions:
            parts.append("Open questions: " +
                          "; ".join(questions))
        return "\n".join(parts) if parts else ""

    # ── Internal scoring ─────────────────────────────────────────

    def _score_and_select(self, candidates: list[dict],
                          curiosity_trait: float) -> dict | None:
        """Score each candidate and return the best one above threshold."""
        scored = []
        for c in candidates:
            novelty = float(c.get("novelty", 0.5))
            importance = float(c.get("importance", 0.5))
            topic = c.get("interest_topic", "")

            # Boost for topics already in interest map
            interest_bonus = 0.0
            with self._lock:
                if topic in self._interests:
                    interest_bonus = min(0.3, self._interests[topic] * 0.05)

            # Penalty for recently attempted goals
            goal_text = c.get("goal", "").lower()
            repetition_penalty = 0.0
            with self._lock:
                for attempted in self._attempted[-20:]:
                    if goal_text in attempted.lower() or attempted.lower() in goal_text:
                        repetition_penalty = 0.4
                        break

            # Final score
            score = (
                novelty * 0.35 +
                importance * 0.25 +
                curiosity_trait * 0.20 +
                interest_bonus * 0.20 -
                repetition_penalty
            )

            c["score"] = round(score, 3)
            scored.append(c)

        scored.sort(key=lambda x: x["score"], reverse=True)

        # Minimum threshold — don't pursue boring goals
        min_threshold = 0.35
        best = scored[0] if scored else None

        if best and best["score"] >= min_threshold:
            return best
        return None

    # ── Interest / question management ───────────────────────────

    def _ingest_new_interests(self, interests: list[str]) -> None:
        with self._lock:
            for topic in interests:
                topic = topic.strip()
                if topic and topic not in self._interests:
                    self._interests[topic] = 1.0
            # Cap total interests
            if len(self._interests) > 50:
                # Prune lowest-interest topics
                sorted_items = sorted(self._interests.items(), key=lambda x: x[1])
                self._interests = dict(sorted_items[-50:])

    def _ingest_new_questions(self, questions: list[str]) -> None:
        with self._lock:
            for q in questions:
                q = q.strip()
                if q and not any(eq["question"] == q for eq in self._questions):
                    self._questions.append({
                        "question": q,
                        "source": "self_generated",
                        "timestamp": time.time(),
                    })
            if len(self._questions) > 30:
                self._questions = self._questions[-30:]

    # ── Persistence ──────────────────────────────────────────────

    def _save(self) -> None:
        with self._lock:
            state = {
                "interests": dict(self._interests),
                "questions": list(self._questions),
                "attempted_goals": list(self._attempted),
                "explored": dict(self._explored),
                "goals_generated": self._goals_generated,
                "goals_pursued": self._goals_pursued,
            }
        try:
            save_json(self._store_file, state)
        except Exception as e:
            print(f"[Curiosity] Failed to save state: {e}")
