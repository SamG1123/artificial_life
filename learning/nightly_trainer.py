"""
Nightly Trainer — self-improvement pipeline that runs during sleep.

While the agent sleeps, this system reviews recent experiences and
uses the LLM to extract actionable insights, update the knowledge
graph, and optionally retrain classifiers when enough data
accumulates.

Pipeline stages:

    1. Harvest  — collect experience entries from recent log files
    2. Analyze  — LLM reviews batches, extracts patterns + lessons
    3. Absorb   — store insights as facts, knowledge-graph relations,
                  and self-improvement strategies
    4. Retrain  — trigger model retraining if data threshold met

Runs incrementally: tracks a watermark so it never re-processes
old entries.
"""

import os
import json
import time
from threading import Lock

from groq import Groq
from dotenv import load_dotenv

from memory.compressed_storage import save_json, load_json, migrate_if_needed

load_dotenv()

# How many experience entries to analyze per sleep session
BATCH_SIZE = 30
# Minimum entries since last training to run a new cycle
MIN_NEW_ENTRIES = 8


class NightlyTrainer:
    """Runs during sleep to learn from logged experiences."""

    def __init__(self, memory, exp_logger, self_improver=None,
                 store_dir: str = "memory_store"):
        self.memory = memory
        self.exp_logger = exp_logger
        self.self_improver = self_improver

        self._store_dir = store_dir
        self._state_file = os.path.join(store_dir, "nightly_trainer_state.json")
        self._lock = Lock()

        migrate_if_needed(self._state_file)
        self._state = load_json(self._state_file, default={
            "last_run": 0,
            "watermark_ts": 0,
            "total_sessions": 0,
            "total_insights": 0,
            "total_kg_relations": 0,
        })

        self._groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ── Main entry point (called from brain during sleep) ────────

    def run_training_session(self) -> dict:
        """Run one nightly training session.

        Returns a report dict summarizing what was learned.
        """
        report = {
            "timestamp": time.time(),
            "entries_processed": 0,
            "insights": [],
            "kg_relations": 0,
            "strategies": 0,
            "retrained": False,
        }

        # 1. Harvest recent entries
        entries = self._harvest_entries()
        if len(entries) < MIN_NEW_ENTRIES:
            return report

        report["entries_processed"] = len(entries)

        # 2. Analyze in batches
        all_insights = []
        for i in range(0, len(entries), BATCH_SIZE):
            batch = entries[i:i + BATCH_SIZE]
            insights = self._analyze_batch(batch)
            all_insights.extend(insights)

        report["insights"] = [ins.get("lesson", "") for ins in all_insights if ins.get("lesson")]

        # 3. Absorb insights
        kg_count = 0
        strat_count = 0
        for insight in all_insights:
            # Store lessons as facts
            lesson = insight.get("lesson", "")
            if lesson:
                self.memory.learn_fact(f"[Training] {lesson}")

            # Add knowledge graph relations
            for triple in insight.get("knowledge", []):
                src = triple.get("subject", "")
                rel = triple.get("relation", "")
                obj = triple.get("object", "")
                if src and rel and obj:
                    self.memory.add_knowledge(src, rel, obj, confidence=0.7)
                    kg_count += 1

            # Store strategies
            strategy = insight.get("strategy", "")
            if strategy and self.self_improver:
                with self.self_improver._lock:
                    strats = self.self_improver._strategies.get(
                        "action_strategies", [])
                    if strategy not in strats:
                        strats.append(strategy)
                        if len(strats) > 30:
                            self.self_improver._strategies["action_strategies"] = strats[-30:]
                        self.self_improver._save_strategies()
                        strat_count += 1

        report["kg_relations"] = kg_count
        report["strategies"] = strat_count

        # 4. Optional retraining via SelfImprover
        if self.self_improver and self.self_improver._should_retrain():
            try:
                retrained = self.self_improver._trigger_retrain()
                report["retrained"] = retrained
            except Exception as e:
                print(f"[NightlyTrainer] Retrain failed: {e}")

        # Update watermark
        if entries:
            max_ts = max(e.get("ts", 0) for e in entries)
            with self._lock:
                self._state["watermark_ts"] = max_ts
                self._state["last_run"] = time.time()
                self._state["total_sessions"] += 1
                self._state["total_insights"] += len(report["insights"])
                self._state["total_kg_relations"] += kg_count
                self._save()

        return report

    # ── Harvest ──────────────────────────────────────────────────

    def _harvest_entries(self) -> list[dict]:
        """Collect experience entries newer than the watermark."""
        watermark = self._state.get("watermark_ts", 0)
        entries = []
        for entry in self.exp_logger.iter_entries(days=3):
            if entry.get("ts", 0) > watermark:
                entries.append(entry)
        # Sort by timestamp, take most recent BATCH_SIZE * 2
        entries.sort(key=lambda e: e.get("ts", 0))
        return entries[-(BATCH_SIZE * 2):]

    # ── Analyze ──────────────────────────────────────────────────

    def _analyze_batch(self, batch: list[dict]) -> list[dict]:
        """Use LLM to extract insights from a batch of experiences."""
        # Prepare summary of the batch
        summaries = []
        for e in batch:
            etype = e.get("type", "step")
            if etype == "episode_summary":
                summaries.append(
                    f"- Goal: \"{e.get('goal','')}\" | "
                    f"Outcome: {e.get('outcome','')} | "
                    f"Steps: {e.get('total_steps',0)} | "
                    f"Success: {e.get('success', False)}"
                )
            else:
                action = e.get("action", {})
                action_str = action.get("action", "unknown") if isinstance(action, dict) else str(action)
                summaries.append(
                    f"- Goal: \"{e.get('goal','')}\" step {e.get('step',0)} | "
                    f"Action: {action_str} | "
                    f"Result: {str(e.get('result',''))[:80]} | "
                    f"Success: {e.get('success', False)}"
                )

        batch_text = "\n".join(summaries[-BATCH_SIZE:])

        prompt = f"""You are reviewing an AI agent's recent experiences to extract lessons for self-improvement.

RECENT EXPERIENCES:
{batch_text}

Analyze these experiences and extract actionable insights. Output a JSON array of insight objects:
[
  {{
    "lesson": "one clear sentence about what was learned (empty if nothing notable)",
    "strategy": "a concrete strategy for doing better next time (empty if nothing to suggest)",
    "knowledge": [
      {{"subject": "entity1", "relation": "relationship_type", "object": "entity2"}}
    ]
  }}
]

Rules:
- Focus on patterns: repeated failures, successful strategies, new capabilities discovered
- Knowledge triples should capture factual relationships learned (e.g. {{"subject": "Chrome", "relation": "is_a", "object": "web browser"}})
- Keep lessons concise and actionable
- Return 1-5 insights maximum. Return [] if nothing notable.
- Output ONLY valid JSON array."""

        try:
            resp = self._groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Output only valid JSON array."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
            if isinstance(result, list):
                return result
            return []
        except Exception as e:
            print(f"[NightlyTrainer] Batch analysis failed: {e}")
            return []

    # ── Context ──────────────────────────────────────────────────

    def get_context_for_llm(self) -> str:
        """Brief summary for brain context."""
        s = self._state
        if s["total_sessions"] == 0:
            return ""
        last_run = time.strftime("%Y-%m-%d %H:%M",
                                 time.localtime(s["last_run"])) if s["last_run"] else "never"
        return (
            f"Nightly training: {s['total_sessions']} sessions, "
            f"{s['total_insights']} insights, "
            f"{s['total_kg_relations']} knowledge relations. "
            f"Last run: {last_run}"
        )

    # ── Persistence ──────────────────────────────────────────────

    def _save(self):
        save_json(self._state_file, self._state)
