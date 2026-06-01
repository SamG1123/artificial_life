"""
Self-Improvement Engine — the agent's ability to analyze its own
performance and actively improve over time.

Improvement loop (runs periodically or on-demand):

    ┌──────────────────────────────────────────────────┐
    │              SELF-IMPROVEMENT CYCLE               │
    │                                                    │
    │  1. Analyze  ──▶  Review recent performance        │
    │  2. Diagnose ──▶  Identify weaknesses / patterns   │
    │  3. Strategize ▶  Generate improvement plans       │
    │  4. Adapt    ──▶  Update strategies + retrain       │
    │  5. Reflect  ──▶  Evaluate if improvements helped   │
    │                                                    │
    └──────────────────────────────────────────────────┘

Capabilities:
  * Performance analysis (success rate trends, failure clustering)
  * Strategy generation (LLM-based reasoning about how to improve)
  * Prompt self-tuning (store learned prompting strategies)
  * Automatic model retraining when enough data accumulates
  * Meta-cognition journal (the agent tracks its own growth)
"""

import os
import json
import time
import csv
from threading import Lock
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

from memory.compressed_storage import save_json, load_json, migrate_if_needed

load_dotenv()


class SelfImprover:
    """Analyzes performance and generates self-improvement strategies."""

    def __init__(self, memory, learning_logger, dataset_builder,
                 model_trainer, store_dir: str = "memory_store"):
        """
        Args:
            memory: MemorySystem instance
            learning_logger: ExperienceLogger instance
            dataset_builder: DatasetBuilder instance
            model_trainer: ModelTrainer instance
            store_dir: persistence directory
        """
        self.memory = memory
        self.logger = learning_logger
        self.dataset_builder = dataset_builder
        self.trainer = model_trainer

        self._store_dir = store_dir
        self._journal_file = os.path.join(store_dir, "improvement_journal.json")
        self._strategies_file = os.path.join(store_dir, "strategies.json")
        self._lock = Lock()

        self._journal: list[dict] = self._load(self._journal_file, [])
        self._strategies: dict = self._load(self._strategies_file, {
            "action_strategies": [],
            "conversation_strategies": [],
            "planning_strategies": [],
            "known_weaknesses": [],
            "known_strengths": [],
            "improvement_goals": [],
            "strategy_trials": {},
        })

        self._groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Thresholds
        self.MIN_EPISODES_FOR_ANALYSIS = 10
        self.RETRAIN_THRESHOLD = 50  # episodes since last train
        self._last_train_episode_count = self._strategies.get(
            "_last_train_count", 0)

    # ── Main improvement cycle ───────────────────────────────────

    def run_improvement_cycle(self) -> dict:
        """Execute a full self-improvement cycle.

        Returns a report dict with analysis results and actions taken.
        """
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
            "analysis": None,
            "strategies_generated": 0,
            "retrained": False,
            "correction_retrained": False,
        }

        # 1. Performance analysis
        analysis = self._analyze_performance()
        report["analysis"] = analysis

        if analysis is None:
            return report

        # 2. Generate improvement strategies
        strategies = self._generate_strategies(analysis)
        report["strategies_generated"] = len(strategies)

        # 3. Check if retraining is warranted
        if self._should_retrain():
            retrained = self._trigger_retrain()
            report["retrained"] = retrained
            report["correction_retrained"] = self._trigger_correction_retrain()

        # 4. Journal the cycle
        self._prune_low_confidence_strategies()
        self._journal_entry(report)

        return report

    # ── Performance analysis ─────────────────────────────────────

    def _analyze_performance(self) -> dict | None:
        """Gather and analyze performance metrics."""
        total = self.memory.episodic.total_episodes
        if total < self.MIN_EPISODES_FOR_ANALYSIS:
            return None

        recent = self.memory.episodic.get_recent(20)
        older = self.memory.episodic.get_recent(50)

        # Compute metrics
        recent_success = sum(1 for e in recent if e["success"]) / max(len(recent), 1)
        overall_success = sum(1 for e in older if e["success"]) / max(len(older), 1)
        avg_steps = sum(e["steps"] for e in recent) / max(len(recent), 1)

        # Cluster failures
        failures = [e for e in recent if not e["success"]]
        failure_goals = [e["goal"] for e in failures]

        # Trend: is performance improving?
        if len(older) >= 20:
            first_half = older[:len(older)//2]
            second_half = older[len(older)//2:]
            first_rate = sum(1 for e in first_half if e["success"]) / max(len(first_half), 1)
            second_rate = sum(1 for e in second_half if e["success"]) / max(len(second_half), 1)
            trend = "improving" if second_rate > first_rate + 0.05 else (
                "declining" if second_rate < first_rate - 0.05 else "stable"
            )
        else:
            trend = "insufficient_data"

        analysis = {
            "total_episodes": total,
            "recent_success_rate": round(recent_success, 2),
            "overall_success_rate": round(overall_success, 2),
            "avg_steps_recent": round(avg_steps, 1),
            "trend": trend,
            "recent_failures": failure_goals[:5],
            "failure_count": len(failures),
        }

        print(f"[SelfImprover] Performance: {recent_success:.0%} recent, "
              f"{overall_success:.0%} overall, trend: {trend}")

        return analysis

    # ── Strategy generation ──────────────────────────────────────

    def _generate_strategies(self, analysis: dict) -> list[str]:
        """Use LLM to generate improvement strategies based on analysis."""
        # Get current strategies and self-notes for context
        current_strats = self._strategies.get("action_strategies", [])[-5:]
        self_notes = self.memory.semantic.get_self_notes()[-5:]
        known_weaknesses = self._strategies.get("known_weaknesses", [])[-5:]

        prompt = f"""You are an AI agent analyzing your own performance to improve.

PERFORMANCE ANALYSIS:
- Total tasks completed: {analysis['total_episodes']}
- Recent success rate: {analysis['recent_success_rate']:.0%}
- Overall success rate: {analysis['overall_success_rate']:.0%}
- Performance trend: {analysis['trend']}
- Average steps per task: {analysis['avg_steps_recent']}
- Recent failures: {analysis['recent_failures']}

CURRENT STRATEGIES:
{json.dumps(current_strats, indent=2) if current_strats else "(none yet)"}

KNOWN WEAKNESSES:
{json.dumps(known_weaknesses, indent=2) if known_weaknesses else "(none identified)"}

SELF-NOTES:
{json.dumps(self_notes, indent=2) if self_notes else "(none)"}

Based on this analysis, generate improvements. Output a JSON object:
{{
  "new_strategies": ["specific actionable strategies to try (0-3 items)"],
  "weaknesses_identified": ["specific weaknesses found (0-2 items)"],
  "strengths_identified": ["things working well (0-2 items)"],
  "improvement_goals": ["concrete goals to work toward (0-2 items)"],
  "self_reflection": "1-2 sentence reflection on overall progress"
}}

Rules:
- Be specific and actionable. "Try harder" is bad. "Break multi-step browser tasks into smaller sub-goals" is good.
- Don't repeat existing strategies
- If performance is good, focus on strengths and subtle improvements
- Output ONLY valid JSON"""

        try:
            resp = self._groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400, temperature=0.5,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
        except Exception as e:
            print(f"[SelfImprover] Strategy generation failed: {e}")
            return []

        # Store new strategies
        all_new = []
        with self._lock:
            for s in result.get("new_strategies", []):
                if s.strip() and s.strip() not in self._strategies["action_strategies"]:
                    self._strategies["action_strategies"].append(s.strip())
                    all_new.append(s.strip())

            for w in result.get("weaknesses_identified", []):
                if w.strip() and w.strip() not in self._strategies["known_weaknesses"]:
                    self._strategies["known_weaknesses"].append(w.strip())

            for st in result.get("strengths_identified", []):
                if st.strip() and st.strip() not in self._strategies["known_strengths"]:
                    self._strategies["known_strengths"].append(st.strip())

            for g in result.get("improvement_goals", []):
                if g.strip():
                    self._strategies["improvement_goals"].append(g.strip())
                    # Keep only recent goals
                    self._strategies["improvement_goals"] = \
                        self._strategies["improvement_goals"][-10:]

            # Cap lists
            for key in ("action_strategies", "known_weaknesses", "known_strengths"):
                self._strategies[key] = self._strategies[key][-20:]

        self._save_strategies()

        # Store reflection as a self-note
        reflection = result.get("self_reflection", "")
        if reflection:
            self.memory.semantic.add_self_note(
                f"[Self-improvement] {reflection}"
            )
            print(f"[SelfImprover] Reflection: {reflection}")

        if all_new:
            print(f"[SelfImprover] New strategies: {all_new}")

        return all_new

    # ── Retraining ───────────────────────────────────────────────

    def _should_retrain(self) -> bool:
        """Check if we have enough new experience to warrant retraining."""
        current = self.memory.episodic.total_episodes
        since_last = current - self._last_train_episode_count
        return since_last >= self.RETRAIN_THRESHOLD

    def _trigger_retrain(self) -> bool:
        """Build dataset and retrain the action classifier."""
        try:
            path = self.dataset_builder.build("action_prediction")
            if path is None:
                return False

            meta = self.trainer.train_action_classifier(
                str(path), run_name=f"self_improve_{time.strftime('%Y%m%d')}",
                epochs=5,
            )

            # Update tracking
            self._last_train_episode_count = self.memory.episodic.total_episodes
            with self._lock:
                self._strategies["_last_train_count"] = self._last_train_episode_count
            self._save_strategies()

            accuracy = meta.get("eval_accuracy", 0)
            self.memory.semantic.add_self_note(
                f"[Self-improvement] Retrained action model — accuracy: {accuracy:.1%}"
            )
            print(f"[SelfImprover] Model retrained — accuracy: {accuracy:.1%}")
            return True

        except Exception as e:
            print(f"[SelfImprover] Retraining failed: {e}")
            return False

    def _trigger_correction_retrain(self) -> bool:
        """Optional correction-learning training when enough correction examples exist."""
        try:
            path = self.dataset_builder.build("correction")
            if path is None:
                return False

            rows = 0
            with open(path, "r", encoding="utf-8", newline="") as f:
                rows = max(0, sum(1 for _ in csv.reader(f)) - 1)
            if rows < 12:
                return False

            meta = self.trainer.train_correction_classifier(
                str(path),
                run_name=f"correction_{time.strftime('%Y%m%d')}",
                epochs=4,
            )
            accuracy = meta.get("eval_accuracy", 0)
            self.memory.semantic.add_self_note(
                f"[Self-improvement] Trained correction model — accuracy: {accuracy:.1%}"
            )
            return True
        except Exception as e:
            print(f"[SelfImprover] Correction retraining failed: {e}")
            return False

    # ── Strategy access (for brain/reasoning to use) ─────────────

    def get_strategies(self, category: str = "action") -> list[str]:
        """Get current strategies for a category."""
        key = f"{category}_strategies"
        with self._lock:
            return list(self._strategies.get(key, []))

    def get_weaknesses(self) -> list[str]:
        with self._lock:
            return list(self._strategies.get("known_weaknesses", []))

    def get_strengths(self) -> list[str]:
        with self._lock:
            return list(self._strategies.get("known_strengths", []))

    def get_improvement_goals(self) -> list[str]:
        with self._lock:
            return list(self._strategies.get("improvement_goals", []))

    def get_context_for_llm(self) -> str:
        """Summary of strategies for injecting into LLM prompts."""
        parts = []
        strats = self.get_strategies("action")
        scored = sorted(
            [(s, self.get_strategy_confidence(s)) for s in strats],
            key=lambda x: x[1],
            reverse=True,
        )
        trusted = [s for s, c in scored if c >= 0.45]
        if strats:
            parts.append("Learned strategies: " + "; ".join(trusted[-3:] if trusted else strats[-2:]))
        weak = self.get_weaknesses()
        if weak:
            parts.append("Known weaknesses: " + "; ".join(weak[-2:]))
        goals = self.get_improvement_goals()
        if goals:
            parts.append("Improvement goals: " + "; ".join(goals[-2:]))
        return "\n".join(parts) if parts else ""

    # ── Strategy validation loop ───────────────────────────────

    def record_strategy_outcome(self, goal: str, success: bool,
                                reward_total: float = 0.0) -> None:
        """Track whether strategies are helping on real episodes."""
        goal_l = (goal or "").lower()
        with self._lock:
            trials = self._strategies.setdefault("strategy_trials", {})
            for s in self._strategies.get("action_strategies", []):
                sl = s.lower()
                # Lightweight matching: if strategy keywords appear in goal, count trial.
                if any(tok for tok in sl.split() if len(tok) > 4 and tok in goal_l):
                    rec = trials.setdefault(s, {
                        "attempts": 0,
                        "successes": 0,
                        "reward_sum": 0.0,
                        "last_used": 0.0,
                    })
                    rec["attempts"] += 1
                    if success:
                        rec["successes"] += 1
                    rec["reward_sum"] += float(reward_total)
                    rec["last_used"] = time.time()

            # Keep table bounded for long runs
            if len(trials) > 100:
                ordered = sorted(
                    trials.items(),
                    key=lambda kv: kv[1].get("last_used", 0.0),
                    reverse=True,
                )[:100]
                self._strategies["strategy_trials"] = dict(ordered)
        self._save_strategies()

    def get_strategy_confidence(self, strategy: str) -> float:
        """Confidence in [0,1] based on success + reward trend."""
        with self._lock:
            rec = self._strategies.get("strategy_trials", {}).get(strategy, {})
        return self._confidence_from_record(rec)

    def _prune_low_confidence_strategies(self) -> None:
        """Demote strategies that consistently underperform."""
        with self._lock:
            action_strategies = list(self._strategies.get("action_strategies", []))
            trials = self._strategies.get("strategy_trials", {})
            kept = []
            demoted = []
            for s in action_strategies:
                rec = trials.get(s, {})
                attempts = int(rec.get("attempts", 0))
                conf = self._confidence_from_record(rec)
                if attempts >= 5 and conf < 0.25:
                    demoted.append(s)
                else:
                    kept.append(s)
            self._strategies["action_strategies"] = kept[-20:]
        if demoted:
            self.memory.semantic.add_self_note(
                "[Self-improvement] Demoted low-confidence strategies: " +
                "; ".join(demoted[:4])
            )
            self._save_strategies()

    @staticmethod
    def _confidence_from_record(rec: dict) -> float:
        attempts = int(rec.get("attempts", 0))
        if attempts <= 0:
            return 0.5
        successes = int(rec.get("successes", 0))
        reward_sum = float(rec.get("reward_sum", 0.0))
        success_rate = successes / max(1, attempts)
        avg_reward = reward_sum / max(1, attempts)
        reward_norm = max(0.0, min(1.0, (avg_reward + 0.5)))
        coverage = min(1.0, attempts / 8.0)
        confidence = (0.55 * success_rate) + (0.30 * reward_norm) + (0.15 * coverage)
        return round(max(0.0, min(1.0, confidence)), 3)

    # ── Journal ──────────────────────────────────────────────────

    def _journal_entry(self, report: dict) -> None:
        """Record an improvement cycle in the journal."""
        with self._lock:
            self._journal.append(report)
            # Keep last 50 entries
            self._journal = self._journal[-50:]
        self._save_journal()

    def get_journal(self, n: int = 10) -> list[dict]:
        """Return recent improvement journal entries."""
        with self._lock:
            return list(self._journal[-n:])

    # ── Persistence ──────────────────────────────────────────────

    def _save_strategies(self):
        with self._lock:
            data = dict(self._strategies)
        try:
            save_json(self._strategies_file, data)
        except Exception as e:
            print(f"[SelfImprover] Failed to save strategies: {e}")

    def _save_journal(self):
        with self._lock:
            data = list(self._journal)
        try:
            save_json(self._journal_file, data)
        except Exception as e:
            print(f"[SelfImprover] Failed to save journal: {e}")

    @staticmethod
    def _load(path: str, default):
        migrate_if_needed(path)
        result = load_json(path)
        return result if result is not None else default
