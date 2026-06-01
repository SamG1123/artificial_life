"""
AgentController — the top-level coordinator for the autonomous AI agent.

Owns every subsystem and wires them together into a coherent whole.
The data-flow through the agent on each cognitive tick looks like:

    ┌──────────────────────────────────────────────────────────────────┐
    │                       AGENT CONTROLLER                          │
    │                                                                  │
    │  ┌───────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐  │
    │  │ Perception│──▶│ World State│──▶│ Cognition│──▶│  Action  │  │
    │  │  (senses) │   │  (reality) │   │  (brain) │   │ (executor│  │
    │  └───────────┘   └─────┬──────┘   └────┬─────┘   └────┬─────┘  │
    │                        │               │               │        │
    │                   ┌────▼────┐     ┌────▼────┐    ┌────▼─────┐  │
    │                   │ Memory  │     │Reasoning│    │ Learning │  │
    │                   │ (recall)│     │(planning│    │(experience│  │
    │                   └─────────┘     └─────────┘    └──────────┘  │
    │                        │                                        │
    │                   ┌────▼────┐                                   │
    │                   │ Emotion │                                   │
    │                   │(feeling)│                                   │
    │                   └─────────┘                                   │
    └──────────────────────────────────────────────────────────────────┘

Lifecycle:
    controller = AgentController()   # builds all subsystems
    controller.start()               # launches threads + loops
    controller.wait()                # blocks until shutdown
    controller.stop()                # graceful teardown

Subsystem layers (initialised in order):
    1. IO          — eyes (vision), ears (speech), mouth (TTS)
    2. Perception  — screen, camera, audio, system sensors
    3. World State — unified reality model
    4. Memory      — short-term, episodic, semantic, vector
    5. Emotion     — personality → mood → emotions → behavior
    6. Reasoning   — planner + policies
    7. Learning    — experience logger, dataset builder, trainer
    8. Action      — automation executor (browser + desktop)
    9. Cognition   — CognitiveBrain (the autonomous mind)
"""

import os
import signal
import threading
import time
from datetime import datetime
from threading import Thread, Event

image_processing = None
voice_recognition = None
tts_module = None
web_support = None
AutomationExecutor = None
BackgroundTaskManager = None
IdleMonitor = None
PerceptionManager = None
from brain import CognitiveBrain
from memory import MemorySystem, MemoryCompressor
from world_state import WorldStateManager
from emotion import PersonalityModel, MoodEngine, EmotionEngine, BehaviorController
from reasoning import ReasoningEngine
from learning import ExperienceLogger, DatasetBuilder, ModelTrainer, SelfImprover, SkillGraph, NightlyTrainer, RewardEngine
from curiosity import CuriosityEngine
from cognition import (
    AttentionSystem,
    DreamEngine,
    DialogueStateTracker,
    NotificationEngine,
    PreferenceLearner,
)
from config import global_goal_queue, global_command_queue
from logging_config import get_logger
from checkpoint import StateCheckpoint
from health_monitor import HealthMonitor

log = get_logger("agent")


class _HeadlessObjectDetection:
    def camera_infer(self, stop_event):
        stop_event.wait()


class _HeadlessSpeechSupport:
    def __init__(self):
        self.sleep = True
        self.brain = None

    def listen(self, stop_event):
        stop_event.wait()


class _HeadlessTextToSpeech:
    def text_to_speech_loop(self, stop_event):
        from queue import Empty

        from config import global_command_queue

        while not stop_event.is_set():
            try:
                global_command_queue.get(timeout=0.5)
            except Empty:
                continue


class _HeadlessPerceptionManager:
    def initialise(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def get_world_state(self) -> dict:
        return {}

    def start_background_loop(self, *args, **kwargs):
        return None


class _HeadlessIdleMonitor:
    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def stats(self) -> dict:
        return {"enabled": False, "idle_seconds": 0, "is_idle_1hr": False}
    
    def idle_seconds(self) -> int:
        """Return number of seconds the user has been idle (stubbed for headless)."""
        return 0


class _HeadlessBackgroundTasks:
    def stats(self) -> dict:
        return {"active": 0, "completed": 0, "cancelled": 0}

    def cancel_all(self, timeout: float = 2.0) -> int:
        return 0


class _HeadlessExecutor:
    def __init__(self, experience_logger):
        self.goal = None
        self.goal_completed = False
        self.action_history: list[dict] = []
        self.exp_logger = experience_logger

    def execute_goal(self, goal: str) -> None:
        self.goal = goal
        self.goal_completed = True
        self.action_history = [
            {
                "action": {"action": "headless_demo"},
                "result": f"Planned in hosted demo mode: {goal}",
                "success": True,
            }
        ]


class AgentController:
    """Top-level controller that owns, wires, and manages every subsystem."""

    # ── Construction ──────────────────────────────────────────────

    def __init__(self, store_dir: str = "memory_store", headless: bool = False):
        self._store_dir = store_dir
        self.headless = headless
        self._stop_event = Event()
        self._threads: list[Thread] = []
        self._started = False

        # ── Layer 1 — IO (raw sensors & effectors) ────────────────
        if self.headless:
            self.eyes = _HeadlessObjectDetection()
            self.ears = _HeadlessSpeechSupport()
            self.mouth = _HeadlessTextToSpeech()
            self.web = None
        else:
            import image_processing as image_processing_mod
            import tts as tts_module_mod
            import voice_recognition as voice_recognition_mod
            import web_support as web_support_mod

            self.eyes = image_processing_mod.ObjectDetection()
            self.web = web_support_mod.WebSupport()
            self.ears = voice_recognition_mod.SpeechSupport()
            self.mouth = tts_module_mod.TextToSpeech()

        # ── Layer 2 — Perception ──────────────────────────────────
        self.world_state = WorldStateManager()

        # ── User Activity & Background Tasks (Perception layer) ──
        if self.headless:
            self.idle_monitor = _HeadlessIdleMonitor()
            self.background_tasks = _HeadlessBackgroundTasks()
            self.perception = _HeadlessPerceptionManager()
            self.perception.initialise()
        else:
            from automation.background_tasks import BackgroundTaskManager as BackgroundTaskManager_mod
            from perception import PerceptionManager as PerceptionManager_mod
            from perception.idle_monitor import IdleMonitor as IdleMonitor_mod

            self.idle_monitor = IdleMonitor_mod()
            self.background_tasks = BackgroundTaskManager_mod()

            self.perception = PerceptionManager_mod(
                vision_model=self.eyes,
                world_state_manager=self.world_state,
            )
            self.perception.initialise()

        # ── Layer 3 — Memory ─────────────────────────────────────
        self.memory = MemorySystem()

        # ── Layer 4 — Emotion ─────────────────────────────────────
        self.personality = PersonalityModel(store_dir=store_dir)
        self.mood_engine = MoodEngine(self.personality)
        self.emotion_engine = EmotionEngine(self.personality, self.mood_engine)
        self.behavior = BehaviorController(
            self.personality, self.mood_engine, self.emotion_engine,
        )

        # ── Layer 5 — Reasoning ──────────────────────────────────
        self.reasoning = ReasoningEngine(
            behavior=self.behavior, store_dir=store_dir,
        )

        # ── Layer 6 — Learning ───────────────────────────────────
        self.exp_logger = ExperienceLogger(store_dir=store_dir)
        self.dataset_builder = DatasetBuilder(self.exp_logger, store_dir=store_dir)
        self.model_trainer = ModelTrainer(store_dir=store_dir)

        # ── Layer 6b — Memory compression + Self-improvement ─────
        self.compressor = MemoryCompressor(self.memory, store_dir=store_dir)
        self.self_improver = SelfImprover(
            memory=self.memory,
            learning_logger=self.exp_logger,
            dataset_builder=self.dataset_builder,
            model_trainer=self.model_trainer,
            store_dir=store_dir,
        )

        # ── Layer 6d — Skill Graph (capability tracking) ─────────
        self.skill_graph = SkillGraph(store_dir=store_dir)

        # ── Layer 6c — Curiosity (autonomous goal generation) ────
        self.curiosity = CuriosityEngine(
            memory=self.memory,
            behavior=self.behavior,
            store_dir=store_dir,
            skill_graph=self.skill_graph,
        )

        # ── Layer 6j — Reward engine (learning signal architecture) ──
        self.reward_engine = RewardEngine(skill_graph=self.skill_graph)

        # ── Layer 6e — Attention (cognitive focus management) ─────
        self.attention = AttentionSystem(store_dir=store_dir)

        # ── Layer 6f — Dreaming (experience-based memory consolidation) ──
        self.dream_engine = DreamEngine(
            memory=self.memory,
            behavior=self.behavior,
            store_dir=store_dir,
        )

        # ── Layer 6h — Dialogue state tracking ───────────────────
        self.dialogue = DialogueStateTracker(store_dir=store_dir)

        # ── Layer 6i — Notifications + Personalization ───────────
        self.notifications = NotificationEngine(store_dir=store_dir)
        self.preferences = PreferenceLearner(memory=self.memory, store_dir=store_dir)

        # ── Layer 6g — Nightly Training (sleep-time learning pipeline) ──
        self.nightly_trainer = NightlyTrainer(
            memory=self.memory,
            exp_logger=self.exp_logger,
            self_improver=self.self_improver,
            store_dir=store_dir,
        )

        # ── Layer 7 — Action ─────────────────────────────────────
        if self.headless:
            self.executor = _HeadlessExecutor(self.exp_logger)
        else:
            from automation.executor import AutomationExecutor as AutomationExecutor_mod

            self.executor = AutomationExecutor_mod(
                reasoning_engine=self.reasoning,
                experience_logger=self.exp_logger,
            )

        # ── Layer 8 — Cognition ──────────────────────────────────
        self.brain = CognitiveBrain(
            executor=self.executor,
            memory=self.memory,
            tts_queue=global_command_queue,
            goal_queue=global_goal_queue,
            perception=self.perception,
            world_state=self.world_state,
            behavior=self.behavior,
            reasoning=self.reasoning,
            compressor=self.compressor,
            self_improver=self.self_improver,
            curiosity=self.curiosity,
            skill_graph=self.skill_graph,
            attention=self.attention,
            dream_engine=self.dream_engine,
            nightly_trainer=self.nightly_trainer,
            dialogue_tracker=self.dialogue,
            notifications=self.notifications,
            preference_learner=self.preferences,
            reward_engine=self.reward_engine,
            idle_monitor=self.idle_monitor,
            background_tasks=self.background_tasks,
        )

        # Wire voice input → brain
        self.ears.brain = self.brain

        # ── Layer 9 — Persistence & Monitoring ─────────────────
        self.checkpoint = StateCheckpoint(
            path=os.path.join(store_dir, "checkpoint.json"),
            interval_ticks=6,
        )
        self.brain._checkpoint = self.checkpoint

        # Restore brain state from checkpoint if available
        saved = self.checkpoint.load()
        if saved:
            self.checkpoint.restore(self.brain, saved)
            log.info("Restored brain from checkpoint.")

        self.health = HealthMonitor(
            brain=self.brain,
            tts_queue=global_command_queue,
            goal_queue=global_goal_queue,
            interval=30.0,
        )

        # Avatar runtime is temporarily disconnected to avoid interference.
        self.avatar = None

        log.info("All subsystems initialised.")

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Launch all background threads and perception loops."""
        if self._started:
            return
        self._started = True

        # Register Ctrl-C handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGBREAK, self._signal_handler)

        # Start user activity monitor
        self.idle_monitor.start()

        # Daemon threads — die automatically if main thread exits
        self._threads = [
            Thread(target=self.mouth.text_to_speech_loop,
                   args=(self._stop_event,), daemon=True, name="tts"),
            Thread(target=self.brain.run,
                   args=(self._stop_event,), daemon=True, name="brain"),
            Thread(target=self.health.run,
                   args=(self._stop_event,), daemon=True, name="health"),
        ]
        if not self.headless:
            self._threads.insert(
                0,
                Thread(target=self.ears.listen,
                       args=(self._stop_event,), daemon=True, name="speech"),
            )
        for t in self._threads:
            t.start()

        if self.avatar is not None:
            avatar_threads = self.avatar.start(self._stop_event)
            self._threads.extend(avatar_threads)

        # Give the health monitor access to managed threads for liveness checks.
        self.health._threads = self._threads

        # Background perception (screen + system, lightweight)
        if not self.headless:
            self.perception.start_background_loop(
                self._stop_event, interval=7.0, sources=["screen", "system"],
            )

        log.info("All threads started.")

    def wait(self) -> None:
        """Block the main thread on the camera loop (OpenCV needs main thread on Windows).

        When the camera window is closed or Ctrl-C is pressed, this returns.
        Skipped when a GUI manages the main thread instead.
        """
        if self.headless:
            self.wait_until_stopped()
            return
        try:
            self.eyes.camera_infer(self._stop_event)
        except KeyboardInterrupt:
            pass

    def wait_until_stopped(self) -> None:
        """Block until the stop event is set (used when GUI owns the main thread)."""
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            pass

    def stop(self) -> None:
        """Graceful shutdown — signal all threads and clean up resources."""
        if self._stop_event.is_set():
            return
        log.info("Shutting down…")
        self._stop_event.set()
        # Persist scheduled tasks before threads stop
        try:
            self.reasoning.planner.scheduler._save()
        except Exception as e:
            log.warning("Failed to persist scheduled tasks: %s", e)
        
        # Stop idle monitor and cancel background tasks
        self.idle_monitor.stop()
        cancelled = self.background_tasks.cancel_all(timeout=2.0)
        if cancelled > 0:
            log.info("Cancelled %d background tasks", cancelled)
        
        self.perception.shutdown()

        if self.headless:
            for t in self._threads:
                t.join(timeout=0.2)
        else:
            for t in self._threads:
                t.join(timeout=3)

        self._started = False
        log.info("Shutdown complete.")

    def run(self) -> None:
        """Convenience runner.

        In avatar mode, keep the process alive without entering the OpenCV
        camera window loop so the desktop overlay/VTube Studio runtime can own
        the experience. Otherwise retain the legacy camera-loop behavior.
        """
        self.start()
        avatar_mode = bool(
            self.avatar is not None
            and getattr(self.avatar, "settings", None)
            and self.avatar.settings.enabled
            and (self.avatar.settings.overlay_enabled or self.avatar.settings.vts_enabled)
        )
        if self.headless:
            self.wait_until_stopped()
        elif avatar_mode:
            self.wait_until_stopped()
        else:
            self.wait()
        self.stop()

    # ── Status & introspection ───────────────────────────────────

    def status(self) -> dict:
        """Return a quick health snapshot of all subsystems."""
        try:
            emotion = self.behavior.emotion.dominant_emotion()
        except Exception:
            try:
                emotion = self.behavior.emotion.dominant()
            except Exception:
                emotion = "neutral"
        base = {
            "running": self._started and not self._stop_event.is_set(),
            "mode": "headless" if self.headless else "desktop",
            "brain_state": self.brain.state.name,
            "current_goal": self.brain.current_goal,
            "mood": self.behavior.mood.get_mood_label(),
            "emotion": emotion,
            "personality": self.personality.summary(),
            "memory_events": len(self.memory.get_recent_events(999)),
            "active_projects": len(self.memory.get_active_projects()),
            "notifications": self.notifications.stats(),
            "preferences_summary": self.preferences.summary(3),
            "skill_graph": self.skill_graph.stats(),
            "reward_engine": "enabled",
            "attention": self.attention.stats(),
            "dreaming": self.dream_engine.stats(),
            "idle_monitor": self.idle_monitor.stats(),
            "background_tasks": self.background_tasks.stats(),
            "threads": {t.name: t.is_alive() for t in self._threads},
        }
        # Merge health monitor snapshot (RAM, CPU, etc.)
        base["health"] = self.health.snapshot()
        return base

    def send_message(self, text: str) -> None:
        """Inject a user message into the cognitive loop (programmatic input)."""
        self.brain.receive_user_message(text)

    def schedule_goal(
        self,
        goal: str,
        run_at: float | datetime | str,
        *,
        priority: int = 0,
        recurrence_seconds: float | None = None,
        world_context: str = "",
    ) -> dict:
        """Schedule a goal for future execution by the brain loop."""
        return self.reasoning.schedule_goal(
            goal,
            run_at,
            world_context=world_context,
            priority=priority,
            recurrence_seconds=recurrence_seconds,
        )

    def list_scheduled_goals(self) -> list[dict]:
        """Return all pending scheduled goals."""
        return self.reasoning.list_scheduled_goals()

    def cancel_scheduled_goal(self, task_id: str) -> bool:
        """Cancel a previously scheduled goal by ID."""
        return self.reasoning.cancel_scheduled_goal(task_id)

    # ── Training entrypoint ──────────────────────────────────────

    def train(self, dataset_type: str = "action_prediction",
              epochs: int = 3) -> dict | None:
        """Build a dataset from logged experience and train a model.

        Args:
            dataset_type: one of action_prediction, outcome_prediction,
                          plan_quality, reasoning.
            epochs: training epochs.

        Returns:
            Run metadata dict, or None on failure.
        """
        path = self.dataset_builder.build(dataset_type)
        if path is None:
            log.info("No training data available yet.")
            return None
        return self.model_trainer.train(path, epochs=epochs)

    # ── Internals ─────────────────────────────────────────────

    def _signal_handler(self, signum, frame):
        log.info("Interrupt received.")
        self.stop()
