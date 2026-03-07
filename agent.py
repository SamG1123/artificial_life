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

import signal
import threading
import time
from threading import Thread, Event

import image_processing
import voice_recognition
import tts as tts_module
import web_support
from automation.executor import AutomationExecutor
from brain import CognitiveBrain
from memory import MemorySystem, MemoryCompressor
from perception import PerceptionManager
from world_state import WorldStateManager
from emotion import PersonalityModel, MoodEngine, EmotionEngine, BehaviorController
from reasoning import ReasoningEngine
from learning import ExperienceLogger, DatasetBuilder, ModelTrainer, SelfImprover
from curiosity import CuriosityEngine
from config import global_goal_queue, global_command_queue


class AgentController:
    """Top-level controller that owns, wires, and manages every subsystem."""

    # ── Construction ──────────────────────────────────────────────

    def __init__(self, store_dir: str = "memory_store"):
        self._store_dir = store_dir
        self._stop_event = Event()
        self._threads: list[Thread] = []
        self._started = False

        # ── Layer 1 — IO (raw sensors & effectors) ────────────────
        self.eyes = image_processing.ObjectDetection()
        self.web = web_support.WebSupport()
        self.ears = voice_recognition.SpeechSupport()
        self.mouth = tts_module.TextToSpeech()

        # ── Layer 2 — Perception ──────────────────────────────────
        self.world_state = WorldStateManager()

        self.perception = PerceptionManager(
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
        self.reasoning = ReasoningEngine(behavior=self.behavior)

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

        # ── Layer 6c — Curiosity (autonomous goal generation) ────
        self.curiosity = CuriosityEngine(
            memory=self.memory,
            behavior=self.behavior,
            store_dir=store_dir,
        )

        # ── Layer 7 — Action ─────────────────────────────────────
        self.executor = AutomationExecutor(
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
        )

        # Wire voice input → brain
        self.ears.brain = self.brain

        print("[Agent] All subsystems initialised.")

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Launch all background threads and perception loops."""
        if self._started:
            return
        self._started = True

        # Register Ctrl-C handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGBREAK, self._signal_handler)

        # Daemon threads — die automatically if main thread exits
        self._threads = [
            Thread(target=self.ears.listen,
                   args=(self._stop_event,), daemon=True, name="speech"),
            Thread(target=self.mouth.text_to_speech_loop,
                   args=(self._stop_event,), daemon=True, name="tts"),
            Thread(target=self.brain.run,
                   args=(self._stop_event,), daemon=True, name="brain"),
        ]
        for t in self._threads:
            t.start()

        # Background perception (screen + system, lightweight)
        self.perception.start_background_loop(
            self._stop_event, interval=5.0, sources=["screen", "system"],
        )

        print("[Agent] All threads started.")

    def wait(self) -> None:
        """Block the main thread on the camera loop (OpenCV needs main thread on Windows).

        When the camera window is closed or Ctrl-C is pressed, this returns.
        Skipped when a GUI manages the main thread instead.
        """
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
        print("[Agent] Shutting down…")
        self._stop_event.set()

        self.perception.shutdown()

        for t in self._threads:
            t.join(timeout=3)

        self._started = False
        print("[Agent] Shutdown complete.")

    def run(self) -> None:
        """Convenience: start → wait (camera loop) → stop."""
        self.start()
        self.wait()
        self.stop()

    # ── Status & introspection ───────────────────────────────────

    def status(self) -> dict:
        """Return a quick health snapshot of all subsystems."""
        return {
            "running": self._started and not self._stop_event.is_set(),
            "brain_state": self.brain.state.name,
            "current_goal": self.brain.current_goal,
            "mood": self.behavior.mood.get_mood_label(),
            "emotion": self.behavior.emotion.dominant_emotion(),
            "personality": self.personality.summary(),
            "memory_events": len(self.memory.get_recent_events(999)),
            "threads": {t.name: t.is_alive() for t in self._threads},
        }

    def send_message(self, text: str) -> None:
        """Inject a user message into the cognitive loop (programmatic input)."""
        self.brain.receive_user_message(text)

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
            print("[Agent] No training data available yet.")
            return None
        return self.model_trainer.train(path, epochs=epochs)

    # ── Internals ─────────────────────────────────────────────────

    def _signal_handler(self, signum, frame):
        print("\n[Agent] Interrupt received.")
        self.stop()
