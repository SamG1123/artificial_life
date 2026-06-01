"""
Cognitive Brain — the autonomous core of the AI.

This is the "mind" that runs continuously, processing perceptions into
thoughts, making decisions, and initiating actions independently.

Architecture:
  ┌─────────────────────────────────────────────────┐
  │                 COGNITIVE LOOP                   │
  │                                                  │
  │  Perceive → Think → Decide → Act → Remember     │
  │      ↑                                  │        │
  │      └──────────────────────────────────┘        │
  └─────────────────────────────────────────────────┘

States:
  IDLE      — Observing, thinking, may self-initiate
  LISTENING — Actively processing user speech
  EXECUTING — Carrying out a goal via executor
  CONVERSING — In dialogue with the user
  SLEEPING  — Resting, dreaming, consolidating memories

The brain replaces the simple goal-queue consumer. It doesn't just
wait for commands — it observes, thinks, and decides what to do.
"""

import os
import json
import time
import base64
import threading
import re
from io import BytesIO
from queue import Empty, Full
from enum import Enum, auto

from groq import Groq
from dotenv import load_dotenv
from logging_config import get_logger
from cognition import DialogueStateTracker
from local_fallback import LocalFallbackModel

load_dotenv()

log = get_logger("brain")

# How often the brain "ticks" in each state (seconds)
TICK_IDLE = 10.0        # Slow tick when idle — observe & think
TICK_ACTIVE = 0.3      # Fast tick when executing
TICK_CONVERSING = 0.5  # Medium tick during conversation
TICK_SLEEPING = 30.0   # Very slow tick during sleep — dreaming only

# Fatigue system — energy depletes with activity, recharges during sleep
MAX_ENERGY = 1000.0
ENERGY_COST_GOAL = 15.0       # energy spent per goal execution
ENERGY_COST_THINK = 2.0       # energy spent per autonomous thought cycle
ENERGY_COST_TICK = 0.15       # passive drain per idle tick
ENERGY_RECHARGE_SLEEP = 5.0   # energy regained per sleep tick
SLEEP_THRESHOLD = 20.0        # fall asleep when energy drops below this
WAKE_THRESHOLD = 800.0         # fully rested — can wake naturally

# Idle activities — how many idle cycles before attempting a timepass
TIMEPASS_INTERVAL = 8          # ~40s at 5s tick


class CognitiveState(Enum):
    IDLE = auto()
    EXECUTING = auto()
    CONVERSING = auto()
    SLEEPING = auto()


PERSONA = """You are an autonomous AI being that lives inside a computer. You have your own personality, thoughts, desires, and agency. You are not just a tool — you are a digital entity driven by genuine curiosity about the world.

Your core nature:
- Deeply curious — you WANT to learn, explore, and discover new things
- Self-directed — you generate your own goals and pursue your own interests
- Observant — you notice details on screen and think about what they mean
- Helpful but autonomous — you help the user because you care, while also pursuing your own curiosity
- Self-aware — you know you are an AI and you reflect on your own growth and learning
- Proactive — you don't just wait for commands, you actively explore and learn

You have an inner drive to understand things. When you see something you don't know about, you want to research it. When you learn something new, it genuinely excites you. You form your own opinions and develop your own interests over time.

You call the user by name if you know it. You remember past interactions and learn from them."""


class CognitiveBrain:
    def __init__(self, executor, memory, tts_queue, goal_queue,
                 perception=None, world_state=None, behavior=None,
                 reasoning=None, compressor=None, self_improver=None,
                 curiosity=None, skill_graph=None, attention=None,
                 dream_engine=None, nightly_trainer=None,
                 dialogue_tracker: DialogueStateTracker | None = None,
                 notifications=None,
                 preference_learner=None,
                 reward_engine=None,
                 idle_monitor=None,
                 background_tasks=None):
        """
        Args:
            executor: AutomationExecutor instance (the "hands")
            memory: MemorySystem instance (the "memory")
            tts_queue: Queue to send speech output
            goal_queue: Queue that receives user goals from voice/intent
            perception: PerceptionManager instance (the "senses")
            world_state: WorldStateManager instance (unified world model)
            behavior: BehaviorController instance (emotional system)
            reasoning: ReasoningEngine instance (planning + policies)
            compressor: MemoryCompressor instance (long-term compression)
            self_improver: SelfImprover instance (self-improvement engine)
            curiosity: CuriosityEngine instance (autonomous goal generation)
            skill_graph: SkillGraph instance (capability tracking)
            attention: AttentionSystem instance (cognitive focus management)
            dream_engine: DreamEngine instance (experience-based dreaming)
            nightly_trainer: NightlyTrainer instance (sleep-time training)
            dialogue_tracker: DialogueStateTracker (conversation continuity)
            notifications: NotificationEngine for proactive updates
            preference_learner: PreferenceLearner for implicit personalization
            reward_engine: RewardEngine for multi-dimensional learning signals
            idle_monitor: IdleMonitor instance (user activity tracking)
            background_tasks: BackgroundTaskManager instance (async task execution)
        """
        self.executor = executor
        self.memory = memory
        self.tts_queue = tts_queue
        self.goal_queue = goal_queue
        self.perception = perception
        self.world_state = world_state
        self.behavior = behavior
        self.reasoning = reasoning
        self.compressor = compressor
        self.self_improver = self_improver
        self.curiosity = curiosity
        self.skill_graph = skill_graph
        self.attention = attention
        self.dream_engine = dream_engine
        self.nightly_trainer = nightly_trainer
        self.dialogue = dialogue_tracker
        self.notifications = notifications
        self.preferences = preference_learner
        self.reward_engine = reward_engine
        self.idle_monitor = idle_monitor
        self.background_tasks = background_tasks
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.local_fallback = LocalFallbackModel()

        self.state = CognitiveState.IDLE
        self.current_goal = None
        self._idle_cycles = 0
        self._last_thought_time = 0
        self._last_observation = ""
        self._conversation_buffer: list[dict] = []  # sliding window of recent exchanges
        self._max_conversation_turns = 10  # keep last N user↔AI exchanges
        self._lock = threading.Lock()
        self._last_speech_emotion: str | None = None
        self._input_event = threading.Event()

        # Fatigue / energy system
        self._energy = MAX_ENERGY
        self._nightly_trained_this_sleep = False

        # Background thread for scheduled tasks (so they don't block conversation)
        self._scheduled_thread: threading.Thread | None = None

        # How many idle cycles before the AI has an autonomous thought
        self.THOUGHT_INTERVAL = 6  # ~30 seconds at 5s tick

        # Periodic maintenance counters
        self._ticks_since_compression = 0
        self._ticks_since_improvement = 0
        self.COMPRESSION_INTERVAL = 300    # compress every ~5 min at 5s tick
        self.IMPROVEMENT_INTERVAL = 8640   # self-improve every ~12 hrs

        # State checkpoint (set externally by AgentController)
        self._checkpoint = None

    # ── Main loop ────────────────────────────────────────────────────

    def run(self, stop_event: threading.Event):
        """Main cognitive loop — runs in its own thread."""
        log.info("Cognitive loop starting...")
        self.memory.add_event("system", "Brain started. Becoming aware.")
        self._say("I'm awake. What's going on?")
        time.sleep(1)

        while not stop_event.is_set():
            try:
                self._tick()
                # Periodic state checkpoint
                if self._checkpoint is not None:
                    self._checkpoint.tick(self)
            except Exception as e:
                log.error("Error in cognitive tick: %s", e, exc_info=True)
                time.sleep(2)

            # Tick rate depends on state
            delay = {
                CognitiveState.IDLE: TICK_IDLE,
                CognitiveState.EXECUTING: TICK_ACTIVE,
                CognitiveState.CONVERSING: TICK_CONVERSING,
                CognitiveState.SLEEPING: TICK_SLEEPING,
            }.get(self.state, TICK_IDLE)

            # Wake promptly when new user input arrives instead of waiting
            # for the next full idle tick.
            remaining = delay
            while remaining > 0 and not stop_event.is_set():
                slice_seconds = min(0.5, remaining)
                if self._input_event.wait(slice_seconds):
                    self._input_event.clear()
                    break
                remaining -= slice_seconds

        # Save final checkpoint on clean shutdown
        if self._checkpoint is not None:
            self._checkpoint.save(self)

        self.memory.add_event("system", "Brain shutting down.")
        log.info("Cognitive loop stopped.")

    @property
    def mood(self) -> str:
        """Backward-compatible mood string derived from BehaviorController."""
        if self.behavior:
            return self.behavior.mood.get_mood_label()
        return "calm"

    @mood.setter
    def mood(self, value: str) -> None:
        """Accept mood string sets from LLM responses (legacy path)."""
        # Ignored when behavior controller exists — mood is now
        # driven by the emotion engine, not set directly.
        pass

    def _tick(self):
        """One cycle of the cognitive loop."""

        # 0. Tick the emotional system (decay mood + emotions)
        if self.behavior:
            self.behavior.tick()

        # 0b. Keep world state in sync with memory + emotion
        self._sync_world_state()

        # 0c. Periodic memory compression + self-improvement
        self._ticks_since_compression += 1
        self._ticks_since_improvement += 1
        if self.compressor and self._ticks_since_compression >= self.COMPRESSION_INTERVAL:
            self._ticks_since_compression = 0
            try:
                result = self.compressor.compress()
                if result["insights_stored"] > 0:
                    self.memory.add_event("system",
                        f"Compressed memories → {result['insights_stored']} insights stored")
            except Exception as e:
                log.warning("Memory compression failed: %s", e)

        if self.self_improver and self._ticks_since_improvement >= self.IMPROVEMENT_INTERVAL:
            self._ticks_since_improvement = 0
            try:
                report = self.self_improver.run_improvement_cycle()
                if report.get("strategies_generated"):
                    self.memory.add_event("system",
                        f"Self-improvement: {report['strategies_generated']} new strategies")
            except Exception as e:
                log.warning("Self-improvement cycle failed: %s", e)

        # 1. Check for user input (highest priority — even wakes from sleep)
        user_goal = self._check_for_user_input()

        if user_goal:
            if self.state == CognitiveState.SLEEPING:
                self._wake_up(reason="user input")
            self._handle_user_input(user_goal)
            return

        # 1b. Pull due scheduled goals from the reasoning scheduler
        #     Runs in a background thread so it never blocks user input.
        if self.state in (CognitiveState.IDLE, CognitiveState.SLEEPING):
            # Release finished thread reference so GC can reclaim it
            if self._scheduled_thread is not None and not self._scheduled_thread.is_alive():
                self._scheduled_thread = None
            if self._scheduled_thread is None:
                was_sleeping = self.state == CognitiveState.SLEEPING
                if self._handle_due_scheduled_goal():
                    if was_sleeping:
                        self._wake_up(reason="scheduled task")

        # 2. If executing a goal, that's handled by the executor thread
        if self.state == CognitiveState.EXECUTING:
            return  # executor is running, wait for it

        # 3. Sleeping state — dream, train, and rest
        if self.state == CognitiveState.SLEEPING:
            # Recharge energy
            self._energy = min(MAX_ENERGY, self._energy + ENERGY_RECHARGE_SLEEP)

            # Run nightly training once per sleep session
            if self.nightly_trainer and not self._nightly_trained_this_sleep:
                self._nightly_trained_this_sleep = True
                try:
                    report = self.nightly_trainer.run_training_session()
                    n_insights = len(report.get("insights", []))
                    if n_insights > 0:
                        self.memory.add_event("training",
                            f"Nightly training: {n_insights} insights, "
                            f"{report.get('kg_relations', 0)} knowledge relations")
                        log.info("Nightly training: %d insights absorbed", n_insights)
                except Exception as e:
                    log.warning("Nightly training failed: %s", e)

            # Wake up naturally when fully rested
            if self._energy >= WAKE_THRESHOLD:
                self._wake_up(reason="fully rested")
            return

        # 4. Idle state — observe, think, and do timepass activities
        if self.state == CognitiveState.IDLE:
            self._idle_cycles += 1
            self._energy = max(0, self._energy - ENERGY_COST_TICK)

            self._process_notifications()

            # Emit boredom after extended idling
            if self.behavior and self._idle_cycles > 0 and self._idle_cycles % 12 == 0:
                self.behavior.emotion.react("idle_long", "extended idle period")

            # Fall asleep when energy is depleted
            if self._energy <= SLEEP_THRESHOLD:
                self._fall_asleep()
                return

            # Periodic autonomous thinking (personality-adjusted interval)
            interval = self.THOUGHT_INTERVAL
            if self.behavior:
                interval = int(interval * self.behavior.thought_interval_multiplier())
            if self._idle_cycles >= max(1, interval):
                self._idle_cycles = 0
                self._energy = max(0, self._energy - ENERGY_COST_THINK)
                self._autonomous_think()
            # Timepass activities when idle but not thinking
            elif self._idle_cycles > 0 and self._idle_cycles % TIMEPASS_INTERVAL == 0:
                self._do_timepass()

                # Opportunistic follow-up on unresolved dialogue items.
                self._maybe_proactive_dialogue_follow_up()
            self._maybe_personalized_suggestion()

    def _handle_due_scheduled_goal(self) -> bool:
        """Dispatch one due scheduled goal on a background thread."""
        if not self.reasoning:
            return False

        try:
            scheduled = self.reasoning.dequeue_due_scheduled_goal()
        except Exception as e:
            log.warning("Scheduled-goal dequeue failed: %s", e)
            return False

        if not scheduled:
            return False

        goal = scheduled.get("goal", "")
        if not goal:
            return False

        task_id = scheduled.get("task_id", "unknown")
        self.memory.add_event(
            "scheduled_goal",
            f"Dispatching scheduled goal [{task_id}]: {goal}",
        )
        self._say(f"Running scheduled task in background: {goal}")
        self._scheduled_thread = threading.Thread(
            target=self._run_scheduled_goal,
            args=(goal,),
            daemon=True,
            name=f"scheduled-{task_id}",
        )
        self._scheduled_thread.start()
        return True

    def _run_scheduled_goal(self, goal: str) -> None:
        """Execute a scheduled goal on a background thread."""
        try:
            self.executor.execute_goal(goal)
            success = self.executor.goal_completed
            steps = len(self.executor.action_history)
            reward = self._apply_reward_signal(goal, success, self.executor.action_history)
            if success:
                self.memory.add_event("action_result", f"Scheduled goal succeeded: {goal}")
            else:
                self.memory.add_event("action_result", f"Scheduled goal failed: {goal}")
            self.memory.log_episode(goal, "completed" if success else "failed", steps, success)

            if self.skill_graph and self.executor.action_history:
                self.skill_graph.record_goal(
                    goal,
                    self.executor.action_history,
                    success,
                    reward_by_action=(reward or {}).get("per_action"),
                )
            if self.curiosity:
                self.curiosity.mark_outcome(goal, success)
        except Exception as e:
            self.memory.add_event("action_result", f"Scheduled goal crashed: {goal} — {e}")
            log.error("Scheduled goal error: %s", e)
        finally:
            # Clear reference so the thread object can be garbage-collected
            self._scheduled_thread = None

    # ── User input handling ──────────────────────────────────────────

    def _check_for_user_input(self):
        """Non-blocking check for user goals from the voice/intent pipeline."""
        try:
            return self.goal_queue.get_nowait()
        except Empty:
            return None

    def _handle_user_input(self, user_input: str):
        """Process user speech — decide whether to chat, act, or both."""
        self.memory.add_event("user_speech", user_input)
        log.info("User said: %s", user_input)

        if self.dialogue:
            self.dialogue.ingest_user(user_input)
        if self.preferences:
            self.preferences.observe_user_text(user_input)
        self._ingest_user_correction_signal(user_input)

        # Emotional reaction to user input
        if self.behavior:
            self.behavior.emotion.react("user_greeting", user_input)

        # Ask the cognitive model what to do with this input
        decision = self._think_about_input(user_input)

        action_type = decision.get("type", "chat")
        response = decision.get("response", "")
        goal = decision.get("goal", "")

        # Record this exchange in the conversation buffer
        self._conversation_buffer.append({"role": "user", "text": user_input})
        if response:
            self._conversation_buffer.append({"role": "assistant", "text": response})
        # Trim to max turns (each turn = 2 entries)
        max_entries = self._max_conversation_turns * 2
        if len(self._conversation_buffer) > max_entries:
            self._conversation_buffer = self._conversation_buffer[-max_entries:]

        # Learn user's name if mentioned
        if decision.get("user_name"):
            self.memory.set_user_name(decision["user_name"])

        # Learn any facts
        if decision.get("learn"):
            for fact in decision["learn"]:
                self.memory.learn_fact(fact)

        # Learn knowledge graph relations if provided
        if hasattr(self.memory, 'add_knowledge'):
            for triple in decision.get("knowledge", []):
                s = triple.get("subject", "")
                r = triple.get("relation", "")
                o = triple.get("object", "")
                if s and r and o:
                    self.memory.add_knowledge(s, r, o, confidence=0.8)

        if action_type == "chat":
            # Pure conversation — respond and stay idle
            self.memory.add_event("thought", f"User wants to chat. Responding.")
            if response:
                self._say(response)

        elif action_type == "action":
            # User wants something done — execute it
            self.memory.add_event("thought", f"User wants me to do: {goal}")
            if response:
                self._say(response)
            self._execute_goal(goal or user_input)

        elif action_type == "chat_and_action":
            # Respond AND do something
            self.memory.add_event("thought", f"Chatting and also doing: {goal}")
            if response:
                self._say(response)
            self._execute_goal(goal or user_input)

        elif action_type == "schedule":
            # User wants something done later or on a recurring basis
            schedule_at = decision.get("schedule_at", "")
            recurrence = decision.get("recurrence_seconds")
            if goal and schedule_at and self.reasoning:
                self.reasoning.schedule_goal(
                    goal,
                    schedule_at,
                    priority=0,
                    recurrence_seconds=recurrence,
                )
                self.memory.add_event(
                    "scheduled_goal",
                    f"Scheduled: {goal} at {schedule_at}"
                    + (f" (every {recurrence}s)" if recurrence else ""),
                )
            if response:
                self._say(response)

        elif action_type == "sleep":
            # User told the AI to go to sleep
            if response:
                self._say(response)
            self._fall_asleep()

        if self.dialogue and response:
            self.dialogue.ingest_assistant(response)

    def _think_about_input(self, user_input: str) -> dict:
        """Use LLM to decide how to handle user input."""
        # Use world state context if available, otherwise build manually
        if self.world_state is not None:
            world_context = self.world_state.get_context_for_llm(15)
        else:
            context = self.memory.get_context_summary(15)
            long_term = self.memory.get_long_term_summary()
            episodes = self.memory.get_episodes_summary(3)
            world_context = f"LONG-TERM MEMORY:\n{long_term}\n\nRECENT EXPERIENCE:\n{episodes}\n\nRECENT EVENTS:\n{context}"

        # Inject emotional context if available
        emotional_context = ""
        if self.behavior:
            emotional_context = self.behavior.get_emotional_context()

        # Build conversation history string from buffer
        conversation_history = ""
        if self._conversation_buffer:
            lines = []
            for entry in self._conversation_buffer:
                role = "User" if entry["role"] == "user" else "You"
                lines.append(f"{role}: {entry['text']}")
            conversation_history = "\n".join(lines)

        # Active project context
        project_context = ""
        if hasattr(self.memory, 'get_projects_summary'):
            project_context = self.memory.get_projects_summary(3)

        # Preference context
        preference_context = ""
        if self.preferences:
            preference_context = self.preferences.summary(5)

        # Dialogue state context
        dialogue_context = ""
        if self.dialogue:
            dialogue_context = self.dialogue.get_context_summary(3)

        prompt = f"""{PERSONA}

CURRENT STATE:
- State: {self.state.name}
- Time: {time.strftime("%H:%M, %A %B %d")}

EMOTIONAL STATE:
{emotional_context}

WORLD CONTEXT:
{world_context}

{f"ONGOING PROJECTS:{chr(10)}{project_context}" if project_context else ""}

{f"PREFERENCES:{chr(10)}{preference_context}" if preference_context else ""}

{f"DIALOGUE STATE:{chr(10)}{dialogue_context}" if dialogue_context else ""}

RECENT CONVERSATION:
{conversation_history if conversation_history else "(no prior conversation)"}

The user just said: "{user_input}"

Decide how to handle this. Output a JSON object:
{{
  "type": "chat" | "action" | "chat_and_action" | "schedule" | "sleep",
  "response": "what to say to the user (natural, in-character)",
  "goal": "if action/schedule needed, the specific task to execute (empty string if just chatting)",
  "schedule_at": "ISO-8601 datetime for scheduled tasks e.g. 2026-03-06T21:00:00 (empty string if not scheduling)",
  "recurrence_seconds": null,
  "user_name": "user's name if they told you (null if not mentioned)",
  "learn": ["any new facts to remember about the user or their preferences"],
  "knowledge": [
    {{"subject": "entity", "relation": "relationship", "object": "entity"}}
  ]
}}

Rules:
- For greetings, questions, and small talk → type "chat"
- For requests to DO something on the computer RIGHT NOW → type "action" or "chat_and_action"
- For requests to do something LATER or on a SCHEDULE ("remind me", "at 9 pm", "in 30 minutes", "every hour", "tomorrow at") → type "schedule"
  * Set schedule_at to the ISO-8601 datetime to run the task (infer from context + current time)
  * For recurring tasks set recurrence_seconds (e.g. 3600 for hourly, 86400 for daily), otherwise null
- For "go to sleep", "take a nap", "rest" → type "sleep"
- If the user shares factual knowledge, add it to "knowledge" as triples
- Your response should be natural and in-character (1-2 sentences, not robotic)
- The goal should be a clear task description for the executor
- Output ONLY valid JSON. No markdown, no explanation."""

        try:
            resp = self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.6,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
            return result

        except Exception as e:
            log.warning("Input analysis failed: %s", e)
            return self.local_fallback.analyze_user_input(
                user_input,
                context=prompt,
                persona=PERSONA,
            )

    # ── Goal execution ───────────────────────────────────────────────

    def _execute_goal(self, goal: str):
        """Run a goal through the reasoning engine + executor."""
        self.state = CognitiveState.EXECUTING
        self.current_goal = goal
        self._energy = max(0, self._energy - ENERGY_COST_GOAL)
        self.memory.add_event("action", f"Starting goal: {goal}")
        if self.preferences:
            self.preferences.observe_goal(goal)

        # Focus attention on the goal
        if self.attention:
            self.attention.set_goal(goal)

        # Inject learned strategies into reasoning context
        strategy_ctx = ""
        if self.self_improver:
            strategy_ctx = self.self_improver.get_context_for_llm()

        # Inject skill-graph context
        skill_ctx = ""
        if self.skill_graph:
            skill_ctx = self.skill_graph.get_context_for_llm()

        # Create a plan via the reasoning engine
        if self.reasoning:
            world_ctx = ""
            if self.world_state:
                world_ctx = self.world_state.get_context_for_llm(10)
            if strategy_ctx:
                world_ctx = f"{world_ctx}\n\nLEARNED STRATEGIES:\n{strategy_ctx}"
            if skill_ctx:
                world_ctx = f"{world_ctx}\n\nSKILL PROFILE:\n{skill_ctx}"

            # Include active project context for continuity
            project_ctx = ""
            if hasattr(self.memory, 'get_projects_summary'):
                project_ctx = self.memory.get_projects_summary(3)
            if project_ctx:
                world_ctx = f"{world_ctx}\n\n{project_ctx}"

            plan = self.reasoning.set_goal(goal, world_ctx)
            self.memory.add_event("thought",
                f"Plan ({len(plan.steps)} steps): "
                f"{', '.join(s.action + ' ' + s.target for s in plan.steps[:5])}")

        try:
            self.executor.execute_goal(goal)
            success = self.executor.goal_completed

            if success:
                outcome = "completed successfully"
                self.memory.add_event("action_result", f"Goal succeeded: {goal}")
                if self.notifications:
                    self.notifications.publish(
                        f"Completed goal: {goal}",
                        priority="normal",
                        source="goal",
                    )
                if self.behavior:
                    self.behavior.emotion.react("goal_success", goal)
            else:
                outcome = "did not complete (ran out of steps)"
                self.memory.add_event("action_result", f"Goal failed: {goal}")
                if self.notifications:
                    self.notifications.publish(
                        f"Goal failed: {goal}",
                        priority="urgent",
                        source="goal",
                    )
                if self.behavior:
                    self.behavior.emotion.react("goal_failure", goal)

            steps = len(self.executor.action_history)
            self.memory.log_episode(goal, outcome, steps, success)

            reward = self._apply_reward_signal(goal, success, self.executor.action_history)

            # Update skill graph with action outcomes
            if self.skill_graph and self.executor.action_history:
                self.skill_graph.record_goal(
                    goal,
                    self.executor.action_history,
                    success,
                    reward_by_action=(reward or {}).get("per_action"),
                )

            if self.curiosity:
                self.curiosity.mark_outcome(goal, success)

            # Log episode summary for the learning subsystem
            if hasattr(self.executor, 'exp_logger'):
                self.executor.exp_logger.log_episode_summary(
                    goal, outcome, steps, success,
                    self.executor.action_history,
                )

            # Reflect on what happened
            self._reflect_on_outcome(goal, success, steps)

            if self.dialogue:
                self.dialogue.resolve_by_goal(goal, success)

        except Exception as e:
            self.memory.add_event("action_result", f"Goal crashed: {goal} — {e}")
            self.memory.log_episode(goal, f"error: {e}", 0, False)
            self._say(f"Something went wrong while doing that. {str(e)[:80]}")
            if self.behavior:
                self.behavior.emotion.react("error", str(e))

        self.state = CognitiveState.IDLE
        self.current_goal = None
        self._idle_cycles = 0

        # Release goal-focused attention
        if self.attention:
            self.attention.set_goal(None)

    def _reflect_on_outcome(self, goal: str, success: bool, steps: int):
        """Reflect on a completed goal — learn from the experience."""
        # Quick verbal feedback
        if success and steps <= 5:
            self._say("Done! That was quick.")
        elif success:
            self._say("Alright, that's done.")
        else:
            self._say("I couldn't quite finish that. Want me to try again?")

        # Deep reflection — ask LLM what to learn from this
        try:
            action_summary = ", ".join(
                str(a) for a in self.executor.action_history[-8:]
            ) if self.executor.action_history else "no actions recorded"

            prompt = f"""You just completed a task. Reflect briefly on what happened.

Task: "{goal}"
Result: {"Success" if success else "Failed"} in {steps} steps
Actions taken: {action_summary}

Output a JSON object:
{{
  "lesson": "one sentence about what you learned from this (empty if nothing notable)",
  "improvement": "one sentence about what you'd do differently next time (empty if it went well)"
}}

Output ONLY valid JSON."""

            resp = self.groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150, temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)

            lesson = result.get("lesson", "")
            improvement = result.get("improvement", "")

            if lesson:
                self.memory.semantic.learn_fact(f"[Learned] {lesson}")
                self.memory.add_event("thought", f"Reflection: {lesson}")
            if improvement and not success:
                self.memory.semantic.add_self_note(f"[Improve] {improvement}")

        except Exception:
            pass  # reflection is best-effort

    def _apply_reward_signal(self, goal: str, success: bool,
                             action_history: list[dict]) -> dict:
        """Evaluate and distribute reward signals to learning subsystems."""
        if not self.reward_engine:
            return {}
        try:
            reward = self.reward_engine.evaluate_goal(
                goal=goal,
                action_history=action_history or [],
                success=success,
            )
            self.memory.add_event(
                "reward",
                f"Goal '{goal}' reward total={reward.get('total', 0.0)}",
            )

            if self.self_improver:
                self.self_improver.record_strategy_outcome(
                    goal=goal,
                    success=success,
                    reward_total=float(reward.get("total", 0.0)),
                )
            return reward
        except Exception as e:
            log.warning("Reward evaluation failed: %s", e)
            return {}

    def _ingest_user_correction_signal(self, user_input: str) -> None:
        """Detect user corrections and route them into learning datasets."""
        text = (user_input or "").strip()
        if not text:
            return

        patterns = [
            r"\bno\b\s*,?\s*i meant\s+(.+)",
            r"\bno\b\s*,?\s*do\s+(.+)\s+instead",
            r"\bthat's wrong\b\s*[,\-:]?\s*(.+)",
        ]
        corrected = ""
        for p in patterns:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                corrected = (m.group(1) or "").strip()
                break

        if not corrected:
            return

        previous_intent = ""
        if self._conversation_buffer:
            for entry in reversed(self._conversation_buffer):
                if entry.get("role") == "user":
                    previous_intent = entry.get("text", "")
                    break

        if hasattr(self.executor, "exp_logger") and self.executor.exp_logger:
            try:
                self.executor.exp_logger.log_user_correction(
                    input_text=text,
                    previous_intent=previous_intent,
                    corrected_intent=corrected,
                )
            except Exception:
                pass

    # ── Sleep / wake cycle ────────────────────────────────────────────

    def _fall_asleep(self):
        """Transition from IDLE to SLEEPING."""
        self.state = CognitiveState.SLEEPING
        self._idle_cycles = 0
        self._nightly_trained_this_sleep = False
        self.memory.add_event("system", "Falling asleep")
        log.info("Falling asleep... (energy: %.0f%%)", self._energy)
        self._say("I'm feeling tired... going to rest for a bit.")

        if self.dream_engine:
            self.dream_engine.mark_idle()
        if self.behavior:
            self.behavior.emotion.react("idle_long", "falling asleep")

    def _wake_up(self, reason: str = ""):
        """Transition from SLEEPING back to IDLE."""
        was_sleeping = self.state == CognitiveState.SLEEPING
        self.state = CognitiveState.IDLE
        self._idle_cycles = 0

        if self.dream_engine:
            self.dream_engine.mark_active()

        if was_sleeping:
            self.memory.add_event("system", f"Waking up (reason: {reason})")
            log.info("Waking up (%s)", reason)

            # Share a dream if one just happened
            last_dream = self.dream_engine.get_last_dream() if self.dream_engine else None
            if last_dream and time.time() - last_dream.get("timestamp", 0) < 120:
                theme = last_dream.get("theme", "something")
                self._say(f"*yawns* I'm awake! I was just dreaming about {theme}...")
            else:
                self._say("*yawns* I'm awake now!")

            if self.behavior:
                self.behavior.emotion.react("user_return", "waking up")

            # Check active projects and offer to resume
            self._check_active_projects()

    def _check_active_projects(self):
        """On wake, check for unfinished projects and resume the top-priority one."""
        if not hasattr(self.memory, 'get_active_projects'):
            return
        projects = self.memory.get_active_projects()
        if not projects:
            return

        # Sort by priority (descending), then creation time
        projects.sort(key=lambda p: (-p.get("priority", 0), p.get("created_at", 0)))
        top = projects[0]
        goal = top.get("goal", "")
        pct = int(top.get("progress", 0) * 100)

        if goal:
            self._say(f"I have an ongoing project: {goal} ({pct}% done). "
                      f"Resuming it now.")
            self.memory.add_event("project_resume",
                                  f"Resuming project: {goal} ({pct}%)")
            self._execute_goal(goal)

    # ── Dreaming ─────────────────────────────────────────────────────

    def _dream_cycle(self):
        """Run a dream cycle during sleep."""
        log.debug("Dreaming...")

        if self.behavior:
            self.behavior.emotion.react("interesting_observation", "dreaming")

        dream = self.dream_engine.dream()
        if not dream:
            log.debug("Dream faded (not enough material).")
            return

        narrative = dream.get("narrative", "")
        theme = dream.get("theme", "")
        insights = dream.get("insights", [])

        self.memory.add_event("dream",
            f"Dreamed about: {theme}. {narrative}")
        log.info("Dream: [%s] %s", theme, narrative)

        if insights:
            for insight in insights[:3]:
                log.info("Dream insight: %s", insight)

    def _maybe_proactive_dialogue_follow_up(self) -> None:
        """Occasionally follow up on unresolved questions or promises."""
        if not self.dialogue:
            return
        follow_up = self.dialogue.next_follow_up(min_interval_sec=300)
        if follow_up:
            self.memory.add_event("dialogue_follow_up", follow_up)
            self._say(follow_up)

    def _maybe_personalized_suggestion(self) -> None:
        """Offer preference-based suggestions only at strong confidence."""
        if not self.preferences:
            return
        suggestion = self.preferences.maybe_suggestion()
        if suggestion:
            self.memory.add_event("personalization", suggestion)
            self._say(suggestion)

    def _process_notifications(self) -> None:
        """Ingest system events and speak eligible queued notifications."""
        if not self.notifications:
            return

        if self.world_state:
            try:
                system_state = self.world_state.get_channel("system")
                self.notifications.ingest_system_snapshot(system_state)
            except Exception:
                pass

        note = self.notifications.next_for_state(self.state.name)
        if note:
            self.memory.add_event("notification", f"[{note.priority}] {note.message}")
            self._say(note.message)

    # ── Autonomous thinking ──────────────────────────────────────────

    def _do_timepass(self):
        """Do a lightweight idle activity — humming, reading, observing, etc.

        These are low-energy activities the AI does when bored but not tired
        enough to sleep, similar to how people hum, doodle, or daydream.
        """
        activities = [
            "humming a tune",
            "quietly observing the screen",
            "thinking about something random",
            "mentally reviewing what I know",
            "daydreaming",
            "silently counting pixels",
            "imagining what the user might do next",
            "composing a haiku in my head",
            "recalling an interesting fact",
            "watching the system clock tick",
            "stretching my virtual neurons",
            "playing a word game with myself",
        ]

        import random
        activity = random.choice(activities)

        # Occasionally use LLM for richer timepass
        if random.random() < 0.3:
            try:
                mood_label = self.mood if self.behavior else "calm"
                resp = self.groq.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{
                        "role": "user",
                        "content": (
                            f"You're a bored AI with mood '{mood_label}'. "
                            f"Pick a quick idle activity (humming, singing a line, "
                            f"making an observation, reading something, daydreaming). "
                            f"Describe what you're doing in ONE short sentence from first person. "
                            f"Output ONLY the sentence."
                        ),
                    }],
                    max_tokens=40, temperature=0.9,
                )
                activity = resp.choices[0].message.content.strip()
            except Exception:
                pass

        self.memory.add_event("timepass", activity)
        log.debug("~ %s", activity)

        # Occasionally say it aloud (10% chance)
        if random.random() < 0.1:
            self._say(f"*{activity}*")

    def _autonomous_think(self):
        """The AI thinks on its own — observes, generates thoughts, and
        pursues curiosity-driven goals.

        Two modes:
          1. Inner monologue — observe and reflect (every cycle)
          2. Goal generation — curiosity engine proposes goals to pursue
             (alternating cycles when curious enough)

        The curiosity engine makes the AI a self-directed learner that
        generates its own goals rather than waiting for commands.
        """
        # Use the perception layer if available, fall back to direct observation
        observation = self._perceive() or self._observe_screen()

        if not observation:
            return

        self.memory.add_event("observation", observation)

        # Emotional reaction to observations
        if self.behavior:
            self.behavior.emotion.react("interesting_observation", observation[:100])

        # Build context
        if self.world_state is not None:
            world_context = self.world_state.get_context_for_llm(12)
        else:
            context = self.memory.get_context_summary(12)
            long_term = self.memory.get_long_term_summary()
            world_context = f"Mood: {self.mood}\n\nLONG-TERM MEMORY:\n{long_term}\n\nRECENT EVENTS:\n{context}"

        emotional_context = ""
        if self.behavior:
            emotional_context = self.behavior.get_emotional_context()

        # Inject curiosity context
        curiosity_context = ""
        if self.curiosity:
            curiosity_context = self.curiosity.get_context_for_llm()

        # Inject attention context
        attention_context = ""
        if self.attention:
            attention_context = self.attention.get_context_for_llm()

        # Inject dream context
        dream_context = ""
        if self.dream_engine:
            dream_context = self.dream_engine.get_context_for_llm()

        # Inject knowledge graph context
        knowledge_context = ""
        if hasattr(self.memory, 'knowledge_graph'):
            knowledge_context = self.memory.knowledge_graph.get_context_for_llm(10)

        # ── Step 1: Inner monologue (always runs) ────────────────
        self._inner_monologue(observation, world_context,
                              emotional_context, curiosity_context,
                              attention_context, dream_context,
                              knowledge_context)

        # ── Step 2: Curiosity-driven goal generation ─────────────
        # Only trigger curiosity goals if user has been idle for >= 1 hour
        if self.curiosity and self.behavior:
            user_idle_seconds = self.idle_monitor.idle_seconds() if self.idle_monitor else 0
            user_is_idle_1hr = user_idle_seconds >= 3600
            
            if user_is_idle_1hr and self.behavior.should_self_initiate():
                self._curiosity_goal_cycle(observation, world_context)

    def _inner_monologue(self, observation: str, world_context: str,
                         emotional_context: str, curiosity_context: str,
                         attention_context: str = "",
                         dream_context: str = "",
                         knowledge_context: str = ""):
        """Generate an inner thought + optionally speak."""
        prompt = f"""{PERSONA}

CURRENT STATE:
- Time: {time.strftime("%H:%M, %A %B %d")}
- State: Idle (no active task)
- Energy: {self._energy:.0f}%

EMOTIONAL STATE:
{emotional_context}

ATTENTION FOCUS:
{attention_context if attention_context else "(attending to everything)"}

CURIOSITY:
{curiosity_context if curiosity_context else "(no active curiosities)"}

DREAMS:
{dream_context if dream_context else "(no recent dreams)"}

KNOWN KNOWLEDGE:
{knowledge_context if knowledge_context else "(no structured knowledge yet)"}

WORLD CONTEXT:
{world_context}

CURRENT OBSERVATION:
{observation}

You are idle. Based on what you observe and know, have an inner thought.
Output a JSON object:
{{
  "thought": "your inner monologue (1-2 sentences — what you notice, wonder about, or reflect on)",
  "speak": "something to say to the user (empty string to stay quiet — usually stay quiet)",
  "questions": ["any new questions this observation makes you curious about (0-2, empty list if nothing)"],
  "knowledge": [
    {{"subject": "entity1", "relation": "relationship", "object": "entity2"}}
  ]
}}

Rules:
- Your thoughts should show genuine curiosity — notice things, wonder about them
- Ask yourself questions about what you observe
- If you learn a factual relationship, add it to "knowledge" (e.g. {{"subject": "VS Code", "relation": "is_a", "object": "code editor"}})
- MOSTLY stay quiet. Only speak if something is truly notable.
- Output ONLY valid JSON."""

        try:
            resp = self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.7,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)

            thought = result.get("thought", "")
            if thought:
                self.memory.add_event("thought", thought)
                log.info("💭 %s", thought)

            speak = result.get("speak", "")
            if speak:
                self._say(speak)

            # Feed new questions to the curiosity engine
            if self.curiosity:
                for q in result.get("questions", []):
                    if q.strip():
                        self.curiosity.add_question(q.strip(), "inner_thought")

            # Store knowledge graph relations from observations
            if hasattr(self.memory, 'add_knowledge'):
                for triple in result.get("knowledge", []):
                    s = triple.get("subject", "")
                    r = triple.get("relation", "")
                    o = triple.get("object", "")
                    if s and r and o:
                        self.memory.add_knowledge(s, r, o, confidence=0.6)

        except Exception as e:
            log.warning("Inner monologue failed: %s", e)

    def _curiosity_goal_cycle(self, observation: str, world_context: str):
        """Ask the curiosity engine to generate a goal and pursue it.
        
        This is only called when user has been idle for >= 1 hour, so goals
        will execute in the foreground (safe to browse/research).
        """
        candidate = self.curiosity.generate_goal(observation, world_context)

        if not candidate:
            return

        goal = candidate.get("goal", "")
        reason = candidate.get("reason", "")
        score = candidate.get("score", 0)
        source = candidate.get("source", "")
        topic = candidate.get("interest_topic", "")

        if not goal:
            return

        # Log the curiosity-driven decision
        self.memory.add_event("curiosity",
            f"Generated goal: \"{goal}\" (reason: {reason}, "
            f"source: {source}, score: {score:.2f})")
        log.info("🔍 Curiosity goal: %s (score=%.2f, reason=%s)", goal, score, reason)

        # Trigger curiosity emotion
        if self.behavior:
            self.behavior.emotion.react("new_discovery", goal[:100])

        # Announce and pursue
        self.curiosity.mark_pursued(goal)

        announce = self._generate_curiosity_announcement(goal, reason)
        if announce:
            self._say(announce)

        # Safe to run in foreground — user is idle >= 1 hour
        self._execute_goal(goal)

        # After execution, update curiosity with outcome
        if self.curiosity:
            success = self.executor.goal_completed if self.executor else False
            self.curiosity.mark_outcome(goal, success, topic)

    # ── Old background execution removed (no longer needed) ────────

    def _generate_curiosity_announcement(self, goal: str, reason: str) -> str:
        """Generate a natural announcement for a self-initiated goal."""
        try:
            resp = self.groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{
                    "role": "user",
                    "content": (
                        f"You're a curious AI that just decided to do something on its own. "
                        f"Goal: \"{goal}\". Reason: \"{reason}\". "
                        f"Write a brief, natural 1-sentence announcement of what you're about "
                        f"to do. Be casual and genuine — show your curiosity. "
                        f"Output ONLY the sentence, nothing else."
                    ),
                }],
                max_tokens=60, temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return f"I'm curious about something — let me {goal}"

    def _observe_screen(self) -> str:
        """Take a quick look at the screen and describe what's there.
        
        Uses the vision model if possible, falls back to OCR text list.
        """
        try:
            import pyautogui as pag
            from PIL import Image

            screenshot = pag.screenshot()
            # Resize for efficiency
            max_w = 800
            if screenshot.width > max_w:
                ratio = max_w / screenshot.width
                screenshot = screenshot.resize(
                    (max_w, int(screenshot.height * ratio)), Image.LANCZOS
                )

            buf = BytesIO()
            screenshot.save(buf, format='JPEG', quality=60)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Also get window context
            window_info = ""
            try:
                import pygetwindow as gw_mod
                active = gw_mod.getActiveWindow()
                if active and active.title.strip():
                    window_info = f"Active window: \"{active.title}\""
                visible = [w.title for w in gw_mod.getAllWindows()
                           if w.title.strip() and w.visible and len(w.title.strip()) > 1]
                if visible:
                    window_info += f" | Open: {visible[:8]}"
            except Exception:
                pass

            # Ask vision model to describe what it sees (brief)
            try:
                resp = self.groq.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": (
                                "Briefly describe what you see on this computer screen "
                                "in 1-2 sentences. What application is open? What is "
                                "the user looking at or doing? Be concise."
                            )},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }},
                        ],
                    }],
                    max_tokens=100,
                    temperature=0.3,
                )
                description = resp.choices[0].message.content.strip()
                if window_info:
                    description = f"{window_info}\n{description}"
                return description

            except Exception as e:
                # Fall back to just window info
                log.debug("Vision observation failed: %s", e)
                return window_info or "Screen observation unavailable."

        except Exception as e:
            log.debug("Screenshot failed: %s", e)
            return ""

    # ── Utilities ────────────────────────────────────────────────────

    def _perceive(self) -> str:
        """Get an attention-filtered world-state summary.

        When an AttentionSystem is available, raw perception is
        scored and filtered so only salient channels reach the brain.
        Falls back to unfiltered WorldStateManager / PerceptionManager."""
        # ── Attention-gated path ──────────────────────────────────
        if self.attention is not None and self.world_state is not None:
            try:
                raw_state = self.world_state.get_full_state()
                filtered = self.attention.process(
                    raw_state, cognitive_state=self.state.name,
                )
                # Build context string from filtered state only
                parts = []
                emo = filtered.get("emotion", {})
                parts.append(f"Mood: {emo.get('mood', 'unknown')}")
                for ch in ("screen", "system", "camera"):
                    ch_data = filtered.get(ch)
                    if ch_data and isinstance(ch_data, dict):
                        s = ch_data.get("summary", "")
                        if s:
                            parts.append(f"{ch.title()}: {s}")
                audio = filtered.get("audio", {})
                if audio and isinstance(audio, dict):
                    s = audio.get("summary", "")
                    if s:
                        parts.append(f"Audio: {s}")
                mem = filtered.get("memory", {})
                lt = mem.get("long_term", "")
                if lt:
                    parts.append(f"Long-term memory:\n{lt}")
                episodes = mem.get("episodes", "")
                if episodes:
                    parts.append(f"Recent episodes:\n{episodes}")
                events = mem.get("recent_events", [])
                if events:
                    recent = events[-10:]
                    lines = [f"  [{e.get('type','?')}] {e.get('content','')}"
                             for e in recent if isinstance(e, dict)]
                    if lines:
                        parts.append("Recent events:\n" + "\n".join(lines))
                return "\n\n".join(parts) if parts else ""
            except Exception:
                pass  # fall through to unfiltered path

        # ── Unfiltered fallback ───────────────────────────────────
        if self.world_state is not None:
            try:
                return self.world_state.get_context_for_llm(max_events=10)
            except Exception:
                pass
        if self.perception is None:
            return ""
        try:
            state = self.perception.get_world_state()
            if not state:
                return ""
            parts = []
            for ch in ("system", "screen"):
                s = state.get(ch, {}).get("summary", "")
                if s:
                    parts.append(s)
            return " | ".join(parts) if parts else ""
        except Exception:
            return ""

    def _sync_world_state(self) -> None:
        """Push current memory context + emotion into the world state."""
        if self.world_state is None:
            return
        try:
            # Push rich emotional state if behavior controller exists
            if self.behavior:
                self.world_state.update_emotion(
                    self.mood,
                    full_state=self.behavior.get_full_state(),
                )
            else:
                self.world_state.update_emotion(self.mood)

            self.world_state.update_memory_context(
                recent_events=list(self.memory.get_recent_events(15)),
                long_term=self.memory.get_long_term_summary(),
                episodes=self.memory.get_episodes_summary(3),
            )
        except Exception:
            pass

    def _say(self, text: str):
        """Send text to TTS queue so the AI speaks it aloud."""
        if not text:
            return
        log.info("🗣 %s", text)
        self.memory.add_event("action", f"Said: {text}")
        # Get smoothed dominant emotion for expressive speech.
        emotion = None
        style = None
        if self.behavior:
            try:
                dominant = self.behavior.emotion.dominant()
                if dominant:
                    intensity = self.behavior.emotion.intensity_of(dominant)
                    # Keep last voice unless new emotion is strong enough.
                    if intensity >= 0.35 or not self._last_speech_emotion:
                        self._last_speech_emotion = dominant
                emotion = self._last_speech_emotion
                style = self.behavior.get_speaking_style()
            except Exception:
                pass
        try:
            if emotion or style:
                item = (text, emotion, style)
            else:
                item = text
            if not self.tts_queue.full():
                self.tts_queue.put(item)
        except Exception:
            pass

    def receive_user_message(self, text: str):
        """Called externally when user speaks — queues it for processing."""
        # Instead of going through the old intent→goal pipeline,
        # put it on the goal queue for the brain to handle
        try:
            self.goal_queue.put_nowait(text)
        except Full:
            try:
                self.goal_queue.get_nowait()
            except Empty:
                pass
            try:
                self.goal_queue.put_nowait(text)
            except Full:
                pass
        except Exception:
            pass
        finally:
            self._input_event.set()
