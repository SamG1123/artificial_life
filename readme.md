# Artificial Life

A project on developing an all rounder AI and actually trying to have its own thought process and be usable for future data and not just past.

## Overview

This repository contains implementations of artificial life simulations, including agent-based models and evolutionary systems while also dividing and modeling each component based on how living organisms function.

## Features

- Agent-based simulation framework
- Environmental interaction systems
- Evolutionary algorithms
- Data visualization tools

## Getting Started

### Prerequisites
- Python 3.8+
- Required dependencies listed in `requirements.txt`

### Installation

```bash
git clone <repository-url>
cd artificial_life
pip install -r requirements.txt
```

### Usage

```bash
python main.py
```

Text-only test mode (no voice input):

```bash
python text_mode.py
```

## Current System Architecture

This is the current system architecture on which this is being built, it is subjected to change as more features and improvements are done in future.

```mermaid
graph TB
    subgraph IO["Layer 1: Input/Output"]
        EYES["Eyes<br/>ObjectDetection"]
        EARS["Ears<br/>SpeechRecognition"]
        MOUTH["Mouth<br/>TextToSpeech"]
        WEB["Web Support"]
    end

    subgraph PERCEPTION["Layer 2: Perception"]
        SCREEN["Screen Perception<br/>OCR + Diff Detection"]
        CAMERA["Camera Perception<br/>Face/Object Detection"]
        AUDIO["Audio Perception<br/>Sound Analysis"]
        SYSTEM["System Perception<br/>CPU/Memory/Battery"]
        IDLE["Idle Monitor<br/>User Activity Tracking"]
        PM["PerceptionManager<br/>Coordinator"]
    end

    subgraph MEMORY_LAYER["Layer 3: Memory"]
        ST["Short-term<br/>Recent Events"]
        EPISODIC["Episodic<br/>Goal Outcomes"]
        SEMANTIC["Semantic<br/>Facts & Knowledge"]
        VECTOR["Vector Memory<br/>Embeddings"]
        MEMORY["MemorySystem<br/>Coordinator"]
    end

    subgraph UNIFIED["Layer 4: Unified World Model"]
        WSM["WorldStateManager<br/>Unified Reality"]
    end

    subgraph EMOTION["Layer 5: Emotion & Personality"]
        PERSONALITY["Personality Model"]
        MOOD["Mood Engine"]
        EMOTION_ENG["Emotion Engine"]
        BEHAVIOR["BehaviorController"]
    end

    subgraph REASONING["Layer 6: Reasoning & Planning"]
        REASONER["ReasoningEngine<br/>Goal Planning"]
        PLANNER["Hierarchical Planner"]
        POLICIES["Policies"]
    end

    subgraph LEARNING["Layer 7: Learning & Self-Improvement"]
        EXP_LOG["ExperienceLogger<br/>Goal Outcomes"]
        DATASET["DatasetBuilder"]
        TRAINER["ModelTrainer"]
        COMPRESSOR["MemoryCompressor<br/>Long-term Consolidation"]
        SELF_IMPROVE["SelfImprover<br/>Strategy Learning"]
        SKILL_GRAPH["SkillGraph<br/>Capability Tracking"]
        REWARD["RewardEngine<br/>Multi-dim Signals"]
        NIGHTLY["NightlyTrainer<br/>Sleep Learning"]
    end

    subgraph COGNITION["Layer 8: Cognition & Autonomy"]
        CURIOSITY["CuriosityEngine<br/>Goal Generation"]
        ATTENTION["AttentionSystem<br/>Focus Management"]
        DREAM["DreamEngine<br/>Memory Consolidation"]
        DIALOGUE["DialogueStateTracker<br/>Conversation"]
        NOTIF["NotificationEngine"]
        PREF["PreferenceLearner"]
        BRAIN["CognitiveBrain<br/>Main Loop"]
    end

    subgraph ACTION["Layer 9: Action Execution"]
        EXECUTOR["AutomationExecutor<br/>Browser + Desktop"]
        BACKGROUND["BackgroundTaskManager<br/>Async Goals"]
    end

    subgraph RUNTIME["Layer 10: Runtime & Persistence"]
        CHECKPOINT["StateCheckpoint<br/>Brain State"]
        HEALTH["HealthMonitor<br/>System Health"]
        AVATAR["Avatar Runtime<br/>VTube Studio Bridge"]
    end

    %% Data Flow
    EYES --> PM
    EARS --> PM
    CAMERA --> PM
    AUDIO --> PM
    SYSTEM --> PM
    IDLE --> PM
    WEB --> PM

    PM --> SCREEN
    PM --> CAMERA
    PM --> AUDIO
    PM --> SYSTEM

    SCREEN --> WSM
    CAMERA --> WSM
    AUDIO --> WSM
    SYSTEM --> WSM

    MEMORY --> WSM
    ST --> MEMORY
    EPISODIC --> MEMORY
    SEMANTIC --> MEMORY
    VECTOR --> MEMORY

    WSM --> BRAIN
    EMOTION --> BRAIN
    REASONING --> BRAIN
    BEHAVIOR --> BRAIN
    CURIOSITY --> BRAIN
    ATTENTION --> BRAIN
    DREAM --> BRAIN

    PERSONALITY --> MOOD
    MOOD --> EMOTION_ENG
    EMOTION_ENG --> BEHAVIOR

    BRAIN --> EXECUTOR
    BRAIN --> BACKGROUND
    EXECUTOR --> EXP_LOG
    BACKGROUND --> EXP_LOG

    EXP_LOG --> DATASET
    EXP_LOG --> COMPRESSOR
    DATASET --> TRAINER
    TRAINER --> SELF_IMPROVE
    SELF_IMPROVE --> REWARD
    REWARD --> SKILL_GRAPH
    SKILL_GRAPH --> CURIOSITY
    MEMORY --> DREAM
    DREAM --> NIGHTLY
    NIGHTLY --> MEMORY

    BRAIN --> DIALOGUE
    BRAIN --> NOTIF
    BRAIN --> PREF
    BRAIN --> MOUTH

    BRAIN --> CHECKPOINT
    BRAIN --> HEALTH
    BRAIN --> AVATAR
```

## Idle-Aware Behavior & Background Tasks

The agent respects your activity and won't interrupt your work with foreground browsing or information gathering.

### How it works

1. **User Activity Monitoring** (Windows only)
   - The agent tracks keyboard + mouse input to determine your activity level
   - Idle threshold: **1 hour of no input** = user is idle

2. **Autonomous Work Scheduling**
   - When the agent wants to explore or research something (**curiosity goals**):
     - **If you're active** (< 1 hour idle): work happens in **background threads** — non-blocking
     - **If you're idle** (≥ 1 hour): work happens in the **foreground** — agent can browse, collect info, etc.
   - **User instructions** and **scheduled tasks** always execute as directed (background or foreground)

3. **Task Management**
   - Background tasks are tracked and can be monitored via `/status` in text mode
   - Tasks auto-clean up after completion
   - On shutdown, all background tasks are cancelled gracefully

### Checking agent activity

```bash
python text_mode.py
you> /status
```

Output will show:
- `idle_monitor`: user idle time in seconds, is user idle >= 1 hour
- `background_tasks`: list of active background curiosity goals being explored

### Background task execution details

When the agent runs a task in background:
- Goal executes on a separate thread pool
- Main brain loop stays responsive
- Skill learning and memory updates still occur
- You can interact with the agent or give new instructions without blocking


