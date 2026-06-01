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

## Avatar Runtime (Desktop Overlay + VTube Studio Bridge)

The agent now includes an avatar runtime that starts automatically with `main.py`.

- Desktop overlay: transparent, always-on-top, optional click-through.
- VTube Studio bridge: WebSocket sender for expression/parameter updates.
- Lip movement: text-driven viseme updates synced to TTS speech lifecycle.

### Using The Default VTube Studio Model

You do not need to download a custom model yet.

1. Install and open VTube Studio.
2. Load the built-in sample/default model inside VTube Studio.
3. Start this agent with `AVATAR_VTS_ENABLED=1`.
4. When VTube Studio asks to allow the plugin, approve it.

The current bridge will drive whichever model is currently loaded in VTube Studio.
By default it only uses generic parameters such as mouth open, mouth form, smile, eye open, and angle, so custom expression files are not required.

### Quick controls

- Right-click the overlay to toggle interactive mode vs click-through mode.
- Global toggle hotkey: `Ctrl+Alt+V` (Windows).
- Keep VTube Studio running on `ws://127.0.0.1:8001` (default).
- On first connect, VTube Studio may ask to allow plugin authentication.
- Toggle interaction mode ON, then drag the avatar with left mouse to reposition.

### Environment flags

Set these before launch (PowerShell example):

```powershell
$env:AVATAR_ENABLED = "1"
$env:AVATAR_OVERLAY_ENABLED = "1"
$env:AVATAR_VTS_ENABLED = "1"
$env:AVATAR_OVERLAY_TOPMOST = "1"
$env:AVATAR_OVERLAY_CLICK_THROUGH = "1"
$env:AVATAR_ASSET_DIR = "avatar_assets"
$env:AVATAR_HOTKEY_TOGGLE = "CTRL+ALT+V"
$env:AVATAR_OVERLAY_STATE_PATH = "memory_store/avatar_overlay_state.json"
$env:AVATAR_LOW_FPS = "30"
python main.py
```

Useful optional settings:

- `AVATAR_OVERLAY_SIZE` (default `300`)
- `AVATAR_OVERLAY_X` / `AVATAR_OVERLAY_Y` (default `40`/`40`)
- `AVATAR_ASSET_DIR` (default `avatar_assets`)
- `AVATAR_HOTKEY_TOGGLE` (default `CTRL+ALT+V`)
- `AVATAR_OVERLAY_STATE_PATH` (default `memory_store/avatar_overlay_state.json`)
- `AVATAR_VTS_HOST` / `AVATAR_VTS_PORT` (default `127.0.0.1`/`8001`)
- `AVATAR_VTS_EXPRESSION_MAP` (optional JSON map: emotion -> expression file)

Example expression map in PowerShell:

```powershell
$env:AVATAR_VTS_EXPRESSION_MAP = '{"amusement":"amused.exp3.json","frustration":"angry.exp3.json"}'
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


