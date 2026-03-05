"""
Desktop GUI for the AI Agent — tkinter-based chat interface with
live camera feed, status panel, and TTS integration.

Usage:
    python gui.py
"""

import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import time
import queue
import cv2
from PIL import Image, ImageTk

from config import global_command_queue, global_goal_queue


# ── Colours / theme ──────────────────────────────────────────────
BG           = "#1e1e2e"
BG_DARKER    = "#181825"
BG_LIGHTER   = "#313244"
FG           = "#cdd6f4"
FG_DIM       = "#6c7086"
ACCENT       = "#89b4fa"
ACCENT2      = "#a6e3a1"
USER_BUBBLE  = "#45475a"
AGENT_BUBBLE = "#313244"
INPUT_BG     = "#45475a"
BORDER       = "#585b70"
FONT_FAMILY  = "Segoe UI"


class AgentGUI:
    """Tkinter GUI that wraps the AgentController."""

    def __init__(self):
        # ── Build the AgentController (all subsystems) ────────────
        from agent import AgentController
        self.agent = AgentController()

        # Queue for GUI-bound messages (brain speech, thoughts, status)
        self._gui_queue: queue.Queue[tuple[str, str]] = queue.Queue()

        # Monkey-patch brain._say to also push to the GUI queue
        self._original_say = self.agent.brain._say
        self.agent.brain._say = self._hooked_say

        # ── Root window ──────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("✦ AI Agent")
        self.root.geometry("1100x720")
        self.root.configure(bg=BG)
        self.root.minsize(800, 500)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Layout: left panel (chat) + right panel (camera + status)
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        self._build_chat_panel()
        self._build_right_panel()

        # Camera state
        self._cap = None
        self._camera_running = False
        self._camera_photo = None  # prevent GC of PhotoImage

        # Periodic GUI updater
        self.root.after(100, self._poll_gui_queue)
        self.root.after(2000, self._update_status)

    # ── Chat panel (left) ────────────────────────────────────────

    def _build_chat_panel(self):
        frame = tk.Frame(self.root, bg=BG)
        frame.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        # Title
        title = tk.Label(frame, text="💬 Chat", font=(FONT_FAMILY, 14, "bold"),
                         bg=BG, fg=ACCENT)
        title.grid(row=0, column=0, sticky="w", pady=(0, 6))

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD, state=tk.DISABLED,
            bg=BG_DARKER, fg=FG, font=(FONT_FAMILY, 11),
            relief=tk.FLAT, borderwidth=0, padx=12, pady=8,
            insertbackground=FG, selectbackground=ACCENT,
        )
        self.chat_display.grid(row=1, column=0, sticky="nsew")

        # Configure text tags for styled messages
        self.chat_display.tag_configure("user_name", foreground=ACCENT, font=(FONT_FAMILY, 10, "bold"))
        self.chat_display.tag_configure("agent_name", foreground=ACCENT2, font=(FONT_FAMILY, 10, "bold"))
        self.chat_display.tag_configure("user_msg", foreground=FG, font=(FONT_FAMILY, 11))
        self.chat_display.tag_configure("agent_msg", foreground=FG, font=(FONT_FAMILY, 11))
        self.chat_display.tag_configure("thought", foreground=FG_DIM, font=(FONT_FAMILY, 10, "italic"))
        self.chat_display.tag_configure("system", foreground=FG_DIM, font=(FONT_FAMILY, 9))
        self.chat_display.tag_configure("timestamp", foreground=FG_DIM, font=(FONT_FAMILY, 8))

        # Input area
        input_frame = tk.Frame(frame, bg=BG)
        input_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        input_frame.columnconfigure(0, weight=1)

        self.input_entry = tk.Entry(
            input_frame, bg=INPUT_BG, fg=FG, font=(FONT_FAMILY, 12),
            relief=tk.FLAT, insertbackground=FG, borderwidth=8,
        )
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.input_entry.bind("<Return>", self._on_send)
        self.input_entry.insert(0, "Type a message…")
        self.input_entry.bind("<FocusIn>", self._on_entry_focus_in)
        self.input_entry.bind("<FocusOut>", self._on_entry_focus_out)

        send_btn = tk.Button(
            input_frame, text="Send", bg=ACCENT, fg=BG_DARKER,
            font=(FONT_FAMILY, 11, "bold"), relief=tk.FLAT,
            activebackground="#b4d0fb", cursor="hand2",
            command=lambda: self._on_send(None), padx=16, pady=4,
        )
        send_btn.grid(row=0, column=1)

    # ── Right panel (camera + status) ────────────────────────────

    def _build_right_panel(self):
        frame = tk.Frame(self.root, bg=BG)
        frame.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        # ── Camera section ────────────────────────────────────────
        cam_title = tk.Label(frame, text="📷 Camera", font=(FONT_FAMILY, 14, "bold"),
                             bg=BG, fg=ACCENT)
        cam_title.grid(row=0, column=0, sticky="w", pady=(0, 4))

        self.camera_label = tk.Label(
            frame, bg=BG_DARKER, relief=tk.FLAT,
            text="Camera off", fg=FG_DIM, font=(FONT_FAMILY, 10),
        )
        self.camera_label.grid(row=1, column=0, sticky="nsew", pady=(0, 8))

        # Camera toggle button
        self.cam_btn = tk.Button(
            frame, text="▶ Start Camera", bg=BG_LIGHTER, fg=FG,
            font=(FONT_FAMILY, 10), relief=tk.FLAT, cursor="hand2",
            command=self._toggle_camera, padx=10, pady=4,
        )
        self.cam_btn.grid(row=2, column=0, sticky="ew", pady=(0, 12))

        # ── Status section ────────────────────────────────────────
        status_title = tk.Label(frame, text="⚡ Status", font=(FONT_FAMILY, 14, "bold"),
                                bg=BG, fg=ACCENT)
        status_title.grid(row=3, column=0, sticky="w", pady=(0, 4))

        self.status_frame = tk.Frame(frame, bg=BG_DARKER, relief=tk.FLAT)
        self.status_frame.grid(row=4, column=0, sticky="nsew")
        frame.rowconfigure(4, weight=0)

        self._status_labels = {}
        for i, key in enumerate(["State", "Mood", "Emotion", "Goal", "Listening"]):
            lbl_key = tk.Label(self.status_frame, text=f"{key}:", bg=BG_DARKER,
                               fg=FG_DIM, font=(FONT_FAMILY, 10, "bold"),
                               anchor="w", padx=10, pady=3)
            lbl_key.grid(row=i, column=0, sticky="w")
            lbl_val = tk.Label(self.status_frame, text="—", bg=BG_DARKER,
                               fg=FG, font=(FONT_FAMILY, 10),
                               anchor="w", padx=6, pady=3)
            lbl_val.grid(row=i, column=1, sticky="w")
            self._status_labels[key] = lbl_val

        # Mic toggle
        self.mic_btn = tk.Button(
            frame, text="🎤 Wake Up", bg=ACCENT2, fg=BG_DARKER,
            font=(FONT_FAMILY, 10, "bold"), relief=tk.FLAT, cursor="hand2",
            command=self._toggle_mic, padx=10, pady=4,
        )
        self.mic_btn.grid(row=5, column=0, sticky="ew", pady=(8, 0))

    # ── Message handling ─────────────────────────────────────────

    def _on_send(self, event):
        text = self.input_entry.get().strip()
        if not text or text == "Type a message…":
            return
        self.input_entry.delete(0, tk.END)
        self._append_chat("You", text, "user")
        # Send to the brain
        self.agent.brain.receive_user_message(text)

    def _hooked_say(self, text: str):
        """Intercept brain._say — push to GUI queue AND original TTS."""
        if text:
            self._gui_queue.put(("agent", text))
        self._original_say(text)

    def _append_chat(self, sender: str, text: str, role: str):
        """Add a message to the chat display."""
        self.chat_display.configure(state=tk.NORMAL)
        ts = time.strftime("%H:%M")

        if role == "user":
            self.chat_display.insert(tk.END, f"\n {ts}  ", "timestamp")
            self.chat_display.insert(tk.END, f"{sender}\n", "user_name")
            self.chat_display.insert(tk.END, f"  {text}\n", "user_msg")
        elif role == "agent":
            self.chat_display.insert(tk.END, f"\n {ts}  ", "timestamp")
            self.chat_display.insert(tk.END, f"Agent\n", "agent_name")
            self.chat_display.insert(tk.END, f"  {text}\n", "agent_msg")
        elif role == "thought":
            self.chat_display.insert(tk.END, f"  💭 {text}\n", "thought")
        else:
            self.chat_display.insert(tk.END, f"  [{sender}] {text}\n", "system")

        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _poll_gui_queue(self):
        """Drain the GUI queue and render messages."""
        try:
            while True:
                role, text = self._gui_queue.get_nowait()
                self._append_chat("Agent", text, role)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_gui_queue)

    # ── Placeholder handling ─────────────────────────────────────

    def _on_entry_focus_in(self, event):
        if self.input_entry.get() == "Type a message…":
            self.input_entry.delete(0, tk.END)
            self.input_entry.configure(fg=FG)

    def _on_entry_focus_out(self, event):
        if not self.input_entry.get().strip():
            self.input_entry.insert(0, "Type a message…")
            self.input_entry.configure(fg=FG_DIM)

    # ── Status updater ───────────────────────────────────────────

    def _update_status(self):
        try:
            self._status_labels["State"].configure(text=self.agent.brain.state.name)
            self._status_labels["Mood"].configure(text=self.agent.brain.mood)
            goal = self.agent.brain.current_goal or "None"
            self._status_labels["Goal"].configure(
                text=goal[:40] + ("…" if len(goal) > 40 else ""))
            listening = "Awake" if not self.agent.ears.sleep else "Sleeping"
            self._status_labels["Listening"].configure(text=listening)
            try:
                emo = self.agent.behavior.emotion.dominant_emotion()
                self._status_labels["Emotion"].configure(text=emo or "neutral")
            except Exception:
                self._status_labels["Emotion"].configure(text="—")
            # Mic button text
            self.mic_btn.configure(
                text="🎤 Wake Up" if self.agent.ears.sleep else "💤 Sleep",
                bg=ACCENT2 if self.agent.ears.sleep else BG_LIGHTER,
            )
        except Exception:
            pass
        self.root.after(2000, self._update_status)

    # ── Camera ───────────────────────────────────────────────────

    def _toggle_camera(self):
        if self._camera_running:
            self._camera_running = False
            self.cam_btn.configure(text="▶ Start Camera")
            self.camera_label.configure(image="", text="Camera off")
        else:
            self._camera_running = True
            self.cam_btn.configure(text="⏹ Stop Camera")
            threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._camera_running = False
            self.cam_btn.configure(text="▶ Start Camera")
            self.camera_label.configure(text="Camera not available")
            return

        while self._camera_running and not self.agent._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection
            try:
                boxes = self.agent.eyes.get_boxes(frame)
                labels = self.agent.eyes.get_labels(frame)
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if i < len(labels):
                        cv2.putText(frame, labels[i], (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
            except Exception:
                pass

            # Convert BGR → RGB → PIL → PhotoImage
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            # Fit to label size
            lbl_w = self.camera_label.winfo_width()
            lbl_h = self.camera_label.winfo_height()
            if lbl_w > 20 and lbl_h > 20:
                img.thumbnail((lbl_w, lbl_h), Image.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            self._camera_photo = photo  # prevent GC
            try:
                self.camera_label.configure(image=photo, text="")
            except tk.TclError:
                break  # window closed

            time.sleep(0.033)  # ~30 fps

        cap.release()
        self._camera_running = False

    # ── Mic toggle ───────────────────────────────────────────────

    def _toggle_mic(self):
        self.agent.ears.sleep = not self.agent.ears.sleep

    # ── Lifecycle ────────────────────────────────────────────────

    def _on_close(self):
        self._camera_running = False
        self.agent.stop()
        self.root.destroy()

    def run(self):
        """Start the agent (background threads) and the GUI (main thread)."""
        # Start agent subsystems (speech, TTS, brain, perception)
        self.agent.start()

        # Welcome message
        self._append_chat("System", "Agent started. All subsystems online.", "system")
        self._append_chat("Agent", "I'm awake. What's going on?", "agent")

        # Tkinter main loop (must run on main thread)
        self.root.mainloop()

        # After GUI closes, clean up
        self.agent.stop()


if __name__ == "__main__":
    app = AgentGUI()
    app.run()
