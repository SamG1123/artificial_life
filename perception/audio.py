"""
Audio Perception — perceives sound (primarily speech).

Responsibilities:
  * Listen for audio input via the microphone
  * Recognise speech and produce transcripts
  * Detect audio events (silence, speech, noise)
  * Output structured audio snapshots

Wraps the existing `voice_recognition.SpeechSupport` recogniser —
does NOT duplicate device handling or speech-recognition logic.
"""

import time
import threading

import speech_recognition as sr

from .base import PerceptionModule, PerceptionData


class AudioPerception(PerceptionModule):
    """Perceives the environment via microphone audio."""

    def __init__(self, recogniser=None, microphone=None):
        """
        Args:
            recogniser: an existing speech_recognition.Recognizer instance
                        (reused from SpeechSupport). Created if None.
            microphone: an existing speech_recognition.Microphone instance.
                        Created if None.
        """
        self._recogniser = recogniser or sr.Recognizer()
        self._microphone = microphone or sr.Microphone()
        self._lock = threading.Lock()
        self._last_transcript = None
        self._last_transcript_time = 0.0

    # ── PerceptionModule interface ───────────────────────────────

    def initialise(self) -> None:
        # Adjust for ambient noise once at startup
        try:
            with self._microphone as source:
                self._recogniser.adjust_for_ambient_noise(source, duration=0.5)
        except Exception:
            pass

    def capture(self) -> dict:
        """Listen for a short utterance and return structured data.

        Non-blocking by design: uses a short timeout so the perception
        loop doesn't stall.
        """
        transcript = None
        confidence = 0.0
        event_type = "silence"

        try:
            with self._microphone as source:
                audio = self._recogniser.listen(
                    source, timeout=1, phrase_time_limit=5
                )
            transcript = self._recogniser.recognize_google(audio)
            confidence = 0.9
            event_type = "speech"

            with self._lock:
                self._last_transcript = transcript
                self._last_transcript_time = time.time()

        except sr.WaitTimeoutError:
            event_type = "silence"
        except sr.UnknownValueError:
            event_type = "noise"
        except sr.RequestError as e:
            event_type = "error"
            transcript = str(e)
            confidence = 0.0

        data = {
            "event": event_type,
            "transcript": transcript,
        }

        summary = ""
        if event_type == "speech" and transcript:
            summary = f'Heard: "{transcript}"'
        elif event_type == "noise":
            summary = "Audio detected but not recognised"
        elif event_type == "error":
            summary = f"Recognition error: {transcript}"

        return PerceptionData(
            source="audio",
            data=data,
            confidence=confidence,
            summary=summary,
        )

    def shutdown(self) -> None:
        pass  # microphone is auto-released by context manager

    # ── Extra helpers ────────────────────────────────────────────

    def feed_transcript(self, text: str):
        """Accept a transcript from an external listener (e.g. the
        existing SpeechSupport.listen thread) so the perception layer
        stays in sync without running its own mic loop."""
        with self._lock:
            self._last_transcript = text
            self._last_transcript_time = time.time()

    def get_last_transcript(self) -> dict | None:
        """Return the most recent transcript without re-listening."""
        with self._lock:
            if self._last_transcript is None:
                return None
            return {
                "transcript": self._last_transcript,
                "timestamp": self._last_transcript_time,
            }
