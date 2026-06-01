import speech_recognition as r
import pyttsx3
import language_processing as lp
import image_processing as ip
from threading import Thread, Event
from config import global_command_queue

class SpeechSupport:
    def __init__(self):
        self.recognizer = r.Recognizer()
        self.microphone = r.Microphone()
        self.engine = pyttsx3.init()
        self.language_processor = lp.LanguageProcessor()
        self.image_processor = ip.ObjectDetection()
        self.wake_word = "hello assistant"
        self.sleep_word = "sleep"
        self.sleep = True
        self.brain = None  # Set by Main after construction

    
    def listen(self, stop_event: Event):
        # Calibrate for ambient noise once at startup
        with self.microphone as source:
            print("[SpeechSupport] Calibrating for ambient noise…")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print(f"[SpeechSupport] Energy threshold set to {self.recognizer.energy_threshold:.0f}")

        # Allow dynamic threshold adjustment to adapt over time
        self.recognizer.dynamic_energy_threshold = True
        last_mode = None

        while not stop_event.is_set():
            mode = "sleep" if self.sleep else "awake"
            if mode != last_mode:
                if self.sleep:
                    print("[SpeechSupport] Waiting for wake word...")
                else:
                    print("[SpeechSupport] Listening...")
                last_mode = mode

            with self.microphone as source:
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                except r.WaitTimeoutError:
                    continue
            try:
                command = self.recognizer.recognize_google(audio)
                print(f"Recognized: {command}")

                if self.sleep:
                    # Sleeping — only listen for the wake word
                    if self.wake_word in command.lower():
                        self.sleep = False
                        print("[SpeechSupport] Woke up!")
                        # Strip the wake word and process the rest (if any)
                        remainder = command.lower().replace(self.wake_word, "").strip()
                        if remainder:
                            self.process_command(remainder)
                    continue

                # Awake — process normally
                if command.lower().strip() == "stop":
                    stop_event.set()
                    break

                if self.sleep_word in command.lower():
                    self.sleep = True
                    print("[SpeechSupport] Going to sleep.")
                    continue

                self.process_command(command)

            except r.UnknownValueError:
                pass  # silence — no speech detected
            except r.RequestError as e:
                print(f"Could not request results; {e}")
    
    def process_command(self, command: str):
        """Route the command through the cognitive brain if available,
        otherwise fall back to the old intent classifier."""
        if self.brain is not None:
            # New path: send everything to the brain's cognitive loop
            print(f"[SpeechSupport] -> Brain: {command}")
            self.brain.receive_user_message(command)
        else:
            # Legacy fallback: use intent classifier
            response = self.language_processor.process_text(command)
            print(f"[SpeechSupport] Response: {response}")
            if response and not global_command_queue.full():
                global_command_queue.put(response)
    
