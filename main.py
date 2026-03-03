import image_processing, web_support, voice_recognition, language_processing, tts
from automation.executor import AutomationExecutor
from threading import Thread, Event
from queue import Queue, Empty
from config import global_goal_queue, global_command_queue


def executor_loop(executor: AutomationExecutor, stop_event: Event):
    """Background thread: pulls goals from the queue and executes them."""
    while not stop_event.is_set():
        try:
            goal = global_goal_queue.get(timeout=0.5)
        except Empty:
            continue

        print(f"\n[ExecutorThread] Received goal: {goal}")
        try:
            executor.execute_goal(goal)
            result_msg = f"Done: {goal}"
        except Exception as e:
            result_msg = f"Failed to execute '{goal}': {e}"
            print(f"[ExecutorThread] Error: {e}")

        # Send result to TTS so the user hears feedback
        if not global_command_queue.full():
            global_command_queue.put(result_msg)

        global_goal_queue.task_done()


class Main:
    def __init__(self):
        self.eyes = image_processing.ObjectDetection()
        self.web = web_support.WebSupport()
        self.ears = voice_recognition.SpeechSupport()
        self.language = language_processing.LanguageProcessor()
        self.tts = tts.TextToSpeech()
        self.executor = AutomationExecutor()

    def thread_init(self, stop_event):
        self.speech_thread = Thread(target=self.ears.listen, args=(stop_event,), daemon=True)
        self.tts_thread = Thread(target=self.tts.text_to_speech_loop, args=(stop_event,), daemon=True)
        self.executor_thread = Thread(target=executor_loop, args=(self.executor, stop_event), daemon=True)

    def run(self):
        self.stop_event = Event()
        self.thread_init(self.stop_event)

        self.speech_thread.start()
        self.tts_thread.start()
        self.executor_thread.start()

        # OpenCV windows are most reliable on the main thread (especially on Windows).
        self.eyes.camera_infer(self.stop_event)

        self.stop_event.set()
        self.speech_thread.join(timeout=3)
        self.tts_thread.join(timeout=3)
        self.executor_thread.join(timeout=3)


if __name__ == "__main__":
    app = Main()
    app.run()