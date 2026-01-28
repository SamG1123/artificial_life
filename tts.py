import pyttsx3
from threading import Thread, Event
from config import global_command_queue

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
    
    def text_to_speech_loop(self, stop_event: Event):
        while not stop_event.is_set():
            if not global_command_queue.empty():
                text = global_command_queue.get()
                self.speak(text)
                global_command_queue.task_done()

    def speak(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()