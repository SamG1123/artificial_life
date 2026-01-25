import speech_recognition as r
import pyttsx3
import language_processing as lp
import image_processing as ip
from threading import Thread, Event

class SpeechSupport:
    def __init__(self):
        self.recognizer = r.Recognizer()
        self.microphone = r.Microphone()
        self.engine = pyttsx3.init()
        self.language_processor = lp.LanguageProcessor()
        self.image_processor = ip.ObjectDetection()

    
    def listen(self, stop_event: Event):
        while not stop_event.is_set():
            with self.microphone as source:
                print("Listening...")
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                except r.WaitTimeoutError:
                    continue
            try:
                command = self.recognizer.recognize_google(audio)
                print(f"Recognized command: {command}")
                if command.lower() == "stop":
                    stop_event.set()
                    break

                #self.process_command(command)
            except r.UnknownValueError:
                print("Could not understand audio")
            except r.RequestError as e:
                print(f"Could not request results; {e}")
    
    def process_command(self, command: str):
        response = self.language_processor.generate_response(command)
        print(f"Response: {response}")
        self.speak(response)
            
