import speech_recognition as r
import pyttsx3


class SpeechSupport:
    def __init__(self):
        self.recognizer = r.Recognizer()
        self.microphone = r.Microphone()
        self.engine = pyttsx3.init()
    
    def listen(self):
        with self.microphone as source:
            audio = self.recognizer.listen(source)
        try:
            self.text = self.recognizer.recognize_google(audio)
            return self.text
        except r.UnknownValueError:
            return "Sorry, I did not understand that."