import speech_recognition as r
import pyttsx3
import language_processing as lp
import image_processing as ip

class SpeechSupport:
    def __init__(self):
        self.recognizer = r.Recognizer()
        self.microphone = r.Microphone()
        self.engine = pyttsx3.init()
        self.language_processor = lp.LanguageProcessor()
        self.image_processor = ip.ObjectDetection()

    
    def listen(self):
        with self.microphone as source:
            audio = self.recognizer.listen(source)
        try:
            self.text = self.recognizer.recognize_google(audio)
            if self.language_processor.similarity_score(self.text, "scan this image") > 0.6:
                text = self.image_processor.ocr_infer(self.image_processor.frame_buffer[-1])
                print("OCR Result:", text)
            
        except r.UnknownValueError:
            return "Sorry, I did not understand that."
    