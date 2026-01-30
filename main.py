import image_processing, web_support, voice_recognition, language_processing, tts
from threading import Thread, Event
from queue import Queue

class main():
    def __init__(self):
        self.eyes = image_processing.ObjectDetection()
        self.web = web_support.WebSupport()
        self.ears = voice_recognition.SpeechSupport()
        self.language = language_processing.LanguageProcessor()
        self.tts = tts.TextToSpeech()
    
    def thread_init(self, stop_event):
        self.speech_thread = Thread(target=self.ears.listen, args=(stop_event,))
        self.tts_thread = Thread(target=self.tts.text_to_speech_loop, args=(stop_event,))

    
    def run(self):
        self.stop_event = Event()
        self.thread_init(self.stop_event)

        self.speech_thread.start()
        self.tts_thread.start()

        # OpenCV windows are most reliable on the main thread (especially on Windows).
        self.eyes.camera_infer(self.stop_event)

        self.stop_event.set()
        self.speech_thread.join()
        self.tts_thread.join()
        

if __name__ == "__main__":
    main = main()
    main.run()