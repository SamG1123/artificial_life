import image_processing, web_support, voice_recognition, language_processing
import os
import sys
from transformers import pipeline
import torch
import multiprocessing
import huggingface_hub
from threading import Thread, Event

class main():
    def __init__(self):
        self.eyes = image_processing.ObjectDetection()
        self.web = web_support.WebSupport()
        self.ears = voice_recognition.SpeechSupport()
        self.language = language_processing.LanguageProcessor()
    
    def thread_init(self, stop_event):
        self.vision_thread = Thread(target=self.eyes.camera_infer, args=(stop_event,))
        self.speech_thread = Thread(target=self.ears.listen, args=(stop_event,))

    
    def run(self):
        self.stop_event = Event()
        self.thread_init(self.stop_event)

        self.vision_thread.start()
        self.speech_thread.start()
        self.vision_thread.join()
        self.speech_thread.join()

if __name__ == "__main__":
    main = main()
    main.run()