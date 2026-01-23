import image_processing, web_support, speech_support, language_processing
import os
import sys
from transformers import pipeline
import torch
import multiprocessing
import huggingface_hub

class main():
    def __init__(self):
        self.eyes = image_processing.ObjectDetection()
        self.web = web_support.WebSupport()
        #self.speech = speech_support.SpeechSupport()
        self.language = language_processing.LanguageProcessor()
    
    def models_init(self):
        self.text_generator = "Qwen/Qwen2.5-3B-Instruct"
        self.device = 0 if torch.cuda.is_available() else -1
        self.text_generator_pipeline = pipeline("text-generation", model=self.text_generator, device=self.device)

    def run(self):
        #self.models_init()
        self.eyes.camera_infer()
        #self.speech.listen()

if __name__ == "__main__":
    main = main()
    main.run()