import image_processing, web_support
import os
import sys
from transformers import pipeline
import torch
import huggingface_hub

class main():
    def __init__(self):
        self.eyes = image_processing.ObjectDetection()
        self.web = web_support.WebSupport()
    
    def models_init(self):
        self.text_generator = "Qwen/Qwen2.5-3B-Instruct"
        self.device = 0 if torch.cuda.is_available() else -1
        self.text_generator_pipeline = pipeline("text-generation", model=self.text_generator, device=self.device)

    def run(self):
        self.models_init()
        while True:
            

if __name__ == "__main__":
    main = main()