import nltk
import spacy
import textblob as tb
from transformers import pipeline
from image_processing import ObjectDetection
from config import global_command_queue, global_goal_queue

class LanguageProcessor:
    def __init__(self):
        self.vision = ObjectDetection()
        self.classifier = pipeline("text-classification", model="./intent_model")

    def classify_intent(self, text: str) -> str:
        """Return the intent label string, e.g. 'PC_CONTROL', 'CHAT', 'VISION_QUERY'."""
        result = self.classifier(text)
        # pipeline returns [{"label": "PC_CONTROL", "score": 0.99}]
        return result[0]["label"]
    
    def classify_entity(self, text: str):
        pass

    def tokenize_text(self, text: str):
        tokens = nltk.word_tokenize(text)
        return tokens
    
    def sentiment_analysis(self, text: str):
        blob = tb.TextBlob(text)
        sentiment = blob.sentiment
        return sentiment
    
    def remove_stopwords(self, text: str):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        filtered_tokens = [token.text for token in doc if not token.is_stop]
        return filtered_tokens
    
    def lemmatize_text(self, text: str):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        lemmatized_tokens = [token.lemma_ for token in doc]
        return lemmatized_tokens
    
    def similarity_score(self, text1: str, text2: str):
        nlp = spacy.load("en_core_web_sm")
        doc1 = nlp(text1)
        doc2 = nlp(text2)
        similarity = doc1.similarity(doc2)
        return similarity
    
    def generate_response(self, prompt: str):
        response = f"Response to: {prompt}"
        return response
    
    def process_text(self, text: str) -> str:
        """Classify intent and route the command accordingly.
        
        Returns a response string (for TTS) describing what happened.
        """
        intent = self.classify_intent(text)
        print(f"[LanguageProcessor] Intent: {intent} | Text: {text}")

        if intent == "CHAT":
            response = self.generate_response(text)
            return response

        elif intent in ("PC_CONTROL", "MEDIA_CONTROL", "TASK_MANAGEMENT",
                         "CREATIVE", "SYSTEM_CONTROL"):
            # Every actionable intent goes to the unified executor
            if not global_goal_queue.full():
                global_goal_queue.put(text)
                return f"Got it, working on: {text}"
            else:
                return "I'm busy with another task, please wait."

        elif intent == "VISION_QUERY":
            # Use the latest camera frame for the vision query
            if self.vision.frame_buffer:
                frame = self.vision.frame_buffer[-1]
                answer = self.vision.ocr_infer(frame, query=text)
                return answer
            else:
                return "I can't see anything right now."

        else:
            return "I'm not sure how to handle that."

    


if __name__ == "__main__":
    processor = LanguageProcessor()
    print(processor.classify_intent("Can you download the latest video from Mr Beast?"))