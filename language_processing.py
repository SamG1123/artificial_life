import nltk
import spacy
import textblob as tb
from transformers import pipeline
from image_processing import ObjectDetection

class LanguageProcessor:
    def __init__(self):
        self.vision = ObjectDetection()
        self.classifier = pipeline("text-classification", model="./intent_model")

    def classify_intent(self, text: str):
        result = self.classifier(text)
        return result
    
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
    
    def process_text(self, text: str):
        intent = self.classify_intent(text)
        entities = self.classify_entity(text)
        sentiment = self.sentiment_analysis(text)


        if intent == 'CHAT':
            response = self.generate_response(text)
        
        elif intent == 'PC_CONTROL':
            steps = self.plan_task(text)
            response = "Executing PC control command."
        
        elif intent == 'VISION_QUERY':
            self.vision.ocr_infer(None)
            response = "Processing vision query."

    


if __name__ == "__main__":
    processor = LanguageProcessor()
    print(processor.classify_intent("Can you download the latest video from Mr Beast?"))