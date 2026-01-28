import nltk
import spacy
import textblob as tb

class LanguageProcessor:
    def __init__(self):
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

if __name__ == "__main__":
    processor = LanguageProcessor()
    print(processor.similarity_score("Scan this Image","Scan this object"))
    print(processor.generate_response("Scan this Image"))