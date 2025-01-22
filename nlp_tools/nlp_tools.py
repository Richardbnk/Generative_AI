from transformers import pipeline
from langdetect import detect, DetectorFactory

# Set up deterministic language detection
DetectorFactory.seed = 0

def detect_language(text):
    """
    Detect the language of the given text.
    """
    return detect(text)

def translate_text(text, src_lang="auto", target_lang="en"):
    """
    Translate text to a target language using a multilingual transformer model.
    """
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-multilingual")
    return translator(text, src_lang=src_lang, tgt_lang=target_lang)[0]['translation_text']

def sentiment_analysis(text):
    """
    Perform sentiment analysis on the given text.
    """
    sentiment_model = pipeline("sentiment-analysis")
    return sentiment_model(text)

# Example usage
texts = [
    "Bonjour tout le monde", # French
    "Hola mundo", # Spanish
    "Hello world", # English
    "こんにちは世界" # Japanese
    "Olá mundo" # Potuguese
]

for text in texts:
    print(f"Original Text: {text}")
    lang = detect_language(text)
    print(f"Detected Language: {lang}")

    translated_text = translate_text(text, target_lang="en")
    print(f"Translated Text: {translated_text}")

    sentiment = sentiment_analysis(translated_text)
    print(f"Sentiment: {sentiment}")
    print("-" * 50)
