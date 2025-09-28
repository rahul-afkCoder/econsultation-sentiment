# app/preprocessing.py
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOP = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Lowercase, remove urls, extra whitespace, and non-alphanumerics (preserve basic punctuation)."""
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'[^a-z0-9\s,.]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def tokenize_and_remove_stopwords(text: str) -> list:
    t = clean_text(text)
    tokens = [w for w in t.split() if w not in STOP]
    return tokens

# small helper for word cloud frequencies
from collections import Counter

def word_frequencies(corpus: list) -> dict:
    tokens = []
    for txt in corpus:
        tokens.extend(tokenize_and_remove_stopwords(txt))
    return dict(Counter(tokens))
