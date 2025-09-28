# app/inference.py
from transformers import pipeline
from typing import Tuple
import os
import logging

logger = logging.getLogger(__name__)

# The design: try to load local fine-tuned models from ./models, otherwise use HF public models

def load_sentiment_pipeline(model_path: str = None):
    try:
        if model_path and os.path.isdir(model_path):
            logger.info(f"Loading local sentiment model from {model_path}")
            return pipeline('sentiment-analysis', model=model_path, device=0
if _has_cuda() else -1)
        # fallback to a small general model
        logger.info("Loading default HF sentiment model (distilbert-base-uncased-finetuned-sst-2-english)")
        return pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    except Exception as e:
        logger.exception('Failed to load sentiment pipeline: %s', e)
        raise

def load_summarizer(model_name: str = 'sshleifer/distilbart-cnn-12-6'):
    try:
        return pipeline('summarization', model=model_name)
    except Exception:
        # fallback: not available -> None
        logger.warning('Summarizer unavailable. Summaries will fallback to truncate.')
        return None

def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

class ModelService:
    def __init__(self, model_dir: str = './models'):
        self.sentiment = load_sentiment_pipeline(model_dir)
        self.summarizer = load_summarizer()
        
    def predict_sentiment(self, text: str) -> Tuple[str, float]:
        pred = self.sentiment(text)[0]
        # pipeline returns label like 'POSITIVE'/'NEGATIVE' — map to desired labels
        label = pred['label'].lower()
        score = float(pred.get('score', 0.0))
        if label == 'positive':
            return 'positive', score
        if label == 'negative':
            return 'negative', score
        return 'neutral', score
    
    def summarize(self, text: str, max_words: int = 20) -> str:
        if not text:
            return ''
        if self.summarizer:
            try:
                s = self.summarizer(text, max_length=45, min_length=5,

do_sample=False)[0]['summary_text']
                # compact to max_words
                words = s.split()
                return ' '.join(words[:max_words])
            except Exception:
                pass
        # fallback: simple truncation to max_words
        words = text.split()
        return ' '.join(words[:max_words])