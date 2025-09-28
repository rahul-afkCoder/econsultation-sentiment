# tests/test_inference.py
from app.inference import ModelService

def test_predict():
    svc = ModelService()
    sentiment, conf = svc.predict_sentiment('This is great')
    assert sentiment in ('positive', 'negative', 'neutral')
    assert 0.0 <= conf <= 1.0