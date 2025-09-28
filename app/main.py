# app/main.py
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from app.inference import ModelService
from app.preprocessing import clean_text, word_frequencies
from app.utils import log_prediction
from io import StringIO
import pandas as pd
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

app = FastAPI(title='eConsultation Sentiment API')
model_service = ModelService()

class SingleRequest(BaseModel):
    comment_id: str | None = None
    comment_text: str
@app.get('/health')
async def health():
    return {'status': 'ok'}
@app.post('/predict_single')
async def predict_single(req: SingleRequest):
    text = clean_text(req.comment_text)
    sentiment, confidence = model_service.predict_sentiment(text)
    summary = model_service.summarize(req.comment_text)
    
    rec = {
        'comment_id': req.comment_id or '',
        'timestamp': pd.Timestamp.now().isoformat(),
        'original_text': req.comment_text,
        'clean_text': text,
        'predicted_sentiment': sentiment,
        'confidence': confidence,
        'summary': summary
    }
    log_prediction(rec)
    return rec

@app.post('/predict_batch')
async def predict_batch(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode()))
    out = []
    for _, row in df.iterrows():
        text = clean_text(row['comment_text'])
        sentiment, confidence = model_service.predict_sentiment(text)
        summary = model_service.summarize(row['comment_text'])
        rec = {
            'comment_id': row.get('comment_id', ''),
            'predicted_sentiment': sentiment,
            'confidence': confidence,
            'summary': summary
        }
        out.append(rec)
        log_prediction({**rec, 'original_text': row['comment_text'], 'timestamp': pd.Timestamp.now().isoformat()})
        return {'predictions': out}

@app.post('/wordcloud')
async def wordcloud(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode()))
    freqs = word_frequencies(df['comment_text'].astype(str).tolist())
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(freqs)
    out_path = 'wordcloud.png'
    wc.to_file(out_path)
    # return image as base64
    with open(out_path, 'rb') as f:
        b = base64.b64encode(f.read()).decode()
    return {'wordcloud_base64': b}