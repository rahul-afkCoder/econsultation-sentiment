# eConsultation Sentiment

## Quick start
1. Create virtualenv and install: `pip install -r requirements.txt`
2. Start the API: `uvicorn app.main:app --reload --port 8000`
3. Open `web/index.html` and use the UI (or curl/postman) to call `/
predict_single`.

## Training
Prepare a CSV with columns `comment_text` and `label` (0/1/2 mapping to
negative/neutral/positive). Then run: