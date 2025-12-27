import os
import torch
import torch.nn as nn
import pickle
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentiment Analysis AI Tool")

# 1. ADD MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. SETUP TEMPLATES
templates = Jinja2Templates(directory="templates")

# 3. BILSTM CLASS DEFINITION
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden_cat))

# 4. LOAD MODELS (BERT & BiLSTM)
try:
    # Load BERT folder
    bert_pipe = pipeline("sentiment-analysis", model="./bert.pt", device=-1)
    
    # Load BiLSTM Assets (Ensure these files are in the ./bilstm/ folder)
    with open('./bilstm/vocab.pkl', 'rb') as f: 
        vocab = pickle.load(f)
    with open('./bilstm/label_encoder.pkl', 'rb') as f: 
        le = pickle.load(f)
    
    bilstm_model = BiLSTMClassifier(len(vocab), 100, 128, len(le.classes_))
    bilstm_model.load_state_dict(torch.load('./bilstm/bilstm_weights.pt', map_location='cpu'))
    bilstm_model.eval()
    print("✅ All models loaded on CPU successfully.")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

class SentimentRequest(BaseModel):
    text: str

# 5. ROUTES
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(request: SentimentRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    # 1. BERT Prediction
    with torch.no_grad():
        bert_res = bert_pipe(request.text, return_token_type_ids=False)[0]
    
    try:
        bert_idx = int(bert_res['label'].split('_')[1])
        bert_string_label = le.inverse_transform([bert_idx])[0]
    except (IndexError, ValueError):
        bert_string_label = bert_res['label']

    # 2. BiLSTM Prediction
    tokens = [vocab.get(w, 1) for w in request.text.lower().split()][:128]
    padded = tokens + [0] * (128 - len(tokens))
    input_tensor = torch.tensor([padded])
    
    with torch.no_grad():
        bilstm_out = bilstm_model(input_tensor)
        
        # Convert raw logits to probabilities
        probabilities = torch.nn.functional.softmax(bilstm_out, dim=1)
        
        # Get the highest probability as the "confidence"
        confidence, bilstm_idx = torch.max(probabilities, 1)
        
        bilstm_label = le.inverse_transform([bilstm_idx.item()])[0]
        bilstm_conf_score = confidence.item()

    return {
        "text": request.text,
        "results": {
            "bert_model": {
                "label": bert_string_label.upper(), 
                "confidence": round(bert_res['score'], 4)
            },
            "bilstm_model": {
                "label": bilstm_label.upper(),
                "confidence": round(bilstm_conf_score, 4)
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)