import os
import warnings
import pickle
import torch
from transformers import pipeline, AutoTokenizer

# 1. Hide unwanted logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator LabelEncoder")

model_path = "./bert.pt"

def run_inference():
    try:
        # 2. Fix the error by loading the tokenizer separately
        # DistilBERT models do not use 'token_type_ids'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 3. Create the pipeline without using 'tokenizer_kwargs'
        classifier = pipeline(
            "sentiment-analysis",
            model=model_path,
            tokenizer=tokenizer,
            device=-1 # Use CPU
        )

        # 4. Load your newly generated label encoder
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)

        text = "im feeling quite sad and sorry for myself but ill snap out of it soon"
        
        # 5. Perform the analysis
        # We handle the token_type_ids issue by telling the tokenizer NOT to return them
        result = classifier(text, return_token_type_ids=False)[0]
        
        # Map result back to human-readable names
        label_idx = int(result['label'].split('_')[1])
        sentiment = le.inverse_transform([label_idx])[0]

        print(f"\n✅ Result: {sentiment.upper()} (Score: {result['score']:.4f})")

    except Exception as e:
        print(f"❌ Loading failed: {e}")

if __name__ == "__main__":
    run_inference()