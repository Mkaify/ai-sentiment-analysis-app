import torch
import torch.nn as nn
import pickle
import sys

# 1. ARCHITECTURE DEFINITION 
# (Must exactly match the training structure)
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
        # Concatenate final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden_cat))

# 2. GLOBAL LOADERS
try:
    with open('vocab.pkl', 'rb') as f: 
        vocab = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f: 
        le = pickle.load(f)

    # Initialize model with CPU mapping for your local machine
    num_classes = len(le.classes_)
    model = BiLSTMClassifier(len(vocab), 100, 128, num_classes)
    
    # Load weights specifically to CPU
    model.load_state_dict(torch.load('bilstm_weights.pt', map_location=torch.device('cpu')))
    model.eval()
    print("✅ Model and Assets loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error: Could not find model files. Ensure .pt and .pkl files are in this folder. ({e})")
    sys.exit()

# 3. PREDICTION FUNCTION
def predict_sentiment(text):
    # Pre-process: Tokenize, Pad, and Convert to Tensor
    tokens = [vocab.get(w, 1) for w in str(text).lower().split()][:128]
    padded = tokens + [0] * (128 - len(tokens))
    input_tensor = torch.tensor([padded])
    
    with torch.no_grad():
        output = model(input_tensor)
        # Get the index of the highest probability
        prediction_idx = torch.argmax(output, dim=1).item()
        
    # Convert index back to original string label (e.g., 'joy', 'sadness')
    return le.inverse_transform([prediction_idx])[0]

# 4. EXECUTION BLOCK
if __name__ == "__main__":
    print("-" * 30)
    print("Sentiment Analysis AI (BiLSTM PyTorch)")
    print("Type 'exit' to quit.")
    print("-" * 30)
    
    while True:
        user_input = input("\nEnter text to analyze: ")
        if user_input.lower() == 'exit':
            break
        
        result = predict_sentiment(user_input)
        print(f"Result: {result.upper()}")