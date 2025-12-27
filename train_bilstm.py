import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
from sklearn.preprocessing import LabelEncoder

# 1. FORCE SYNCHRONOUS CUDA (Must be before importing torch in a fresh runtime)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 2. DATA PREP
df = pd.read_csv('/content/train_processed.csv')
train_df = df[df['split'] == 'train'].reset_index(drop=True)
test_df = df[df['split'] == 'test'].reset_index(drop=True)

# Important: LabelEncoder must result in 0 to (num_classes - 1)
le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['label'])
test_df['label'] = le.transform(test_df['label'])
num_classes = len(le.classes_)

# 3. VOCABULARY (Safe Indexing)
all_text = ' '.join(train_df['text_cleaned'].astype(str)).split()
word_freq = pd.Series(all_text).value_counts()
vocab = {word: i+2 for i, word in enumerate(word_freq.index[:10000])}
vocab['<PAD>'] = 0
vocab['<OOV>'] = 1

class SentimentDataset(Dataset):
    def __init__(self, dataframe, vocab, max_len=128):
        self.data = dataframe
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.loc[idx, 'text_cleaned'])
        tokens = [self.vocab.get(w, 1) for w in text.split()][:self.max_len]
        padded = tokens + [0] * (self.max_len - len(tokens))
        # Ensure label is a Long tensor for CrossEntropy
        return torch.tensor(padded), torch.tensor(self.data.loc[idx, 'label'], dtype=torch.long)

# 4. ARCHITECTURE
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        # vocab_size must match len(vocab)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden_cat))

# 5. INITIALIZATION & TRAINING
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CHECK: vocab size must be exactly len(vocab)
model = BiLSTMClassifier(len(vocab), 100, 128, num_classes).to(device)

train_loader = DataLoader(SentimentDataset(train_df, vocab), batch_size=32, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Training on {device}...")
for epoch in range(5):
    model.train()
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete.")

# 6. SAVE EVERYTHING
torch.save(model.state_dict(), 'bilstm_weights.pt')
with open('vocab.pkl', 'wb') as f: pickle.dump(vocab, f)
with open('label_encoder.pkl', 'wb') as f: pickle.dump(le, f)