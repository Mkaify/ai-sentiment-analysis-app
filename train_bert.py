import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# We will use the 'train_processed.csv' and 'test_processed.csv' files available.

# 1. Load train and test data directly from the processed CSVs
# Assuming train_processed.csv and test_processed.csv are comma-separated with a header.
# The previous 'sep=;' and 'names=['text', 'label']' caused incorrect parsing.
train_df = pd.read_csv('/content/train_processed.csv')
test_df = pd.read_csv('/content/test_processed.csv')

# 2. Encode labels
le = LabelEncoder()
train_df['label_encoded'] = le.fit_transform(train_df['label'])
test_df['label_encoded'] = le.transform(test_df['label']) # Use transform for consistency with train set encoder

# Create Datasets, selecting only the necessary columns and renaming 'label_encoded' to 'label' for the Trainer
train_ds = Dataset.from_pandas(train_df[['text', 'label_encoded']].rename(columns={'label_encoded': 'label'}))
test_ds = Dataset.from_pandas(test_df[['text', 'label_encoded']].rename(columns={'label_encoded': 'label'}))

# 3. Tokenize
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_fn(batch):
    # Tokenize the 'text' column
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)

# 4. Remove the original 'text' column after tokenization
train_ds = train_ds.remove_columns(["text"])
test_ds = test_ds.remove_columns(["text"])

# 5. Set the format for PyTorch, explicitly listing the columns the Trainer expects
train_ds.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])
test_ds.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])

# Load PyTorch Model
num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)

# Training Arguments (PyTorch Backend)
args = TrainingArguments(
    output_dir="./pytorch_bert",
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    # The 'use_cpu' argument is not standard in recent versions of TrainingArguments and can be removed.
    # Trainer automatically uses GPU if available.
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)

trainer.train()
trainer.save_model("/content/bert.pt")