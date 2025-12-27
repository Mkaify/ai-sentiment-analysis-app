from transformers import AutoTokenizer

# 1. Use the original model name you trained with (e.g., 'bert-base-uncased')
model_name = "bert-base-uncased" 

# 2. Download the tokenizer files
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Save them into your local bert.pt folder
tokenizer.save_pretrained("./bert.pt")

print("✅ Tokenizer files have been added to ./bert.pt")