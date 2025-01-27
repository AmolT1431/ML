import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset

# 1. Load and Prepare Dataset
data = [
    {"text": "Math: 90, Science: 85", "labels": {"Math": 90, "Science": 85}},
    {"text": "English: 75, History: 88", "labels": {"English": 75, "History": 88}}
]

# Convert to a dataset
texts = [item['text'] for item in data]
labels = [item['labels'] for item in data]

# Convert labels into a token-friendly format
def preprocess_data(text, labels):
    token_labels = []
    for subject, mark in labels.items():
        pattern = f"{subject}: {mark}"
        start_idx = text.find(pattern)
        if start_idx != -1:
            token_labels.append((subject, mark, start_idx))
    return token_labels

processed_data = [
    {"text": item["text"], "labels": preprocess_data(item["text"], item["labels"])}
    for item in data
]

# 2. Tokenize Data
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    # Add custom logic to map labels to token indices if required
    return tokens

dataset = Dataset.from_dict({"text": texts})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 3. Load Pre-trained Model
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(labels[0]))

# 4. Define Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# 5. Train the Model
trainer.train()

# 6. Save the Model
model.save_pretrained("./custom_model3")
tokenizer.save_pretrained("./custom_model3")

# 7. Test the Model
test_text = "Physics: 92, Chemistry: 89"
inputs = tokenizer(test_text, return_tensors="pt")
outputs = model(**inputs)
print(outputs)
