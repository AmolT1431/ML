import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("glue", "mrpc")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",  # Updated
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Reduced batch size
    per_device_eval_batch_size=2,  # Reduced batch size
    num_train_epochs=1,  # Reduced epochs for testing
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train and Save Model
trainer.train()
model.save_pretrained("./custom_model")
tokenizer.save_pretrained("./custom_model")

print("Training complete and model saved.")
