from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset("imdb")

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split dataset into train and test
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    evaluation_strategy="epoch",     # evaluation strategy to adopt during training
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=64,   # batch size for evaluation
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset            # evaluation dataset
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

# Label mapping
label_map = {0: "Negative", 1: "Positive"}

def get_label_name(label_index):
    return label_map[label_index]

# Testing the model
def predict(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = get_label_name(predicted_class_id)
    return predicted_label

# Example predictions
texts = [
    "I love learning new languages",
    "Debugging is so frustrating"
]

for text in texts:
    prediction = predict(text)
    print(f"Text: {text} -> Prediction: {prediction}")
