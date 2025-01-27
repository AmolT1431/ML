from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
import torch
import torch.nn as nn
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("conll2003")

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define a function to tokenize and align labels
def tokenize_and_align_labels(examples):
    # Tokenize the input texts
    texts = [" ".join(tokens) for tokens in examples['tokens']]  # Join tokenized words into a single string
    
    tokenized_inputs = tokenizer(texts, truncation=True, padding=True, is_split_into_words=True)
    
    # Align the labels
    labels = examples['ner_tags']
    
    # Ensure labels match the tokenized inputs (add padding to labels)
    aligned_labels = []
    for i, label in enumerate(labels):
        # Add -100 to labels that correspond to padding tokens
        new_labels = label + [-100] * (len(tokenized_inputs['input_ids'][i]) - len(label))
        aligned_labels.append(new_labels)
    
    tokenized_inputs['labels'] = aligned_labels
    return tokenized_inputs

# Apply the tokenization function
dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Load the model
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=9)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define the loss function with ignore_index for padding tokens
loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=None,
)

# Train the model
trainer.train()
