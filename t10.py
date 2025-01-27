from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Step 1: Create raw data
data = {
    "text": [
        "Math: 90 Science: 85 English: 88",
        "History: 75 Geography: 80 Math: 95"
    ],
    "labels": [
        {"Math": 90, "Science": 85, "English": 88},
        {"History": 75, "Geography": 80, "Math": 95}
    ]
}

# Step 2: Convert raw data to a Hugging Face dataset
dataset = Dataset.from_dict(data)

# Step 3: Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=2)  # Background and target tokens

# Step 4: Tokenization and label alignment
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=128, return_offsets_mapping=True
    )
    labels = []
    for i, label_dict in enumerate(examples["labels"]):
        # Create default label IDs for the sequence
        label_ids = [-100] * len(tokenized_inputs["input_ids"][i])

        # Align the label dictionary with tokenized offsets
        for subject, mark in label_dict.items():
            pattern = f"{subject}: {mark}"
            start_idx = examples["text"][i].find(pattern)
            if start_idx != -1:
                end_idx = start_idx + len(pattern)

                # Align with token offsets
                for idx, (token_start, token_end) in enumerate(tokenized_inputs["offset_mapping"][i]):
                    if token_start >= start_idx and token_end <= end_idx:
                        label_ids[idx] = 1  # Class ID for target token

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    tokenized_inputs.pop("offset_mapping")  # Remove offset mapping
    return tokenized_inputs


# Apply tokenization and alignment
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Step 7: Train the model
trainer.train()

trainer.save_model("./saved_model")
tokenizer.save_pretrained("./saved_model")
