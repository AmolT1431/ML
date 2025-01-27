from transformers import Trainer, AutoTokenizer, AutoModelForTokenClassification, TrainingArguments
from evaluate import load  # Correct import for loading metrics
from datasets import load_dataset, DatasetDict

# Assuming id_to_label and label_to_id are defined somewhere in your code
id_to_label = {0: 'O', 1: 'B-PER', 2: 'I-PER', -100: 'IGNORE'}
label_to_id = {'O': 0, 'B-PER': 1, 'I-PER': 2}

# 2. Load the Dataset
dataset_path = r"F:\work\ML\structured_dataset.json"  # Replace with your dataset path
dataset = load_dataset("json", data_files=dataset_path)

# Split the dataset into train and validation sets
dataset = dataset["train"].train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# 3. Load Tokenizer and Model
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_labels = len(id_to_label)  # Ensure num_labels is defined
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

# 4. Preprocess the Data
def tokenize_and_align_labels(examples):
    # Tokenize the inputs
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        is_split_into_words=False  # Keep it False to ensure correct tokenization
    )

    all_labels = []

    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get word IDs for each token
        label_ids = []

        # Make sure that labels are aligned to the tokens
        for word_idx in word_ids:
            if word_idx is None:
                # If it's a special token, we use -100 (ignore label during loss computation)
                label_ids.append(-100)
            elif word_idx < len(labels):
                # If the word_idx is within the bounds of the labels list, get the corresponding label
                label_ids.append(label_to_id.get(labels[word_idx], -100))
            else:
                # Handle cases where the tokenized input has more tokens than labels
                label_ids.append(-100)

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

# Tokenize the dataset
tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)

# 5. Initialize Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
)

# 7. Train the Model
trainer.train()

# 8. Save the Model
trainer.save_model("./trained_model")

# 9. Evaluate the Model
metric = load("accuracy")  # Correct function to load the metric

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    true_labels = [
        [id_to_label[label] for label in label_set if label != -100]
        for label_set in labels
    ]
    true_predictions = [
        [id_to_label[pred] for pred, label in zip(pred_set, label_set) if label != -100]
        for pred_set, label_set in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results

trainer.compute_metrics = compute_metrics
eval_results = trainer.evaluate()
print(eval_results)