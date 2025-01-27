from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Define the dataset (repeated for clarity)
data = {
    "tokens": [
        ["Applied", "Mathematics", ":", "45", "PASS"],
        ["Discrete", "Mathematical", "Structures", ":", "26","PASS"],
        ["Data", "Structures", ":", "30", ":","PASS"],
        ["Computer", "Networks", "-", "I", ":", "27","PASS"],
        ["Microprocessors", ":", "29","PASS"],
        ["C", "Programming", ":", "35","PASS"],
        ["Soft", "Skills", ":", "23","PASS"],
    ],
    "labels": [
        [1, 1, 0, 2, 3],  # Applied Mathematics -> B-SUBJECT, Marks -> B-MARK, Result -> B-RESULT
        [1, 1, 0, 2, 3],  # Discrete Mathematical Structures -> B-SUBJECT, Marks -> B-MARK, Result -> B-RESULT
        [1, 1, 0, 2, 3],  # Data Structures -> B-SUBJECT, Marks -> B-MARK, Result -> B-RESULT
        [1, 1, 0, 2, 3],  # Computer Networks -> B-SUBJECT, Marks -> B-MARK, Result -> B-RESULT
        [1, 1, 0, 2, 3],  # Microprocessors -> B-SUBJECT, Marks -> B-MARK, Result -> B-RESULT
        [1, 1, 0, 2, 3],  # C Programming -> B-SUBJECT, Marks -> B-MARK, Result -> B-RESULT
        [1, 1, 0, 2, 3],  # Soft Skills -> B-SUBJECT, Marks -> B-MARK, Result -> B-RESULT
    ]
}

# Convert dataset to Huggingface format
dataset = Dataset.from_dict(data)

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Tokenization and Label Alignment Function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,   # Enable truncation
        padding=True,      # Enable padding to ensure uniform length
    )
    
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get word IDs for current example
        label_ids = []
        
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Ignore special tokens
            else:
                if word_id < len(label):
                    label_ids.append(label[word_id])
                else:
                    label_ids.append(-100)
                
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Load the pre-trained BERT model
num_labels = 5  # Number of unique labels in your dataset
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,  # Reduce learning rate slightly
    per_device_train_batch_size=8,
    num_train_epochs=10,  # Increase epochs for better convergence
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
)

# Trainer Initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train the Model
trainer.train()

# Save the Model and Tokenizer
trainer.save_model("./saved_model")
tokenizer.save_pretrained("./saved_model")

# --- Inference Section ---
# Load the Model and Tokenizer for Inference
model = BertForTokenClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizerFast.from_pretrained("./saved_model")

# Prepare the Input for Inference
text = "Data Structures Mid Semester Evaluation 30 PASS 65 PASS End Semester Examination 35 PASS 73279 Computer Networks - I Mid Semester Evaluation 27 PASS 136 PASS End Semester Examination 40 PASS Practical 46 PASS Termwork 23 PASS "
inputs = tokenizer(
    text.split(),
    return_tensors="pt",
    is_split_into_words=True,
    padding=True,
    truncation=True,
)

# Get Predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

# Map Predictions Back to Tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predicted_labels = predictions[0].tolist()

# Define Label Mapping
id_to_label = {
    0: "O",          # Other (non-entity tokens)
    1: "B-SUBJECT",  # Subject name
    2: "B-MARK",     # Marks
    3: "B-RESULT",   # Result (PASS)
}

# Map predictions to labels
predicted_labels_named = [id_to_label[label] for label in predicted_labels]

# Extract Subjects and Marks (Updated)
subjects_marks = {}
current_subject = []
current_marks = None

# Iterate over tokens and predicted labels
for token, label in zip(tokens, predicted_labels_named):
    # Check for a new subject
    if label == "B-SUBJECT":
        current_subject = [token]  # Start a new subject
    elif label == "B-MARK" and current_subject:
        current_marks = token
        subjects_marks[" ".join(current_subject)] = current_marks
        current_subject = []  # Reset after capturing the subject and marks

# If a subject has no marks, it might still need to be captured.
if current_subject and current_marks is not None:
    subjects_marks[" ".join(current_subject)] = current_marks

# Display the result
print("Subjects and Marks:")
for subject, mark in subjects_marks.items():
    print(f"{subject}: {mark}")
