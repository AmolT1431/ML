from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
import torch

# 1. Define the dataset
data = {
    "tokens": [
        ["Applied", "Mathematics", "End", "Semester", "Examination", "35", "PASS", "82", "PASS", "Mid", "Semester", "Evaluation", "28", "PASS", "Termwork", "25", "PASS", "73277"],
        ["Data", "Structures", "End", "Semester", "Examination", "40", "PASS", "85", "PASS", "Mid", "Semester", "Evaluation", "30", "PASS", "Termwork", "28", "PASS", "73278"],
        ["Computer", "Networks", "End", "Semester", "Examination", "42", "PASS", "89", "PASS", "Mid", "Semester", "Evaluation", "32", "PASS", "Termwork", "30", "PASS", "73279"],
        ["Microprocessors", "End", "Semester", "Examination", "38", "PASS", "80", "PASS", "Mid", "Semester", "Evaluation", "25", "PASS", "Termwork", "27", "PASS", "73280"],
        ["C", "Programming", "End", "Semester", "Examination", "34", "PASS", "76", "PASS", "Mid", "Semester", "Evaluation", "24", "PASS", "Termwork", "23", "PASS", "73281"],
        ["Soft", "Skills", "End", "Semester", "Examination", "50", "PASS", "90", "PASS", "Mid", "Semester", "Evaluation", "35", "PASS", "Termwork", "33", "PASS", "73282"],
        ["Artificial", "Intelligence", "End", "Semester", "Examination", "45", "PASS", "88", "PASS", "Mid", "Semester", "Evaluation", "27", "PASS", "Termwork", "29", "PASS", "73283"],
        ["Machine", "Learning", "End", "Semester", "Examination", "40", "PASS", "84", "PASS", "Mid", "Semester", "Evaluation", "30", "PASS", "Termwork", "28", "PASS", "73284"],
        ["Database", "Systems", "End", "Semester", "Examination", "39", "PASS", "81", "PASS", "Mid", "Semester", "Evaluation", "28", "PASS", "Termwork", "26", "PASS", "73285"],
        ["Operating", "Systems", "End", "Semester", "Examination", "41", "PASS", "83", "PASS", "Mid", "Semester", "Evaluation", "31", "PASS", "Termwork", "29", "PASS", "73286"],
        ["Data", "Science", "End", "Semester", "Examination", "47", "PASS", "91", "PASS", "Mid", "Semester", "Evaluation", "33", "PASS", "Termwork", "32", "PASS", "73287"],
        ["Digital", "Logic", "End", "Semester", "Examination", "36", "PASS", "79", "PASS", "Mid", "Semester", "Evaluation", "26", "PASS", "Termwork", "24", "PASS", "73288"],
        ["Software", "Engineering", "End", "Semester", "Examination", "43", "PASS", "86", "PASS", "Mid", "Semester", "Evaluation", "29", "PASS", "Termwork", "27", "PASS", "73289"],
        ["Cyber", "Security", "End", "Semester", "Examination", "44", "PASS", "87", "PASS", "Mid", "Semester", "Evaluation", "30", "PASS", "Termwork", "28", "PASS", "73290"],
        ["Computer", "Graphics", "End", "Semester", "Examination", "39", "PASS", "82", "PASS", "Mid", "Semester", "Evaluation", "25", "PASS", "Termwork", "27", "PASS", "73291"],
        ["Web", "Development", "End", "Semester", "Examination", "37", "PASS", "80", "PASS", "Mid", "Semester", "Evaluation", "28", "PASS", "Termwork", "25", "PASS", "73292"],
        ["Cloud", "Computing", "End", "Semester", "Examination", "42", "PASS", "84", "PASS", "Mid", "Semester", "Evaluation", "31", "PASS", "Termwork", "29", "PASS", "73293"],
        ["Computer", "Vision", "End", "Semester", "Examination", "40", "PASS", "83", "PASS", "Mid", "Semester", "Evaluation", "29", "PASS", "Termwork", "27", "PASS", "73294"],
        ["Compiler", "Design", "End", "Semester", "Examination", "43", "PASS", "85", "PASS", "Mid", "Semester", "Evaluation", "30", "PASS", "Termwork", "28", "PASS", "73295"],
        ["Discrete", "Mathematics", "End", "Semester", "Examination", "41", "PASS", "82", "PASS", "Mid", "Semester", "Evaluation", "28", "PASS", "Termwork", "26", "PASS", "73296"],
        ["Network", "Security", "End", "Semester", "Examination", "44", "PASS", "86", "PASS", "Mid", "Semester", "Evaluation", "32", "PASS", "Termwork", "30", "PASS", "73297"]
    ],
    "labels": [
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]
}


# Convert dataset to Huggingface format
dataset = Dataset.from_dict(data)

# 2. Load the pre-trained BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# 3. Tokenization and Label Alignment Function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get word IDs for current example
        label_ids = []
        
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Special tokens should be ignored
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
# 4. Load the pre-trained BERT model
num_labels = 5  # Number of unique labels in your dataset
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# 5. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
)

# 6. Trainer Initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# 7. Train the Model
trainer.train()

# 8. Save the Model and Tokenizer
trainer.save_model("./saved_model")
tokenizer.save_pretrained("./saved_model")

# --- Inference Section ---
# 9. Load the Model and Tokenizer for Inference
model = BertForTokenClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizerFast.from_pretrained("./saved_model")

# 10. Prepare the Input for Inference
# 10. Prepare the Input for Inference
text = "Data Structures Mid Semester Evaluation 30 PASS 65 PASS End Semester Examination 35 PASS 73279 Computer Networks - I Mid Semester Evaluation 27 PASS 136 PASS End Semester Examination 40 PASS Practical 46 PASS Termwork 23 PASS "
inputs = tokenizer(
    text.split(),
    return_tensors="pt",
    is_split_into_words=True,
    padding=True,
    truncation=True,
)

# 11. Get Predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

# 12. Map Predictions Back to Tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predicted_labels = predictions[0].tolist()

# Define Label Mapping
id_to_label = {
    0: "O",          # Other (non-entity tokens)
    1: "B-SUBJECT",  # Subject name
    2: "B-MARK",     # Marks
    3: "B-RESULT",   # Result (PASS)5   4: "P-PMARK",
    5: "P-RESULT",
}

# Map predictions to labels
predicted_labels_named = [id_to_label[label] for label in predicted_labels]

actual_labels = [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0]  # Example from your dataset for this input

# Print Tokens with Actual Labels and Predicted Labels
print("Tokens with Actual Labels and Predicted Labels:")
for token, actual_label, predicted_label in zip(tokens, actual_labels, predicted_labels_named):
    print(f"Token: {token} | Actual Label: {id_to_label.get(actual_label, 'O')} | Predicted Label: {predicted_label}")