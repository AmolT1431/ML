import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
from torch.utils.data import DataLoader
from transformers import AdamW

# Sample labeled data for text classification
data = [
    {"text": "73276 Applied Mathematics End Semester Examination 1 FAIL", "label": "Subject"},
    {"text": "Termwork 12 PASS", "label": "Marks"},
    {"text": "Mid Semester Evaluation 13 PASS", "label": "Marks"},
    {"text": "End Semester Examination 10 FAIL", "label": "Marks"},
    {"text": "Dr. A. D. Shinde College of Engineering", "label": "Other"},
    {"text": "Discrete Mathematical Structures 10 FAIL", "label": "Subject"},
    {"text": "Soft Skills Termwork 11 PASS", "label": "Subject"},
]

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Split data into features and labels
texts = df["text"]
labels = df["label"]

# Mapping labels to numeric values for BERT
label_map = {'Subject': 0, 'Marks': 1, 'Other': 2}
labels = labels.map(label_map)

# Load a pre-trained BERT model for sequence classification
model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_map), 
                                                      output_attentions=False, output_hidden_states=False)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize and encode the text data
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=512)

# Convert into Torch datasets
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, y_train.tolist())
test_dataset = CustomDataset(test_encodings, y_test.tolist())

# Set up the DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training the model
optimizer = AdamW(model.parameters(), lr=5e-5)

# Train for 3 epochs
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}")

# Evaluate the model
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Print classification report with zero_division set to 1
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=list(label_map.keys()), 
                            labels=list(label_map.values()), zero_division=1))


# Test the classifier on new data
new_lines = [
    "73277 Discrete Mathematical Structures End Semester Examination 10 FAIL",
    "Soft Skills Termwork 11 PASS",
    "Dr. A. D. Shinde College of Engineering",
]

# Vectorize new data and predict
new_encodings = tokenizer(new_lines, truncation=True, padding=True, max_length=512, return_tensors='pt')
new_encodings = {key: value.to(device) for key, value in new_encodings.items()}

model.eval()
with torch.no_grad():
    outputs = model(**new_encodings)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

# Output predictions for new data
print("\nText Classification Predictions:")
for line, pred in zip(new_lines, predictions):
    label = list(label_map.keys())[pred.item()]
    print(f"{line} -> {label}")

# ------------------------------
# Named Entity Recognition (NER)
# ------------------------------

print("\nNamed Entity Recognition (NER):")

# Use a pre-trained NLP model for NER
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")

# Example text for NER
ner_text = """
73276 Applied Mathematics End Semester Examination 1 FAIL
Termwork 12 PASS
Mid Semester Evaluation 13 PASS
End Semester Examination 10 FAIL
"""

# Extract named entities
entities = ner_pipeline(ner_text)

for entity in entities:
    print(f"Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']:.2f}")
