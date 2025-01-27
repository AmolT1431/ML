from transformers import BertForTokenClassification, BertTokenizerFast
import torch

# Define Label Mapping (Ensure that your labels are correctly aligned)
# Define Label Mapping
id_to_label = {
    0: "O",          # Other (non-entity tokens)
    1: "B-SUBJECT",  # Subject name
    2: "B-MARK",     # Marks
    3: "B-RESULT",   # Result (PASS)
    4:"P-PMARK",
    5:"P-RESULT",
}

# 9. Load the Model and Tokenizer for Inference
model = BertForTokenClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizerFast.from_pretrained("./saved_model")

# 10. Prepare the Input for Inference
text = "Data Structures Mid Semester Evaluation 30 PASS 65 PASS End Semester Examination 35 PASS 73279 Computer Networks - I Mid Semester Evaluation 27 PASS 136 PASS End Semester Examination 40 PASS Practical 46 PASS Termwork 23 PASS"
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

# Map the predicted labels to their corresponding names
predicted_labels_named = [id_to_label[label] for label in predicted_labels]


actual_labels = [1,1,2,2,2,5,3,4,0,0,0,0,0,0,0,0,0,0]  # Example from your dataset for this input

# Print Tokens with Actual Labels and Predicted Labels
print("Tokens with Actual Labels and Predicted Labels:")
for token, actual_label, predicted_label in zip(tokens, actual_labels, predicted_labels_named):
    print(f"Token: {token} | Actual Label: {id_to_label.get(actual_label, 'O')} | Predicted Label: {predicted_label}")