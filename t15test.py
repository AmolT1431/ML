from transformers import BertForTokenClassification, BertTokenizerFast
import torch

# 1. Load the saved model and tokenizer
model = BertForTokenClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizerFast.from_pretrained("./saved_model")

# 2. Define the input text
text = "Fundamentals of Electronics and Computer End Semester Evaluation UtyExm 19 FAIL 51 FAIL 71820 Basic Mechanical Engineering End Semester Evaluation UtyExm 19 FAIL 47 FAIL Sem - 1 Result - FAIL B.Tech.CBCS Part 1 Semester 2 72504 Engineering Graphics End Semester Examination 6 FAIL 45 FAIL Sem - 2 Result - FAIL Part - 1 Result - FAIL ATKT B.Tech.CBCS Part 2 Semester 3 73276 Applied Mathematics Termwork 10 PASS 24 FAIL Mid Semester Evaluation 13 PASS End Semester Examination 1 FAIL 73277 Discrete Mathematical Structures Termwork 12 PASS 30 FAIL Mid Semester Evaluation 12 PASS End Semester Examination 6 FAIL 73278"
# 3. Tokenize the input
inputs = tokenizer(
    text.split(),
    return_tensors="pt",
    is_split_into_words=True,
    padding=True,
    truncation=True,
)

# 4. Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

# 5. Map predictions back to tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predicted_labels = predictions[0].tolist()

# Define label mapping
id_to_label = {
    0: "O",          # Other (non-entity tokens)
    1: "B-SUBJECT",  # Subject name
    2: "E-ESUBJECT", # Marks
    3: "B-SEM",      # Result (PASS)
    4: "E-ESEM",     # Final marks
    5: "B-MARK",     # Marks
    6: "B-RESULT"    # Final result (PASS/FAIL)
}

# Map predictions to human-readable labels
predicted_labels_named = [id_to_label[label] for label in predicted_labels]

# Print only tokens and predicted labels
# for token, predicted in zip(tokens, predicted_labels_named):
#     print(f"Token: {token}, Predicted: {predicted}")
subject_list = []
subject = ""

# Iterate through tokens and predicted labels
for token, predicted in zip(tokens, predicted_labels_named):
    if predicted == "B-SUBJECT":
        subject += token+" "  # Build the subject token by token
    elif subject:  # If we have an existing subject and the predicted label is not B-SUBJECT
        subject_list.append(subject)  # Append the subject to the list
        subject = ""  # Reset subject to start a new one

# To handle cases where the last subject needs to be added
if subject:
    subject_list.append(subject)

# Print the final subject list
for item in subject_list:
    print(item)
