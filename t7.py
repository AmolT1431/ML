import spacy
from spacy.training.example import Example
from spacy.util import minibatch
import random

# Corrected TRAIN_DATA with proper offsets
TRAIN_DATA = [
    (
        "73276 Applied Mathematics End Semester Examination 45 PASS",
        {"entities": [(6, 26, "SUBJECT"), (27, 53, "CATEGORY"), (54, 56, "MARKS")]},
    ),
    (
        "73276 Applied Mathematics Termwork 22 PASS",
        {"entities": [(6, 26, "SUBJECT"), (27, 35, "CATEGORY"), (36, 38, "MARKS")]},
    ),
    (
        "73276 Applied Mathematics Mid Semester Evaluation 29 PASS",
        {"entities": [(6, 26, "SUBJECT"), (27, 52, "CATEGORY"), (53, 55, "MARKS")]},
    ),
    (
        "73277 Discrete Mathematical Structures Mid Semester Evaluation 26 PASS",
        {"entities": [(6, 37, "SUBJECT"), (38, 63, "CATEGORY"), (64, 66, "MARKS")]},
    ),
    (
        "73277 Discrete Mathematical Structures End Semester Examination 45 PASS",
        {"entities": [(6, 37, "SUBJECT"), (38, 65, "CATEGORY"), (66, 68, "MARKS")]},
    ),
    (
        "73277 Discrete Mathematical Structures Termwork 22 PASS",
        {"entities": [(6, 37, "SUBJECT"), (38, 46, "CATEGORY"), (47, 49, "MARKS")]},
    ),
    (
        "73278 Data Structures Mid Semester Evaluation 30 PASS",
        {"entities": [(6, 20, "SUBJECT"), (21, 46, "CATEGORY"), (47, 49, "MARKS")]},
    ),
    (
        "73278 Data Structures End Semester Examination 35 PASS",
        {"entities": [(6, 20, "SUBJECT"), (21, 48, "CATEGORY"), (49, 51, "MARKS")]},
    ),
    (
        "73279 Computer Networks - I Mid Semester Evaluation 27 PASS",
        {"entities": [(6, 27, "SUBJECT"), (28, 53, "CATEGORY"), (54, 56, "MARKS")]},
    ),
    (
        "73279 Computer Networks - I End Semester Examination 40 PASS",
        {"entities": [(6, 27, "SUBJECT"), (28, 55, "CATEGORY"), (56, 58, "MARKS")]},
    ),
    (
        "73280 Microprocessors Mid Semester Evaluation 29 PASS",
        {"entities": [(6, 21, "SUBJECT"), (22, 47, "CATEGORY"), (48, 50, "MARKS")]},
    ),
    (
        "73280 Microprocessors End Semester Examination 41 PASS",
        {"entities": [(6, 21, "SUBJECT"), (22, 49, "CATEGORY"), (50, 52, "MARKS")]},
    ),
    (
        "73281 C Programming Practical 35 PASS",
        {"entities": [(6, 18, "SUBJECT"), (19, 28, "CATEGORY"), (29, 31, "MARKS")]},
    ),
    (
        "73281 C Programming Termwork 47 PASS",
        {"entities": [(6, 18, "SUBJECT"), (19, 27, "CATEGORY"), (28, 30, "MARKS")]},
    ),
    (
        "73282 Soft Skills Termwork 23 PASS",
        {"entities": [(6, 16, "SUBJECT"), (17, 26, "CATEGORY"), (27, 29, "MARKS")]},
    ),
    (
        "73282 Soft Skills External 23 PASS",
        {"entities": [(6, 16, "SUBJECT"), (17, 25, "CATEGORY"), (26, 28, "MARKS")]},
    ),
]

# Step 1: Load Blank Model
nlp = spacy.blank("en")  # Create a blank English model

# Step 2: Add NER Pipeline Component
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Step 3: Add Labels to the NER Component
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])  # Add entity label (e.g., "SUBJECT", "CATEGORY", "MARKS")

# Step 4: Train the Model
optimizer = nlp.begin_training()
n_iter = 20  # Number of training iterations

for epoch in range(n_iter):
    random.shuffle(TRAIN_DATA)  # Shuffle the training data
    losses = {}

    # Minibatch the training data
    batches = minibatch(TRAIN_DATA, size=2)
    for batch in batches:
        for text, annotations in batch:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.5, losses=losses)  # Update the model
    print(f"Losses at epoch {epoch + 1}: {losses}")

# Step 5: Save the Trained Model
output_dir = "custom_ner_model"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")

# Step 6: Test the Model
print("\nTesting the model...")
nlp = spacy.load(output_dir)  # Load the trained model

test_text = """
73281 C Programming Termwork 47 PASS
73280 Microprocessors End Semester Examination 41 PASS
73278 Data Structures End Semester Examination 35 PASS
"""

doc = nlp(test_text)

# Print Entities Detected
print("Detected Entities:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
