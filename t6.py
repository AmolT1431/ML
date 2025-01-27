import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

# Load spaCy's language model
nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"])

# Add a new entity recognizer to the pipeline (if not already present)
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Define training data
TRAIN_DATA = [
    ("Machine learning is fascinating.", {"entities": [(0, 16, "PATTERN1")]}),
    ("Artificial intelligence is the future.", {"entities": [(0, 22, "PATTERN2")]}),
]

# Add labels to the NER
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable other pipeline components during training
pipe_exceptions = ["ner"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Start training
with nlp.disable_pipes(*unaffected_pipes):
    optimizer = nlp.begin_training()
    for itn in range(30):  # Number of iterations
        losses = {}
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            examples = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in zip(texts, annotations)]
            nlp.update(examples, drop=0.5, losses=losses)
        print(f"Iteration {itn}, Losses: {losses}")

# Save the trained model
nlp.to_disk("./custom_ner_model")

# Test the trained model
nlp = spacy.load("./custom_ner_model")
doc = nlp("Machine learning and artificial intelligence are popular topics.")
for ent in doc.ents:
    print(ent.text, ent.label_)