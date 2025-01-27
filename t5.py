import json

# Example data
dataset = [
    {
        "text": "Applied Mathematics End Semester Examination 45 PASS 96 PASS Termwork 22 PASS Mid Semester Evaluation 29 PASS 73277",
        "labels": [
            "B-SUB", "I-SUB", "I-SUB", "B-EXAMTYPE", "I-EXAMTYPE",
            "B-MARK", "B-RESULT", "B-MARK", "B-RESULT", "B-EXAMTYPE",
            "B-MARK", "B-RESULT", "B-EXAMTYPE", "I-EXAMTYPE",
            "B-MARK", "B-RESULT", "B-CODE"
        ]
    }
]

# Save the dataset to a JSON file
with open("structured_dataset.json", "w") as file:
    json.dump(dataset, file, indent=4)

print("Dataset saved to structured_dataset.json")
