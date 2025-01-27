import re

# Function to extract name, subjects, and marks from the file
def extract_info_from_file(file_path):
    # Read the file content
    with open(file_path, 'r') as file:
        text = file.read()

    # Extract Name (assumed to be under the 'Name' field)
    name_match = re.search(r"Name\s*:\s*([A-Za-z\s]+)", text)
    name = name_match.group(1) if name_match else "Unknown"

    # Extract Subjects and Marks using a regular expression pattern
    subject_marks_pattern = r"(\d+)\s+([A-Za-z\s&]+)\s+End Semester\s+Evaluation\s+UtyExm\s+(\d+)"
    subjects_and_marks = re.findall(subject_marks_pattern, text)

    # Prepare result in a structured format
    result = {
        "Name": name,
        "Subjects and Marks": [(subject.strip(), mark) for _, subject, mark in subjects_and_marks]
    }

    return result

# Example usage
file_path = r'F:\work\ML\tt.txt'  # Provide the path to your .txt file
info = extract_info_from_file(file_path)
print(info)
