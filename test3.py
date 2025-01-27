from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = [
    ("Buy now", "ham"),
    ("Congratulations, you won a prize!", "spam"),
    ("Free vacation to Hawaii", "spam"),
    ("Limited time offer!", "ham"),
    ("You have a new message", "ham"),
    ("You have won $1000", "spam"),
    ("Important updates", "ham")
]

# Separate data into texts and labels
texts, labels = zip(*data)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)  # Feature matrix
y = labels  # Labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define class weights (optional)
class_weights = {'ham': 1, 'spam': 3}

# Initialize and fit the model with class weights
model = LogisticRegression(class_weight=class_weights)
model.fit(X_train, y_train)

# Predict labels for the test data
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example usage: Predicting a new message
new_message = ["Win a million dollars now!"]
new_message_vectorized = vectorizer.transform(new_message)
prediction = model.predict(new_message_vectorized)
print(f"Prediction for new message: {prediction[0]}")
