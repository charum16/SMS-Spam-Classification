import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset from the specified location
data = pd.read_csv(r'C:\Users\charu\OneDrive\Pictures\Desktop\spam.csv', encoding='latin-1')

# Preprocess the data
data['label'] = data['v1'].map({'ham': 0, 'spam': 1})  # Convert labels to binary
X = data['v2']  # Feature (the message)
y = data['label']  # Target (the label)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = classifier.predict(X_test_vectorized)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(classifier, 'sms_spam_classifier.pkl')

# Load the model for predictions
loaded_model = joblib.load('sms_spam_classifier.pkl')

# Function to classify custom input messages
def classify_messages(messages):
    messages_vectorized = vectorizer.transform(messages)
    predictions = loaded_model.predict(messages_vectorized)
    return predictions

# Custom input for classification
while True:
    user_input = input("Enter a message to classify (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    prediction = classify_messages([user_input])
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    print(f"The message: '{user_input}' is classified as: {result}")
