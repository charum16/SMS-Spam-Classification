
# SMS Spam Classification

The **SMS Spam Classifier** is a Machine Learning project that automatically detects whether a given SMS message is **spam** or **ham** (not spam). It uses NLP techniques for text preprocessing and a Multinomial Naive Bayes classifier for prediction.

This project demonstrates end-to-end processing — from raw data cleaning to model training, testing, and evaluation.

---

## 📊 Dataset

The dataset consists of SMS messages labeled as:

* `spam`: unwanted promotional or phishing messages
* `ham`: legitimate messages

Each message is a text string and the label is either "spam" or "ham".

---

## 🚀 Features

* Clean and preprocess raw SMS data
* Convert text data into numerical format using **TF-IDF Vectorizer**
* Train a **Naive Bayes** classifier
* Evaluate the model using:

  * Accuracy
  * Confusion Matrix
  * Precision, Recall, F1 Score

---

## 🧰 Tech Stack

* **Language**: Python
* **Libraries**:

  * `pandas` for data handling
  * `numpy` for numeric operations
  * `sklearn` for model training and evaluation
  * `nltk` for natural language preprocessing

---

## 🧪 Model Pipeline

1. **Load Dataset** – Read labeled SMS messages.
2. **Preprocess Text** – Clean, tokenize, and remove stopwords.
3. **Feature Extraction** – Use `TfidfVectorizer` to convert text into vectors.
4. **Train-Test Split** – Divide data into training and testing sets.
5. **Model Training** – Use `MultinomialNB` from `sklearn`.
6. **Evaluation** – Assess performance with metrics and confusion matrix.
