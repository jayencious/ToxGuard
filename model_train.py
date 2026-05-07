import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib

print("Loading the merged dataset...")

df = pd.read_csv("final_merged_dataset.csv")

# To drop any NaN values that might have slipped through
df = df.dropna(subset=["text"])

X = df["text"]
y = df["label"]

# Train-test split
print("Splitting the data into training and testing sets...")

# We will split the data: 80% for training and 20% for testing.
# stratify=y ensures that the 80/20 split maintains the same ratio of clean/toxic labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorization
print("Vectorizing text using TF-IDF Vectorization...")
# We use max_features=5000 to keep the model lightweight and fast for our API
# ngram_range=(1,2) means it looks at single words and two-word phrases (e.g., "not good")
vectorizer = TfidfVectorizer(
    max_features=10000, ngram_range=(1, 3), stop_words="english", min_df=5
)

# Fitting on the training data and transforming it
X_train_tfidf = vectorizer.fit_transform(X_train)

# Only transform the testing data (no fitting, to prevent the data leakage)
X_test_tfidf = vectorizer.transform(X_test)

# Training on the Naive Bayes Model
print("Training Multinomial Naive Bayes Model...")

# model = MultinomialNB(alpha=0.1)
model = LogisticRegression(
    class_weight="balanced", max_iter=1000
)

model.fit(X_train_tfidf, y_train)

# Evaluation of the model
print("\nEvaluation of the Model's Performance...")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy:   {accuracy * 100:.2f}%")
print(f"Precision:  {precision * 100:.2f}% (Crucial for minimizing the false positives)")
print(f"Recall:     {recall * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Exporting the .pkl (pickle) files
print("\nExporting the model and vectorizer to .pkl files...")

# We use joblib because it is highly optimized for scikit-learn objects
# that contain large NumPy arrays
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(model, "logistic_model.pkl")

print("SUCCESS! 'tfidf_vectorizer.pkl' and 'logistic_model.pkl' files have been saved.")