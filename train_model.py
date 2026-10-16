
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Dataset paths
FAKE_PATH = "data/Fake.csv"
REAL_PATH = "data/True.csv"

# Check dataset files
if not os.path.exists(FAKE_PATH) or not os.path.exists(REAL_PATH):
    raise FileNotFoundError(
        "Please put Fake.csv and True.csv inside the data folder."
    )

# Load datasets
fake = pd.read_csv(FAKE_PATH)
real = pd.read_csv(REAL_PATH)

# Create labels: 0 = Fake, 1 = Real
fake["label"] = 0
real["label"] = 1

# Combine title and article text
data = pd.concat([fake, real], ignore_index=True)

data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")
data["content"] = (
    data["title"].astype(str) + " " + data["text"].astype(str)
).str.strip()

# Remove empty articles and duplicates
data = data[data["content"].str.len() > 0]
data = data.drop_duplicates(subset=["content"])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data["content"]
y = data["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# TF-IDF + Logistic Regression pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
print("Training model... Please wait.")
model.fit(X_train, y_train)

# Evaluate on held-out test data
predictions = model.predict(X_test)

print("\nAccuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Fake", "Real"]
))

# Save the trained model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fake_news_model.joblib")

print("\nModel saved successfully!")