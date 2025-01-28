import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Load data from SQLite
def load_data(db_path="imdb_reviews.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM imdb_reviews", conn)
    conn.close()
    return df

# Preprocess data and train model
def train_model(df):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(df['review_text'], df['sentiment'], test_size=0.2, random_state=42)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    predictions = model.predict(X_test_tfidf)
    print(classification_report(y_test, predictions))

    # Save the model and vectorizer
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    print("Model and vectorizer saved!")

if __name__ == "__main__":
    df = load_data()
    train_model(df)
