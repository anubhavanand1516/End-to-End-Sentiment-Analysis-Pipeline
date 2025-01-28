import sqlite3
import pandas as pd
from datasets import load_dataset

# Load dataset
def download_dataset():
    print("Downloading IMDB dataset...")
    dataset = load_dataset("imdb")
    df = pd.concat([dataset['train'].to_pandas(), dataset['test'].to_pandas()])
    df = df.rename(columns={"text": "review_text", "label": "sentiment"})
    df['sentiment'] = df['sentiment'].map({0: "negative", 1: "positive"})
    return df

# Store dataset in SQLite
def store_in_database(df, db_path="imdb_reviews.db"):
    print(f"Storing data in {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS imdb_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT,
            sentiment TEXT
        )
    """)
    df.to_sql("imdb_reviews", conn, if_exists="replace", index=False)
    conn.close()
    print("Data successfully stored.")

if __name__ == "__main__":
    df = download_dataset()
    store_in_database(df)
