import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load data from SQLite database
def load_data(db_path="imdb_reviews.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM imdb_reviews", conn)
    conn.close()
    return df

# Perform EDA
def generate_eda_report(df):
    print("\n--- Basic Information ---")
    print(df.info())
    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Sentiment Distribution ---")
    print(df['sentiment'].value_counts())

    # Sentiment Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='sentiment', data=df, palette='viridis')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

    # Review Length Analysis
    df['review_length'] = df['review_text'].apply(len)
    print("\n--- Average Review Length ---")
    print(df.groupby('sentiment')['review_length'].mean())

    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='review_length', hue='sentiment', bins=50, kde=True, palette='viridis')
    plt.title("Review Length Distribution by Sentiment")
    plt.xlabel("Review Length")
    plt.ylabel("Count")
    plt.show()

    # WordCloud for Positive and Negative Reviews
    positive_reviews = " ".join(df[df['sentiment'] == 'positive']['review_text'])
    negative_reviews = " ".join(df[df['sentiment'] == 'negative']['review_text'])

    wordcloud_pos = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(positive_reviews)
    wordcloud_neg = WordCloud(width=800, height=400, background_color="white", colormap="magma").generate(negative_reviews)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud_pos, interpolation="bilinear")
    plt.title("WordCloud for Positive Reviews")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud_neg, interpolation="bilinear")
    plt.title("WordCloud for Negative Reviews")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    df = load_data()
    generate_eda_report(df)
