# Sentiment Analysis with Flask API

This project performs sentiment analysis on IMDB movie reviews. It includes scripts for dataset acquisition, model training, and serving predictions via a Flask API.

---

## Project Setup

### Prerequisites
- Python 3.7 or above
- SQLite

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/anubhavanand1516/anubhavanand1516-End-to-End-Sentiment-Analysis-Pipeline.git
   cd End-to-End-Sentiment-Analysis-Pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the database:
   - Download the IMDB dataset and store it in an SQLite database by running the script:
     ```bash
     python store_dataset.py
     ```

---

## Data Acquisition

- **Dataset Used**: IMDB Movie Reviews dataset from the [Hugging Face Datasets Library](https://huggingface.co/datasets/imdb).
- The dataset was downloaded and converted into a pandas DataFrame using the `datasets` library. It was then stored in a SQLite database (`imdb_reviews.db`) for efficient retrieval.

Steps to acquire and prepare the dataset:
1. Run the script to download and process the dataset:
   ```bash
   python store_dataset.py
   ```

---

## Run Instructions

### 1. Train the Model
Train the sentiment analysis model using the preprocessed data:
```bash
python train_model.py
```

- The script:
  - Loads data from the SQLite database.
  - Splits the dataset into training and testing sets.
  - Vectorizes text data using TF-IDF.
  - Trains a Logistic Regression model.
  - Saves the model and vectorizer as `model.pkl` and `vectorizer.pkl`.

---

### 2. Start the Flask Server
Launch the Flask API server to serve predictions:
```bash
python app.py
```

By default, the server will run at `http://127.0.0.1:5000`.

---

### 3. Test the Endpoint
Send POST requests to the `/predict` endpoint with movie reviews to get sentiment predictions.

#### Example cURL Command:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"review_text": "This movie was amazing and full of heartwarming moments!"}' http://127.0.0.1:5000/predict
```

#### Example Response:
```json
{
    "sentiment_prediction": "positive"
}
```

#### Batch Prediction (Optional):
If extended, send multiple reviews in a single request:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"reviews": ["This movie was amazing!", "I hated the plot."]}' http://127.0.0.1:5000/predict
```

---

## Model Info

- **Model Used**: Logistic Regression
- **Feature Extraction**: TF-IDF vectorization with a max of 5000 features.
- **Performance**: Final evaluation on the test set yielded the following results:
  - Example (customize with actual results):
    ```
    Precision: 0.88
    Recall: 0.87
    F1-Score: 0.87
    ```
- The model was trained on the IMDB dataset and demonstrates robust performance for binary sentiment classification.

---

## Additional Assets (Optional)

- If available, include:
  - for generating Exploratory Data Analysis (EDA):
  - Run
      ```bash
       python eda.py
       ```
  

---

