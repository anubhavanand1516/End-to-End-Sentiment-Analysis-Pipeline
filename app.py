from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review_text = data.get("review_text", "")

    if not review_text:
        return jsonify({"error": "No review text provided"}), 400

    # Preprocess and predict
    review_tfidf = vectorizer.transform([review_text])
    prediction = model.predict(review_tfidf)[0]

    return jsonify({"sentiment_prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)

