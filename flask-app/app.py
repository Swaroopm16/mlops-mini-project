import pickle
from pathlib import Path

from flask import Flask, render_template, request

from preprocessing import normalize_text

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"


def load_pickle(file_name):
    with open(MODEL_DIR / file_name, "rb") as file:
        return pickle.load(file)


model = load_pickle("model.pkl")
vectorizer = load_pickle("vectorizer.pkl")


def predict_sentiment(text):
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    return "Positive" if prediction == 1 else "Negative"

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    #clean
    cleaned_text = normalize_text(text)
    #bow
    

    #predict
    prediction = predict_sentiment(cleaned_text)
    return render_template(
        'index.html',
        original_text=text,
        cleaned_text=cleaned_text,
        prediction=prediction,
    )


if __name__ == '__main__':
    app.run(debug=True)