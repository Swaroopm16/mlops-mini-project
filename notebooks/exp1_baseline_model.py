import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import os

# ── NLTK downloads (safe to run every time) ──────────────────────────────────
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(
    'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/refs/heads/main/tweet_emotions.csv'
).drop(columns=['tweet_id'])

# ── Text preprocessing ────────────────────────────────────────────────────────

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    """Convert text to lower case."""
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()   # fixed: raw string
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise

# ── Preprocess ────────────────────────────────────────────────────────────────
df = normalize_text(df)

# Keep only happiness and sadness
df = df[df['sentiment'].isin(['happiness', 'sadness'])]

# Encode labels: sadness=0, happiness=1
df['sentiment'] = df['sentiment'].map({'sadness': 0, 'happiness': 1})  # fixed: map() instead of replace()

# ── Feature extraction ────────────────────────────────────────────────────────
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['content'])
y = df['sentiment']

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── MLflow / DagsHub setup ────────────────────────────────────────────────────
import dagshub
dagshub.init(repo_owner='Swaroopm16', repo_name='mlops-mini-project', )

mlflow.set_tracking_uri('https://dagshub.com/Swaroopm16/mlops-mini-project.mlflow')
mlflow.set_experiment("Logistic Regression Baseline")

# ── Training & logging ────────────────────────────────────────────────────────
with mlflow.start_run():
    # Log preprocessing parameters
    mlflow.log_param("vectorizer", "bag of words")
    mlflow.log_param("num_features", 1000)
    mlflow.log_param("test_size", 0.2)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    mlflow.log_param("model", "Logistic Regression")

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1_score",  f1)

    # Log model — fixed: use 'name' instead of deprecated 'artifact_path'
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name="Tweet-Sentiment-LR"   # fixed: was 'Diabetes-Random-Forest'
    )

    # Log this script as an artifact
    mlflow.log_artifact(__file__)

    # Print results
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")