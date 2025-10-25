

import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# --- optional NLP tools ---
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk_available = True
except Exception:
    nltk_available = False

# --- preprocessing helpers ---
def ensure_nltk():
    """Download small NLTK corpora if missing."""
    if not nltk_available:
        return False
    try:
        stopwords.words("english")
        nltk.data.find("corpora/wordnet")
    except Exception:
        nltk.download("stopwords")
        nltk.download("punkt")
        nltk.download("wordnet")
        nltk.download("omw-1.4")
    return True

def clean_text(text, lemmatizer=None, stop_words=None):
    """Basic text cleaning: lowercase, remove punctuation, tokenize-ish, remove stops, lemmatize."""
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    # remove URLs
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    # remove html tags
    t = re.sub(r"<.*?>", " ", t)
    # remove non-alphanumeric characters (keep spaces)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    # collapse multiple spaces
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    tokens = t.split()  # simple split
    if stop_words:
        tokens = [w for w in tokens if w not in stop_words]
    if lemmatizer:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# --- core training/eval functions ---
def load_dataset(csv_path, text_col="text", label_col="label"):
    df = pd.read_csv(csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{text_col}' and '{label_col}'")
    df = df[[text_col, label_col]].dropna()
    return df

def normalize_labels(labels):
    """Convert labels to 0/1 ints. Accepts 0/1, 'fake'/'real' (case-insensitive)."""
    def map_label(x):
        if isinstance(x, (int, np.integer)):
            return int(x)
        s = str(x).strip().lower()
        if s in ("fake", "0", "false", "f"):
            return 0
        if s in ("real", "1", "true", "t"):
            return 1
        # fallback: try to parse numeric
        try:
            return int(float(s))
        except Exception:
            # treat unknowns as fake (0) — change if you prefer
            return 0
    return labels.apply(map_label)

def build_pipeline():
    """Return a sklearn Pipeline with TF-IDF and LogisticRegression."""
    vect = TfidfVectorizer(
        max_features=20000,        # limit features
        ngram_range=(1,2),         # unigrams + bigrams
        min_df=3,                  # ignore very rare tokens
        stop_words=None            # we already remove stopwords in preprocessing optionally
    )
    clf = LogisticRegression(
        solver="saga",
        max_iter=2000,
        class_weight="balanced",  # help with class imbalance
        random_state=42
    )
    pipeline = Pipeline([
        ("tfidf", vect),
        ("clf", clf)
    ])
    return pipeline

def train_and_evaluate(df, text_col="text", label_col="label", test_size=0.2, random_state=42):
    # optional NLTK setup
    if ensure_nltk():
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
    else:
        stop_words = None
        lemmatizer = None

    # clean texts
    print("Cleaning texts...")
    df["clean_text"] = df[text_col].astype(str).apply(lambda t: clean_text(t, lemmatizer, stop_words))

    # prepare labels
    y = normalize_labels(df[label_col])
    X = df["clean_text"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline()
    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Predicting on test set...")
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, preds))
    return pipeline, (X_test, y_test, preds)

def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    print(f"Saved pipeline to: {path}")

def load_pipeline(path):
    p = joblib.load(path)
    print(f"Loaded pipeline from: {path}")
    return p

def predict_texts(pipeline, texts):
    """Return predicted labels and probabilities for a list of raw texts."""
    # try to pre-clean like training (optional best-effort)
    if ensure_nltk():
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
    else:
        stop_words = None
        lemmatizer = None
    cleaned = [clean_text(t, lemmatizer, stop_words) for t in texts]
    preds = pipeline.predict(cleaned)
    probs = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(cleaned)
    return preds, probs

# --- CLI ---
def main(args):
    csv_path = Path(args.data)
    if not csv_path.exists():
        print(f"Data file not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    df = load_dataset(csv_path, text_col=args.text_col, label_col=args.label_col)
    pipeline, _ = train_and_evaluate(df, text_col=args.text_col, label_col=args.label_col,
                                     test_size=args.test_size, random_state=args.random_state)

    out_path = Path(args.model)
    save_pipeline(pipeline, out_path)

    if args.demo:
        demo_samples = [
            "President signs new bill that helps build more schools.",
            "Shocking! Study proves that drinking gasoline cures all diseases."
        ]
        preds, probs = predict_texts(pipeline, demo_samples)
        print("\nDemo predictions:")
        for i, txt in enumerate(demo_samples):
            print("TEXT:", txt)
            lab = preds[i]
            prob_text = f" (proba: {probs[i].max():.3f})" if probs is not None else ""
            print("PRED:", lab, prob_text)
            print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple Fake News detector.")
    parser.add_argument("--data", type=str, default="fake_news_dataset.csv", help="CSV file with text & label columns")
    parser.add_argument("--text-col", type=str, default="text", help="Name of the text column")
    parser.add_argument("--label-col", type=str, default="label", help="Name of the label column")
    parser.add_argument("--model", type=str, default="fake_news_pipeline.joblib", help="Output path for saved pipeline")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--demo", action="store_true", help="Run demo predictions after training")
    args = parser.parse_args()
    main(args)
