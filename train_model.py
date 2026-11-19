import os
import sys
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Ensure imports work regardless of cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from preprocessing import load_dataset, inspect_dataset, clean_dataset, get_class_distribution


def build_pipelines():
    """Return a dict of name -> sklearn Pipeline."""
    def tfidf():
        return TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english'
        )
    return {
        'MultinomialNB': Pipeline([
            ('tfidf', tfidf()),
            ('clf', MultinomialNB(alpha=0.1))
        ]),
        'LogisticRegression': Pipeline([
            ('tfidf', tfidf()),
            ('clf', LogisticRegression(max_iter=1000, C=5, random_state=42))
        ]),
        'LinearSVC': Pipeline([
            ('tfidf', tfidf()),
            ('clf', LinearSVC(C=1.0, max_iter=2000, random_state=42))
        ]),
    }


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    """Fit pipeline on raw text, evaluate on test set, return metrics dict."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='spam', zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label='spam', zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label='spam', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    cr = classification_report(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"Model: {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (spam)")
    print(f"  Recall    : {rec:.4f}  (spam)")
    print(f"  F1-score  : {f1:.4f}  (spam)")
    print(f"\nConfusion Matrix (ham | spam):")
    print(cm)
    print(f"\nClassification Report:")
    print(cr)

    return {'name': name, 'pipeline': pipeline, 'accuracy': acc,
            'precision': prec, 'recall': rec, 'f1': f1}


def select_best(results):
    """Pick best model: highest combined score of F1 + spam recall."""
    def score(r):
        return r['f1'] * 0.6 + r['recall'] * 0.4
    best = max(results, key=score)
    print(f"\n{'=' * 60}")
    print(f"Best model selected: {best['name']}")
    print(f"  F1={best['f1']:.4f}  Recall={best['recall']:.4f}  Accuracy={best['accuracy']:.4f}")
    print(f"{'=' * 60}")
    return best


def main():
    data_path = os.path.join(SCRIPT_DIR, 'data', 'spam.csv')
    model_path = os.path.join(SCRIPT_DIR, 'models', 'spam_detection_model.pkl')

    print("Loading dataset...")
    df = load_dataset(data_path)
    inspect_dataset(df)

    print("\nCleaning dataset...")
    df = clean_dataset(df)
    get_class_distribution(df)

    X = df['text']
    y = df['label']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")

    pipelines = build_pipelines()
    results = []
    for name, pipeline in pipelines.items():
        metrics = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(metrics)

    best = select_best(results)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best['pipeline'], model_path)
    print(f"\nModel saved to {model_path}")
    print("Training complete!")


if __name__ == '__main__':
    main()
