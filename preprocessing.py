import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_dataset(filepath):
    """Load spam.csv with pandas, trying latin-1 encoding as fallback."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')
    return df


def inspect_dataset(df):
    """Print shape, column names, dtypes, and first 5 rows."""
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print("=" * 60)


def clean_dataset(df):
    """Drop Unnamed cols, handle NaN, deduplicate, normalise labels, strip whitespace."""
    # Rename v1 -> label, v2 -> text if those column names are present
    col_map = {}
    if 'v1' in df.columns:
        col_map['v1'] = 'label'
    if 'v2' in df.columns:
        col_map['v2'] = 'text'
    if col_map:
        df = df.rename(columns=col_map)

    # Drop Unnamed columns
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"Dropped columns: {unnamed_cols}")

    # Keep only label and text
    df = df[['label', 'text']].copy()

    print(f"Shape before cleaning: {df.shape}")

    # Drop rows with NaN in either column
    df = df.dropna(subset=['label', 'text'])

    # Normalise labels to lowercase 'ham' / 'spam'
    df['label'] = df['label'].str.strip().str.lower()
    valid_labels = df['label'].isin(['ham', 'spam'])
    removed_invalid = (~valid_labels).sum()
    if removed_invalid > 0:
        print(f"Removed {removed_invalid} rows with invalid labels.")
    df = df[valid_labels].copy()

    # Clean text: strip leading/trailing whitespace, collapse multiple spaces
    df['text'] = df['text'].str.strip()
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)

    # Remove duplicates
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['text'])
    after_dedup = len(df)
    print(f"Removed {before_dedup - after_dedup} duplicate records.")

    print(f"Shape after cleaning: {df.shape}")
    return df.reset_index(drop=True)


def get_class_distribution(df):
    """Return value_counts of the label column."""
    dist = df['label'].value_counts()
    print("\nClass distribution:")
    print(dist)
    print(f"Spam ratio: {dist.get('spam', 0) / len(df) * 100:.2f}%")
    return dist


def split_and_vectorize(df):
    """
    80:20 stratified split, TF-IDF fitted only on train.
    Returns X_train, X_test, y_train, y_test, vectorizer.
    """
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english'
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"\nTrain size: {X_train_tfidf.shape}, Test size: {X_test_tfidf.shape}")
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
