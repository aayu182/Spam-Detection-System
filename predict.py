import os
import numpy as np

try:
    import joblib
except ImportError:
    import sklearn.externals.joblib as joblib


def load_model(model_path):
    """
    Load a joblib-saved sklearn pipeline.
    Returns the pipeline, or None if the file does not exist.
    """
    if not os.path.isfile(model_path):
        return None
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def validate_input(text):
    """
    Return True if text is non-empty and not purely whitespace.
    """
    if text is None:
        return False
    return bool(text.strip())


def predict_message(pipeline, text):
    """
    Predict whether a message is SPAM or HAM.

    Returns a dict:
        {
            'label': 'SPAM' or 'HAM',
            'probability': float (0-1) or None,
            'confidence': str description
        }
    """
    if pipeline is None:
        return {'label': 'UNKNOWN', 'probability': None, 'confidence': 'Model not loaded'}

    raw_label = pipeline.predict([text])[0]  # 'spam' or 'ham'
    label = 'SPAM' if raw_label == 'spam' else 'HAM'

    probability = None
    confidence = 'N/A'

    # Try predict_proba (works for NB and LR, not LinearSVC by default)
    if hasattr(pipeline, 'predict_proba'):
        try:
            proba = pipeline.predict_proba([text])[0]
            classes = pipeline.classes_
            spam_index = list(classes).index('spam') if 'spam' in classes else 1
            probability = float(proba[spam_index])
            if probability >= 0.8:
                confidence = 'Very High'
            elif probability >= 0.6:
                confidence = 'High'
            elif probability >= 0.4:
                confidence = 'Moderate'
            else:
                confidence = 'Low'
        except Exception:
            probability = None
            confidence = 'N/A'
    elif hasattr(pipeline, 'decision_function'):
        # For LinearSVC: use sigmoid of decision score as pseudo-probability
        try:
            score = pipeline.decision_function([text])[0]
            # sigmoid transform
            probability = float(1 / (1 + np.exp(-score)))
            if probability >= 0.8:
                confidence = 'Very High'
            elif probability >= 0.6:
                confidence = 'High'
            elif probability >= 0.4:
                confidence = 'Moderate'
            else:
                confidence = 'Low'
        except Exception:
            probability = None
            confidence = 'N/A'

    return {'label': label, 'probability': probability, 'confidence': confidence}
