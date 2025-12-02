import joblib
from src.config.config import MODEL_PATH

def save_model(final_model):
    """Save trained model object into disk."""
    joblib.dump(final_model, MODEL_PATH)

def load_model():
    """Load model object from disk and return it."""
    return joblib.load(MODEL_PATH)