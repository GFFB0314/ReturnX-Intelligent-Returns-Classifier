"""
Inference Module for ReturnX Project.

This module handles:
1. Loading the trained XGBoost model and artifacts.
2. Preprocessing new complaints (text cleaning + feature engineering).
3. Generating predictions with confidence scores.

Aligns with 'notebooks/03_nlp_feature_engineering.ipynb' logic.
"""

import os
import joblib
import numpy as np
import logging
import sys
from typing import Dict, Tuple
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import streamlit as st

from src.preprocessing import clean_review_text, calculate_word_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Manual cache for internal use (especially for API)
_artifacts_cache = None

def load_model_artifacts() -> Tuple[object, LabelEncoder, TfidfVectorizer]:
    """
    Loads all necessary model artifacts for inference:
    - Best XGBoost Model (best_model.pkl)
    - Label Encoder (le.pkl)
    - TF-IDF Vectorizer (tfidf.pkl)
    
    Returns:
        Tuple containing (model, label_encoder, tfidf_vectorizer)
    """
    global _artifacts_cache
    if _artifacts_cache is not None:
        return _artifacts_cache

    logger.info("Loading model artifacts...")
    
    base_path = "src"
    paths = {
        "model": os.path.join(base_path, "best_model.pkl"),
        "le": os.path.join(base_path, "le.pkl"),
        "tfidf": os.path.join(base_path, "tfidf.pkl")
    }
    
    # Verify all files exist
    for name, path in paths.items():
        if not os.path.exists(path):
            error_msg = f"Artifact '{name}' not found at {path}. Run training first."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
    try:
        model = joblib.load(paths["model"])
        le = joblib.load(paths["le"])
        tfidf = joblib.load(paths["tfidf"])
        
        logger.info("All artifacts loaded successfully.")
        _artifacts_cache = (model, le, tfidf)
        return _artifacts_cache
        
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        raise e

def predict_category(
    complaint_text: str,
    age: int,
    rating: int,
    model: object,
    label_encoder: LabelEncoder,
    tfidf: TfidfVectorizer
) -> Dict:
    """
    Predicts the return category for a given complaint.
    
    Args:
        complaint_text (str): Raw customer complaint.
        age (int): Customer age.
        rating (int): Product rating (1-5).
        model: Trained XGBClassifier (or pipeline).
        label_encoder: Fitted LabelEncoder.
        tfidf: Fitted TfidfVectorizer.
        
    Returns:
        Dict: Classification result with category, confidence, and probability distribution.
    """
    # Step 1: Clean text
    cleaned_text = clean_review_text(complaint_text)
    
    # Step 2: Calculate word count from ORIGINAL text
    word_count = calculate_word_count(complaint_text)
    
    # Step 3: Create numeric features (Order: age, rating, word_count)
    numeric_features = np.array([[age, rating, word_count]], dtype=np.float32)
    
    # Step 4: Vectorize cleaned text
    X_text = tfidf.transform([cleaned_text])
    
    # Step 5: Combine features using hstack (CSR matrix + CSR matrix)
    X = hstack([csr_matrix(numeric_features), X_text], dtype=np.float32)
    
    # Step 6: Probability Prediction
    try:
        y_proba = model.predict_proba(X)[0]
        y_pred_idx = np.argmax(y_proba)
        
        # Determine category label
        category = label_encoder.inverse_transform([y_pred_idx])[0]
        confidence = float(y_proba[y_pred_idx])
        
        # Build probability dictionary
        all_categories = label_encoder.classes_
        probabilities = {
            cat: float(prob) for cat, prob in zip(all_categories, y_proba)
        }
        
        return {
            "category": category,
            "confidence": confidence,
            "probabilities": probabilities
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise e
