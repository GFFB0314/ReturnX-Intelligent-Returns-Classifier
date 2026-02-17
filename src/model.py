"""
Model Training Module for ReturnX Project.

This module handles:
1. Building the classification pipeline (Imbalanced-Learn Pipeline + SMOTE + XGBoost).
2. Training the model using GridSearchCV for hyperparameter tuning.
3. Saving the best performing model and necessary artifacts for inference.

Aligns with 'notebooks/03_nlp_feature_engineering.ipynb'.
"""

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline
from scipy.sparse import hstack, csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from xgboost import XGBClassifier

from src.logging_utils import setup_logger

# Configure logging using shared utility
logger = setup_logger(__name__)


def build_pipeline():
    """
    Constructs the training pipeline architecture.
    Detailed construction happens dynamically in train_model to handle sparse matrix dimensions.
    kept as placeholder for future modularity if needed.
    """


# pylint: disable=too-many-locals
def train_model(df: pd.DataFrame):
    """
    Trains the XGBoost model using GridSearchCV and saves artifacts.

    Args:
        df (pd.DataFrame): Preprocessed dataframe containing 'clean_text', 'age',
                           'rating', 'word_count', 'return_category'.
    """
    logger.info("Starting model training process...")

    try:
        # 1. Label Encoding
        logger.info("Encoding target labels...")
        le = LabelEncoder()
        y = le.fit_transform(df["return_category"])

        # 2. Text Vectorization (Fit & Save)
        logger.info("Fitting TfidfVectorizer...")
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            stop_words=None,
            min_df=5,
            max_df=0.95,
        )
        # pylint: disable=invalid-name
        X_text = tfidf.fit_transform(df["clean_text"])

        # Save vectorizer immediately
        tfidf_path = "src/tfidf.pkl"
        joblib.dump(tfidf, tfidf_path)
        logger.info("TfidfVectorizer saved to %s", tfidf_path)

        # 3. Prepare Numeric Features & Combine
        numeric_features = df[["age", "rating", "word_count"]]
        X_numeric = csr_matrix(numeric_features.values)

        X = hstack([X_numeric, X_text], dtype=np.float32)
        logger.info("Feature matrix created. Shape: %s", X.shape)

        # 4. Determine Indices for ColumnTransformer
        # Numeric are first 3 columns, text are the rest
        num_indices = [0, 1, 2]
        text_indices = list(range(3, X.shape[1]))

        # 5. Build Pipeline
        logger.info(
            "Building imbalanced-learn pipeline (ColumnTransformer -> SMOTE -> XGBoost)..."
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric_features", MaxAbsScaler(), num_indices),
                ("text_features", MaxAbsScaler(), text_indices),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

        pipeline = IMBPipeline(
            steps=[
                ("processor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                (
                    "xgbc",
                    XGBClassifier(
                        objective="multi:softprob",
                        random_state=42,
                        tree_method="hist",
                        n_estimators=200,
                        n_jobs=1,
                    ),
                ),
            ]
        )

        # 6. GridSearchCV
        logger.info("Configuring GridSearchCV...")
        param_grid = {
            "xgbc__max_depth": [4, 6],
            "xgbc__learning_rate": [0.1],
            "xgbc__subsample": [0.8, 1.0],
            "xgbc__colsample_bytree": [0.8, 1.0],
        }

        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        model = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=kf,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
        )

        logger.info("Fitting model (this may take a while)...")
        model.fit(X, y)

        logger.info("Best CV F1_MACRO Score: %.4f", model.best_score_)
        logger.info("Best Params: %s", model.best_params_)

        # 7. Save Artifacts
        best_estimator = model.best_estimator_

        # Save independent best model for deployment (not gitignored)
        best_model_path = "src/best_model.pkl"
        joblib.dump(best_estimator, best_model_path)
        logger.info("Best estimator saved to %s (Deployment Artifact)", best_model_path)

        # Save LabelEncoder
        le_path = "src/le.pkl"
        joblib.dump(le, le_path)
        logger.info("LabelEncoder saved to %s", le_path)

    except (ValueError, TypeError, RuntimeError) as e:
        logger.error("Training failed: %s", e)
        raise e


# pylint: disable=unused-argument, invalid-name
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data.
    """
