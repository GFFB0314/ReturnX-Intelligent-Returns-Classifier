"""
FastAPI application for ReturnX Intelligent Returns Classifier.
Provides endpoints for health checks and return category prediction.
"""

import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import load_model_artifacts, predict_category

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ReturnX Intelligent Returns API",
    description="Automated routing and classification of e-commerce returns.",
    version="1.0.0",
)

# Load artifacts once at startup
try:
    model, label_encoder, tfidf = load_model_artifacts()
    logger.info("Model artifacts loaded successfully for API.")
except (IOError, ValueError, RuntimeError) as e:
    logger.error("Failed to load artifacts: %s", e)
    raise RuntimeError("API could not start: Artifacts missing.") from e


class PredictionRequest(BaseModel):
    """Data model for a return category prediction request."""

    complaint_text: str
    age: int = 40
    rating: int = 3


class PredictionResponse(BaseModel):
    """Data model for a return category prediction response."""

    category: str
    confidence: float
    probabilities: Dict[str, float]


@app.get("/")
def read_root() -> Dict[str, str]:
    """Root endpoint providing a welcome message."""
    return {"message": "Welcome to ReturnX API. Visit /docs for documentation."}


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint to verify API availability."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> Dict[str, Any]:
    """
    Predicts the return category for a given customer complaint.
    """
    if not request.complaint_text.strip():
        raise HTTPException(status_code=400, detail="Complaint text cannot be empty.")

    try:
        result = predict_category(
            complaint_text=request.complaint_text,
            age=request.age,
            rating=request.rating,
            model=model,
            label_encoder=label_encoder,
            tfidf=tfidf,
        )
        return result
    except (ValueError, TypeError) as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Internal prediction error.") from e
