from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import load_model_artifacts, predict_category
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ReturnX Intelligent Returns API",
    description="Automated routing and classification of e-commerce returns.",
    version="1.0.0"
)

# Load artifacts once at startup
try:
    model, label_encoder, tfidf = load_model_artifacts()
    logger.info("Model artifacts loaded successfully for API.")
except Exception as e:
    logger.error(f"Failed to load artifacts: {e}")
    raise RuntimeError("API could not start: Artifacts missing.")

class PredictionRequest(BaseModel):
    complaint_text: str
    age: int = 40
    rating: int = 3

class PredictionResponse(BaseModel):
    category: str
    confidence: float
    probabilities: dict

@app.get("/")
def read_root():
    return {"message": "Welcome to ReturnX API. Visit /docs for documentation."}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.complaint_text.strip():
        raise HTTPException(status_code=400, detail="Complaint text cannot be empty.")
    
    try:
        result = predict_category(
            complaint_text=request.complaint_text,
            age=request.age,
            rating=request.rating,
            model=model,
            label_encoder=label_encoder,
            tfidf=tfidf
        )
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")
