"""
FastAPI ML Model API with Monitoring
Production-ready invoice payment prediction API with Prometheus metrics and PostgreSQL logging
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import joblib
import numpy as np
import time
import logging
from contextlib import asynccontextmanager
from config import settings
from database import init_db, log_prediction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and scaler
model = None
scaler = None
feature_names = None

# Custom Prometheus metrics
prediction_counter = Counter(
    'ml_predictions_total', 
    'Total number of predictions made',
    ['prediction_class']
)

payment_delay_probability_histogram = Histogram(
    'ml_payment_delay_probability', 
    'Distribution of late payment probabilities'
)

late_payments_detected_gauge = Gauge(
    'ml_late_payments_detected_current', 
    'Current number of late payments predicted (last 100 predictions)'
)

model_latency = Histogram(
    'ml_model_inference_latency_seconds',
    'Model inference latency in seconds'
)


# Request/Response models
class InvoiceFeatures(BaseModel):
    """Input features for invoice payment prediction"""
    invoice_amount: float = Field(..., gt=0, description="Invoice amount in dollars")
    days_until_due: int = Field(..., ge=1, le=180, description="Days until invoice is due")
    customer_age_days: float = Field(..., ge=0, description="Customer account age in days")
    previous_invoices_count: int = Field(..., ge=0, description="Number of previous invoices")
    previous_late_payments: int = Field(..., ge=0, description="Number of previous late payments")
    average_payment_delay: float = Field(..., ge=0, description="Average days payment was late historically")
    customer_credit_score: float = Field(..., ge=300, le=850, description="Customer credit score (300-850)")
    invoice_age_days: int = Field(..., ge=0, description="Days since invoice was issued")
    outstanding_balance: float = Field(..., ge=0, description="Customer's outstanding balance")
    payment_history_score: float = Field(..., ge=0, le=1, description="Payment history quality score (0-1)")
    industry_risk_score: float = Field(..., ge=0, le=1, description="Industry risk score (0-1)")
    economic_indicator: float = Field(..., description="Economic indicator (market conditions)")
    
    class Config:
        schema_extra = {
            "example": {
                "invoice_amount": 5000.00,
                "days_until_due": 30,
                "customer_age_days": 730.0,
                "previous_invoices_count": 15,
                "previous_late_payments": 2,
                "average_payment_delay": 3.5,
                "customer_credit_score": 720.0,
                "invoice_age_days": 5,
                "outstanding_balance": 8000.00,
                "payment_history_score": 0.75,
                "industry_risk_score": 0.4,
                "economic_indicator": 0.2
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response"""
    will_pay_late: bool
    late_payment_probability: float
    risk_level: str
    recommended_action: str
    timestamp: str
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting Invoice Payment Prediction API...")
    load_model()
    init_db()
    logger.info("✓ API ready to serve predictions")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Invoice Payment API...")


# Initialize FastAPI app
app = FastAPI(
    title="Invoice Payment Prediction API",
    version=settings.APP_VERSION,
    description="Production ML API for predicting invoice payment delays with monitoring",
    lifespan=lifespan
)

# Setup Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def load_model():
    """Load ML model and preprocessing artifacts"""
    global model, scaler, feature_names
    
    try:
        model = joblib.load(settings.MODEL_PATH)
        scaler = joblib.load('models/scaler.pkl')
        
        with open('models/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        logger.info(f"✓ Model loaded from {settings.MODEL_PATH}")
        logger.info(f"✓ Features: {feature_names}")
        
    except FileNotFoundError:
        logger.error("✗ Model files not found. Please run train.py first!")
        raise RuntimeError("Model not found. Run training pipeline first.")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        raise


def get_risk_level(probability: float) -> str:
    """Determine risk level based on late payment probability"""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"


def get_recommended_action(probability: float, days_until_due: int) -> str:
    """Recommend action based on prediction"""
    if probability < 0.3:
        return "No action needed - customer likely to pay on time"
    elif probability < 0.5:
        return "Send payment reminder 3-5 days before due date"
    elif probability < 0.7:
        return "Contact customer proactively, offer payment plan if needed"
    else:
        if days_until_due > 14:
            return "High risk - contact immediately, consider requiring deposit"
        else:
            return "Very high risk - escalate to collections team"


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Invoice Payment Prediction API",
        "version": settings.APP_VERSION,
        "status": "operational",
        "description": "Predicts whether invoices will be paid on time to optimize cash flow management",
        "endpoints": {
            "predictions": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    scaler_loaded = scaler is not None
    
    if not (model_loaded and scaler_loaded):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "scaler_loaded": scaler_loaded,
        "version": settings.APP_VERSION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_payment(features: InvoiceFeatures):
    """
    Predict whether an invoice will be paid late
    
    Returns:
    - Binary prediction (will_pay_late: true/false)
    - Probability score (0-1)
    - Risk level (LOW/MEDIUM/HIGH)
    - Recommended action
    """
    start_time = time.time()
    
    try:
        # Convert features to array
        feature_dict = features.dict()
        feature_array = np.array([[
            feature_dict['invoice_amount'],
            feature_dict['days_until_due'],
            feature_dict['customer_age_days'],
            feature_dict['previous_invoices_count'],
            feature_dict['previous_late_payments'],
            feature_dict['average_payment_delay'],
            feature_dict['customer_credit_score'],
            feature_dict['invoice_age_days'],
            feature_dict['outstanding_balance'],
            feature_dict['payment_history_score'],
            feature_dict['industry_risk_score'],
            feature_dict['economic_indicator']
        ]])
        
        # Scale features
        feature_scaled = scaler.transform(feature_array)
        
        # Make prediction
        prediction = int(model.predict(feature_scaled)[0])
        probability = float(model.predict_proba(feature_scaled)[0][1])
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Update Prometheus metrics
        prediction_counter.labels(prediction_class=str(prediction)).inc()
        payment_delay_probability_histogram.observe(probability)
        model_latency.observe(time.time() - start_time)
        
        # Log to database (async in background)
        try:
            log_prediction(feature_dict, prediction, probability, latency_ms)
        except Exception as e:
            logger.warning(f"Failed to log prediction to DB: {e}")
        
        # Build response
        risk_level = get_risk_level(probability)
        recommended_action = get_recommended_action(probability, feature_dict['days_until_due'])
        
        response = PredictionResponse(
            will_pay_late=bool(prediction),
            late_payment_probability=round(probability, 4),
            risk_level=risk_level,
            recommended_action=recommended_action,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            model_version=settings.APP_VERSION
        )
        
        logger.info(
            f"Prediction: {prediction} | Probability: {probability:.4f} | "
            f"Risk: {risk_level} | Latency: {latency_ms:.2f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/stats", tags=["Statistics"])
async def get_stats():
    """Get API statistics"""
    return {
        "message": "Statistics endpoint",
        "note": "View detailed metrics at /metrics for Prometheus"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
