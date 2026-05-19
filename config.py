"""
Configuration management for Invoice Payment Prediction API
Handles environment variables and application settings
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Application
    APP_NAME: str = "Invoice Payment Prediction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Database
    DATABASE_URL: str = "postgresql://mluser:mlpassword@postgres:5432/ml_predictions"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "invoice_payment_prediction"
    
    # Model
    MODEL_PATH: str = "models/payment_model.pkl"
    MODEL_NAME: str = "invoice_payment_model"
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 8000
    
    # Prediction thresholds
    LATE_PAYMENT_THRESHOLD: float = 0.5
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
