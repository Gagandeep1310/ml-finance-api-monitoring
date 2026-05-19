"""
Database connection and prediction logging
Stores all invoice payment predictions in PostgreSQL for monitoring and analysis
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()


class Prediction(Base):
    """Model for storing invoice payment predictions"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Input features - Invoice details
    invoice_amount = Column(Float)
    days_until_due = Column(Integer)
    customer_age_days = Column(Float)
    previous_invoices_count = Column(Integer)
    previous_late_payments = Column(Integer)
    average_payment_delay = Column(Float)
    customer_credit_score = Column(Float)
    invoice_age_days = Column(Integer)
    outstanding_balance = Column(Float)
    payment_history_score = Column(Float)
    industry_risk_score = Column(Float)
    economic_indicator = Column(Float)
    
    # Prediction outputs
    prediction = Column(Integer)  # 0=on-time, 1=late
    late_payment_probability = Column(Float)
    
    # Metadata
    model_version = Column(String, default="1.0.0")
    api_latency_ms = Column(Float)


# Database engine and session
engine = None
SessionLocal = None


def init_db():
    """Initialize database connection and create tables"""
    global engine, SessionLocal
    
    try:
        engine = create_engine(
            settings.DATABASE_URL,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        engine = None
        SessionLocal = None


def get_db():
    """Get database session"""
    if SessionLocal is None:
        return None
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_prediction(features: dict, prediction: int, probability: float, latency_ms: float):
    """
    Log invoice payment prediction to database
    
    Args:
        features: Input feature dictionary
        prediction: Model prediction (0=on-time, 1=late)
        probability: Late payment probability score
        latency_ms: API latency in milliseconds
    """
    if SessionLocal is None:
        logger.warning("Database not available, skipping prediction logging")
        return
    
    db = SessionLocal()
    try:
        prediction_record = Prediction(
            invoice_amount=features.get('invoice_amount'),
            days_until_due=features.get('days_until_due'),
            customer_age_days=features.get('customer_age_days'),
            previous_invoices_count=features.get('previous_invoices_count'),
            previous_late_payments=features.get('previous_late_payments'),
            average_payment_delay=features.get('average_payment_delay'),
            customer_credit_score=features.get('customer_credit_score'),
            invoice_age_days=features.get('invoice_age_days'),
            outstanding_balance=features.get('outstanding_balance'),
            payment_history_score=features.get('payment_history_score'),
            industry_risk_score=features.get('industry_risk_score'),
            economic_indicator=features.get('economic_indicator'),
            prediction=prediction,
            late_payment_probability=probability,
            model_version=settings.APP_VERSION,
            api_latency_ms=latency_ms
        )
        
        db.add(prediction_record)
        db.commit()
        
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")
        db.rollback()
    finally:
        db.close()
