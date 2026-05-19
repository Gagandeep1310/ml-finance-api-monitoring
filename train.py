"""
ML Model Training Pipeline for Invoice Payment Prediction
Trains a Random Forest classifier to predict late invoice payments
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime
from config import settings


def generate_synthetic_data(n_samples=10000):
    """Generate synthetic invoice payment dataset"""
    np.random.seed(42)
    
    # Generate features related to invoice payment behavior
    data = {
        'invoice_amount': np.random.exponential(scale=5000, size=n_samples),
        'days_until_due': np.random.randint(7, 90, size=n_samples),
        'customer_age_days': np.random.exponential(scale=730, size=n_samples),  # ~2 years avg
        'previous_invoices_count': np.random.poisson(lam=10, size=n_samples),
        'previous_late_payments': np.random.poisson(lam=2, size=n_samples),
        'average_payment_delay': np.random.exponential(scale=5, size=n_samples),  # avg days late
        'customer_credit_score': np.random.normal(700, 100, size=n_samples),
        'invoice_age_days': np.random.randint(0, 60, size=n_samples),
        'outstanding_balance': np.random.exponential(scale=10000, size=n_samples),
        'payment_history_score': np.random.uniform(0, 1, size=n_samples),  # 0=bad, 1=good
        'industry_risk_score': np.random.uniform(0, 1, size=n_samples),
        'economic_indicator': np.random.normal(0, 1, size=n_samples),  # market conditions
    }
    
    df = pd.DataFrame(data)
    
    # Clip credit score to realistic range
    df['customer_credit_score'] = np.clip(df['customer_credit_score'], 300, 850)
    
    # Generate late payment labels (will they pay late? 1=yes, 0=no)
    # ~25% base late payment rate, influenced by various factors
    late_probability = (
        0.15 +  # Base rate
        0.20 * (df['invoice_amount'] > 10000) +  # Large invoices
        0.25 * (df['previous_late_payments'] > 3) +  # History of late payments
        0.20 * (df['average_payment_delay'] > 7) +  # Historically slow payer
        0.15 * (df['customer_credit_score'] < 600) +  # Poor credit
        0.10 * (df['outstanding_balance'] > 20000) +  # High debt
        0.15 * (df['payment_history_score'] < 0.3) +  # Poor payment history
        0.10 * (df['industry_risk_score'] > 0.7) -  # Risky industry
        0.20 * (df['customer_age_days'] > 1095)  # Long-term customers more reliable
    )
    late_probability = np.clip(late_probability, 0, 1)
    df['will_pay_late'] = np.random.binomial(1, late_probability)
    
    return df


def train_model():
    """Train invoice payment prediction model with MLflow tracking"""
    
    print("=" * 60)
    print("🚀 Starting ML Model Training Pipeline")
    print("   Use Case: Invoice Payment Prediction")
    print("=" * 60)
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("invoice_payment_prediction")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"payment_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # 1. Generate data
        print("\n📊 Step 1: Generating synthetic invoice payment dataset...")
        df = generate_synthetic_data(n_samples=50000)
        print(f"   Dataset shape: {df.shape}")
        print(f"   Late payment rate: {df['will_pay_late'].mean():.2%}")
        
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("late_payment_rate", df['will_pay_late'].mean())
        
        # 2. Split features and target
        print("\n🔧 Step 2: Preparing features...")
        X = df.drop('will_pay_late', axis=1)
        y = df['will_pay_late']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"   Train set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        
        # 3. Handle class imbalance with SMOTE
        print("\n⚖️  Step 3: Balancing dataset with SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"   Balanced train set: {len(X_train_balanced)} samples")
        print(f"   Late payment rate after SMOTE: {y_train_balanced.mean():.2%}")
        
        # 4. Feature scaling
        print("\n📏 Step 4: Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        # 5. Train model
        print("\n🤖 Step 5: Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train_balanced)
        print("   ✓ Model training completed")
        
        # Log hyperparameters
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "used_smote": True
        })
        
        # 6. Evaluate model
        print("\n📈 Step 6: Evaluating model performance...")
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n   Confusion Matrix:")
        print(f"   {cm}")
        print(f"\n   TN: {cm[0][0]} | FP: {cm[0][1]}")
        print(f"   FN: {cm[1][0]} | TP: {cm[1][1]}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n   Top 5 Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        # 7. Save model and scaler
        print("\n💾 Step 7: Saving model artifacts...")
        os.makedirs('models', exist_ok=True)
        
        # Save scaler
        joblib.dump(scaler, 'models/scaler.pkl')
        print("   ✓ Scaler saved to models/scaler.pkl")
        
        # Save model with MLflow
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="invoice_payment_model"
        )
        
        # Also save locally for API
        joblib.dump(model, settings.MODEL_PATH)
        print(f"   ✓ Model saved to {settings.MODEL_PATH}")
        
        # Log feature names
        feature_names_path = 'models/feature_names.txt'
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(X.columns))
        mlflow.log_artifact(feature_names_path)
        
        print("\n" + "=" * 60)
        print("✅ Training pipeline completed successfully!")
        print("=" * 60)
        print(f"\n📊 MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"🔗 View results: {settings.MLFLOW_TRACKING_URI}")
        print("\n💡 Business Impact:")
        print(f"   • Precision {precision:.1%} means {precision:.1%} of predicted late payments are correct")
        print(f"   • Recall {recall:.1%} means we catch {recall:.1%} of actual late payments")
        print(f"   • This helps companies prioritize collections and improve cash flow")
        
        return model, scaler, feature_importance


if __name__ == "__main__":
    train_model()
