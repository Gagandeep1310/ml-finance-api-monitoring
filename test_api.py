# Invoice Payment Prediction API - Test Script
import requests
import random

API_URL = "http://localhost:8000"

def generate_invoice():
    """Generate a random invoice for testing"""
    return {
        "invoice_amount": round(random.uniform(100, 10000), 2),
        "days_until_due": random.randint(7, 90),
        "customer_age_days": float(random.randint(30, 2000)),
        "previous_invoices_count": random.randint(0, 50),
        "previous_late_payments": random.randint(0, 10),
        "average_payment_delay": round(random.uniform(0, 15), 2),
        "customer_credit_score": float(random.randint(300, 850)),
        "invoice_age_days": random.randint(0, 60),
        "outstanding_balance": round(random.uniform(0, 50000), 2),
        "payment_history_score": round(random.uniform(0, 1), 3),
        "industry_risk_score": round(random.uniform(0, 1), 3),
        "economic_indicator": round(random.uniform(-2, 2), 3)
    }

if __name__ == "__main__":
    print("Testing Invoice Payment Prediction API")
    print("="*60)
    
    # Health check
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"✓ Health Status: {response.json()['status']}")
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        print("   Make sure it's running: docker-compose up -d")
        exit(1)
    
    # Test prediction with good customer
    print("\n1️⃣ Testing: Good Customer (Low Risk)")
    good_invoice = generate_invoice()
    good_invoice.update({
        "customer_credit_score": 750.0,
        "previous_late_payments": 0,
        "payment_history_score": 0.9,
        "average_payment_delay": 0.5
    })
    
    response = requests.post(f"{API_URL}/predict", json=good_invoice)
    result = response.json()
    
    print(f"   Invoice: ${good_invoice['invoice_amount']:.2f}")
    print(f"   Credit Score: {good_invoice['customer_credit_score']}")
    print(f"   → Will Pay Late: {result['will_pay_late']}")
    print(f"   → Probability: {result['late_payment_probability']:.2%}")
    print(f"   → Risk: {result['risk_level']}")
    
    # Test prediction with risky customer
    print("\n2️⃣ Testing: Risky Customer (High Risk)")
    risky_invoice = generate_invoice()
    risky_invoice.update({
        "invoice_amount": 15000.0,
        "customer_credit_score": 550.0,
        "previous_late_payments": 5,
        "payment_history_score": 0.3,
        "average_payment_delay": 12.0,
        "outstanding_balance": 35000.0
    })
    
    response = requests.post(f"{API_URL}/predict", json=risky_invoice)
    result = response.json()
    
    print(f"   Invoice: ${risky_invoice['invoice_amount']:.2f}")
    print(f"   Credit Score: {risky_invoice['customer_credit_score']}")
    print(f"   → Will Pay Late: {result['will_pay_late']}")
    print(f"   → Probability: {result['late_payment_probability']:.2%}")
    print(f"   → Risk: {result['risk_level']}")
    print(f"   → Action: {result['recommended_action']}")
    
    print("\n✅ Testing completed!")
    print("📊 View API docs: http://localhost:8000/docs")
