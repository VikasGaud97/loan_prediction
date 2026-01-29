from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import os

app = FastAPI(title="Loan Approval AI Engine")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PATH HANDLING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bank_loan_ai_v1.joblib")

# Global variables
model = None
best_threshold = 0.87
required_features = []

# Model Loading
if os.path.exists(MODEL_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
        model = bundle['model']
        best_threshold = bundle.get('best_threshold', 0.87)
        required_features = bundle.get('features', [])
        print(f"✅ Model Loaded Successfully! (Threshold: {best_threshold})")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Model file not found at {MODEL_PATH}")

# Input Schema (Updated to include Emp_Status_Num and DTI_Ratio)
class UserInput(BaseModel):
    Age: int
    Income: float
    Credit_Score: int
    Loan_Amount: float
    Emp_Status_Num: int  # Added based on your formulas
    DTI_Ratio: float     # Added based on your formulas

@app.get("/")
def health_check():
    return {"status": "Active", "model_loaded": model is not None}

@app.post("/predict")
def predict_loan(data: UserInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    try:
        # 1. Get raw input data
        raw_data = data.model_dump()
        df = pd.DataFrame([raw_data])
        
        # 2. EXACT FEATURE ENGINEERING (As per your provided formulas)
        df['Emp_Credit_Stability'] = df['Emp_Status_Num'] * df['Credit_Score']
        df['Income_Debt_Ratio'] = df['Income'] / (df['DTI_Ratio'] + 0.01)
        df['Loan_Eligibility_Factor'] = df['Income'] * df['Credit_Score']
        df['Net_Financial_Health'] = (df['Income'] * df['Emp_Status_Num']) - df['DTI_Ratio']
        
        # 3. Filter and Order Features
        if required_features:
            # Check if any required feature is missing
            missing = [f for f in required_features if f not in df.columns]
            if missing:
                raise ValueError(f"Missing engineered features: {missing}")
            final_df = df[required_features]
        else:
            final_df = df
        
        # 4. Model Prediction
        prob = model.predict_proba(final_df)[:, 1][0]
        is_approved = float(prob) >= best_threshold
        
        return {
            "status": "Success",
            "decision": "Approved" if is_approved else "Rejected",
            "probability": round(float(prob), 4),
            "threshold": best_threshold
        }
        
    except Exception as e:
        # Detailed error for Render logs
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)