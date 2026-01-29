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

# --- IMPROVED PATH HANDLING ---
# Isse Docker aur Local dono mein path sahi rehta hai
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bank_loan_ai_v1.joblib")

# Global variables
model = None
best_threshold = 0.5
required_features = []

# Model Loading with Error Handling
if os.path.exists(MODEL_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
        model = bundle['model']
        best_threshold = bundle.get('best_threshold', 0.5)
        required_features = bundle.get('features', [])
        print(f"✅ Model Loaded Successfully! (Threshold: {best_threshold})")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Model file not found at {MODEL_PATH}")

# Input Schema
class UserInput(BaseModel):
    Age: int
    Income: float
    Credit_Score: int
    Loan_Amount: float

@app.get("/")
def health_check():
    return {"status": "Active", "model_loaded": model is not None}

@app.post("/predict")
def predict_loan(data: UserInput):
    # 1. Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    try:
        # 2. Convert to dict (Pydantic v2 recommends model_dump)
        raw_data = data.model_dump()
        
        # 3. FEATURE ENGINEERING
        # (Income * Credit_Score) / Loan_Amount
        stability_score = (raw_data['Income'] * raw_data['Credit_Score']) / raw_data['Loan_Amount']
        
        # 4. Prepare DataFrame
        input_dict = raw_data.copy()
        input_dict['Emp_Credit_Stability'] = stability_score
        
        input_df = pd.DataFrame([input_dict])
        
        # Ensure feature order matches training
        final_df = input_df[required_features] if required_features else input_df
        
        # 5. Model Prediction
        prob = model.predict_proba(final_df)[:, 1][0]
        is_approved = float(prob) >= best_threshold
        
        return {
            "status": "Success",
            "decision": "Approved" if is_approved else "Rejected",
            "probability": round(float(prob), 4),
            "threshold": best_threshold,
            "input_summary": raw_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)