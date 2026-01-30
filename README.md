 **Bank Loan Approval AI Engine**
 
An enterprise-grade Machine Learning API designed to predict loan approval probabilities with high precision. This system leverages XGBoost, wrapped in a FastAPI backend, and is fully Dockerized for seamless cloud deployment.

**Key Highlights**

Performance: Achieved a 99.97% Test AUC, ensuring high reliability for financial decisions.

Business Logic: Implementation of a custom 0.87 classification threshold to minimize financial defaults and maximize bank ROI.

Modern Stack: Built with FastAPI for low-latency responses and uv for blazing-fast dependency management.

Scalable: Containerized using Docker with a multi-stage build to keep the production image lightweight.

**Domain Intelligence & Feature Engineering**
Unlike "black-box" models, this engine incorporates specific Finance Domain knowledge through real-time feature computation:

Emp_Credit_Stability: Measures borrower reliability by combining employment status and credit history.

Income_Debt_Ratio: Evaluates the borrower's disposable income relative to their debt pressure.

Loan_Eligibility_Factor: A consolidated metric of earning power and creditworthiness.

Net_Financial_Health: A risk-adjusted view of the borrower's financial standing after accounting for current debt ratios.

 **Architecture**
 
User Request (JSON) ➔ FastAPI (Uvicorn) ➔ Real-time Feature Engineering ➔ XGBoost Inference ➔ Risk-Adjusted Decision

**How to Run It**

**Option 1: Running with Docker (use to deploy on clouds)**
Build the Image:

Bash
docker build -t loan-app .
Run the Container:

Bash
docker run -p 8000:8000 loan-app
Access Documentation: Open http://localhost:8000/docs in your browser.

**Option 2: Local Installation**
Clone & Install:

Bash
git clone https://github.com/YOUR_USERNAME/loan-prediction-app.git
pip install -r requirements.txt
Launch Server:

Bash
python -m uvicorn app.main:app --reload

**API Specification**
Endpoint: POST /predict

Sample Payload:

JSON
{
  "Age": 32,
  "Income": 65000,
  "Credit_Score": 720,
  "Loan_Amount": 25000,
  "Emp_Status_Num": 1,
  "DTI_Ratio": 0.35
}
Successful Response:

JSON
{
  "status": "Success",
  "decision": "Approved",
  "probability": 0.9145,
  "threshold": 0.87
}
Developed with a focus on Financial Risk Management & Scalable AI.
