**Bank Loan Approval AI Engine**

An enterprise-grade Machine Learning API designed to predict loan approval probabilities with high precision. This system leverages XGBoost, wrapped in a FastAPI backend, and is fully Dockerized for seamless cloud deployment.

 ***Key Highlights***
 
Performance: Achieved a 99.97% Test AUC, ensuring high reliability for financial decisions.

Business Logic: Implementation of a custom 0.87 classification threshold to minimize financial defaults and maximize bank ROI.

Modern Stack: Built with FastAPI for low-latency responses and uv for blazing-fast dependency management.

Scalable: Containerized using Docker with a multi-stage build to keep the production image lightweight.

 ***Domain Intelligence & Feature Engineering***
 
Unlike "black-box" models, this engine incorporates specific Finance Domain knowledge:

Stability Index: Created a synthetic feature Emp_Credit_Stability using the formula (Income * Credit_Score) / Loan_Amount. This represents a borrower's relative strength against the requested debt.

EDA Insights: Conducted extensive Exploratory Data Analysis to identify non-linear relationships between Age, Income, and Credit History.

Risk-Averse Tuning: The threshold was moved from 0.5 to 0.87 after a profitability audit, prioritizing Precision to protect the bank's capital.

**Architecture***

User Request (JSON) ➔ FastAPI (Uvicorn) ➔ Feature Engineering Pipeline ➔ XGBoost Inference ➔ Risk-Adjusted Decision

 **How to Run It**
 
**Option 1: Running with Docker**

Make sure you have Docker installed on your system.

Build the Image:

Bash
docker build -t loan-app .
Run the Container:

Bash
docker run -p 8000:8000 loan-app
Access Documentation: Open http://localhost:8000/docs in your browser to test the API via Swagger UI.

**Option 2: Local Installation (For Development)**

Clone the Repository:

Bash
git clone https://github.com/YOUR_USERNAME/loan-prediction-app.git
cd loan-prediction-app
Install Dependencies:

Bash
pip install -r requirements.txt
Launch the Server:

Bash
python -m uvicorn app.main:app --reload
API Specification
Endpoint: POST /predict

**Sample Payload:**

JSON
{
  "Age": 32,
  "Income": 65000,
  "Credit_Score": 720,
  "Loan_Amount": 25000
}
Response:

JSON
{
  "status": "Success",
  "decision": "Approved",
  "probability": 0.9145,
  "threshold": 0.87
}
