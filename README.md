# DonorPulse – Explainable AI for Fundraising Risk and Donor Retention

DonorPulse is a data analytics and machine learning project designed to help fundraising organisations monitor campaign performance, identify operational risks, predict donor conversion likelihood, and prioritise follow-ups.

The project combines an interactive Streamlit dashboard with machine learning and explainable AI techniques to turn fundraising interaction data into actionable insights.

> **Note:** This project currently uses a synthetic dataset created for demonstration and learning purposes. It is not validated for real-world fundraising decisions.

## Key Features

### 1. Campaign Overview
- Monitor total donations, interactions, and conversion rates.
- Compare campaign performance.
- Track monthly fundraising trends.

### 2. Campaign Health Monitoring
- Analyse campaign and agent performance.
- Identify declining conversion trends.
- Monitor duplicate contacts, invalid contacts, and pending follow-ups.

### 3. Anomaly Detection
- Identify unusual patterns in fundraising operations.
- Flag potential duplicate-contact spikes, invalid-contact increases, and missed follow-ups.
- Highlight areas that may require investigation.

### 4. Conversion Prediction
- Use a Logistic Regression model to estimate donor conversion likelihood.
- Explore prediction results for individual interactions.

### 5. Explainable AI
- Use SHAP to understand which features influence model predictions.
- Explore individual feature contributions.
- Make model outputs easier to interpret.

### 6. Follow-up Recommendations
- Prioritise eligible donor follow-ups.
- Generate recommended actions based on interaction history.
- Filter and export recommendations for further analysis.

## Tech Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## Project Structure

```text
DonorPulse/
├── app.py
├── data/
│   ├── donorpulse_synthetic_dataset.csv
│   └── final_donor_recommendations.csv
├── models/
│   └── conversion_model.pkl
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_campaign_health.ipynb
│   ├── 04_anomaly_detection.ipynb
│   ├── 05_conversion_prediction.ipynb
│   ├── 06_explainable_ai.ipynb
│   └── 07_followup_recommendations.ipynb
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/TrishaM77/DonorPulse.git
cd DonorPulse
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install streamlit pandas numpy plotly scikit-learn shap matplotlib seaborn jupyter
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

The dashboard should open in your browser.

## Machine Learning and Analytics

The project explores:
- Exploratory Data Analysis (EDA)
- Campaign performance monitoring
- Rule-based operational anomaly detection
- Isolation Forest anomaly detection in the analysis notebook
- Logistic Regression for conversion prediction
- SHAP-based model explainability
- Donor follow-up prioritisation

## Future Improvements

- Integrate real-time fundraising data.
- Add automated call-quality analysis.
- Improve model performance and probability calibration.
- Validate recommendations with real operational feedback.
- Deploy the dashboard for wider organisational use.

## Author

**Trisha Mathur**

Computer Science Engineering Student | Aspiring Data Analyst

[GitHub](https://github.com/TrishaM77)