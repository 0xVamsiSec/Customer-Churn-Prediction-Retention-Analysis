# Customer Churn Prediction & Retention Analysis

End-to-end, fully executed data science project: raw telecom data → validated cleaning →
EDA → feature engineering → Logistic Regression → customer-level risk scoring → Power BI.

Every metric in this README was produced by actually running the code in this repository
(no simulated results). Source datasets are immutable; every stage is reproducible.

## Business Problem

A subscription telecom company loses roughly one in four customers. Acquiring a new
customer costs far more than retaining one, so the retention team needs to know,
**before** a customer leaves, who is at risk and why — and needs a prioritized contact
list rather than a blanket campaign.

## Objective

1. Predict each customer's probability of churn.
2. Identify the drivers associated with churn (observed associations, not causes).
3. Segment customers into operational risk bands for retention prioritization.
4. Deliver an editable Power BI report for executives and the retention team.

## Tech Stack

- **Python 3.14** — pandas 3.0.5, NumPy 2.5.2, scikit-learn 1.9.0, joblib 1.6.0
- **Visualization** — Matplotlib 3.11.1, Seaborn 0.13.2
- **BI** — Microsoft Power BI (dataset, DAX measures, dashboard documentation)
- **Dev** — Jupyter Notebook, PowerShell pipeline script, pytest

## Workflow

```text
Data Acquisition
→ Data Quality Audit        (read-only, SHA-256 verified)
→ Cleaning                  (asserted TotalCharges repair, target encoding)
→ EDA                       (churn rates, segments, correlations)
→ Feature Engineering       (leakage review, redundancy diagnostics)
→ Logistic Regression       (baseline, class-weight CV, feature ablation)
→ Risk Scoring              (all 7,043 customers scored by the saved pipeline)
→ Power BI                  (dashboard dataset and editable three-page report)
→ Scoring app               (single profile and batch CSV)
```

## Key Results (verified)

| Metric | Value |
|---|---|
| Customers analyzed | 7,043 |
| Overall churn rate | 26.54% (1,869 churned / 5,174 retained) |
| Test ROC-AUC | **0.8452** |
| Test F1 (churn class) | **0.6242** |
| Test recall (churn class) | **0.7995** |
| Test precision / accuracy | 0.5120 / 0.7445 |
| Confusion matrix (threshold 0.50) | TN 750 · FP 285 · FN 75 · TP 299 |
| High + Critical risk customers | **2,854** (40.5% of base) |
| Actual churn in Critical segment | **68.35%** (vs 3.91% in Low) |

Strongest verified risk patterns (observed, not causal):

- **Month-to-month contracts churn at 42.71%** vs 2.83% for two-year contracts.
- **Customers with ≤6 months tenure churn at 52.94%** vs 14.04% at 25+ months.
- Highest-risk segment: **0–6 months + Fiber optic → 74.19% churn (460/620)**.
- Electronic-check payment: 45.29% churn vs ~16% for automatic payments.
- Largest model coefficients: two-year contract (OR 0.20), fiber optic (OR 3.67),
  ≤6-month tenure (OR 2.43).

## Important Limitations

- **Observational dataset** — all findings are associations; no causal conclusions.
- **Historical snapshot** — no temporal data, so churn dynamics over time can't be studied.
- **Random (stratified) split**, not temporal validation — no drift assessment possible.
- **Probability calibration not performed** — treat scores as risk rankings.
- **Threshold 0.40 is operational** — the right business threshold depends on the cost of
  outreach vs the value of a saved customer (out-of-fold analysis suggests 0.60 maximizes F1).
- Scores for training customers are in-sample; use them for triage, not calibrated probabilities.

## Run the scoring app

From the project root, with the Python environment set up:

```powershell
.\.venv\Scripts\python.exe -m app.server
```

Open [the local app](http://127.0.0.1:8765/). Edit the preloaded example and select **Score customer**, or download the example CSV and upload it for batch scoring. The app loads the saved model, accepts profiles without a churn outcome, and returns a risk score, risk band, and 0.40 flag. CSV uploads can include an optional `customerID` and stay on the local machine. Pass `--port 8766` if port 8765 is in use.

## Open the Power BI report

Run `python powerbi/build_report.py --configure-csv` once from your checkout. Then open [Churn Retention.pbip](powerbi/Churn%20Retention.pbip) in a current version of Power BI Desktop and select **Refresh**. The [Power BI setup guide](powerbi/REPORT_SETUP.md) lists the expected totals for the three report pages.

For a five-minute demo and implementation talking points, see the [project walkthrough](docs/DEMO_WALKTHROUGH.md).

## Project Structure

```text
├── data/
│   ├── raw/customer_churn.csv               # Raw data - NEVER modified (SHA-256 frozen)
│   └── processed/
│       ├── cleaned_customer_churn.csv       # Phase 3 output (7,043 × 21)
│       └── model_ready_data.csv             # Phase 5 output (7,043 × 26)
├── notebooks/                               # 01 cleaning … 05 risk scoring (all executed)
├── app/                                     # Local browser app and inference adapter
├── src/
│   ├── data_quality_audit.py                # Phase 2 read-only audit
│   ├── data_cleaning.py                     # Phase 3 cleaning + validation
│   ├── eda.py                               # Phase 4 analysis engine
│   ├── feature_engineering.py               # Phase 5 features, leakage & selection
│   ├── preprocessing.py                     # Phase 6 feature groups + sklearn pipeline
│   ├── train.py                             # Phase 6 training/experiments
│   ├── evaluate.py                          # Phase 6 metrics + figures (consistency-checked)
│   ├── risk_scoring.py                      # Phase 7 scoring + Power BI outputs
│   └── project_validation.py                # Phase 8 end-to-end integrity check
├── models/
│   ├── churn_model.pkl                      # Frozen pipeline (preprocessing + classifier)
│   └── model_metadata.json                  # Frozen metadata (all values from artifacts)
├── outputs/
│   ├── figures/                             # 11 numbered figures
│   ├── predictions/                         # customer_risk_scores.csv, high_risk_customers.csv
│   └── reports/                             # Audit → cleaning → EDA → features → model → risk
├── powerbi/
│   ├── powerbi_dataset.csv                  # 7,043 × 29 dashboard dataset
│   ├── Churn Retention.pbip                  # Open in Power BI Desktop
│   ├── Churn Retention.Report/               # Three editable report pages
│   ├── Churn Retention.SemanticModel/        # Import, types, and DAX measures
│   ├── data_dictionary.csv
│   ├── dax_measures.md                      # Audited DAX measures
│   └── dashboard_documentation.md           # Original 3-page dashboard design
├── docs/
│   ├── PROJECT_DOCUMENTATION.md             # Full technical walkthrough
│   ├── INTERVIEW_PREPARATION.md             # Q&A grounded in this project's numbers
│   └── DEMO_WALKTHROUGH.md                   # Five-minute project demo
├── tests/
├── run_pipeline.ps1                         # One-command reproducible pipeline
├── requirements.txt
└── README.md
```

## How to Run

Run these commands from the project root with Python 3.14 and PowerShell. The requirements pin the versions used to train and load the saved model.

```powershell
# 1. Environment setup (once)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Data cleaning            -> data/processed/cleaned_customer_churn.csv
.\.venv\Scripts\python.exe src\data_cleaning.py

# 3. EDA                      -> outputs/reports/eda_report.txt + figures 01-08
.\.venv\Scripts\python.exe src\eda.py

# 4. Feature engineering      -> data/processed/model_ready_data.csv + feature reports
.\.venv\Scripts\python.exe src\feature_engineering.py

# 5. Model training           -> models/churn_model.pkl + model reports + figures 09-11
.\.venv\Scripts\python.exe src\train.py

# 6. Risk scoring             -> outputs/predictions/* + powerbi/* + risk reports
.\.venv\Scripts\python.exe src\risk_scoring.py
```

Or run the whole validated pipeline (stops on first failure):

```powershell
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1
```

Final integrity check (also freezes `models/model_metadata.json`):

```powershell
.\.venv\Scripts\python.exe src\project_validation.py
```

## Data Integrity

The input is IBM’s public [Telco Customer Churn sample](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv). The bundled raw CSV matches the upstream file byte for byte. See [data attribution](data/README.md) for licensing and the derived artifacts.

| File | SHA-256 (frozen) |
|---|---|
| `data/raw/customer_churn.csv` | `16320C9C1EC72448DB59AA0A26A0B95401046BEF5D02FD3AEB906448E3055E91` |
| `data/processed/cleaned_customer_churn.csv` | `38E6F2DD278033A4617F880E8F11304C5FC6DCE2C05ECEF3F2A8BF48EEAA8996` |
| `data/processed/model_ready_data.csv` | `DF02696964C95DE935070C92B147D591B100EA87746870A9EAE7F6D8CEFA4F73` |

The raw file was never modified — verified by SHA-256 before and after every phase.
