# Project Documentation — Customer Churn Prediction & Retention Analysis

End-to-end technical walkthrough. Every number cited here was produced by actually
executing the project code (seeds fixed; no simulated results).

---

## 1. Business problem

A telecom subscription business loses ~26.5% of customers. Because retention is cheaper
than acquisition, the business needs (a) a forward-looking per-customer churn risk score,
(b) an understanding of which customer characteristics are associated with churn, and
(c) a prioritized, operational contact list — not a blanket campaign.

## 2. Dataset

- `data/raw/customer_churn.csv` — 7,043 rows × 21 columns (Telco-style customer records).
- Target: `Churn` (No/Yes → 0/1 in cleaning). Class split: 5,174 retained / 1,869 churned
  (26.5370% churn rate) — moderately imbalanced.
- Features: customer demographics (gender, SeniorCitizen, Partner, Dependents), account
  (tenure, Contract, PaperlessBilling, PaymentMethod), services (Phone/Internet + 6 add-ons),
  charges (MonthlyCharges, TotalCharges).
- Integrity: raw file SHA-256 `16320C9C…5E91` — byte-for-byte unchanged through the whole
  project (verified after every phase).

## 3. Data-quality issues (Phase 2, read-only audit)

Real findings from the audit (no assumptions):

- 0 standard NaN values, 0 duplicate rows, 0 duplicate customer IDs.
- `TotalCharges` stored as **text**: exactly **11** non-convertible values, each exactly
  **one whitespace character `' '`**; all 11 rows have `tenure = 0` and `Churn = "No"`
  (new customers with no billing history yet).
- No categorical whitespace issues, no normalization collisions, no invalid categories.
- IQR outliers: tenure 0, MonthlyCharges 0 (earlier claims of outliers were simulation
  artifacts and were rejected).

## 4. Cleaning strategy (Phase 3)

- The 11 whitespace `TotalCharges` cells were set to `0.0` **only after asserting** all three
  audit conditions (single whitespace char, `tenure == 0`, `Churn == "No"`); any other
  pattern would abort the pipeline. Affected customerIDs recorded in the cleaning report.
- `TotalCharges` → `float64`; `Churn` → 0/1 (validated: only {0,1} present).
- Whitespace stripping ran defensively; **no categorical value actually required correction**,
  so categories were left untouched (no lowercasing/normalization).
- Nothing dropped: 0 rows removed, 0 outliers removed. Final: 7,043 × 21, 0 missing,
  0 duplicates. Output: `data/processed/cleaned_customer_churn.csv`
  (SHA-256 `38E6F2DD…A8996`).

## 5. EDA findings (Phase 4)

Key observed rates (all from the actual file; `outputs/reports/eda_report.txt`):

- Contract: Month-to-month **42.71%** (1,655/3,875), One year 11.27%, Two year **2.83%**.
- Tenure: ≤6 months **52.94%**, 25+ months 14.04%.
- Internet: Fiber optic **41.89%**, DSL 18.96%, No internet 7.41%.
- Payment: Electronic check **45.29%**; automatic methods ≤16.71%.
- Add-ons: no OnlineSecurity 41.77% vs 14.61% with; no TechSupport 41.64% vs 15.17%.
- Senior citizens 41.68% vs 23.61%; paperless billing 33.57% vs 16.33%.
- Correlations: tenure↔TotalCharges **0.826** (accumulation), tenure↔churn **−0.352**,
  MonthlyCharges↔churn 0.193.
- Highest-churn observed segment: **0–6 months + Fiber optic = 74.19% (460/620)**.

## 6. Feature engineering (Phase 5)

- Reviewed all 21 original columns in a feature inventory (predictive signal, redundancy,
  leakage, decision) — nothing removed merely to reduce columns.
- Engineered: `TenureGroup` (≤6 / 7–12 / 13–24 / 25+ months; tenure=0 in first group),
  `MonthlyChargeBand` (quartiles computed from the actual data: 18.25 / 35.5 / 70.35 /
  89.85 / 118.75), `ServiceCount` (count of `Yes` across the 7 optional add-ons;
  "No internet/phone service" → 0), `HasSecuritySupport` (OnlineSecurity or TechSupport),
  `AvgMonthlyCharge` (TotalCharges/tenure; tenure=0 uses MonthlyCharges).
- **Leakage review** (`feature_leakage_review.csv`): every candidate is a pre-churn
  attribute → low risk; `customerID` excluded (identifier); `Churn` is target-only.
- Redundancy documented, not deleted: `corr(MonthlyCharges, AvgMonthlyCharge) = 0.9963`
  (VIF ≈ 136), binned features duplicate continuous ones; final treatment decided by
  modeling evidence (ablation).
- Output: `data/processed/model_ready_data.csv` (7,043 × 26: customerID + 24 candidates + Churn).

## 7. Preprocessing (Phase 6)

Inside a single sklearn `Pipeline` (fitted on training data only, never the full dataset):

- Numeric (6): median imputation → `StandardScaler`.
- Categorical (18): most-frequent imputation → `OneHotEncoder(handle_unknown="ignore",
  drop="first")` → reference-category odds ratios; unseen categories at predict time
  encode as the reference (documented).
- `customerID` kept in the dataset but out of the feature matrix; `Churn` never a predictor.
- 24 candidate features → **39 transformed features**.

## 8. Logistic Regression (Phase 6)

- Split: `train_test_split(test_size=0.20, random_state=42, stratify=y)` →
  5,634 train / 1,409 test; churn-rate gap 0.0083 pct points (stratification preserved).
- Baseline `DummyClassifier(strategy="prior")`: accuracy 0.7346 but precision/recall/F1 = 0,
  ROC-AUC 0.5 — documents the no-skill floor under imbalance (accuracy alone misleads).
- Model: `LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)`.

## 9. Class-weight decision (CV only — test set untouched)

5-fold stratified CV (shuffle, seed 42) on the **training set**:

| class_weight | ROC-AUC | F1 | Recall | Precision | Accuracy |
|---|---|---|---|---|---|
| None | 0.8468 | 0.5900 | 0.5298 | 0.6671 | 0.8051 |
| **balanced** | 0.8467 | **0.6293** | **0.7920** | 0.5223 | 0.7526 |

ROC-AUC essentially unchanged (gap 0.0001 ≤ 0.003) → decided by CV F1. For retention
campaigns, catching churners (recall 0.79 vs 0.53) at equal ranking quality is worth the
precision trade. **Chosen: `class_weight="balanced"`.**

## 10. Model evaluation

- Primary-model CV (training): accuracy 0.7526±0.0135, precision 0.5223±0.0171,
  recall 0.7920±0.0339, F1 0.6293±0.0215, ROC-AUC 0.8467±0.0115.
- Feature ablation (CV on training): A originals-only ROC-AUC 0.8459/F1 0.6296;
  **B originals+engineered 0.8467/0.6293**; C originals+ServiceCount+HasSecuritySupport
  0.8459/0.6292. B selected by CV ROC-AUC → engineered features kept on evidence.
- Held-out test (single pass, one prediction/probability array):
  **accuracy 0.7445, precision 0.5120, recall 0.7995, F1 0.6242, ROC-AUC 0.8452**;
  confusion matrix TN 750 / FP 285 / FN 75 / TP 299.
- **Metric consistency**: accuracy/precision/recall/F1 mathematically re-derived from the
  confusion matrix and asserted equal to sklearn's values; ROC-AUC from
  `predict_proba[:, 1]` (never binary predictions). Saved model reload reproduces
  predictions and probabilities exactly.

## 11. Threshold analysis

Out-of-fold **training** probabilities (business-selection basis; test table descriptive only):

| Threshold | Precision | Recall | F1 | Flagged % |
|---|---|---|---|---|
| 0.30 | 0.4301 | 0.9177 | 0.5857 | 56.62% |
| 0.40 | 0.4708 | 0.8676 | 0.6104 | 48.90% |
| 0.50 | 0.5223 | 0.7920 | 0.6295 | 40.24% |
| 0.60 | 0.5750 | 0.7077 | **0.6345** | 32.66% |
| 0.70 | 0.6394 | 0.5906 | 0.6140 | 24.51% |

0.60 maximizes F1; 0.40 maximizes recall per flagged-dollar. The operational threshold
must come from outreach cost vs saved-customer value — not from test-set tuning. Phase 7
uses 0.40 for flagging (specification) while keeping the raw probability separate.

## 12. Risk scoring (Phase 7)

Saved pipeline (never retrained) scored all 7,043 customers:

| Segment | Customers | % | Avg risk | Actual churned | Actual churn rate |
|---|---|---|---|---|---|
| Low (<0.25) | 2,790 | 39.61% | 0.0991 | 109 | 3.91% |
| Medium | 1,399 | 19.86% | 0.3787 | 259 | 18.51% |
| High | 1,489 | 21.14% | 0.6303 | 568 | 38.15% |
| Critical (≥0.75) | 1,365 | 19.38% | 0.8498 | 933 | **68.35%** |

Perfectly monotonic. High+Critical = **2,854** customers exported as the sorted contact
list (`high_risk_customers.csv`). Probability (continuous), 0.40 flag (binary) and segment
(band) are kept as three distinct columns.

## 13. Power BI design

`powerbi/powerbi_dataset.csv` (7,043 × 29) — one row per customer with scores + attributes;
`data_dictionary.csv` documents every column; `dax_measures.md` provides audited measures
(Total Customers, Churned/Retained, Churn Rate, Average Churn Risk, Low/Medium/High/Critical
Risk Customers, High + Critical, 0.40 Churn Flags / Flag Rate, Average Monthly Charges,
flag precision, expected churners, revenue at risk); `dashboard_documentation.md` specifies
three pages: **Executive Overview** (6 KPI cards, churn split, segment distribution, churn
by contract/tenure, 4 slicers), **Churn Drivers** (8 observed-association visuals),
**Customer Risk** (distribution, high/critical KPIs, top-N probability ranking, customer
table sorted by probability descending).

## 14. Business recommendations (from `retention_strategy_report.txt`)

- **Priority 1 — Critical (1,365; 68.35% historical churn; $109,523.90 monthly revenue pool)**:
  proactive calls within 24–48h, save-desk offers, contract-upgrade incentives,
  executive escalation for the top decile.
- **Priority 2 — High (1,489; 38.15%; $109,332.50 monthly revenue pool)**: targeted campaigns,
  security/support trials (no-security churn 41.77% vs 14.61% with), payment migration to
  automatic methods.
- **Priority 3 — Medium (1,399; 18.51%)**: low-cost digital engagement, pulse surveys,
  loyalty milestones, quarterly re-scoring.
- **Standard — Low (2,790; 3.91%)**: lifecycle communication, referrals, careful upsell.
- All actions are proposals based on observed associations; validate via A/B tests.

## 15. Limitations

1. Observational, cross-sectional snapshot — no causal claims anywhere in the project.
2. Class imbalance (26.54% churners) handled via class weighting; threshold economics
   still business-dependent.
3. Random stratified split, not temporal — no drift/seasonality assessment possible.
4. Probabilities not calibrated — use as risk rankings.
5. In-sample scores for the 5,634 training customers; only 1,409 test customers are
   out-of-sample.
6. `AvgMonthlyCharge` nearly duplicates `MonthlyCharges` (r = 0.9963, VIF ≈ 136) — kept
   because ablation showed no harm, but coefficients for correlated numerics (TotalCharges,
   AvgMonthlyCharge) are fold-unstable in sign and must not be over-interpreted.
7. No time, campaign, or competitor data; single-provider sample.

## 16. Future improvements

- Probability calibration (Platt/isotonic) and a decision-curve/cost analysis for
  threshold selection.
- Temporal validation split and drift monitoring once timestamped data exists.
- Gradient-boosted trees (LightXGB/CatBoost) as a benchmark; SHAP for local explanations.
- Survival analysis (time-to-churn) instead of binary classification.
- Uplift modeling to measure which retention actions actually work per segment.
- Automated retraining + monitoring pipeline (data tests, fairness checks, alerting).
