# Interview Preparation — Customer Churn Prediction & Retention Analysis

Every answer below is grounded in this project's actual implementation and verified
numbers (7,043 customers; test ROC-AUC 0.8452; F1 0.6242; recall 0.7995). No fabricated
results. Where a number is quoted, it comes from a real artifact in `outputs/reports/`
or `outputs/predictions/`.

---

## Modeling choices

### Q1. Why Logistic Regression?

- **Business fit.** The deliverable is not just a label but a **probability** for each of
  the 7,043 customers, which drives risk bands (Low <0.25, Medium, High, Critical ≥0.75)
  and a prioritized contact list. Logistic regression outputs calibrated-shape
  probabilities directly and is easy to explain to business stakeholders.
- **Interpretability.** Coefficients convert to odds ratios (`exp(coef)`): two-year
  contract OR = 0.20, fiber optic OR = 3.67, ≤6-month tenure OR = 2.43
  (`outputs/reports/logistic_coefficients.csv`). A retention team can act on that.
- **Strong baseline performance.** Test ROC-AUC **0.8452** with a linear model — near the
  CV ceiling (0.8467). For 21 columns of mostly categorical data, that leaves little room
  for a more complex model to justify its cost.
- **Regulatory/audit friendliness**: every score is reproducible from a frozen pipeline
  (`models/churn_model.pkl`, SHA-verified datasets, `random_state=42`).

### Q2. Why not Random Forest (or gradient boosting)?

- I did not reject ensembles on principle — I **did not need them yet**. The project's
  goal was a validated, interpretable, production-ready baseline; logistic regression
  already achieved ROC-AUC 0.8452.
- Class-weight CV showed the problem is about **ranking + operating point**, not capacity:
  recall went 0.5298 → 0.7920 just by re-weighting, with ROC-AUC unchanged (0.8468 →
  0.8467). That signals the model already captures the signal; more capacity would buy
  little ranking power and cost interpretability.
- Trees lose the per-feature odds-ratio story the business asked for, and with 39
  transformed features from ~7k rows, tree ensembles overfit more easily and require
  tuning infrastructure I deliberately kept out of scope.
- I documented gradient-boosted trees + SHAP as a **future benchmark** (see
  `docs/PROJECT_DOCUMENTATION.md` §16), not as a silent replacement.

### Q3. Why use `class_weight="balanced"`?

The dataset is imbalanced: 1,869 churned vs 5,174 retained (26.54% churn rate). Without
compensation the loss is dominated by the majority class. I **measured** it with 5-fold
stratified CV on the training set (`model_training_report.txt`):

| class_weight | ROC-AUC | F1 | Recall | Precision | Accuracy |
|---|---|---|---|---|---|
| None | 0.8468 | 0.5900 | 0.5298 | 0.6671 | 0.8051 |
| **balanced** | 0.8467 | **0.6293** | **0.7920** | 0.5223 | 0.7526 |

Ranking quality (ROC-AUC) was identical (gap 0.0001), while recall jumped from 0.53 to
0.79 — for retention, missing a churner costs far more than a wasted outreach. Decision
criterion was CV F1; the test set stayed untouched.

### Q4. Why stratify the train/test split?

Churn is a 26.54%/73.46% split. A plain random split could distort that ratio in the
1,409-row test set and make metrics unstable. `train_test_split(test_size=0.20,
random_state=42, stratify=y)` produced 5,634 train / 1,409 test with churn rates
26.5353% vs 26.5436% — a 0.0083 percentage-point gap, asserted in the report (≤0.5 pt
required). Stratification also guarantees every CV fold has both classes in proportion,
which is why the class-weight CV numbers are trustworthy.

## Metrics

### Q5. What is ROC-AUC?

ROC-AUC is the probability that the model ranks a randomly chosen churner above a
randomly chosen non-churner. It is threshold-independent and insensitive to class
prevalence. In this project it is the primary metric because retention uses **rankings**
(who to call first), not hard labels. Test ROC-AUC = **0.8452**, computed from
`predict_proba[:, 1]` — never from binary predictions. CV estimate was 0.8467 ± 0.0115,
so the test value is consistent with training-time expectations.

### Q6. Precision vs recall?

- **Recall (TPR)** = TP/(TP+FN) = 299/374 = **0.7995**: of customers who truly churned,
  the model catches ~80%.
- **Precision** = TP/(TP+FP) = 299/584 = **0.5120**: of flagged customers, ~51% would
  actually churn — meaning ~285 (FP) are contacted "unnecessarily".
- In churn retention the asymmetry favors recall: one saved customer outweighs several
  wasted calls, and a false positive's cost is a discount/call, while a false negative's
  cost is a lost customer (and acquisition cost is 5–10× retention cost). That is why
  the operational flag threshold was set at 0.40 (recall 0.8676 in OOF analysis) rather
  than the F1-maximizing 0.60.

### Q7. What is F1?

The harmonic mean of precision and recall — it punishes imbalance between them, unlike
the arithmetic mean. Test F1 = **0.6242** (precision 0.5120, recall 0.7995). I used CV F1
to choose the class weight (0.6293 balanced vs 0.5900 None) because it summarizes the
precision/recall trade in one number; but F1 alone is threshold-sensitive, so it never
replaced ROC-AUC as the ranking metric.

### Q8. What does the confusion matrix mean here?

At the default 0.50 threshold on the 1,409-row test set:

| | Predicted retained | Predicted churn |
|---|---|---|
| **Actual retained (1,035)** | TN = 750 | FP = 285 |
| **Actual churned (374)** | FN = 75 | TP = 299 |

- 299 churners correctly flagged (recall 0.7995); 75 slipped through (20.05%).
- 285 retained customers flagged — the price of high recall; each is one unnecessary
  retention offer, not a lost customer.
- Accuracy 0.7445 looks *worse* than the 0.7346 prior baseline, but the baseline catches
  **zero** churners (precision/recall/F1 = 0, ROC-AUC 0.5) — the matrix is exactly why
  accuracy alone misleads under imbalance.
- All four cells were mathematically cross-checked against accuracy/precision/recall/F1
  (`model_training_report.txt` consistency check).

### Q9. Why not optimize accuracy?

`DummyClassifier(strategy="prior")` achieves accuracy **0.7346** by predicting "retained"
for everyone — beating my model's 0.7445 on accuracy alone while having zero business
value (ROC-AUC 0.5, F1 0). Under 26.54% prevalence, accuracy rewards ignoring the
minority class entirely. The project therefore optimized ROC-AUC (ranking) and F1/recall
(operating point), and the baseline is documented to prove the point.

## Leakage & pipeline

### Q10. What is data leakage? How did you prevent it?

Leakage = any information in training that would not exist at prediction time, inflating
metrics. Concretely in this project:

1. `customerID` is an identifier — kept in the dataset but excluded from the feature
   matrix (`make_xy` asserts this).
2. `Churn` (the target) never enters `X` — asserted in code.
3. A leakage review of all 21 original columns + 5 engineered ones
   (`feature_leakage_review.csv`) confirmed every predictor is a pre-churn attribute
   (contract, tenure, charges, services) recorded before the outcome window.
4. The `Pipeline` is fitted **only on training rows** — scalers/encoders never see test
   data. Even CV refits preprocessing inside each fold.
5. Post-outcome proxies were checked: `TotalCharges` accumulates *before* churn
   (tenure × price), so it's legitimate; a post-churn column like "final refund amount"
   would have been dropped.

### Q11. Why use a sklearn Pipeline?

Because scoring 7,043 customers later must repeat **exactly** the training-time
transformations. The saved object is one artifact:
`Pipeline(ColumnTransformer(num: median-impute→StandardScaler; cat: most-frequent-impute→OneHotEncoder(handle_unknown="ignore", drop="first")) → LogisticRegression(C=1.0, class_weight=balanced, lbfgs, max_iter=1000))`.
That gives: no train/test preprocessing skew, leak-free CV, a single `joblib.dump` that
reproduces the exact test metrics (verified: reload → same TN/FP/FN/TP and same
probabilities for all 7,043 rows, `atol=1e-12`), and one deployment surface
(`model.predict_proba(X)`).

### Q12. Why StandardScaler?

The 6 numeric features live on different scales (tenure ≤ 72, TotalCharges up to ~8,685,
ServiceCount 0–7). Logistic regression's L2 penalty (C=1.0) penalizes coefficients
equally, so unscaled features would be shrunk inconsistently and `lbfgs` convergence
would suffer. Scaling puts all coefficients on comparable z-score units — which is also
what makes coefficient magnitudes comparable across features. It's fit on training data
only, inside the pipeline.

### Q13. Why OneHotEncoder (with drop="first" and handle_unknown="ignore")?

The 18 categorical features are nominal (Contract, PaymentMethod, InternetService, …);
ordinal integers would invent fake orderings. `drop="first"` avoids the dummy-variable
trap (perfect collinearity) and makes the remaining category the **reference level** —
that's why `Contract_Two year` has coefficient −1.609 *relative to month-to-month* (OR
0.20). `handle_unknown="ignore"` means an unseen category at scoring time maps to
all-zeros (reference) instead of crashing production scoring. Output: 24 candidates →
**39 transformed features**.

## Features

### Q14. Why keep TotalCharges? Wasn't it corrupted?

Yes — 11 rows had `' '` instead of numbers. The audit (Phase 2) proved all 11 have
`tenure = 0` and `Churn = "No"`: brand-new customers with no billing history. Setting
them to **0.0** is therefore semantically correct, and the repair was *asserted* (any
row violating the pattern would abort cleaning). TotalCharges stays informative: it
correlates 0.826 with tenure and represents accumulated value/lock-in. Dropping it would
have thrown away signal from 7,032 clean rows because of 11 explainable cells.

### Q15. What is multicollinearity / what does VIF mean?

VIF (variance inflation factor) measures how much a numeric feature's variance is
inflated by linear dependence on the others; VIF ≈ 1 is clean, >10 is severe. Diagnostics
on the training numeric block found `MonthlyCharges` VIF ≈ 135.72 and
`AvgMonthlyCharge` VIF ≈ 135.58, correlation **0.9963** — `AvgMonthlyCharge`
(TotalCharges/tenure) is almost a copy of MonthlyCharges. I documented the redundancy
instead of silently deleting: a feature **ablation** (CV on training) tested set C
(originals + ServiceCount + HasSecuritySupport) vs set B (everything). AUC: B 0.8467 vs
C 0.8459 → B kept, evidence-based. The consequence is documented: coefficients of
correlated numerics are fold-unstable in sign and must not be over-interpreted;
`AvgMonthlyCharge` was kept out of the headline model drivers for that reason.

### Q16. What does a coefficient mean? An odds ratio?

With standardized numerics and dropped-first dummies, a coefficient is the change in
log-odds of churn per one-unit change, **holding all other included variables constant**.
`exp(coef)` = odds ratio. Real examples from the final model:

- `Contract_Two year`: coef −1.609 → **OR 0.20** — two-year customers have ~80% lower
  churn odds than otherwise-identical month-to-month customers.
- `InternetService_Fiber optic`: coef +1.299 → **OR 3.67**.
- `TenureGroup_≤6 months`: coef +0.887 → **OR 2.43**.
- `PaymentMethod_Electronic check`: coef +0.391 → **OR 1.48**.

Because of the collinearity above, I present the categorical/behavioral ORs as
robust and flag the correlated numeric coefficients as unstable.

### Q17. Why isn't correlation causation?

Everything here is observational: customers weren't randomly assigned contracts or fiber.
Fiber-optic churn (41.89%) is higher partly because of self-selection — customers with
higher demands/budgets pick fiber, and price sensitivity differs. Confounders
(tenure, contract type) drive multiple variables at once. Concretely: I *never* claim
"upgrading to a two-year contract reduces churn by 80%"; I claim two-year customers
churned at 2.83% vs 42.71% in this historical data. That's why every driver visual in
Power BI is labeled an association, and the retention report proposes A/B tests before
scaling any action.

## Thresholds & scoring

### Q18. How did you choose the threshold?

Separately from the model. Out-of-fold probabilities on the **training set** (never the
test set) were scanned:

| Threshold | Precision | Recall | F1 | Flagged % |
|---|---|---|---|---|
| 0.30 | 0.4301 | 0.9177 | 0.5857 | 56.62% |
| 0.40 | 0.4708 | 0.8676 | 0.6104 | 48.90% |
| 0.50 | 0.5223 | 0.7920 | 0.6295 | 40.24% |
| 0.60 | 0.5750 | 0.7077 | **0.6345** | 32.66% |
| 0.70 | 0.6394 | 0.5906 | 0.6140 | 24.51% |

0.60 maximizes F1; **0.40 was chosen as the operational flag** because a retention team
with limited budget wants maximum recall per flagged customer (0.8676), accepting
precision 0.4708. The right threshold ultimately depends on outreach cost vs saved-customer
value — a business decision, documented as such. Probability, flag, and risk segment are
kept as three separate columns precisely so the threshold can change without re-modeling.

### Q19. What does `predict_proba` actually mean? Probability vs risk segment?

`churn_probability` is the model's estimated probability the customer churns, from
`predict_proba[:, 1]` (e.g., a customer at 0.87). The **risk segment** is a business band
of that probability: Low <0.25, Medium 0.25–<0.50, High 0.50–<0.75, Critical ≥0.75. The
**0.40 flag** is a third, independent concept: a binary operational cut. Three different
consumers: ranking (probability), capacity planning (segment counts), campaign lists
(flag). Distribution result: Low 2,790 (avg 0.0991, actual churn 3.91%) → Critical
1,365 (avg 0.8498, actual churn 68.35%) — perfectly monotonic
(`outputs/reports/customer_risk_report.txt`).

## Power BI & deployment

### Q20. How does Power BI consume the model output?

There is **no live Python inside Power BI**. The pattern is batch scoring: the frozen
pipeline scores all 7,043 customers once (`src/risk_scoring.py`), writing
`powerbi/powerbi_dataset.csv` — 7,043 rows × 29 columns, one row per customer, with
model outputs (`churn_probability`, `predicted_churn_at_0.40`, `risk_segment`) joined to
the customer attributes. Power BI imports the CSV (`Get Data > Text/CSV`),
`customerID` is the unique key, and all KPIs are DAX measures over that single table
(`powerbi/dax_measures.md`): e.g. `Churn Rate = DIVIDE([Churned Customers], [Total
Customers], 0)`, `High + Critical Risk Customers = CALCULATE(COUNTROWS(...),
powerbi_dataset[risk_segment] IN {"High","Critical"})`. A scheduled refresh re-runs
scoring upstream when new data arrives.

### Q21. How would you deploy this model in a real company?

1. **Serving**: export the pipeline (already a single joblib artifact) behind a REST
   endpoint (FastAPI) for on-demand scoring, or keep the batch pattern: nightly job
   scores the CRM table and writes scores to the warehouse.
2. **Data contracts**: enforce the exact schema (`model_ready_data.csv` columns), log
   feature distributions, and validate inputs (great-expectations-style tests) before
   scoring — the project already asserts schema/hash integrity at every stage.
3. **Monitoring**: track input drift (tenure/charge distributions), prediction drift
   (flag rate), and realized churn vs predicted by band; alert when segment churn rates
   deviate from the validated profile (3.91% / 18.51% / 38.15% / 68.35%).
4. **Retraining policy**: scheduled refits with the same `run_pipeline.ps1` structure,
   champion/challenger with the AUC 0.8452 baseline, and a temporal holdout once
   timestamped data exists (current split is random, not temporal — a documented
   limitation).
5. **Governance**: version model + metadata (`models/model_metadata.json`), keep the
   raw data immutable (SHA-256 `16320C9C…`), and log every scored batch.

---

## Rapid-fire numbers (know these cold)

- 7,043 customers; 1,869 churned (26.5370%); raw data never modified (SHA-256 frozen).
- 11 whitespace TotalCharges, all tenure=0, all Churn=No → 0.0 (asserted, not assumed).
- Split 5,634/1,409, stratified, seed 42; gap 0.0083 pp.
- 24 candidate features → 39 after preprocessing (6 numeric scaled, 18 categorical OHE).
- Class-weight CV: None F1 0.5900/recall 0.5298 vs balanced F1 0.6293/recall 0.7920.
- Test: accuracy 0.7445, precision 0.5120, recall 0.7995, F1 0.6242, ROC-AUC 0.8452.
- Confusion matrix: TN 750, FP 285, FN 75, TP 299.
- Ablation: originals 0.8459 → +engineered **0.8467** (chosen) → subset 0.8459.
- VIF: MonthlyCharges ≈ 135.72, AvgMonthlyCharge ≈ 135.58, corr 0.9963.
- Risk bands: 2,790 / 1,399 / 1,489 / 1,365; High+Critical **2,854**; 0.40 flags **3,465**.
- Key ORs: Two year 0.20 · Fiber 3.67 · ≤6 months 2.43 · E-check 1.48.
- Worst observed segment: 0–6 months + fiber = 74.19% churn (460/620).
