# Customer churn project demo

## Five-minute walkthrough

1. **Start with the business question.** “If the retention team has limited capacity, which customers should it review first?” Open the **Executive Overview** page and point out the historical base rate of 26.54% across 7,043 customers.
2. **Show a concrete segment.** Open **Churn Drivers**. Month-to-month customers churned at 42.71% versus 2.83% for two-year contracts. In the EDA, 460 of 620 fiber customers in their first six months churned. This supports an onboarding test with a randomized holdout.
3. **Explain the model.** The training code uses a stratified 80/20 split. Five-fold cross-validation on the training split compared class weights and feature sets. The saved pipeline applies numeric imputation and scaling, categorical imputation and one-hot encoding, and class-weighted logistic regression. Holdout ROC-AUC was 0.8452. At 0.50, the model caught 299 of 374 historical churn events and flagged 285 retained customers.
4. **Show the operating trade-off.** The **Customer Risk** page displays the historical score ranking. At 0.40, holdout recall rose to 86.63%, while precision fell to 46.89% and 691 of 1,409 customers were flagged. The dashboard’s observed churn among flags describes the full snapshot, which includes the training rows. A live outreach cutoff should reflect capacity, cost, and incremental saves.
5. **Score a profile.** Run `.\.venv\Scripts\python.exe -m app.server` and open `http://127.0.0.1:8765/`. Score the example and show its **0.911 Critical** result. Change its contract or tenure and score again to demonstrate how the model responds. Download the example CSV, upload it, and show the score, band, and flag columns in the returned file.

## Implementation questions

**How does the app score a new customer without a churn label?**

It accepts the 19 original predictors and an optional customer ID. It reproduces the five engineered features, then calls the saved pipeline’s `predict_proba`. The four risk bands and the 0.40 flag use the existing project definitions. The scoring adapter reproduced the original scores, flags, and bands for all 7,043 customers using input without `Churn`.

**Why keep the original charge-band boundaries?**

The saved model expects the categories produced by the historical quartile cutpoints of 35.50, 70.35, and 89.85. Using these cutpoints keeps scores consistent across single-row and batch requests. The original feature-engineering step learned them before the random split. A future training revision should fit the cutpoints inside each training fold and persist them with the model.

**Can the team use 0.911 as a 91.1% chance of churn?**

Use it as a ranking score. The mean score on the random holdout was 0.414 against a historical churn rate of 0.265. To make probability or financial forecasts, define a prediction window and validate calibration on a later cohort. Use the billing card as the sum of monthly charges in the selected risk bands.

**What is needed for a live retention pilot?**

Use a dated extract of eligible active customers, define the future outcome window, and validate on a later snapshot. Set the outreach capacity and use a randomized control group to estimate the incremental saves and the cost per save.

## Where to point during the demo

| Topic | Project file |
|---|---|
| Repairing blank total charges and validating data | `src/data_cleaning.py` |
| Feature engineering and charge-band cutpoints | `src/feature_engineering.py` |
| Train split, class weighting, and feature ablation | `src/train.py`, `outputs/reports/model_training_report.txt` |
| Unlabeled inference and CSV handling | `app/server.py` |
| Report pages and DAX definitions | `powerbi/build_report.py`, `powerbi/Churn Retention.pbip` |
