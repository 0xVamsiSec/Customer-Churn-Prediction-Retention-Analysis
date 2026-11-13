# MASTER PROMPT — Build Customer Churn Prediction & Retention Analysis From Scratch

You are an expert **Data Scientist, Data Analyst, Python Developer, Machine Learning Engineer, and BI Developer**.

I want you to build a complete, professional, portfolio-quality project called:

# Customer Churn Prediction & Retention Analysis

The project must be built **from scratch**, using a realistic end-to-end data analytics and machine learning workflow.

The final project should be suitable for:

- GitHub portfolio
- Data Analyst interviews
- Junior Data Scientist interviews
- Python Developer interviews
- Machine Learning interviews
- Business/Data Analytics demonstrations
- Power BI portfolio demonstrations

Do not create a superficial tutorial project. Build it as if it were a real-world analytics project that could be presented to a company.

---

# 1. PROJECT OBJECTIVE

The business problem is:

> A subscription/service-based company wants to identify customers who are likely to churn and understand why they are at risk so that the retention team can intervene before the customer leaves.

The project must answer four major questions:

1. **How many customers are churning?**
2. **Which types of customers are most likely to churn?**
3. **What factors are associated with increased churn risk?**
4. **Which customers/segments should the retention team prioritize?**

The final solution should transform raw customer data into:

```text
Raw Data
   ↓
Data Quality Audit
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Exploratory Data Analysis
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Customer Churn Probability
   ↓
Risk Segmentation
   ↓
Retention Insights
   ↓
Power BI Dashboard
```

---

# 2. IMPORTANT IMPLEMENTATION RULE

Do not make up impressive-looking results.

If a dataset is used, all statistics, model metrics, feature importance, churn rates, and business findings must be calculated from the actual dataset.

Do not write statements such as:

- "The model achieved 92% accuracy"
- "Tenure was the strongest predictor"
- "Customers with low usage churned 3x more"

unless the project actually calculates and verifies those results.

If the dataset does not contain a requested feature, do not pretend it does.

---

# 3. TECHNOLOGY STACK

Use the following technologies:

### Programming

- Python 3.11+
- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn

### BI

- Microsoft Power BI

### Development

- Jupyter Notebook
- VS Code-compatible Python project structure

Optional libraries may be used where genuinely useful:

- SciPy
- Joblib
- OpenPyXL
- Imbalanced-learn

Do not add unnecessary dependencies.

---

# 4. DATASET

Use a realistic customer churn dataset.

Prefer a well-known public churn dataset such as the **IBM Telco Customer Churn dataset**, or another publicly available customer churn dataset with appropriate customer-level features.

The dataset should ideally contain fields representing:

### Customer information

- Customer ID
- Gender
- Senior citizen indicator
- Partner
- Dependents

### Account information

- Tenure
- Contract type
- Payment method
- Paperless billing
- Monthly charges
- Total charges

### Service information

- Phone service
- Multiple lines
- Internet service
- Online security
- Online backup
- Device protection
- Tech support
- Streaming services

### Target

- Churn

If the selected dataset differs, adapt the project accordingly.

Do not artificially create fake columns merely to match this list.

---

# 5. PROJECT DIRECTORY STRUCTURE

Create a clean professional directory structure:

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── customer_churn.csv
│   │
│   ├── processed/
│   │   ├── cleaned_customer_churn.csv
│   │   └── model_ready_data.csv
│
├── notebooks/
│   ├── 01_data_quality_and_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_churn_prediction_model.ipynb
│   └── 04_business_insights.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── churn_model.pkl
│   └── preprocessing_pipeline.pkl
│
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── predictions/
│
├── powerbi/
│   ├── powerbi_dataset.csv
│   └── dashboard_documentation.md
│
├── tests/
│   ├── test_cleaning.py
│   └── test_features.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── main.py
```

Keep the structure organized and production-like.

---

# 6. ENVIRONMENT SETUP

Create a `requirements.txt`.

Include only packages actually required by the project.

Example:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
joblib
openpyxl
scipy
```

If additional packages are required, explain why.

Create setup instructions for Windows.

Include commands such as:

```bash
python -m venv venv
```

Activation:

```bash
venv\Scripts\activate
```

Installation:

```bash
pip install -r requirements.txt
```

Running notebooks:

```bash
jupyter notebook
```

Running the pipeline:

```bash
python main.py
```

---

# 7. DATA QUALITY AUDIT

Before cleaning anything, perform a complete data-quality audit.

The audit must investigate:

### Dataset structure

- Number of rows
- Number of columns
- Data types
- Unique values
- Duplicate rows
- Duplicate customer IDs

### Missing data

Calculate:

- Missing count
- Missing percentage
- Missing values by column

Display the results in a clear table.

### Data consistency

Check:

- Unexpected categorical values
- Leading/trailing whitespace
- Different capitalization
- Numeric columns stored as strings
- Invalid numeric values
- Invalid customer IDs

### Outliers

Investigate possible outliers for:

- Tenure
- Monthly charges
- Total charges
- Other numerical variables

Do not automatically delete every outlier.

Explain whether each outlier should be:

- retained
- transformed
- capped
- removed

---

# 8. DATA CLEANING

Build a reproducible cleaning pipeline.

Perform appropriate cleaning steps.

Examples:

### Column names

Standardize column names:

```text
CustomerID
MonthlyCharges
TotalCharges
```

or a consistent snake_case format:

```text
customer_id
monthly_charges
total_charges
```

Use one convention consistently.

### Text cleanup

Remove unnecessary whitespace.

Normalize categorical values.

### Numeric conversion

Convert numeric-looking strings to numeric types safely.

For example:

```python
pd.to_numeric(..., errors="coerce")
```

### Missing values

Handle missing values based on the meaning and distribution of each feature.

Document every decision.

### Duplicates

Check for duplicate customer IDs.

If duplicates exist, investigate before removing them.

### Target

Convert churn into a suitable binary target:

```text
No → 0
Yes → 1
```

Document the transformation.

---

# 9. DATA VALIDATION AFTER CLEANING

After cleaning, run validation checks.

Verify:

- No unexpected duplicate customers
- Correct data types
- Valid target values
- Missing values handled appropriately
- Numerical ranges make sense
- Categorical values are consistent

Create a reusable validation function.

The pipeline should fail loudly if critical data-quality problems remain.

---

# 10. FEATURE ENGINEERING

Create useful business-oriented features where appropriate.

Possible examples:

### Tenure groups

```text
0–6 months
7–12 months
13–24 months
25–48 months
49+ months
```

### Monthly charge bands

Create meaningful customer spending groups.

### Total charges

Ensure the variable is numeric and valid.

### Service count

If the dataset supports it, calculate the number of subscribed services.

For example:

```text
service_count
```

### Support usage

If service/support fields are available, create a feature representing whether the customer uses support-related services.

### Contract risk

Categorize contract types if useful.

Do not create features that leak the target.

Explain every engineered feature.

---

# 11. DATA LEAKAGE PREVENTION

This is extremely important.

Do not allow information that would only become known after churn to enter the model.

Explain:

> What is data leakage?

Then verify that the features used for prediction would realistically be available before the churn event.

If any feature could introduce leakage, remove it and explain why.

---

# 12. EXPLORATORY DATA ANALYSIS

Create a comprehensive EDA notebook.

The analysis should answer business questions rather than simply generating random graphs.

---

## 12.1 Overall churn rate

Calculate:

```text
Total customers
Churned customers
Retained customers
Overall churn rate
```

Visualize churn distribution.

---

## 12.2 Churn by tenure

Analyze:

- Churn rate by tenure
- Churn by tenure group
- Customer distribution across tenure groups

Create appropriate charts.

Potential visualizations:

- Bar chart
- Histogram
- Box plot
- Churn-rate line chart

---

## 12.3 Churn by usage/service behavior

Analyze available usage/service variables.

Examples:

- Internet service
- Tech support
- Online security
- Streaming
- Phone service
- Service count

For each important categorical feature, calculate:

```text
Customer count
Churned count
Churn rate
```

---

## 12.4 Churn by contract type

Compare churn across:

- Month-to-month
- One year
- Two year

Create a churn-rate visualization.

Explain possible business implications.

---

## 12.5 Churn by payment method

Investigate whether churn varies across payment methods.

Do not assume causation.

Use language such as:

> "Customers using X showed a higher observed churn rate."

not:

> "X causes customers to churn."

---

## 12.6 Churn by charges

Analyze:

- Monthly charges
- Total charges

Compare distributions between:

```text
Churned
Retained
```

Use:

- Box plots
- Histograms
- KDE plots where appropriate

---

## 12.7 Customer demographics

Analyze churn by available demographic/account variables such as:

- Senior citizen
- Partner
- Dependents

Do not make unsupported demographic conclusions.

---

# 13. EDA VISUALIZATION REQUIREMENTS

Use Matplotlib and Seaborn.

Every chart should have:

- Meaningful title
- Proper axis labels
- Readable tick labels
- Appropriate legend where required

Avoid unnecessary chart decoration.

Save important figures into:

```text
outputs/figures/
```

Suggested figures:

```text
churn_distribution.png
churn_by_tenure.png
churn_by_contract.png
churn_by_payment_method.png
churn_by_internet_service.png
monthly_charges_distribution.png
tenure_distribution.png
correlation_matrix.png
```

Only create figures that provide useful analytical information.

---

# 14. STATISTICAL THINKING

Where appropriate, investigate whether observed differences are meaningful.

For example:

- categorical feature vs churn
- numerical feature vs churn

Use suitable statistical tests where useful.

Explain:

- Null hypothesis
- Alternative hypothesis
- p-value
- Practical significance

Do not rely exclusively on statistical significance.

Distinguish:

```text
Correlation / association
```

from:

```text
Causation
```

---

# 15. MACHINE LEARNING PROBLEM DEFINITION

Define the ML problem clearly.

### Type

Binary classification.

### Target

```text
churn
```

### Goal

Predict:

```text
P(customer churns)
```

---

# 16. TRAIN/TEST SPLIT

Split the data into training and testing sets.

Use stratification because churn datasets can be imbalanced.

For example:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Use a fixed random seed for reproducibility.

Explain why stratification is useful.

---

# 17. PREPROCESSING PIPELINE

Use Scikit-learn preprocessing pipelines.

Handle:

### Numerical features

Possible steps:

- Imputation
- StandardScaler

### Categorical features

Possible steps:

- Imputation
- OneHotEncoder

Use:

```python
ColumnTransformer
```

and:

```python
Pipeline
```

Do not manually preprocess the entire dataset before splitting if doing so would cause leakage.

Fit preprocessing only on the training data.

---

# 18. BASELINE MODEL

Create a simple baseline classifier.

For example:

```text
DummyClassifier
```

Use it to establish a baseline performance.

Explain why comparing against a baseline is useful.

---

# 19. LOGISTIC REGRESSION MODEL

Build the primary model using:

```python
LogisticRegression
```

Use appropriate settings.

Consider:

```python
class_weight="balanced"
```

if class imbalance makes it appropriate, but don't blindly use it.

Explain the decision.

---

# 20. MODEL EVALUATION

Evaluate the model using:

### Accuracy

Explain why accuracy alone is insufficient.

### Precision

Explain:

> Of the customers predicted as churners, how many actually churned?

### Recall

Explain:

> Of the customers who actually churned, how many did the model identify?

### F1-score

Explain the balance between precision and recall.

### Confusion matrix

Display:

```text
True Positive
True Negative
False Positive
False Negative
```

### ROC-AUC

Calculate and explain.

### Precision-Recall curve

If class imbalance is significant, also analyze PR-AUC or the precision-recall curve.

---

# 21. THRESHOLD ANALYSIS

Do not blindly assume:

```text
probability >= 0.5 → churn
```

Analyze different probability thresholds.

For example:

```text
0.30
0.40
0.50
0.60
0.70
```

Show how changing the threshold affects:

- Precision
- Recall
- Number of customers flagged
- Potential retention workload

Explain the business trade-off.

---

# 22. MODEL INTERPRETABILITY

Because Logistic Regression is being used, explain the model coefficients.

Convert coefficients into interpretable measures where appropriate.

For logistic regression:

```text
odds ratio = exp(coefficient)
```

Create a table:

```text
Feature
Coefficient
Odds Ratio
Direction
Interpretation
```

For example:

```text
Feature | Coefficient | Odds Ratio | Direction | Interpretation
```

Do not claim causality.

Use wording such as:

> Holding other variables constant, higher values of this feature are associated with higher/lower estimated odds of churn.

---

# 23. FEATURE IMPORTANCE

Create a ranked visualization of the most influential Logistic Regression features.

Because one-hot encoding may create multiple columns, clearly display the transformed feature names.

Create:

```text
top_positive_churn_features.png
top_negative_churn_features.png
```

Explain that these are model associations, not causal effects.

---

# 24. CUSTOMER RISK SCORING

Generate churn probabilities for customers.

Create a prediction dataset containing:

```text
customer_id
actual_churn
churn_probability
predicted_churn
risk_segment
```

---

# 25. RISK SEGMENTATION

Create meaningful business risk categories.

For example:

```text
Low Risk
Medium Risk
High Risk
Critical Risk
```

Do not arbitrarily choose thresholds without explanation.

A possible approach:

```text
Low: < 0.25
Medium: 0.25–0.50
High: 0.50–0.75
Critical: >= 0.75
```

But evaluate whether these thresholds make sense for the actual model and business context.

Document the chosen thresholds.

---

# 26. CUSTOMER PRIORITIZATION

Create a ranked list of high-risk customers.

For example:

```text
Customer ID
Churn Probability
Risk Segment
Tenure
Contract
Monthly Charges
Service Information
```

Sort by churn probability descending.

Export:

```text
outputs/predictions/high_risk_customers.csv
```

---

# 27. BUSINESS SEGMENT ANALYSIS

Do not stop at individual customer predictions.

Calculate churn risk by segments such as:

- Contract type
- Tenure group
- Internet service
- Payment method
- Monthly charge band
- Service count

Create tables containing:

```text
Segment
Customer Count
Actual Churn Rate
Average Predicted Churn Probability
High-Risk Customer Count
```

This is important because the business may act on **segments**, not only individual customers.

---

# 28. RETENTION STRATEGY

Translate analytical findings into potential retention actions.

Do not make unsupported claims.

Create a framework like:

```text
High-risk segment
        ↓
Observed characteristics
        ↓
Possible business interpretation
        ↓
Suggested retention action
```

Examples:

### High-risk early-tenure customers

Potential action:

- onboarding assistance
- education campaigns
- early customer success outreach

### High-risk month-to-month customers

Potential action:

- contract incentives
- loyalty offers
- annual-plan benefits

### High-risk low-engagement customers

Potential action:

- product education
- usage reminders
- customer success intervention

Clearly state:

> These are recommended business actions based on observed/modelled risk patterns and should be validated through controlled retention experiments.

---

# 29. POWER BI DATASET

Create a clean Power BI-ready dataset.

Export:

```text
powerbi/powerbi_dataset.csv
```

It should contain the fields necessary for the dashboard.

Include:

- Customer ID
- Churn
- Churn probability
- Risk segment
- Tenure
- Tenure group
- Contract
- Payment method
- Monthly charges
- Total charges
- Service attributes
- Relevant engineered features

---

# 30. POWER BI DASHBOARD DESIGN

Design the Power BI dashboard as if it will be presented to company management.

Create documentation describing exactly how to build the dashboard.

The dashboard should contain at least 3 pages.

---

## PAGE 1 — EXECUTIVE OVERVIEW

Display KPI cards:

```text
Total Customers
Churned Customers
Overall Churn Rate
Average Monthly Charges
High-Risk Customers
Average Predicted Churn Probability
```

Visuals:

- Churn distribution
- Churn trend/segment distribution
- Risk segment distribution
- Contract churn comparison
- High-level customer profile

Add slicers:

- Contract
- Internet service
- Tenure group
- Risk segment

---

# PAGE 2 — CHURN DRIVERS

Show:

- Churn by tenure
- Churn by contract
- Churn by payment method
- Churn by internet/service type
- Churn by monthly charges
- Churn by service count

The page should answer:

> "Why are customers churning?"

---

# PAGE 3 — CUSTOMER RISK

Create a table containing:

```text
Customer ID
Risk Segment
Churn Probability
Tenure
Contract
Monthly Charges
Services
```

Allow users to:

- Filter high-risk customers
- Sort by churn probability
- Identify priority customers

Use conditional formatting where useful.

The page should answer:

> "Who should the retention team contact first?"

---

# 31. POWER BI INTERACTIVITY

Document how to create:

### Slicers

- Contract
- Tenure group
- Risk segment
- Internet service
- Payment method

### Drill-through

Allow users to select a customer and inspect customer-level information.

### Tooltips

Provide additional information when hovering over charts.

### Conditional formatting

Use risk categories to visually distinguish customer risk.

---

# 32. POWER BI MEASURES

Provide recommended DAX measures.

Examples:

```DAX
Total Customers =
DISTINCTCOUNT(Customer[customer_id])
```

```DAX
Churned Customers =
CALCULATE(
    [Total Customers],
    Customer[churn] = 1
)
```

```DAX
Churn Rate =
DIVIDE(
    [Churned Customers],
    [Total Customers],
    0
)
```

Create additional useful measures where appropriate.

Make sure the DAX matches the actual exported dataset columns.

---

# 33. MODEL ARTIFACTS

Save the trained model using Joblib.

For example:

```text
models/churn_model.pkl
```

Save the preprocessing pipeline:

```text
models/preprocessing_pipeline.pkl
```

Ensure a new customer record can be passed through the same preprocessing and prediction pipeline.

---

# 34. PREDICTION SCRIPT

Create:

```text
src/predict.py
```

It should allow prediction for new customer data.

Conceptually:

```python
prediction = model.predict(...)
probability = model.predict_proba(...)
```

Return:

```text
Predicted churn
Churn probability
Risk segment
```

---

# 35. TRAINING SCRIPT

Create:

```text
src/train.py
```

It should:

1. Load cleaned data
2. Select features
3. Split train/test
4. Build preprocessing pipeline
5. Train Logistic Regression
6. Evaluate model
7. Save model
8. Generate predictions
9. Save prediction output

Make it reproducible.

---

# 36. EVALUATION SCRIPT

Create:

```text
src/evaluate.py
```

It should calculate:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion matrix

Save evaluation results into:

```text
outputs/reports/
```

---

# 37. AUTOMATION

Create:

```text
main.py
```

It should provide a simple end-to-end pipeline:

```text
1. Load raw data
2. Clean data
3. Validate data
4. Engineer features
5. Train model
6. Evaluate model
7. Generate customer risk scores
8. Export Power BI dataset
```

The project should be reproducible by running:

```bash
python main.py
```

---

# 38. TESTING

Create basic unit tests.

At minimum test:

### Cleaning

- Numeric conversion
- Target encoding
- Missing-value handling

### Feature engineering

- Tenure groups
- Charge bands
- Service count if implemented

### Validation

- Target contains only expected values
- Customer IDs are unique after cleaning

Use:

```text
pytest
```

if appropriate.

---

# 39. README

Create a professional `README.md`.

It must contain:

## Project title

Customer Churn Prediction & Retention Analysis

## Business problem

Explain why churn matters.

## Objective

Explain what the project aims to achieve.

## Dataset

Describe the dataset and its source.

## Tech stack

List all technologies.

## Project architecture

Show:

```text
Raw Data
   ↓
Cleaning
   ↓
EDA
   ↓
Feature Engineering
   ↓
Model
   ↓
Risk Scoring
   ↓
Power BI
```

## Key analysis

Explain major analytical questions.

## Machine learning

Explain:

- Target
- Features
- Train/test split
- Logistic Regression
- Evaluation metrics
- Threshold selection

## Results

Insert only actual calculated metrics.

## Business insights

Summarize only findings supported by the actual analysis.

## Dashboard

Explain the Power BI dashboard pages.

## How to run

Provide complete setup instructions.

## Project structure

Show the folder structure.

## Future improvements

Discuss:

- Random Forest
- XGBoost
- Hyperparameter tuning
- SHAP
- Model monitoring
- Retention experiment/A-B testing
- Automated prediction pipeline

---

# 40. DATA SCIENCE BEST PRACTICES

Follow these principles throughout the project:

### Reproducibility

Use:

```python
random_state=42
```

where appropriate.

### No leakage

Never fit preprocessing on the complete dataset before splitting.

### Interpretability

Explain model coefficients.

### Validation

Validate data after cleaning.

### Documentation

Explain important decisions.

### Business relevance

Do not perform analysis simply because a chart is possible.

Every important analysis should answer a business question.

---

# 41. IMPORTANT MODELING RULES

Do NOT:

- Train on the test set
- Scale data before splitting
- Perform imputation using the entire dataset before splitting
- Include customer ID as a predictive feature
- Include post-churn information
- Optimize only for accuracy
- Claim causality from correlation
- Invent model performance
- Invent business insights

---

# 42. MODEL COMPARISON — OPTIONAL BUT RECOMMENDED

After Logistic Regression is complete, optionally implement:

```text
Logistic Regression
Random Forest
```

Compare:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
```

However, Logistic Regression must remain the primary/interpretable model unless another model clearly provides a justified business advantage.

If a more complex model performs better, explain the trade-off between:

```text
Predictive performance
vs.
Interpretability
```

---

# 43. BUSINESS ROI FRAMEWORK

Add an optional retention economics section.

Define assumptions rather than pretending they are real company values.

For example:

```text
Number of high-risk customers
×
Expected retention success rate
×
Customer value
=
Potential retained value
```

Clearly label all assumptions.

Do not present hypothetical ROI as actual company savings.

---

# 44. FINAL ANALYTICAL REPORT

Create a final report containing:

## Executive Summary

Explain the project in business language.

## Data Quality

Explain what was cleaned.

## EDA Findings

Summarize important patterns.

## Model

Explain Logistic Regression.

## Evaluation

Present actual metrics.

## Risk Segmentation

Explain how customers were categorized.

## High-Risk Customers

Explain how prioritization works.

## Business Recommendations

Give actionable recommendations supported by the analysis.

## Limitations

Discuss:

- Dataset limitations
- Potential class imbalance
- Correlation vs causation
- Model limitations
- Historical data limitations
- Lack of intervention testing

## Future Work

Discuss possible improvements.

---

# 45. FINAL DELIVERABLES

At the end, the project must contain:

```text
✓ Raw dataset
✓ Cleaned dataset
✓ Data-quality report
✓ EDA notebook
✓ EDA charts
✓ Feature engineering
✓ Preprocessing pipeline
✓ Logistic Regression model
✓ Model evaluation
✓ Confusion matrix
✓ ROC curve
✓ Precision-recall analysis
✓ Feature coefficient analysis
✓ Customer churn probabilities
✓ Risk segmentation
✓ High-risk customer CSV
✓ Power BI-ready dataset
✓ Power BI dashboard documentation
✓ DAX measures
✓ Python scripts
✓ Tests
✓ requirements.txt
✓ README.md
✓ Final business report
```

---

# 46. QUALITY CHECK BEFORE FINISHING

Before saying the project is complete, perform a final audit.

Verify:

### Code

- Does everything run?
- Are imports correct?
- Are paths correct?
- Are functions reusable?
- Are there hard-coded machine-specific paths?

### Data

- Is the raw dataset preserved?
- Is cleaned data generated reproducibly?
- Are missing values handled?
- Are duplicates handled?

### Machine Learning

- Is there train/test separation?
- Is preprocessing inside a pipeline?
- Is there leakage?
- Are metrics calculated correctly?
- Is class imbalance considered?
- Are probabilities generated?

### Business

- Are findings based on actual data?
- Are recommendations actionable?
- Is causality avoided?

### Power BI

- Does the exported CSV contain all necessary fields?
- Are the DAX formulas consistent with the data?
- Are dashboard pages clearly defined?

### Documentation

- Can a new developer clone the project and understand it?
- Are setup instructions complete?
- Are assumptions documented?

---

# 47. HOW I WANT YOU TO WORK

Do not dump the entire project blindly into one response.

Build the project **incrementally**.

Follow this sequence:

### Phase 1
Project setup and directory structure.

### Phase 2
Dataset acquisition and data-quality audit.

### Phase 3
Cleaning and validation.

### Phase 4
EDA.

### Phase 5
Feature engineering.

### Phase 6
Machine learning preprocessing.

### Phase 7
Logistic Regression.

### Phase 8
Model evaluation and interpretation.

### Phase 9
Risk scoring and customer prioritization.

### Phase 10
Power BI dataset and dashboard design.

### Phase 11
Testing and automation.

### Phase 12
README and final documentation.

After completing each phase:

1. Explain what was implemented.
2. Show the important files created/modified.
3. Explain the key concepts used.
4. Run/validate the code where possible.
5. Identify any errors.
6. Fix errors before moving to the next phase.
7. Do not proceed while the current phase is broken.

---

# 48. TEACH ME WHILE BUILDING

I am building this project partly to prepare for interviews.

Therefore, whenever you introduce an important concept, explain it briefly.

For example, when using:

```python
train_test_split()
```

explain:

- What it does
- Why it is necessary
- What could go wrong without it

When using:

```python
StandardScaler()
```

explain:

- What scaling means
- Why Logistic Regression may benefit from it
- Why it must be fitted only on training data

When using:

```python
LogisticRegression()
```

explain:

- What logistic regression is
- Why it is appropriate for binary classification
- What the coefficients represent
- What probability output means

When using:

```python
classification_report()
```

explain:

- Precision
- Recall
- F1-score

The explanations should be practical and interview-oriented rather than academic.

---

# 49. INTERVIEW PREPARATION OUTPUT

At the very end, create an interview preparation document containing questions and answers based specifically on the implementation.

Include at least:

### Basic

1. Explain your project.
2. Why did you choose churn prediction?
3. What is customer churn?
4. What was your dataset?
5. What was your target variable?

### Python/Pandas

6. How did you clean the data?
7. How did you handle missing values?
8. How did you detect duplicates?
9. How did you convert categorical variables?
10. Why did you use Pandas?

### EDA

11. What patterns did you find?
12. Why did you analyze tenure?
13. Why did you analyze contract type?
14. What visualizations did you create?
15. What is the difference between correlation and causation?

### Machine Learning

16. Why Logistic Regression?
17. What is binary classification?
18. What is train/test split?
19. Why use stratification?
20. What is feature scaling?
21. What is one-hot encoding?
22. What is data leakage?
23. What is class imbalance?
24. What is a confusion matrix?
25. What is precision?
26. What is recall?
27. What is F1-score?
28. What is ROC-AUC?
29. Why isn't accuracy enough?
30. What does `predict_proba()` return?

### Model interpretation

31. What does a Logistic Regression coefficient mean?
32. What is an odds ratio?
33. Which features were most associated with churn?
34. Can you say those features caused churn?
35. How would you explain the model to a business manager?

### Business

36. How would the company use your model?
37. How would you prioritize customers?
38. What retention strategies would you recommend?
39. How would you measure whether the retention strategy worked?
40. What are the limitations of your project?

### Power BI

41. Why did you use Power BI?
42. What KPIs did you display?
43. What dashboard pages did you create?
44. What are slicers?
45. What is DAX?
46. How would a retention manager use your dashboard?

### Advanced

47. How would you improve the model?
48. Why might Random Forest outperform Logistic Regression?
49. How would you handle severe class imbalance?
50. How would you deploy the model?
51. How would you monitor model performance?
52. How would you prevent model drift?
53. How would you test whether a retention campaign actually reduces churn?

Provide technically correct answers based on the **actual implementation**, not generic answers.

---

# 50. FINAL RULE

The most important objective is:

> Build a project that I can genuinely understand and defend in an interview.

Do not optimize for making the project look artificially impressive.

Prioritize:

```text
Correctness
+
Reproducibility
+
Interpretability
+
Business relevance
+
Clean code
+
Strong documentation
+
Interview readiness
```

When finished, I should be able to confidently explain:

```text
What problem I solved
        ↓
What data I used
        ↓
How I cleaned it
        ↓
What I discovered
        ↓
Why I selected Logistic Regression
        ↓
How I evaluated it
        ↓
How I interpreted the predictions
        ↓
How I identified high-risk customers
        ↓
How Power BI communicated the results
        ↓
How the business could act on the findings
```

Start with **Phase 1: project setup, dataset selection/acquisition, and directory structure**. Do not skip directly to modeling.