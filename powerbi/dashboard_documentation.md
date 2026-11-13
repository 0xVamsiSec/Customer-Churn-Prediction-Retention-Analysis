# Power BI Dashboard Documentation - Telco Customer Churn & Retention (FINAL)

## Data source

* File: `powerbi/powerbi_dataset.csv` - one row per customer, 7,043 rows x 29 columns.
* Import: Get Data > Text/CSV. Types: `Churn`, `predicted_churn_at_0.40`, `ServiceCount`,
  `HasSecuritySupport`, `SeniorCitizen`, `tenure` = Whole Number; `churn_probability`,
  `MonthlyCharges`, `TotalCharges`, `AvgMonthlyCharge` = Decimal; text columns = Text.
* `customerID` is the unique key - never aggregate or deduplicate it away.
* Column definitions: `powerbi/data_dictionary.csv`. Measures: `powerbi/dax_measures.md` (audited).
* No date column exists (single snapshot) - no time-series visuals are possible.

Concepts (do not mix up):
| Column | Meaning |
|---|---|
| `churn_probability` | Continuous model risk score (0-1) |
| `predicted_churn_at_0.40` | Binary operational flag: probability >= 0.40 |
| `risk_segment` | Business band: Low <0.25, Medium 0.25-<0.50, High 0.50-<0.75, Critical >=0.75 |
| `Churn` | Actual historical outcome (1 = churned) - retrospective views only |

---

## Page 1 - Executive Overview

Purpose: leadership snapshot of churn exposure and model-flagged risk.

### KPI cards (top row, large cards)

| Card | Measure |
|---|---|
| Total Customers | `Total Customers` |
| Churn Rate | `Churn Rate` (formatted as %) |
| Churned Customers | `Churned Customers` |
| High + Critical Risk | `High + Critical Risk Customers` |
| Average Churn Risk | `Average Churn Risk` |
| 0.40 Churn Flags | `0.40 Churn Flags` |

### Visuals

| Visual | Type | Fields |
|---|---|---|
| Overall churn split | Donut | Legend: `Churn` (0 = Retained, 1 = Churned); Values: `Total Customers` |
| Risk segment distribution | Donut | Legend: `risk_segment` (order Low > Medium > High > Critical); Values: `Total Customers` |
| Churn by Contract | Clustered bar | Axis: `Contract`; Values: `Churn Rate` |
| Churn by TenureGroup | Clustered bar | Axis: `TenureGroup` (sorted <=6, 7-12, 13-24, 25+); Values: `Churn Rate` |

### Slicers (right panel, sync across pages)

| Slicer | Field | Style |
|---|---|---|
| Contract | `Contract` | Dropdown |
| Risk Segment | `risk_segment` | Multi-select tile |
| InternetService | `InternetService` | Dropdown |
| TenureGroup | `TenureGroup` | Dropdown |

Interactions: cross-filtering on between visuals; format `Churn Rate` as percentage with 2 decimals.

---

## Page 2 - Churn Drivers

Purpose: observed HISTORICAL churn differences across attributes. These are descriptive
associations - not causal effects (stated in a text box on the page).

All visuals are clustered bar/column charts with Axis = category field, Values = `Churn Rate`
(plus `Total Customers` as a secondary column where useful).

| # | Visual | Axis field |
|---|---|---|
| 1 | Churn by Contract | `Contract` |
| 2 | Churn by InternetService | `InternetService` |
| 3 | Churn by PaymentMethod | `PaymentMethod` |
| 4 | Churn by TenureGroup | `TenureGroup` |
| 5 | Churn by MonthlyChargeBand | `MonthlyChargeBand` |
| 6 | Churn by ServiceCount | `ServiceCount` (0-7) |
| 7 | Churn by OnlineSecurity | `OnlineSecurity` |
| 8 | Churn by TechSupport | `TechSupport` |

Layout: 2 columns x 4 rows; consistent axis sorting (descending churn rate where categorical).
Header text box: "All figures show observed historical churn differences in this dataset.
They do not establish causal relationships."

---

## Page 3 - Customer Risk

Purpose: operational triage - who to contact first.

| Visual | Type | Fields |
|---|---|---|
| Risk distribution | Donut | `risk_segment`; Values: `Total Customers` |
| High/Critical customer count | Card | `High + Critical Risk Customers` (with `High + Critical Risk %` subtitle) |
| Flag quality | Cards | `0.40 Flag Precision`, `Expected Churners Among 0.40 Flags` |
| Probability distribution | Column histogram | X: `churn_probability` binned 0.05; Y: `Total Customers` |
| Probability ranking (top 25) | Bar | Y: `customerID` (TopN filter 25 by `churn_probability`); Values: `churn_probability` |
| Revenue at risk | Card | `Total Monthly Revenue at Risk (High + Critical)` |

### Customer risk table (main visual)

Source fields (mapped to dataset columns):

| Column shown | Dataset field |
|---|---|
| Customer ID | `customerID` |
| Risk Segment | `risk_segment` |
| Churn Probability | `churn_probability` |
| Predicted Churn @ 0.40 | `predicted_churn_at_0.40` |
| Contract | `Contract` |
| Tenure | `tenure` |
| Internet Service | `InternetService` |
| Monthly Charges | `MonthlyCharges` |

Default sorting: `churn_probability` **descending** (highest-risk customers first).
Add a TopN / visual-level filter option for outreach batches (e.g., Top 100).

Recommended page filter: `risk_segment IN {High, Critical}` for the contact-list view;
clear the filter for the distribution visuals.

---

## Reading guide & caveats

* `Churn` = 1 marks customers who already churned historically; the model was trained on this
  outcome. Driver pages describe the past; Page 3 describes forward-looking risk scores for
  the current customer base.
* Scores for the 5,634 training customers are in-sample; use scores as triage priorities,
  not calibrated outcome probabilities.
* The 0.40 threshold is operational; business threshold selection requires cost/benefit
  analysis (see `outputs/reports/retention_strategy_report.txt`).
