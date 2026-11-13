# DAX Measures - Churn Risk Dashboard (audited)

Table: `powerbi_dataset` (one row per customer; column definitions in `data_dictionary.csv`).
All measures below reference only columns that exist in that table:
`Churn`, `churn_probability`, `risk_segment`, `predicted_churn_at_0.40`, `MonthlyCharges`, `customerID`.

Create these as measures (Table tools > New measure):

## Core counts & rates

```dax
Total Customers = COUNTROWS ( powerbi_dataset )

Churned Customers =
CALCULATE ( COUNTROWS ( powerbi_dataset ), powerbi_dataset[Churn] = 1 )

Retained Customers =
CALCULATE ( COUNTROWS ( powerbi_dataset ), powerbi_dataset[Churn] = 0 )

Churn Rate =
DIVIDE ( [Churned Customers], [Total Customers], 0 )

Average Churn Risk =
AVERAGE ( powerbi_dataset[churn_probability] )

Average Monthly Charges =
AVERAGE ( powerbi_dataset[MonthlyCharges] )
```

## Risk segment measures

```dax
Low Risk Customers =
CALCULATE ( COUNTROWS ( powerbi_dataset ), powerbi_dataset[risk_segment] = "Low" )

Medium Risk Customers =
CALCULATE ( COUNTROWS ( powerbi_dataset ), powerbi_dataset[risk_segment] = "Medium" )

High Risk Customers =
CALCULATE ( COUNTROWS ( powerbi_dataset ), powerbi_dataset[risk_segment] = "High" )

Critical Risk Customers =
CALCULATE ( COUNTROWS ( powerbi_dataset ), powerbi_dataset[risk_segment] = "Critical" )

High + Critical Risk Customers =
CALCULATE ( COUNTROWS ( powerbi_dataset ),
            powerbi_dataset[risk_segment] IN { "High", "Critical" } )

High + Critical Risk % =
DIVIDE ( [High + Critical Risk Customers], [Total Customers], 0 )
```

## Operational flag (0.40 threshold) measures

```dax
0.40 Churn Flags =
CALCULATE ( COUNTROWS ( powerbi_dataset ),
            powerbi_dataset[predicted_churn_at_0.40] = 1 )

0.40 Flag Rate =
DIVIDE ( [0.40 Churn Flags], [Total Customers], 0 )

Churned Among 0.40 Flags =
CALCULATE ( COUNTROWS ( powerbi_dataset ),
            powerbi_dataset[predicted_churn_at_0.40] = 1,
            powerbi_dataset[Churn] = 1 )

0.40 Flag Precision =
DIVIDE ( [Churned Among 0.40 Flags], [0.40 Churn Flags], 0 )

Expected Churners Among 0.40 Flags =
SUMX ( FILTER ( powerbi_dataset, powerbi_dataset[predicted_churn_at_0.40] = 1 ),
       powerbi_dataset[churn_probability] )
```

## Revenue-at-risk measures (for segment analysis)

```dax
Total Monthly Revenue at Risk (High + Critical) =
CALCULATE ( SUM ( powerbi_dataset[MonthlyCharges] ),
            powerbi_dataset[risk_segment] IN { "High", "Critical" } )

Avg Monthly Revenue at Risk (High + Critical) =
CALCULATE ( AVERAGE ( powerbi_dataset[MonthlyCharges] ),
            powerbi_dataset[risk_segment] IN { "High", "Critical" } )
```

## Segment-analysis usage

All driver visuals slice the same base measures by category columns, e.g.
`Actual Churn Rate` by `Contract`, `TenureGroup`, `InternetService`, `PaymentMethod`,
`MonthlyChargeBand`, `ServiceCount`, `OnlineSecurity`, `TechSupport`:

```dax
Actual Churn Rate = DIVIDE ( [Churned Customers], [Total Customers], 0 )
```

## Audit notes

* `[Churn]` is the ACTUAL historical outcome (retrospective views only - never a model input).
* `churn_probability` is a model risk score (not a guaranteed outcome probability).
* `0.40` is the operational flagging threshold specified in Phase 7; from the Phase-6
  out-of-fold analysis, 0.60 maximized F1 - the final business threshold should follow
  outreach cost/benefit.
* Driver visuals show observed historical associations - they do not establish causation.
