"""Customer risk scoring & Power BI preparation (Phase 7).

Uses the SAVED Phase-6 pipeline (models/churn_model.pkl - never retrained) to
score all 7,043 customers from data/processed/model_ready_data.csv, then
produces:

  * outputs/predictions/customer_risk_scores.csv
  * outputs/predictions/high_risk_customers.csv   (High + Critical, prob desc)
  * powerbi/powerbi_dataset.csv
  * powerbi/data_dictionary.csv
  * powerbi/dax_measures.md
  * powerbi/dashboard_documentation.md
  * outputs/reports/customer_risk_report.txt

Concepts kept strictly separate:
  * model probability  - continuous churn_probability from predict_proba[:, 1]
  * classification threshold - fixed operational cut at 0.40 for flagging
  * business risk segment    - probability bands (Low/Medium/High/Critical)

Caveat documented in the report: scores for the 5,634 training customers are
IN-SAMPLE (the model saw them during fitting); only the 1,409 test customers
have out-of-sample scores. Scoring is descriptive triage, not a performance
estimate, and associations are not causal effects.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

import src.feature_engineering as fe
import src.preprocessing as prep

MODEL_PATH = prep.MODELS_DIR / "churn_model.pkl"
SCORES_PATH = prep.PROJECT_ROOT / "outputs" / "predictions" / "customer_risk_scores.csv"
HIGH_RISK_PATH = prep.PROJECT_ROOT / "outputs" / "predictions" / "high_risk_customers.csv"
PBI_DIR = prep.PROJECT_ROOT / "powerbi"
PBI_DATASET_PATH = PBI_DIR / "powerbi_dataset.csv"
PBI_DICTIONARY_PATH = PBI_DIR / "data_dictionary.csv"
PBI_DAX_PATH = PBI_DIR / "dax_measures.md"
PBI_DOC_PATH = PBI_DIR / "dashboard_documentation.md"
REPORT_PATH = prep.REPORT_DIR / "customer_risk_report.txt"
RETENTION_REPORT_PATH = prep.REPORT_DIR / "retention_strategy_report.txt"

SCORE_COLS = ["customerID", "Churn", "churn_probability", "predicted_churn_at_0.40", "risk_segment"]
FLAG_THRESHOLD = 0.40
SEGMENT_LABELS = ["Low", "Medium", "High", "Critical"]
SEGMENT_RULES = (
    "Low: probability < 0.25 | Medium: 0.25 <= p < 0.50 | "
    "High: 0.50 <= p < 0.75 | Critical: p >= 0.75"
)
HIGH_RISK_SEGMENTS = ["High", "Critical"]

BIZ_SEGMENTS = ["Contract", "TenureGroup", "InternetService", "PaymentMethod", "MonthlyChargeBand"]

COMBO_FILTERS = {
    "Month-to-month + Fiber optic":
        lambda d: (d["Contract"] == "Month-to-month") & (d["InternetService"] == "Fiber optic"),
    "0–6 months + Fiber optic":
        lambda d: (d["TenureGroup"] == "≤6 months") & (d["InternetService"] == "Fiber optic"),
    "Month-to-month + 0–6 months":
        lambda d: (d["Contract"] == "Month-to-month") & (d["TenureGroup"] == "≤6 months"),
}

PBI_COLUMNS = fe.MODEL_READY_ORDER + ["churn_probability", "predicted_churn_at_0.40", "risk_segment"]


# ---------------------------------------------------------------------------
# Model + scoring
# ---------------------------------------------------------------------------

def load_and_verify_model(path: Path = MODEL_PATH):
    model = joblib.load(path)
    steps = list(model.named_steps.keys())
    assert steps == ["preprocess", "classifier"], f"Unexpected pipeline steps: {steps}"
    clf = model.named_steps["classifier"]
    assert clf.__class__.__name__ == "LogisticRegression"
    return model


def assign_risk_segment(proba: np.ndarray) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    return np.select(
        [p < 0.25, p < 0.50, p < 0.75, p >= 0.75],
        SEGMENT_LABELS,
        default="UNASSIGNED",
    )


def score_customers(df: pd.DataFrame, model) -> pd.DataFrame:
    """Score every customer. Probability and binary flag stay separate columns."""
    X = df[prep.ALL_CANDIDATE_FEATURES]
    proba = model.predict_proba(X)[:, 1]
    scores = pd.DataFrame({
        "customerID": df[prep.ID_COL].astype(str),
        "Churn": df[prep.TARGET].astype(int),
        "churn_probability": proba,
        "predicted_churn_at_0.40": (proba >= FLAG_THRESHOLD).astype(int),
        "risk_segment": assign_risk_segment(proba),
    })
    assert not scores["risk_segment"].eq("UNASSIGNED").any(), "Some probabilities fell outside all segment bands"
    assert scores["customerID"].is_unique
    assert scores["churn_probability"].between(0, 1).all()
    assert scores["churn_probability"].notna().all()
    assert set(scores["predicted_churn_at_0.40"].unique()) <= {0, 1}
    assert set(scores["risk_segment"].unique()) <= set(SEGMENT_LABELS)
    return scores


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def segment_analysis(scores: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    t = (
        scores.groupby("risk_segment", observed=True)
        .agg(customers=("customerID", "count"),
             avg_predicted_risk=("churn_probability", "mean"),
             actual_churned=("Churn", "sum"))
        .reindex(SEGMENT_LABELS, fill_value=0)
        .reset_index()
    )
    t["pct_customers"] = (t["customers"] / len(scores) * 100).round(4)
    t["avg_predicted_risk"] = t["avg_predicted_risk"].round(4)
    t["actual_churn_rate_pct"] = (t["actual_churned"] / t["customers"] * 100).round(4)
    checks = {
        "customers_sum_to_total": int(t["customers"].sum()) == len(scores),
        "churned_sum_to_total": int(t["actual_churned"].sum()) == int(scores["Churn"].sum()),
        "all_segments_nonempty": bool((t["customers"] > 0).all()),
    }
    if not all(checks.values()):
        raise ValueError(f"Segment analysis failed: {checks}")
    return t, checks


def business_segment_analysis(df: pd.DataFrame, scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """count / avg predicted risk / actual historical churn rate per segment."""
    d = df[[prep.ID_COL]].merge(scores, on="customerID").drop(columns="customerID")
    d = pd.concat([d, df[BIZ_SEGMENTS].reset_index(drop=True)], axis=1)

    def _table(frame: pd.DataFrame, key: str) -> pd.DataFrame:
        t = (
            frame.groupby(key, observed=True)
            .agg(customers=("Churn", "count"),
                 avg_predicted_risk=("churn_probability", "mean"),
                 actual_churned=("Churn", "sum"))
            .reset_index()
        )
        t["actual_churn_rate_pct"] = (t["actual_churned"] / t["customers"] * 100).round(4)
        t["avg_predicted_risk"] = t["avg_predicted_risk"].round(4)
        return t.sort_values("avg_predicted_risk", ascending=False).reset_index(drop=True)

    out = {col: _table(d, col) for col in BIZ_SEGMENTS}
    for name, fn in COMBO_FILTERS.items():
        sub = d[fn(d)]
        out[name] = pd.DataFrame([{
            "customers": len(sub),
            "avg_predicted_risk": round(float(sub["churn_probability"].mean()), 4),
            "actual_churned": int(sub["Churn"].sum()),
            "actual_churn_rate_pct": round(float(sub["Churn"].mean() * 100), 4),
        }])
    return out


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def save_predictions(scores: pd.DataFrame) -> tuple[Path, Path]:
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores[SCORE_COLS].to_csv(SCORES_PATH, index=False, encoding="utf-8")
    high = scores[scores["risk_segment"].isin(HIGH_RISK_SEGMENTS)].sort_values(
        "churn_probability", ascending=False
    )
    high[SCORE_COLS].to_csv(HIGH_RISK_PATH, index=False, encoding="utf-8")
    return SCORES_PATH, HIGH_RISK_PATH


def build_powerbi_dataset(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    pbi = df.merge(scores[SCORE_COLS], on=["customerID", "Churn"], how="inner", validate="one_to_one")
    assert len(pbi) == len(df) == len(scores)
    pbi = pbi[PBI_COLUMNS]
    assert not pbi.isna().sum().sum(), "Power BI dataset has missing values"
    return pbi


COLUMN_DESCRIPTIONS = {
    "customerID": ("Identifier", "Unique customer identifier (join key for Power BI)"),
    "Churn": ("Target", "Actual historical churn: 1 = churned, 0 = retained"),
    "churn_probability": ("Model output", "Predicted churn probability from the saved Logistic Regression pipeline, 0-1 continuous"),
    "predicted_churn_at_0.40": ("Model output", "Binary flag: 1 if churn_probability >= 0.40 (operational flagging threshold), else 0"),
    "risk_segment": ("Model output", "Business risk band of churn_probability: Low <0.25, Medium 0.25-<0.50, High 0.50-<0.75, Critical >=0.75"),
}


def build_data_dictionary(pbi: pd.DataFrame) -> pd.DataFrame:
    from src.feature_engineering import FEATURE_MEANINGS

    rows = []
    for col in PBI_COLUMNS:
        if col in COLUMN_DESCRIPTIONS:
            source, desc = COLUMN_DESCRIPTIONS[col]
        else:
            source = "Engineered" if col in fe.ENGINEERED_FEATURES else "Original"
            desc = FEATURE_MEANINGS[col]
        rows.append({
            "Column": col,
            "Data Type": str(pbi[col].dtype),
            "Source": source,
            "Description": desc,
        })
    return pd.DataFrame(rows)


def save_powerbi(pbi: pd.DataFrame) -> tuple[Path, Path, Path, Path]:
    PBI_DIR.mkdir(parents=True, exist_ok=True)
    pbi.to_csv(PBI_DATASET_PATH, index=False, encoding="utf-8")
    build_data_dictionary(pbi).to_csv(PBI_DICTIONARY_PATH, index=False, encoding="utf-8")
    PBI_DAX_PATH.write_text(dax_measures_text(), encoding="utf-8")
    PBI_DOC_PATH.write_text(dashboard_documentation_text(), encoding="utf-8")
    return PBI_DATASET_PATH, PBI_DICTIONARY_PATH, PBI_DAX_PATH, PBI_DOC_PATH


def dax_measures_text() -> str:
    return """# DAX Measures - Churn Risk Dashboard (audited)

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
"""


def dashboard_documentation_text() -> str:
    return """# Power BI Dashboard Documentation - Telco Customer Churn & Retention (FINAL)

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
"""


# ---------------------------------------------------------------------------
# Report + orchestration
# ---------------------------------------------------------------------------

def validate_outputs(df: pd.DataFrame, scores: pd.DataFrame, pbi: pd.DataFrame, model) -> list[str]:
    results = []

    def check(name, ok, detail=""):
        if not ok:
            raise ValueError(f"RISK SCORING VALIDATION FAILED: {name} {detail}")
        results.append(f"PASS  {name}" + (f"  ({detail})" if detail else ""))

    check("exactly 7043 customers scored", len(scores) == 7043, f"{len(scores)}")
    check("customerID unique", bool(scores["customerID"].is_unique))
    check("probability within [0,1]", bool(scores["churn_probability"].between(0, 1).all()),
          f"[{scores['churn_probability'].min():.6f}, {scores['churn_probability'].max():.6f}]")
    check("no missing risk scores", int(scores[["churn_probability", "risk_segment"]].isna().sum().sum()) == 0)
    check("flag column in {0,1}", set(scores["predicted_churn_at_0.40"].unique()) <= {0, 1})
    check("segments in {Low, Medium, High, Critical}", set(scores["risk_segment"].unique()) == set(SEGMENT_LABELS))

    high_expected = scores["risk_segment"].isin(HIGH_RISK_SEGMENTS)
    high_csv = pd.read_csv(HIGH_RISK_PATH)
    check("High+Critical count == high-risk export rows",
          int(high_expected.sum()) == len(high_csv), f"{int(high_expected.sum())} == {len(high_csv)}")
    check("high-risk export sorted by probability desc",
          bool(high_csv["churn_probability"].is_monotonic_decreasing))
    check("high-risk export segments only High/Critical",
          set(high_csv["risk_segment"].unique()) == set(HIGH_RISK_SEGMENTS))

    check("Power BI dataset row count matches", len(pbi) == len(scores) == len(df), f"{len(pbi)}")
    merged = pbi.merge(scores[SCORE_COLS], on="customerID", suffixes=("", "_s"), validate="one_to_one")
    proba_match = np.allclose(merged["churn_probability"], merged["churn_probability_s"], atol=1e-12)
    seg_match = (merged["risk_segment"] == merged["risk_segment_s"]).all()
    flag_match = (merged["predicted_churn_at_0.40"] == merged["predicted_churn_at_0.40_s"]).all()
    check("Power BI dataset matches risk-score dataset", bool(proba_match and seg_match and flag_match))

    reloaded = load_and_verify_model(MODEL_PATH)
    repro_proba = reloaded.predict_proba(df[prep.ALL_CANDIDATE_FEATURES])[:, 1]
    check("saved model reproduces scores", bool(np.allclose(repro_proba, scores["churn_probability"], atol=1e-12)))

    check("raw dataset unchanged",
          prep.raw_sha256().lower() == "16320C9C1EC72448DB59AA0A26A0B95401046BEF5D02FD3AEB906448E3055E91".lower())
    check("cleaned dataset unchanged",
          prep.clean_sha256().lower() == "38E6F2DD278033A4617F880E8F11304C5FC6DCE2C05ECEF3F2A8BF48EEAA8996".lower())
    check("model-ready dataset unchanged",
          prep.sha256_of(prep.MODEL_READY_PATH).lower() == "DF02696964C95DE935070C92B147D591B100EA87746870A9EAE7F6D8CEFA4F73".lower())
    return results


def write_risk_report(df, scores, seg_table, seg_checks, biz, validation) -> Path:
    lines = []
    flagged = int(scores["predicted_churn_at_0.40"].sum())
    high = scores[scores["risk_segment"].isin(HIGH_RISK_SEGMENTS)].sort_values(
        "churn_probability", ascending=False
    )

    lines.append("=== CUSTOMER RISK SCORING REPORT (PHASE 7) ===")
    lines.append(f"Model            : {MODEL_PATH} (saved Phase-6 pipeline, NOT retrained)")
    lines.append(f"Classifier       : {model_description()}")
    lines.append(f"Scored source    : {prep.MODEL_READY_PATH}")
    lines.append(f"Cleaned SHA-256  : {prep.clean_sha256()}")
    lines.append("")
    lines.append("--- SCORING OVERVIEW ---")
    lines.append(f"Customers scored          : {len(scores)}")
    lines.append(f"Probability range         : [{scores['churn_probability'].min():.6f}, {scores['churn_probability'].max():.6f}]")
    lines.append(f"Mean predicted risk       : {scores['churn_probability'].mean():.4f}")
    lines.append(f"Flagged at threshold 0.40 : {flagged} ({flagged / len(scores) * 100:.2f}%)")
    lines.append(f"High + Critical (segments): {len(high)} ({len(high) / len(scores) * 100:.2f}%)")
    lines.append("")
    lines.append("--- CONCEPT SEPARATION ---")
    lines.append("Model probability   : continuous churn_probability = predict_proba[:, 1]")
    lines.append("Classification flag : predicted_churn_at_0.40 = (probability >= 0.40)")
    lines.append("Business segment    : risk_segment = probability band (rules below)")
    lines.append(SEGMENT_RULES)
    lines.append("0.40 is the operational flagging threshold specified for this phase. From the Phase-6")
    lines.append("OOF table, 0.60 maximizes F1 (0.6345) while 0.40 trades precision for recall (0.8676);")
    lines.append("the final business threshold should follow outreach cost/benefit, not test-set tuning.")
    lines.append("")
    lines.append("--- RISK SEGMENT ANALYSIS ---")
    lines.append(seg_table.to_string(index=False))
    lines.append(f"Totals: customers={int(seg_table['customers'].sum())} (== 7043: {seg_checks['customers_sum_to_total']}), "
                 f"churned={int(seg_table['actual_churned'].sum())} (== 1869: {seg_checks['churned_sum_to_total']})")
    lines.append("")
    lines.append("--- BUSINESS SEGMENTS (count / avg predicted risk / actual historical churn) ---")
    for name, t in biz.items():
        lines.append(f"\n{name}:")
        lines.append(t.to_string(index=False))
    lines.append("")
    lines.append("--- HIGH-RISK EXPORT (High + Critical, sorted by probability desc) ---")
    lines.append(f"File: {HIGH_RISK_PATH} ({len(high)} customers)")
    lines.append("Top 10 preview:")
    lines.append(high.head(10)[SCORE_COLS].to_string(index=False))
    lines.append("")
    lines.append("--- POWER BI ARTIFACTS ---")
    for p in [PBI_DATASET_PATH, PBI_DICTIONARY_PATH, PBI_DAX_PATH, PBI_DOC_PATH]:
        lines.append(f"  {p}  exists={p.exists()}")
    lines.append("")
    lines.append("--- METHODOLOGY & LIMITATIONS ---")
    lines.append("1. The saved pipeline was applied as-is; no retraining or refitting occurred.")
    lines.append("2. Scores for the 5,634 training customers are IN-SAMPLE (seen during fitting);")
    lines.append("   only the 1,409 test customers are out-of-sample. Use scores for triage, not as")
    lines.append("   calibrated outcome probabilities.")
    lines.append("3. Segment churn rates are historical observations; higher predicted risk and higher")
    lines.append("   observed churn are associations, not causal effects.")
    lines.append("4. Threshold 0.40 is an operational specification; business threshold selection")
    lines.append("   requires cost/benefit analysis (outreach cost vs retained revenue).")
    lines.append("")
    lines.append("--- VALIDATION ---")
    lines.extend(validation)
    lines.append("")
    lines.append("PHASE 7 RISK SCORING: SUCCESS")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT_PATH


def model_description() -> str:
    model = load_and_verify_model()
    clf = model.named_steps["classifier"]
    return (f"Pipeline(preprocess -> {clf.__class__.__name__}, class_weight={clf.class_weight}, "
            f"C={clf.C}, solver={clf.solver}, max_iter={clf.max_iter})")


def write_retention_strategy(df: pd.DataFrame, scores: pd.DataFrame) -> Path:
    """Retention action plan per risk priority, built from ACTUAL scoring output."""
    d = df[[prep.ID_COL, "MonthlyCharges", "Contract", "TenureGroup", "InternetService",
            "PaymentMethod", "OnlineSecurity", "TechSupport"]].merge(
        scores, on=prep.ID_COL, validate="one_to_one")
    seg = (
        d.groupby("risk_segment", observed=True)
        .agg(customers=("customerID", "count"),
             avg_predicted_risk=("churn_probability", "mean"),
             actual_churned=("Churn", "sum"),
             avg_monthly_charges=("MonthlyCharges", "mean"),
             total_monthly_charges=("MonthlyCharges", "sum"))
        .reindex(SEGMENT_LABELS)
        .reset_index()
    )
    seg["pct_customers"] = (seg["customers"] / len(scores) * 100).round(2)
    seg["actual_churn_rate_pct"] = (seg["actual_churned"] / seg["customers"] * 100).round(2)
    seg["avg_predicted_risk"] = seg["avg_predicted_risk"].round(4)
    seg["avg_monthly_charges"] = seg["avg_monthly_charges"].round(2)
    seg["total_monthly_charges"] = seg["total_monthly_charges"].round(2)

    crit = seg.iloc[3]
    high = seg.iloc[2]
    med = seg.iloc[1]
    low = seg.iloc[0]

    # Within Critical: where is the risk concentrated? (actual overlap counts)
    crit_d = d[d["risk_segment"] == "Critical"]
    crit_m2m = int((crit_d["Contract"] == "Month-to-month").sum())
    crit_fiber = int((crit_d["InternetService"] == "Fiber optic").sum())
    crit_echeck = int((crit_d["PaymentMethod"] == "Electronic check").sum())
    crit_new = int((crit_d["TenureGroup"] == "≤6 months").sum())

    coef_path = prep.REPORT_DIR / "logistic_coefficients.csv"
    coef = pd.read_csv(coef_path).head(10)

    lines = []
    A = lines.append
    A("=== RETENTION STRATEGY REPORT (PHASE 8) ===")
    A("Basis: actual risk scores of all 7,043 customers (models/churn_model.pkl applied as-is)")
    A("All figures are observed values from data/processed/model_ready_data.csv + scored output.")
    A("")
    A("--- SEGMENT OVERVIEW ---")
    A(seg.to_string(index=False))
    A("")
    A("--- TOP MODEL DRIVERS (from logistic_coefficients.csv; associations, not causes) ---")
    A(coef.to_string(index=False))
    A("Reading: positive coefficients increase churn odds, negative decrease (odds ratio = exp(coef)),")
    A("each holding all other included variables constant.")
    A("")
    A(f"--- PRIORITY 1: CRITICAL-RISK CUSTOMERS (probability >= 0.75) ---")
    A(f"Customers            : {int(crit['customers'])} ({crit['pct_customers']}% of base)")
    A(f"Average risk         : {crit['avg_predicted_risk']:.4f}")
    A(f"Historical churn     : {int(crit['actual_churned'])} of {int(crit['customers'])} ({crit['actual_churn_rate_pct']}%)")
    A(f"Monthly revenue pool : ${crit['total_monthly_charges']:,.2f} (avg ${crit['avg_monthly_charges']:.2f}/customer)")
    A(f"Concentration        : {crit_m2m} month-to-month | {crit_fiber} fiber optic | "
          f"{crit_echeck} electronic check | {crit_new} tenure <= 6 months")
    A("Proposed actions:")
    A("  1. Proactive human outreach within 24-48 hours (call, not email) - highest-risk pool.")
    A("  2. Save-desk retention offers: targeted discounts or plan changes; prioritize")
    A("     month-to-month + fiber customers (largest high-risk overlap in this segment).")
    A("  3. Contract upgrade incentive to one/two-year terms (two-year contracts show the")
    A("     lowest observed churn: 2.83%).")
    A("  4. Executive/senior-agent escalation for the top probability decile.")
    A("  5. Track save rate per offer; A/B test before scaling (not yet proven to reduce churn).")
    A("")
    A(f"--- PRIORITY 2: HIGH-RISK CUSTOMERS (probability 0.50-<0.75) ---")
    A(f"Customers            : {int(high['customers'])} ({high['pct_customers']}% of base)")
    A(f"Average risk         : {high['avg_predicted_risk']:.4f}")
    A(f"Historical churn     : {int(high['actual_churned'])} of {int(high['customers'])} ({high['actual_churn_rate_pct']}%)")
    A(f"Monthly revenue pool : ${high['total_monthly_charges']:,.2f} (avg ${high['avg_monthly_charges']:.2f}/customer)")
    A("Proposed actions:")
    A("  1. Targeted email/SMS campaign within 1-2 weeks with a clear value proposition.")
    A("  2. Offer security/support add-on trials: customers WITHOUT OnlineSecurity showed")
    A("     41.77% observed churn vs 14.61% with it; TechSupport 41.64% vs 15.17%.")
    A("  3. Payment migration nudge: move electronic-check users to automatic payments")
    A("     (electronic check observed churn 45.29% vs 15.24-16.71% for automatic methods).")
    A("  4. Monitor engagement signals; escalate to Priority 1 behavior if risk rises on re-score.")
    A("")
    A(f"--- PRIORITY 3: MEDIUM-RISK CUSTOMERS (probability 0.25-<0.50) ---")
    A(f"Customers            : {int(med['customers'])} ({med['pct_customers']}% of base)")
    A(f"Average risk         : {med['avg_predicted_risk']:.4f}")
    A(f"Historical churn     : {int(med['actual_churned'])} of {int(med['customers'])} ({med['actual_churn_rate_pct']}%)")
    A("Proposed actions:")
    A("  1. Low-cost digital engagement: onboarding/education content, feature tips.")
    A("  2. Satisfaction pulse survey to catch fixable friction early.")
    A("  3. Loyalty/rewards enrollment for tenure milestones.")
    A("  4. Include in quarterly re-scoring; promote to higher priority if risk increases.")
    A("")
    A(f"--- STANDARD MANAGEMENT: LOW-RISK CUSTOMERS (probability < 0.25) ---")
    A(f"Customers            : {int(low['customers'])} ({low['pct_customers']}% of base)")
    A(f"Average risk         : {low['avg_predicted_risk']:.4f}")
    A(f"Historical churn     : {int(low['actual_churned'])} of {int(low['customers'])} ({low['actual_churn_rate_pct']}%)")
    A("Proposed actions:")
    A("  1. Standard lifecycle communication; no expensive retention spend.")
    A("  2. Referral and advocacy programs (satisfied, stable base).")
    A("  3. Careful upsell of add-ons (ServiceCount associates with retention) without")
    A("     pushing overall price into the high monthly-charge bands.")
    A("  4. Re-score quarterly to detect risk drift early.")
    A("")
    A("--- GOVERNANCE & CAVEATS ---")
    A("1. These are PROPOSED actions based on observed associations in historical data;")
    A("   none has been proven to reduce churn in this dataset. Validate via A/B tests.")
    A("2. Re-score the base regularly (model: models/churn_model.pkl, pipeline: run_pipeline.ps1).")
    A("3. The 0.40 operational threshold trades precision for recall; the final business")
    A("   threshold should reflect outreach cost vs expected retained revenue.")
    A("4. Training-set customers are scored in-sample; treat per-customer scores as triage")
    A("   priorities, not calibrated probabilities.")
    A("")
    A("PHASE 8 RETENTION STRATEGY: GENERATED FROM ACTUAL SCORING OUTPUT")
    RETENTION_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RETENTION_REPORT_PATH


def run_all(verbose: bool = True) -> dict:
    def log(text=""):
        if verbose:
            print(text)

    model = load_and_verify_model()
    log(f"Model loaded & verified: {MODEL_PATH}")
    log(f"  {model_description()}")

    df = prep.load_model_ready()
    scores = score_customers(df, model)
    log(f"Scored {len(scores)} customers (probability + 0.40 flag + segment)")

    seg_table, seg_checks = segment_analysis(scores)
    log("\nRisk segment analysis:")
    log(seg_table.to_string(index=False))

    s_path, h_path = save_predictions(scores)
    log(f"\nSaved: {s_path}")
    log(f"Saved: {h_path} ({int(scores['risk_segment'].isin(HIGH_RISK_SEGMENTS).sum())} customers)")

    biz = business_segment_analysis(df, scores)
    log("\nBusiness segment analysis computed for: " + ", ".join(biz.keys()))

    pbi = build_powerbi_dataset(df, scores)
    pbi_paths = save_powerbi(pbi)
    log(f"\nPower BI artifacts written: {[str(p) for p in pbi_paths]}")

    validation = validate_outputs(df, scores, pbi, model)
    log("\nValidation:")
    for r in validation:
        log(r)

    report = write_risk_report(df, scores, seg_table, seg_checks, biz, validation)
    log(f"\nReport written: {report}")

    retention = write_retention_strategy(df, scores)
    log(f"Retention strategy report: {retention}")

    return {"model": model, "scores": scores, "segment_table": seg_table,
            "business_segments": biz, "powerbi": pbi, "validation": validation,
            "report": report, "retention_report": retention}


if __name__ == "__main__":
    run_all()
