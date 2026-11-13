"""Feature engineering & feature selection for the Telco churn project.

Reads data/processed/cleaned_customer_churn.csv (read-only) and produces:
  * outputs/reports/feature_inventory.csv          (pre-engineering review)
  * outputs/reports/feature_leakage_review.csv     (leakage assessment)
  * outputs/reports/feature_selection_report.csv   (final feature metadata)
  * outputs/reports/feature_engineering_report.txt (validation summary)
  * data/processed/model_ready_data.csv            (customerID + candidates + target)

No model training, no scaler/encoder/imputer fitting happens here - that is
Phase 6. Categorical values are NOT one-hot encoded in this file; the
model-ready dataset keeps raw (human-readable) values.

All engineered features are derived from information known BEFORE a customer
churns (pre-churn billing and service attributes). No causal claims.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_customer_churn.csv"
MODEL_READY_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_data.csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
INVENTORY_PATH = REPORT_DIR / "feature_inventory.csv"
LEAKAGE_PATH = REPORT_DIR / "feature_leakage_review.csv"
SELECTION_PATH = REPORT_DIR / "feature_selection_report.csv"
TEXT_REPORT_PATH = REPORT_DIR / "feature_engineering_report.txt"

ID_COL = "customerID"
TARGET = "Churn"

ORIGINAL_PREDICTORS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# ServiceCount contributors: the 7 OPTIONAL add-on services.
#   Yes -> 1, No -> 0, "No internet service" / "No phone service" -> 0.
# PhoneService (base voice, ~90% Yes) and InternetService (captured separately
# as its own categorical predictor) are deliberately NOT counted.
SERVICE_COUNT_COLUMNS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

TENURE_LABELS = ["≤6 months", "7–12 months", "13–24 months", "25+ months"]
TENURE_BINS = [-1, 6, 12, 24, np.inf]
MC_BAND_LABELS = ["Low", "Medium-Low", "Medium-High", "High"]

ENGINEERED_FEATURES = [
    "TenureGroup", "MonthlyChargeBand", "ServiceCount",
    "HasSecuritySupport", "AvgMonthlyCharge",
]

MODEL_READY_ORDER = [
    ID_COL,
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "TenureGroup",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies",
    "ServiceCount", "HasSecuritySupport",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "MonthlyChargeBand", "TotalCharges", "AvgMonthlyCharge",
    TARGET,
]

NUMERIC_CANDIDATES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "ServiceCount", "HasSecuritySupport", "AvgMonthlyCharge",
]

FEATURE_MEANINGS = {
    "customerID": "Unique customer identifier",
    "gender": "Customer gender (Female / Male)",
    "SeniorCitizen": "1 if customer is a senior citizen (65+), else 0",
    "Partner": "Customer has a spouse or partner (Yes / No)",
    "Dependents": "Customer has dependents such as children (Yes / No)",
    "tenure": "Number of months the customer has been with the company",
    "PhoneService": "Subscribes to home phone service (Yes / No)",
    "MultipleLines": "Multiple phone lines for phone customers (Yes / No / No phone service)",
    "InternetService": "Internet service type (DSL / Fiber optic / No)",
    "OnlineSecurity": "Online security add-on (Yes / No / No internet service)",
    "OnlineBackup": "Online backup add-on (Yes / No / No internet service)",
    "DeviceProtection": "Device protection add-on (Yes / No / No internet service)",
    "TechSupport": "Tech support add-on (Yes / No / No internet service)",
    "StreamingTV": "TV streaming add-on (Yes / No / No internet service)",
    "StreamingMovies": "Movie streaming add-on (Yes / No / No internet service)",
    "Contract": "Contract term (Month-to-month / One year / Two year)",
    "PaperlessBilling": "Bill delivered electronically (Yes / No)",
    "PaymentMethod": "How the customer pays (electronic check, mailed check, automatic bank transfer, automatic credit card)",
    "MonthlyCharges": "Current monthly charge in USD",
    "TotalCharges": "Total amount billed to date in USD (accumulates with tenure)",
    "Churn": "TARGET: 1 if the customer left, 0 if retained (known from Phase 3 encoding)",
    "TenureGroup": "Engineered: relationship-age band derived from tenure (tenure=0 falls in the first group)",
    "MonthlyChargeBand": "Engineered: MonthlyCharges quartile band (boundaries computed from this dataset)",
    "ServiceCount": "Engineered: number of optional add-on services subscribed (0-7; see SERVICE_COUNT_COLUMNS)",
    "HasSecuritySupport": "Engineered: 1 if customer has OnlineSecurity and/or TechSupport, else 0",
    "AvgMonthlyCharge": "Engineered: average monthly spend = TotalCharges/tenure; for tenure=0 uses MonthlyCharges (no billing history yet)",
}

REDUNDANCY_NOTES = {
    "customerID": "None (identifier, not a predictor)",
    "gender": "None structural; weak standalone signal - keep, let modeling decide",
    "SeniorCitizen": "None",
    "Partner": "None",
    "Dependents": "None",
    "tenure": "Overlaps engineered TenureGroup (binned copy); corr 0.826 with TotalCharges",
    "PhoneService": "Low variance (most customers Yes); prerequisite for MultipleLines",
    "MultipleLines": "'No phone service' category is structurally determined by PhoneService",
    "InternetService": "'No' drives 'No internet service' in 6 add-on columns",
    "OnlineSecurity": "'No internet service' determined by InternetService; summarized by ServiceCount",
    "OnlineBackup": "'No internet service' determined by InternetService; summarized by ServiceCount",
    "DeviceProtection": "'No internet service' determined by InternetService; summarized by ServiceCount",
    "TechSupport": "'No internet service' determined by InternetService; summarized by ServiceCount",
    "StreamingTV": "'No internet service' determined by InternetService; summarized by ServiceCount",
    "StreamingMovies": "'No internet service' determined by InternetService; summarized by ServiceCount",
    "Contract": "None",
    "PaperlessBilling": "None",
    "PaymentMethod": "None",
    "MonthlyCharges": "Overlaps engineered MonthlyChargeBand (binned copy) and AvgMonthlyCharge",
    "TotalCharges": "corr 0.826 with tenure (multicollinearity concern for linear models); keep - assess at training",
    "Churn": "Target - never a predictor",
    "TenureGroup": "Binned copy of tenure (exact duplication of information at group level)",
    "MonthlyChargeBand": "Binned copy of MonthlyCharges (quartiles)",
    "ServiceCount": "Additive summary of the 7 add-on columns; overlaps individual add-on flags",
    "HasSecuritySupport": "Overlaps OnlineSecurity, TechSupport and ServiceCount",
    "AvgMonthlyCharge": "Derived from TotalCharges/tenure; overlaps MonthlyCharges and TotalCharges",
}


def sha256_of(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cleaned(path: Path = CLEAN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert set(df[TARGET].unique()) == {0, 1}, "Churn must be 0/1"
    return df


# ---------------------------------------------------------------------------
# 1. Feature inventory (pre-engineering review)
# ---------------------------------------------------------------------------

def _eda_spread(df: pd.DataFrame, col: str) -> str:
    if col in ("tenure", "MonthlyCharges", "TotalCharges"):
        r = df[col].corr(df[TARGET])
        return f"Pearson r with churn = {r:.4f}"
    t = df.groupby(col, observed=True)[TARGET].mean()
    spread = (t.max() - t.min()) * 100
    lo_cat = t.idxmin()
    hi_cat = t.idxmax()
    return (
        f"Churn-rate spread {spread:.2f} pts across categories "
        f"({t[lo_cat] * 100:.2f}% for {lo_cat} vs {t[hi_cat] * 100:.2f}% for {hi_cat})"
    )


PREDICTIVE_NOTES = {
    "customerID": "None - unique per row",
    "Churn": "Target variable",
}


def feature_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [ID_COL] + ORIGINAL_PREDICTORS + [TARGET]:
        dtype = str(df[col].dtype)
        if col in ("tenure", "MonthlyCharges", "TotalCharges"):
            uniq = f"range [{df[col].min()}, {df[col].max()}], {df[col].nunique()} unique values"
        else:
            vals = df[col].astype(str).value_counts()
            uniq = f"{len(vals)} unique: {', '.join(vals.index)}"
        predictive = PREDICTIVE_NOTES.get(col) or _eda_spread(df, col)
        leakage = (
            "None - identifier" if col == ID_COL
            else "Target" if col == TARGET
            else "Low - pre-churn billing/service attribute"
        )
        if col == ID_COL:
            decision = "Keep in dataset for customer-level reporting only; EXCLUDE from feature matrix"
        elif col == TARGET:
            decision = "Target - exclude from feature matrix"
        else:
            decision = "KEEP as candidate predictor (EDA supports signal; modeling phase finalizes)"
        rows.append({
            "Feature": col,
            "Data type": dtype,
            "Meaning": FEATURE_MEANINGS[col],
            "Unique values / range": uniq,
            "Potential predictive value": predictive,
            "Potential redundancy": REDUNDANCY_NOTES[col],
            "Leakage risk": leakage,
            "Final decision": decision,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Engineered features
# ---------------------------------------------------------------------------

def add_tenure_group(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    out["TenureGroup"] = pd.cut(
        out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, right=True
    )
    counts = out["TenureGroup"].value_counts().reindex(TENURE_LABELS, fill_value=0)
    # Independent recomputation via explicit masks (cross-check of pd.cut)
    masks = {
        "≤6 months": out["tenure"].between(0, 6),
        "7–12 months": out["tenure"].between(7, 12),
        "13–24 months": out["tenure"].between(13, 24),
        "25+ months": out["tenure"] >= 25,
    }
    checks = {
        "no_missing": bool(out["TenureGroup"].isna().sum() == 0),
        "all_groups_nonempty": bool((counts > 0).all()),
        "counts_sum_to_dataset": bool(int(counts.sum()) == len(out)),
        "exactly_one_group_per_customer": bool(
            pd.concat(masks.values(), axis=1).sum(axis=1).eq(1).all()
        ),
        "counts_match_independent_masks": all(
            int(counts[label]) == int(mask.sum()) for label, mask in masks.items()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"TenureGroup validation failed: {checks}")
    info = {"counts": {k: int(v) for k, v in counts.items()},
            "tenure_zero_in_first_group": int((out["tenure"] == 0).sum())}
    return out, {**checks, **info}


def add_monthly_charge_band(df: pd.DataFrame) -> tuple[pd.DataFrame, list[float], dict]:
    out = df.copy()
    banded, boundaries = pd.qcut(
        out["MonthlyCharges"], q=4, labels=MC_BAND_LABELS, retbins=True, duplicates="raise"
    )
    out["MonthlyChargeBand"] = banded
    if not np.all(np.diff(boundaries) > 0):
        raise ValueError(f"MonthlyCharges quartile boundaries not strictly increasing: {boundaries}")
    counts = out["MonthlyChargeBand"].value_counts().reindex(MC_BAND_LABELS, fill_value=0)
    checks = {
        "no_missing": bool(out["MonthlyChargeBand"].isna().sum() == 0),
        "counts_sum_to_dataset": bool(int(counts.sum()) == len(out)),
        "boundaries_strictly_increasing": True,
    }
    info = {"counts": {k: int(v) for k, v in counts.items()}}
    return out, [round(float(b), 4) for b in boundaries], {**checks, **info}


def add_service_count(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    contrib = 0
    for col in SERVICE_COUNT_COLUMNS:
        vals = out[col].astype(str)
        contrib = contrib + (vals == "Yes").astype("int64")
    out["ServiceCount"] = contrib

    zero_expected = int((out[SERVICE_COUNT_COLUMNS].astype(str) != "Yes").all(axis=1).sum())
    stats = {
        "min": int(out["ServiceCount"].min()),
        "max": int(out["ServiceCount"].max()),
        "mean": round(float(out["ServiceCount"].mean()), 4),
        "median": float(out["ServiceCount"].median()),
        "missing": int(out["ServiceCount"].isna().sum()),
        "contributors": SERVICE_COUNT_COLUMNS,
        "treatment": "Yes=1, No=0, 'No internet service'/'No phone service'=0; "
                     "PhoneService & InternetService excluded (see docstring)",
        "zero_count_matches_all_not_yes": bool(int((out["ServiceCount"] == 0).sum()) == zero_expected),
    }
    if stats["missing"] != 0 or not stats["zero_count_matches_all_not_yes"]:
        raise ValueError(f"ServiceCount validation failed: {stats}")
    return out, stats


def add_has_security_support(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["HasSecuritySupport"] = (
        (out["OnlineSecurity"].astype(str) == "Yes") | (out["TechSupport"].astype(str) == "Yes")
    ).astype("int64")
    return out


def add_avg_monthly_charge(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    tenure = out["tenure"].astype("float64")
    out["AvgMonthlyCharge"] = np.where(
        tenure > 0,
        out["TotalCharges"] / tenure.where(tenure > 0, 1),
        out["MonthlyCharges"],
    )
    zero_rows = out["tenure"] == 0
    ok_zero = bool(np.allclose(
        out.loc[zero_rows, "AvgMonthlyCharge"], out.loc[zero_rows, "MonthlyCharges"]
    ))
    nonzero = ~zero_rows
    ok_ratio = bool(np.allclose(
        out.loc[nonzero, "AvgMonthlyCharge"],
        out.loc[nonzero, "TotalCharges"] / out.loc[nonzero, "tenure"],
    ))
    stats = {
        "min": round(float(out["AvgMonthlyCharge"].min()), 4),
        "max": round(float(out["AvgMonthlyCharge"].max()), 4),
        "mean": round(float(out["AvgMonthlyCharge"].mean()), 4),
        "median": round(float(out["AvgMonthlyCharge"].median()), 4),
        "missing": int(out["AvgMonthlyCharge"].isna().sum()),
        "tenure_zero_rows_use_monthlycharges": ok_zero,
        "nonzero_rows_match_total_div_tenure": ok_ratio,
        "tenure_zero_rows": int(zero_rows.sum()),
    }
    if stats["missing"] != 0 or not (ok_zero and ok_ratio):
        raise ValueError(f"AvgMonthlyCharge validation failed: {stats}")
    return out, stats


# ---------------------------------------------------------------------------
# 3. Redundancy analysis
# ---------------------------------------------------------------------------

def redundancy_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = NUMERIC_CANDIDATES + ["tenure", "MonthlyCharges", "TotalCharges"]
    corr = df[list(dict.fromkeys(cols))].corr(method="pearson").round(4)
    notes = [
        f"corr(tenure, TotalCharges) = {corr.loc['tenure', 'TotalCharges']:.4f}: TotalCharges "
        "structurally accumulates as roughly tenure x average monthly price, so the pair is "
        "necessarily correlated - this is expected, not a data error.",
        "Why keep BOTH: tenure measures relationship length; TotalCharges measures lifetime "
        "billing value. Two customers with equal tenure can have very different TotalCharges "
        "(price tier), and equal TotalCharges can hide very different tenures. Each carries "
        "information the other does not fully determine.",
        "Multicollinearity concern: mainly affects linear models (inflated coefficient "
        "variance, unstable coefficients). Tree-based and boosted models are largely "
        "unaffected by pairwise collinearity.",
        "Assessment plan (Phase 6, not now): compute VIF on the numeric block, compare "
        "regularized linear models with/without TotalCharges, and inspect coefficient "
        "stability. No feature is dropped preemptively.",
        f"corr(TotalCharges, AvgMonthlyCharge) = {corr.loc['TotalCharges', 'AvgMonthlyCharge']:.4f}, "
        f"corr(tenure, AvgMonthlyCharge) = {corr.loc['tenure', 'AvgMonthlyCharge']:.4f}: "
        "AvgMonthlyCharge partially de-couples price level from tenure but still overlaps "
        "MonthlyCharges; flagged as redundancy concern for linear models.",
    ]
    return corr, notes


# ---------------------------------------------------------------------------
# 4. Leakage review
# ---------------------------------------------------------------------------

def leakage_review(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in MODEL_READY_ORDER:
        if col == ID_COL:
            risk, included, reason = "None (identifier)", "No", (
                "Identifier; kept in the dataset for customer-level predictions and reporting "
                "but excluded from the model feature matrix (no predictive content, "
                "memorization risk)"
            )
        elif col == TARGET:
            risk, included, reason = "Target variable", "No (target)", (
                "Prediction target; by definition known only after/at churn - must never "
                "be a predictor"
            )
        else:
            derived = col in ENGINEERED_FEATURES
            risk, included = "Low", "Yes"
            reason = (
                f"{'Engineered from' if derived else 'Pre-churn billing/service attribute:'} "
                f"{FEATURE_MEANINGS[col].split(':')[0] if derived else FEATURE_MEANINGS[col]}"
                " - known before churn occurs"
            )
            if col in ("tenure", "TotalCharges", "MonthlyCharges"):
                reason = "Pre-churn billing attribute: " + FEATURE_MEANINGS[col]
        rows.append({"Feature": col, "Leakage Risk": risk, "Included": included, "Reason": reason})
    out = pd.DataFrame(rows)
    assert out.loc[out["Feature"] == TARGET, "Included"].iloc[0] != "Yes"
    assert out.loc[out["Feature"] == ID_COL, "Included"].iloc[0] == "No"
    return out


# ---------------------------------------------------------------------------
# 5. Final selection + model-ready dataset
# ---------------------------------------------------------------------------

def feature_selection_report(df: pd.DataFrame, mc_boundaries: list[float]) -> pd.DataFrame:
    inv = feature_inventory(df)
    leak = leakage_review(df)
    inv_map = inv.set_index("Feature")
    leak_map = leak.set_index("Feature")

    rows = []
    for col in MODEL_READY_ORDER:
        original = col in ORIGINAL_PREDICTORS or col in (ID_COL, TARGET)
        dtype = str(df[col].dtype)
        if col == ID_COL:
            included, reason = "No (dataset only)", "Identifier for customer-level joins/predictions"
        elif col == TARGET:
            included, reason = "No (target)", "Prediction target"
        elif col in ENGINEERED_FEATURES:
            included = "Yes"
            reason = {
                "TenureGroup": "EDA: churn 52.94% (0-6m) vs 14.04% (25+m) - early-life risk is the strongest single pattern; binned form aids interpretability",
                "MonthlyChargeBand": "EDA: churn rises from 10.89% (<=35) to 36.33% (75-95) by charge level; quartile band computed from this dataset",
                "ServiceCount": "EDA: add-on ownership associates with retention (security/support ~3x lower observed churn); count captures bundling breadth",
                "HasSecuritySupport": "EDA: OnlineSecurity No 41.77% vs Yes 14.61%; TechSupport No 41.64% vs Yes 15.17% - protective add-ons merit a targeted flag",
                "AvgMonthlyCharge": "Charge-to-tenure relationship: separates current price from lifetime average spend; tenure=0 uses MonthlyCharges",
            }[col]
        else:
            included = "Yes"
            reason = "EDA shows signal: " + str(inv_map.loc[col, "Potential predictive value"])
        rows.append({
            "Feature": col,
            "Original/Engineered": "Engineered" if col in ENGINEERED_FEATURES else "Original",
            "Data Type": dtype,
            "Included in Model": included,
            "Reason": reason,
            "Leakage Risk": str(leak_map.loc[col, "Leakage Risk"]),
            "Redundancy Concern": REDUNDANCY_NOTES[col],
        })
    rep = pd.DataFrame(rows)
    assert (rep["Feature"] == ID_COL).sum() == 1 and (rep["Feature"] == TARGET).sum() == 1
    assert not rep.loc[rep["Included in Model"].str.startswith("Yes"), "Feature"].isin([ID_COL, TARGET]).any()
    return rep


def validate_model_ready(df: pd.DataFrame, require_order: bool = False) -> list[str]:
    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            raise ValueError(f"VALIDATION FAILED: {name} {detail}")
        results.append(f"PASS  {name}" + (f"  ({detail})" if detail else ""))

    check("row count == 7043", len(df) == 7043, f"{len(df)}")
    if require_order:
        check("column order as designed", list(df.columns) == MODEL_READY_ORDER)
    else:
        check("required columns present, nothing extra",
              set(df.columns) == set(MODEL_READY_ORDER), f"{len(df.columns)} columns")
    check("customerID unique / no duplicates", df[ID_COL].is_unique)
    check("no missing values anywhere", int(df.isna().sum().sum()) == 0)
    check("Churn in {0,1}", set(df[TARGET].unique()) == {0, 1})
    check("TenureGroup labels exact", set(df["TenureGroup"].astype(str)) == set(TENURE_LABELS),
          f"{sorted(set(df['TenureGroup'].astype(str)))}")
    check("TenureGroup group counts > 0",
          bool((df["TenureGroup"].value_counts() > 0).all()),
          str({k: int(v) for k, v in df["TenureGroup"].value_counts().items()}))
    check("tenure=0 in first group", bool((df.loc[df["tenure"] == 0, "TenureGroup"].astype(str) == "≤6 months").all()),
          f"{int((df['tenure'] == 0).sum())} rows")
    check("MonthlyChargeBand labels exact", set(df["MonthlyChargeBand"].astype(str)) == set(MC_BAND_LABELS))
    check("ServiceCount in [0,7]", bool(df["ServiceCount"].between(0, 7).all()),
          f"min={df['ServiceCount'].min()}, max={df['ServiceCount'].max()}")
    check("HasSecuritySupport in {0,1}", set(df["HasSecuritySupport"].unique()) == {0, 1})
    check("AvgMonthlyCharge positive", bool((df["AvgMonthlyCharge"] > 0).all()),
          f"range [{df['AvgMonthlyCharge'].min():.4f}, {df['AvgMonthlyCharge'].max():.4f}]")
    check("no leakage features present",
          not {"Churn_encoded_from_target", "ChurnDate", "ExitDate", "PostChurn"}.intersection(df.columns))
    return results


def finalize_and_save(df: pd.DataFrame, verbose: bool = True) -> dict[str, Path]:
    """Validate everything and write all Phase-5 artifacts. Single code path."""
    lines: list[str] = []

    def log(text: str = "") -> None:
        lines.append(text)
        if verbose:
            print(text)

    log("=== FEATURE ENGINEERING & SELECTION REPORT ===")
    log(f"Source                 : {CLEAN_PATH}")
    log(f"Cleaned SHA-256        : {sha256_of(CLEAN_PATH)}")

    inventory = feature_inventory(df)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(INVENTORY_PATH, index=False, encoding="utf-8")
    log(f"\nFeature inventory      : {len(inventory)} features reviewed -> {INVENTORY_PATH}")

    df, tg_checks = add_tenure_group(df)
    log("\n--- TenureGroup ---")
    log(f"Labels                 : {TENURE_LABELS}")
    log(f"Counts                 : {tg_checks['counts']}")
    log(f"tenure=0 rows          : {tg_checks['tenure_zero_in_first_group']} (all in first group)")
    log("Validations            : " + ", ".join(f"{k}={v}" for k, v in tg_checks.items()
                                                    if isinstance(v, bool)))

    df, mc_bounds, mc_checks = add_monthly_charge_band(df)
    log("\n--- MonthlyChargeBand ---")
    log("Method                 : quartiles (qcut q=4) of MonthlyCharges computed from this dataset")
    log(f"Boundaries             : {mc_bounds}")
    log(f"Counts                 : {mc_checks['counts']}")
    log("Validations            : " + ", ".join(f"{k}={v}" for k, v in mc_checks.items()))

    df, sc_stats = add_service_count(df)
    log("\n--- ServiceCount ---")
    log(f"Contributors           : {sc_stats['contributors']}")
    log(f"Treatment              : {sc_stats['treatment']}")
    log(f"Stats                  : min={sc_stats['min']}, max={sc_stats['max']}, "
        f"mean={sc_stats['mean']}, median={sc_stats['median']}, missing={sc_stats['missing']}")

    df = add_has_security_support(df)
    log("\n--- HasSecuritySupport ---")
    log(f"Counts                 : {df['HasSecuritySupport'].value_counts().to_dict()}")

    df, amc_stats = add_avg_monthly_charge(df)
    log("\n--- AvgMonthlyCharge ---")
    log(f"Stats                  : min={amc_stats['min']}, max={amc_stats['max']}, "
        f"mean={amc_stats['mean']}, median={amc_stats['median']}, missing={amc_stats['missing']}")
    log(f"tenure=0 handling      : {amc_stats['tenure_zero_rows']} rows use MonthlyCharges "
        f"(verified={amc_stats['tenure_zero_rows_use_monthlycharges']})")

    corr, red_notes = redundancy_analysis(df)
    log("\n--- Redundancy analysis (numeric block, Pearson) ---")
    log(corr.to_string())
    for i, n in enumerate(red_notes, 1):
        log(f"{i}. {n}")

    leak = leakage_review(df)
    leak.to_csv(LEAKAGE_PATH, index=False, encoding="utf-8")
    log(f"\nLeakage review         : {len(leak)} rows -> {LEAKAGE_PATH}")
    log(f"Excluded (not predictors): {leak.loc[leak['Included'] != 'Yes', 'Feature'].tolist()}")

    selection = feature_selection_report(df, mc_bounds)
    selection.to_csv(SELECTION_PATH, index=False, encoding="utf-8")
    n_included = int(selection["Included in Model"].str.startswith("Yes").sum())
    log(f"\nFeature selection report: {len(selection)} rows, {n_included} candidate predictors -> {SELECTION_PATH}")

    log("\n--- Final validation of model-ready frame ---")
    for r in validate_model_ready(df):
        log(r)

    model_ready = df[MODEL_READY_ORDER].copy()
    assert list(model_ready.columns) == MODEL_READY_ORDER
    log("\n--- Final validation of saved model-ready frame (exact order) ---")
    for r in validate_model_ready(model_ready, require_order=True):
        log(r)
    model_ready.to_csv(MODEL_READY_PATH, index=False, encoding="utf-8")
    log(f"\nModel-ready dataset    : {model_ready.shape[0]} rows x {model_ready.shape[1]} cols -> {MODEL_READY_PATH}")
    log("Boundary note          : no encoder/scaler/imputer fitted, no model trained (Phase 6 work)")

    log("\nPHASE 5 FEATURE ENGINEERING: SUCCESS")
    TEXT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"model_ready": MODEL_READY_PATH, "inventory": INVENTORY_PATH,
            "leakage": LEAKAGE_PATH, "selection": SELECTION_PATH,
            "text_report": TEXT_REPORT_PATH}


if __name__ == "__main__":
    finalize_and_save(load_cleaned())
