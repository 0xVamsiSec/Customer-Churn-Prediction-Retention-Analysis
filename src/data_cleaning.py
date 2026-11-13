"""Reproducible, validation-driven cleaning for the raw Telco churn dataset.

Reads data/raw/customer_churn.csv (never modified) and produces
data/processed/cleaned_customer_churn.csv plus a cleaning report under
outputs/reports/.

Cleaning policy (based on the real Phase-2 audit, not assumptions):
  * TotalCharges: non-convertible values are asserted to be exactly one
    whitespace character, with tenure == 0 and Churn == "No", before being
    set to 0. Any other condition aborts the pipeline.
  * Text columns: leading/trailing whitespace stripped defensively; no
    category values are altered unless a real inconsistency is found.
  * Duplicates: verified, never removed.
  * Outliers: reported, never removed.
Every cleaning decision is asserted before it is applied; validation
failures raise ValueError so the pipeline fails loudly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_customer_churn.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "data_cleaning_report.txt"

ID_COL = "customerID"
TARGET = "Churn"

# Text columns stripped defensively. TotalCharges is handled separately
# (numeric conversion) and is NOT in this list.
TEXT_COLUMNS = [
    "customerID", "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "Churn",
]

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]

# Generous business-plausible bounds; catches garbage without rejecting real data.
RANGE_CHECKS = {
    "tenure": (0, 200),
    "MonthlyCharges": (0, 1000),
    "TotalCharges": (0, 100000),
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def convert_total_charges(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int, list[str], pd.DataFrame]:
    """Safely convert TotalCharges to numeric.

    Non-convertible raw values must be exactly one whitespace character,
    and every affected row must have tenure == 0 and Churn == "No"
    (verified new customers with no billing history). Only then are those
    values set to 0. Anything else raises AssertionError so the pipeline
    stops instead of guessing.
    """
    cleaned = df.copy()
    raw_values = cleaned["TotalCharges"]
    coerced = pd.to_numeric(raw_values, errors="coerce")
    invalid_mask = coerced.isna()

    if invalid_mask.any():
        invalid_view = cleaned.loc[invalid_mask]
        non_ws = [
            repr(v)
            for v in raw_values.loc[invalid_mask]
            if not (isinstance(v, str) and len(v) == 1 and v.isspace())
        ]
        assert not non_ws, (
            "Non-numeric TotalCharges values that are NOT a single whitespace "
            f"character: {non_ws}"
        )
        assert invalid_view["tenure"].eq(0).all(), (
            "Unexplained non-numeric TotalCharges with tenure != 0: "
            f"{invalid_view.loc[invalid_view['tenure'].ne(0), ['customerID', 'tenure', 'TotalCharges']].to_dict('records')}"
        )
        assert invalid_view["Churn"].eq("No").all(), (
            "Unexplained non-numeric TotalCharges for churned customers: "
            f"{invalid_view.loc[invalid_view['Churn'].ne('No'), ['customerID', 'Churn', 'TotalCharges']].to_dict('records')}"
        )

    invalid_rows = cleaned.loc[
        invalid_mask, ["customerID", "tenure", "TotalCharges", "Churn"]
    ].copy()
    affected_ids = invalid_rows["customerID"].tolist()

    cleaned["TotalCharges"] = coerced
    n_converted = int(invalid_mask.sum())
    if n_converted:
        cleaned.loc[invalid_mask, "TotalCharges"] = 0.0

    return cleaned, n_converted, affected_ids, invalid_rows


def strip_text_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Strip leading/trailing whitespace from text columns.

    Returns the cleaned copy and a per-column count of values actually changed.
    Categories are NOT lowercased or rewritten; if a column needs no change it
    is left untouched.
    """
    cleaned = df.copy()
    changes: dict[str, int] = {}
    for col in TEXT_COLUMNS:
        original = cleaned[col]
        stripped = original.str.strip()
        changed = int(original.ne(stripped).sum())
        changes[col] = changed
        if changed:
            cleaned[col] = stripped
    return cleaned, changes


def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Encode Churn as No -> 0, Yes -> 1. Fails loudly on unexpected values."""
    cleaned = df.copy()
    counts_before = {str(k): int(v) for k, v in cleaned[TARGET].value_counts().items()}
    encoded = cleaned[TARGET].map({"No": 0, "Yes": 1})
    if encoded.isna().any():
        unexpected = sorted(cleaned.loc[encoded.isna(), TARGET].unique().tolist())
        raise ValueError(f"Unexpected Churn values found: {unexpected}")
    cleaned[TARGET] = encoded.astype("int64")
    return cleaned, counts_before


def count_iqr_outliers(series: pd.Series) -> dict:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())
    return {
        "column": series.name,
        "lower_bound": round(float(lower), 4),
        "upper_bound": round(float(upper), 4),
        "outlier_count": count,
    }


def validate_dataset(df: pd.DataFrame, stage: str) -> list[str]:
    """Run all critical validation checks; raise ValueError on any failure."""
    errors: list[str] = []
    notes: list[str] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if ID_COL in df.columns and not df[ID_COL].is_unique:
        n_dup_ids = int(df["customerID"].duplicated().sum())
        errors.append(f"customerID not unique: {n_dup_ids} duplicate IDs")

    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        errors.append(f"{dup_rows} duplicate full rows")

    if TARGET in df.columns:
        bad_target = sorted(set(df[TARGET].unique().tolist()) - {0, 1})
        if bad_target:
            errors.append(f"Unexpected target values: {bad_target}")

    for col in NUMERIC_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"{col} is not numeric (dtype={df[col].dtype})")

    missing = df.isna().sum()
    cols_with_missing = missing[missing > 0]
    if len(cols_with_missing):
        errors.append(f"Unexpected missing values: {cols_with_missing.to_dict()}")

    for col, (low, high) in RANGE_CHECKS.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            out_of_range = int(((df[col] < low) | (df[col] > high)).sum())
            if out_of_range:
                errors.append(f"{col}: {out_of_range} values outside [{low}, {high}]")
            else:
                notes.append(
                    f"{col} range OK: [{df[col].min()}, {df[col].max()}] within [{low}, {high}]"
                )

    if errors:
        raise ValueError(
            f"VALIDATION FAILED at stage '{stage}':\n- " + "\n- ".join(errors)
        )
    return [f"[{stage}] All critical checks passed."] + notes


def clean() -> tuple[pd.DataFrame, list[str]]:
    """Run the full cleaning pipeline and write outputs. Idempotent."""
    lines: list[str] = []

    def log(text: str = "") -> None:
        lines.append(text)
        print(text)

    raw_hash_before = sha256_of(RAW_PATH)
    df = load_raw()
    original_shape = df.shape
    log("=== DATA CLEANING REPORT ===")
    log(f"Raw file           : {RAW_PATH}")
    log(f"Raw SHA-256 before : {raw_hash_before}")
    log(f"Original shape     : {original_shape[0]} rows x {original_shape[1]} columns")

    # -- Stage 1: structural validation of the raw frame --
    pre_errors = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not pre_errors, f"Raw data missing required columns: {pre_errors}"

    # -- Stage 2: TotalCharges conversion (asserted on RAW values, not silent) --
    df, n_converted, affected_ids, invalid_rows = convert_total_charges(df)
    log("\n--- TOTALCHARGES CONVERSION ---")
    log(f"Non-convertible raw values found   : {n_converted}")
    log("Asserted: every non-numeric raw value is exactly one whitespace char (' ')")
    log("Asserted: every affected row has tenure == 0")
    log("Asserted: every affected row has Churn == 'No'")
    log("Raw values of affected cells (repr):")
    log(f"  {[repr(v) for v in invalid_rows['TotalCharges']]}")
    log("Affected customerIDs:")
    for cid in affected_ids:
        log(f"  - {cid}")
    log(f"Action: {n_converted} verified new-customer rows set to 0.0 (0 months billed).")

    # -- Stage 3: whitespace stripping (defensive; audit found 0 categorical issues) --
    df, ws_changes = strip_text_columns(df)
    ws_total = sum(ws_changes.values())
    cat_changed = {k: v for k, v in ws_changes.items() if v}
    log("\n--- TEXT CLEANING (DEFENSIVE STRIP) ---")
    log(f"Text values changed                : {ws_total}")
    log(
        "Category values changed            : "
        f"{cat_changed or 'none - no categorical value required correction'}"
    )
    log("No lowercase/normalization applied to categories (audit found none needed).")

    # -- Stage 4: target encoding --
    df, counts_before = encode_target(df)
    log("\n--- TARGET ENCODING ---")
    log("Mapping applied: No -> 0, Yes -> 1")
    log(f"Counts before encoding             : {counts_before}")
    log(f"Counts after encoding              : { {int(k): int(v) for k, v in df[TARGET].value_counts().items()} }")

    # -- Stage 5: duplicates (verified, not removed) --
    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df["customerID"].duplicated().sum())
    log("\n--- DUPLICATES ---")
    log(f"Duplicate full rows                : {dup_rows} (nothing removed)")
    log(f"Duplicate customerIDs              : {dup_ids} (nothing removed)")

    # -- Stage 6: outliers (reported, not removed) --
    log("\n--- OUTLIERS (IQR, REPORT ONLY - NOTHING REMOVED) ---")
    for col in ["tenure", "MonthlyCharges"]:
        stats = count_iqr_outliers(df[col])
        log(
            f"{col}: {stats['outlier_count']} IQR outliers "
            f"(bounds [{stats['lower_bound']}, {stats['upper_bound']}]) - retained"
        )

    # -- Stage 7: validation (fails loudly) --
    log("\n--- VALIDATION ---")
    for msg in validate_dataset(df, "final"):
        log(msg)

    final_shape = df.shape
    missing_per_col = df.isna().sum()
    log("\n--- FINAL STATE ---")
    log(f"Final shape                        : {final_shape[0]} rows x {final_shape[1]} columns")
    log(f"Rows removed                       : {original_shape[0] - final_shape[0]}")
    log(f"Columns removed                    : {original_shape[1] - final_shape[1]}")
    log(f"Final missing values (total)       : {int(missing_per_col.sum())}")
    log(f"Final missing values (per column)  : {missing_per_col[missing_per_col > 0].to_dict() or 'none'}")
    log("Final dtypes:")
    for col, dtype in df.dtypes.items():
        log(f"  {col:<20} {dtype}")

    # -- Stage 8: save cleaned dataset --
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    log(f"\nSaved cleaned dataset              : {CLEAN_PATH}")

    # -- Stage 9: raw integrity --
    raw_hash_after = sha256_of(RAW_PATH)
    unchanged = raw_hash_before == raw_hash_after
    log(f"Raw SHA-256 after                  : {raw_hash_after}")
    log(f"Raw file byte-for-byte unchanged   : {unchanged}")
    if not unchanged:
        raise RuntimeError("Raw dataset was modified during cleaning - ABORT.")

    log("\nPHASE 3 CLEANING: SUCCESS")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to: {REPORT_PATH}")

    return df, lines


if __name__ == "__main__":
    clean()
