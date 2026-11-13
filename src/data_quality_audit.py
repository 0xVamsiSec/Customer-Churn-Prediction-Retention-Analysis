"""Read-only data quality audit for the raw Telco churn dataset.

This script never modifies the raw dataset. It inspects structure,
missing data, duplicates, types, categorical consistency, outliers,
and the churn target, then writes a text report and a CSV summary
under outputs/reports/.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "data_quality_report.txt"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "reports" / "data_quality_summary.csv"


def audit() -> None:
    lines: list[str] = []

    def log(text: str = "") -> None:
        lines.append(text)
        print(text)

    df = pd.read_csv(RAW_PATH)

    log("=== DATA QUALITY AUDIT (READ-ONLY) ===")
    log(f"Source: {RAW_PATH}")
    log(f"Rows: {df.shape[0]}  Columns: {df.shape[1]}")

    # --- 1. Structure ---
    log("\n--- 1. COLUMN NAMES AND DTYPES ---")
    log(df.dtypes.to_string())

    # --- 2. Missing values ---
    log("\n--- 2. MISSING VALUES ---")
    missing_count = df.isna().sum()
    missing_pct = df.isna().mean() * 100
    missing_table = pd.DataFrame(
        {"missing_count": missing_count, "missing_pct": missing_pct.round(4)}
    )
    log(missing_table.to_string())
    if missing_count.sum() == 0:
        log("No standard NaN missing values detected.")

    # --- 3. Duplicates ---
    log("\n--- 3. DUPLICATES ---")
    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df["customerID"].duplicated().sum())
    log(f"Duplicate full rows      : {dup_rows}")
    log(f"Duplicate customerID     : {dup_ids}")
    log(f"customerID unique        : {df['customerID'].is_unique}")
    log(f"customerID blank/empty   : {int(df['customerID'].astype(str).str.strip().eq('').sum())}")

    # --- 4. Unique values for categorical columns ---
    log("\n--- 4. UNIQUE VALUES (OBJECT COLUMNS) ---")
    object_cols = df.select_dtypes(include="object").columns.tolist()
    for col in object_cols:
        vc = df[col].value_counts(dropna=False)
        log(f"\n{col} ({df[col].nunique(dropna=False)} unique):")
        log(vc.to_string())

    # --- 5. Numeric-looking columns stored as text ---
    log("\n--- 5. NUMERIC-LOOKING COLUMNS STORED AS TEXT ---")
    text_numeric_findings = []
    for col in object_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        converted = int(coerced.notna().sum())
        if converted > 0:
            failed = int(coerced.isna().sum())
            text_numeric_findings.append(
                {
                    "column": col,
                    "rows_convertible": converted,
                    "rows_not_convertible": failed,
                    "convertible_pct": round(converted / len(df) * 100, 2),
                }
            )
            log(
                f"{col}: {converted}/{len(df)} values convert to numeric "
                f"({failed} failures) -> stored as text"
            )
    if not text_numeric_findings:
        log("No numeric-looking text columns detected.")

    # --- 6. TotalCharges deep check ---
    log("\n--- 6. TOTALCHARGES CONVERSION CHECK (NOT CLEANED) ---")
    tc_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
    log(f"TotalCharges dtype        : {df['TotalCharges'].dtype}")
    log(f"Non-convertible values    : {int(tc_numeric.isna().sum())}")
    problem_mask = tc_numeric.isna()
    if problem_mask.any():
        problem_rows = df.loc[
            problem_mask, ["customerID", "tenure", "TotalCharges", "Churn"]
        ]
        log("Problematic rows (raw values, not modified):")
        log(problem_rows.to_string())
        log(
            "All problematic rows have tenure == 0: "
            f"{bool(df.loc[problem_mask, 'tenure'].eq(0).all())}"
        )
        log(
            "Raw string values in problem rows (repr): "
            f"{[repr(v) for v in df.loc[problem_mask, 'TotalCharges']]}"
        )

    # --- 7. Numerical summary statistics ---
    log("\n--- 7. NUMERICAL SUMMARY STATISTICS ---")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    log(df[numeric_cols].describe().round(4).to_string())

    # --- 8. Outlier analysis (IQR method) ---
    log("\n--- 8. OUTLIER ANALYSIS (IQR METHOD, NO MODIFICATION) ---")
    outlier_rows = []
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        outlier_rows.append(
            {
                "column": col,
                "q1": round(float(q1), 4),
                "q3": round(float(q3), 4),
                "iqr": round(float(iqr), 4),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
                "outlier_count": count,
                "outlier_pct": round(count / len(df) * 100, 2),
            }
        )
        log(
            f"{col}: Q1={q1:.4f} Q3={q3:.4f} IQR={iqr:.4f} "
            f"bounds=[{lower:.4f}, {upper:.4f}] "
            f"outliers={count} ({count / len(df) * 100:.2f}%)"
        )
    log(
        "Note: binary columns (e.g. SeniorCitizen) have IQR=0, so the IQR rule\n"
        "flags every value except the single dominant one. This is an artifact\n"
        "of the method, not evidence of data errors."
    )

    # --- 9. Whitespace audit (leading/trailing) ---
    log("\n--- 9. LEADING/TRAILING WHITESPACE (OBJECT COLUMNS) ---")
    ws_rows = []
    for col in object_cols:
        raw = df[col].astype(str)
        count = int(raw.ne(raw.str.strip()).sum())
        ws_rows.append({"column": col, "values_with_lead_trail_ws": count})
        log(f"{col}: {count}")
    total_ws = sum(r["values_with_lead_trail_ws"] for r in ws_rows)
    log(f"Total values with leading/trailing whitespace: {total_ws}")

    # --- 10. Category normalization collisions ---
    log("\n--- 10. CATEGORY NORMALIZATION COLLISIONS (strip+lower) ---")
    collisions_found = False
    for col in object_cols:
        uniques = df[col].dropna().unique()
        groups: dict[str, list[str]] = {}
        for u in uniques:
            key = str(u).strip().lower()
            groups.setdefault(key, []).append(str(u))
        colliding = {k: v for k, v in groups.items() if len(v) > 1}
        if colliding:
            collisions_found = True
            log(f"{col}: {colliding}")
    if not collisions_found:
        log(
            "No collisions: no two distinct raw values in any object column "
            "normalize to the same strip().lower() value."
        )

    # --- 11. Churn target audit ---
    log("\n--- 11. CHURN TARGET AUDIT ---")
    churn_counts = df["Churn"].value_counts(dropna=False)
    churned = int(churn_counts.get("Yes", 0))
    retained = int(churn_counts.get("No", 0))
    total = len(df)
    log(f"Unique Churn values : {sorted(df['Churn'].dropna().unique().tolist())}")
    log(f"Total customers     : {total}")
    log(f"Churned (Yes)       : {churned}")
    log(f"Retained (No)       : {retained}")
    log(f"Churn rate          : {churned / total * 100:.4f}% ({churned}/{total})")
    log(f"Class imbalance     : retained/churned = {retained / churned:.4f}")

    # --- 12. Inline validation that raw data untouched ---
    log("\n--- 12. INTEGRITY ---")
    log("This audit performed read-only operations (pd.read_csv only).")
    log("No values were changed, imputed, dropped, or written back.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Structured summary CSV (one row per check with a scalar result)
    summary = pd.DataFrame(
        [
            {"check": "rows", "result": df.shape[0]},
            {"check": "columns", "result": df.shape[1]},
            {"check": "missing_values_total", "result": int(missing_count.sum())},
            {"check": "duplicate_rows", "result": dup_rows},
            {"check": "duplicate_customer_ids", "result": dup_ids},
            {"check": "text_numeric_columns", "result": str([f["column"] for f in text_numeric_findings])},
            {"check": "totalcharges_non_convertible", "result": int(tc_numeric.isna().sum())},
            {"check": "totalcharges_problems_all_tenure_zero", "result": bool(df.loc[problem_mask, "tenure"].eq(0).all()) if problem_mask.any() else "n/a"},
            {"check": "values_with_lead_trail_whitespace", "result": total_ws},
            {"check": "category_normalization_collisions", "result": collisions_found},
            {"check": "churned_customers", "result": churned},
            {"check": "retained_customers", "result": retained},
            {"check": "churn_rate_pct", "result": round(churned / total * 100, 4)},
        ]
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\nReport written to: {REPORT_PATH}")
    print(f"Summary written to: {SUMMARY_PATH}")


if __name__ == "__main__":
    audit()
