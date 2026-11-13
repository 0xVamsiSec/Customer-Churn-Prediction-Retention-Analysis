"""Final project validation (Phase 8).

Two responsibilities:
  1. build_model_metadata()  - freeze models/model_metadata.json from ACTUAL
     artifacts (saved pipeline object, test_metrics.csv, model_training_report.txt,
     customer_risk_scores.csv). Verifies - without retraining - that the saved
     pipeline still reproduces the validated test metrics and the Phase-7 scores.
  2. run_validation()        - end-to-end integrity check across data, model,
     predictions, Power BI outputs, reports and notebooks, writing
     outputs/reports/final_project_validation.txt (includes a project tree,
     .venv excluded).

No model fitting happens here; predict/predict_proba only.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

import src.preprocessing as prep
import src.risk_scoring as rs

METADATA_PATH = prep.MODELS_DIR / "model_metadata.json"
FINAL_REPORT_PATH = prep.REPORT_DIR / "final_project_validation.txt"

FROZEN_HASHES = {
    "raw": "16320C9C1EC72448DB59AA0A26A0B95401046BEF5D02FD3AEB906448E3055E91",
    "cleaned": "38E6F2DD278033A4617F880E8F11304C5FC6DCE2C05ECEF3F2A8BF48EEAA8996",
    "model_ready": "DF02696964C95DE935070C92B147D591B100EA87746870A9EAE7F6D8CEFA4F73",
}

REPORTS = [
    "outputs/reports/data_quality_report.txt",
    "outputs/reports/data_cleaning_report.txt",
    "outputs/reports/eda_report.txt",
    "outputs/reports/feature_engineering_report.txt",
    "outputs/reports/feature_inventory.csv",
    "outputs/reports/feature_leakage_review.csv",
    "outputs/reports/feature_selection_report.csv",
    "outputs/reports/model_training_report.txt",
    "outputs/reports/logistic_coefficients.csv",
    "outputs/reports/test_metrics.csv",
    "outputs/reports/customer_risk_report.txt",
    "outputs/reports/retention_strategy_report.txt",
]

NOTEBOOKS = [
    "notebooks/01_data_cleaning.ipynb",
    "notebooks/02_exploratory_data_analysis.ipynb",
    "notebooks/03_feature_engineering.ipynb",
    "notebooks/04_churn_prediction_model.ipynb",
    "notebooks/05_risk_scoring_powerbi.ipynb",
]

FIGURES = [f"outputs/figures/{n}" for n in [
    "01_churn_distribution.png", "02_numeric_features_by_churn.png",
    "03_categorical_churn_rates.png", "04_tenure_group_analysis.png",
    "05_contract_analysis.png", "06_monthly_charges_bands.png",
    "07_correlation_heatmap.png", "08_segment_churn_heatmaps.png",
    "09_confusion_matrix.png", "10_roc_curve.png", "11_precision_recall_curve.png",
]]

PBI_REQUIRED_FIELDS = [
    "customerID", "Churn", "churn_probability", "predicted_churn_at_0.40", "risk_segment",
    "tenure", "TenureGroup", "Contract", "PaymentMethod", "InternetService",
    "MonthlyCharges", "MonthlyChargeBand", "TotalCharges", "ServiceCount",
    "HasSecuritySupport", "AvgMonthlyCharge",
]

TREE_EXCLUDE = {".venv", ".git", "__pycache__", ".ipynb_checkpoints"}


# ---------------------------------------------------------------------------
# 1. Model freeze / metadata
# ---------------------------------------------------------------------------

def _parse_training_report() -> dict:
    text = (prep.REPORT_DIR / "model_training_report.txt").read_text(encoding="utf-8")

    def find(pattern: str) -> str:
        m = re.search(pattern, text)
        assert m, f"Pattern not found in model_training_report.txt: {pattern}"
        return m.group(1)

    train = re.search(r"Train: (\d+) rows, churn rate ([\d.]+)% \((\d+) churned\)", text)
    test = re.search(r"Test : (\d+) rows, churn rate ([\d.]+)% \((\d+) churned\)", text)
    assert train and test, "Split lines not found in model_training_report.txt"
    return {
        "train_rows": int(train.group(1)),
        "train_churn_rate_pct": float(train.group(2)),
        "train_churned": int(train.group(3)),
        "test_rows": int(test.group(1)),
        "test_churn_rate_pct": float(test.group(2)),
        "test_churned": int(test.group(3)),
        "final_feature_set": find(r"Final model: ([^,]+), class_weight=(\w+)"),
        "class_weight": find(r"Final model: [^,]+, class_weight=(\w+)"),
        "n_transformed_features": int(find(r"Transformed features after preprocessing: (\d+)")),
    }


def build_model_metadata(verbose: bool = True) -> dict:
    """Freeze models/model_metadata.json from actual artifacts (no retraining)."""
    model = rs.load_and_verify_model()
    clf = model.named_steps["classifier"]

    df = pd.read_csv(prep.MODEL_READY_PATH)
    X, y, _ = prep.make_xy(df)
    X_train, X_test, y_train, y_test = prep.make_split(X, y)

    # Reproduction check 1: held-out test metrics must match test_metrics.csv
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    recomputed = {
        "accuracy": round(float(accuracy_score(y_test, pred_test)), 4),
        "precision": round(float(precision_score(y_test, pred_test, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred_test, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred_test, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba_test)), 4),
    }
    tn = int(((pred_test == 0) & (y_test == 0)).sum())
    fp = int(((pred_test == 1) & (y_test == 0)).sum())
    fn = int(((pred_test == 0) & (y_test == 1)).sum())
    tp = int(((pred_test == 1) & (y_test == 1)).sum())

    saved = pd.read_csv(prep.REPORT_DIR / "test_metrics.csv")
    lr_row = saved[saved["model"].str.startswith("LogisticRegression")].iloc[0]
    for k in recomputed:
        assert np.isclose(recomputed[k], float(lr_row[k]), atol=1e-9), \
            f"Test metric {k} does not reproduce: {recomputed[k]} vs {lr_row[k]}"
    assert (tn, fp, fn, tp) == (int(lr_row["tn"]), int(lr_row["fp"]), int(lr_row["fn"]), int(lr_row["tp"]))

    # Reproduction check 2: all 7043 scores must match the Phase-7 artifact
    scores = pd.read_csv(rs.SCORES_PATH)
    proba_all = model.predict_proba(X)[:, 1]
    assert np.allclose(proba_all, scores["churn_probability"].values, atol=1e-12), \
        "Saved model does not reproduce customer_risk_scores.csv probabilities"

    report = _parse_training_report()
    pkl_mtime = MODEL_SRC_PATH().stat().st_mtime
    metadata = {
        "model_type": "sklearn Pipeline: ColumnTransformer (numeric: median impute + StandardScaler; "
                      "categorical: most-frequent impute + OneHotEncoder(handle_unknown='ignore', drop='first')) "
                      "-> LogisticRegression",
        "trained_on": "data/processed/model_ready_data.csv (7,043 x 26)",
        "classifier": {
            "type": type(clf).__name__,
            "C": float(clf.C),
            "solver": clf.solver,
            "class_weight": clf.class_weight,
            "max_iter": int(clf.max_iter),
        },
        "feature_set_choice": report["final_feature_set"],
        "n_candidate_features": len(prep.ALL_CANDIDATE_FEATURES),
        "n_transformed_features": report["n_transformed_features"],
        "split": {
            "test_size": prep.TEST_SIZE,
            "random_state": prep.RANDOM_STATE,
            "stratified_on_target": True,
            "train_rows": report["train_rows"],
            "train_churn_rate_pct": report["train_churn_rate_pct"],
            "test_rows": report["test_rows"],
            "test_churn_rate_pct": report["test_churn_rate_pct"],
        },
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": prep.RANDOM_STATE,
            "note": "class weight and feature set selected on CV (training data) only",
        },
        "test_metrics": recomputed,
        "confusion_matrix_threshold_050": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "operational_flag_threshold": rs.FLAG_THRESHOLD,
        "sklearn_version": sklearn.__version__,
        "data_integrity": {
            "raw_sha256": FROZEN_HASHES["raw"],
            "cleaned_sha256": FROZEN_HASHES["cleaned"],
            "model_ready_sha256": FROZEN_HASHES["model_ready"],
        },
        "reproducibility_verified": True,
        "model_file_created_utc": datetime.fromtimestamp(pkl_mtime, tz=timezone.utc).isoformat(),
        "metadata_frozen_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if verbose:
        print(f"Model metadata frozen: {METADATA_PATH}")
        print(json.dumps(metadata["test_metrics"], indent=2))
    return metadata


def MODEL_SRC_PATH() -> Path:
    return prep.MODELS_DIR / "churn_model.pkl"


# ---------------------------------------------------------------------------
# 2. End-to-end validation
# ---------------------------------------------------------------------------

class Checker:
    def __init__(self):
        self.results: list[tuple[bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            raise ValueError(f"PROJECT VALIDATION FAILED: {name} {detail}")
        self.results.append((True, f"PASS  {name}" + (f"  ({detail})" if detail else "")))

    def section(self, title: str) -> str:
        return f"--- {title} ---"


def _notebook_status(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    executed = [c for c in code if c.get("execution_count")]
    errors = [o for c in code for o in c.get("outputs", []) if o["output_type"] == "error"]
    if len(executed) == len(code) and not errors:
        return f"OK  {len(code)}/{len(code)} cells executed, 0 errors"
    return f"FAIL  {len(executed)}/{len(code)} executed, {len(errors)} errors"


def build_tree(root: Path) -> str:
    lines = [f"{root.name}/"]

    def walk(d: Path, prefix: str) -> None:
        entries = sorted(
            [p for p in d.iterdir() if p.name not in TREE_EXCLUDE],
            key=lambda p: (p.is_file(), p.name.lower()),
        )
        for i, p in enumerate(entries):
            connector = "`-- " if i == len(entries) - 1 else "|-- "
            if p.is_dir():
                lines.append(f"{prefix}{connector}{p.name}/")
                walk(p, prefix + ("    " if i == len(entries) - 1 else "|   "))
            else:
                lines.append(f"{prefix}{connector}{p.name}  ({p.stat().st_size:,} bytes)")

    walk(root, "")
    return "\n".join(lines)


def run_validation(verbose: bool = True) -> dict:
    c = Checker()
    lines: list[str] = []

    def log(text: str = "") -> None:
        lines.append(text)
        if verbose:
            print(text)

    log("=== FINAL PROJECT VALIDATION (PHASE 8) ===")
    log(f"Generated : {datetime.now(tz=timezone.utc).isoformat()}")
    log(f"Project   : {prep.PROJECT_ROOT}")
    log("")

    # --- Data ---
    log(c.section("DATA"))
    raw = pd.read_csv(prep.RAW_PATH)
    cleaned = pd.read_csv(prep.CLEAN_PATH)
    ready = pd.read_csv(prep.MODEL_READY_PATH)
    c.check("raw dataset exists", prep.RAW_PATH.exists(), f"{raw.shape[0]} x {raw.shape[1]}")
    c.check("cleaned dataset exists", prep.CLEAN_PATH.exists(), f"{cleaned.shape[0]} x {cleaned.shape[1]}")
    c.check("model-ready dataset exists", prep.MODEL_READY_PATH.exists(), f"{ready.shape[0]} x {ready.shape[1]}")
    c.check("row counts consistent (7043 across all)",
           raw.shape[0] == cleaned.shape[0] == ready.shape[0] == 7043)
    c.check("column counts (21 cleaned / 26 model-ready)",
           cleaned.shape[1] == 21 and ready.shape[1] == 26)
    c.check("raw dataset unchanged", prep.raw_sha256().upper() == FROZEN_HASHES["raw"])
    c.check("cleaned dataset unchanged", prep.clean_sha256().upper() == FROZEN_HASHES["cleaned"])
    c.check("model-ready dataset unchanged",
           prep.sha256_of(prep.MODEL_READY_PATH).upper() == FROZEN_HASHES["model_ready"])

    # --- Model ---
    log(c.section("MODEL"))
    model = rs.load_and_verify_model()
    c.check("model file exists", MODEL_SRC_PATH().exists())
    c.check("pipeline steps = preprocess + classifier", list(model.named_steps) == ["preprocess", "classifier"])
    c.check("classifier is LogisticRegression",
           model.named_steps["classifier"].__class__.__name__ == "LogisticRegression")
    X, y, ids = prep.make_xy(ready)
    sample_proba = model.predict_proba(X.iloc[:50])[:, 1]
    c.check("predict_proba works", len(sample_proba) == 50 and bool(np.isfinite(sample_proba).all())
            and bool(((sample_proba >= 0) & (sample_proba <= 1)).all()))
    sample_pred = model.predict(X.iloc[:50])
    c.check("predict works", len(sample_pred) == 50)
    c.check("metadata file exists", METADATA_PATH.exists())
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        c.check("metadata reproduces test metrics",
                meta.get("reproducibility_verified") is True
                and np.isclose(meta["test_metrics"]["roc_auc"], 0.8452)
                and meta["confusion_matrix_threshold_050"] == {"tn": 750, "fp": 285, "fn": 75, "tp": 299})

    # --- Predictions ---
    log(c.section("PREDICTIONS"))
    scores = pd.read_csv(rs.SCORES_PATH)
    proba_all = model.predict_proba(X)[:, 1]
    c.check("7043 customers scored", len(scores) == 7043, f"{len(scores)}")
    c.check("customerID unique", bool(scores["customerID"].is_unique))
    c.check("probabilities in [0,1]", bool(scores["churn_probability"].between(0, 1).all()),
            f"[{scores['churn_probability'].min():.6f}, {scores['churn_probability'].max():.6f}]")
    c.check("no missing values in scores", int(scores.isna().sum().sum()) == 0)
    c.check("risk segments valid", set(scores["risk_segment"]) == set(rs.SEGMENT_LABELS))
    c.check("saved model reproduces all scores", bool(np.allclose(proba_all, scores["churn_probability"], atol=1e-12)))
    c.check("0.40 flags reproduce", bool(np.array_equal((proba_all >= rs.FLAG_THRESHOLD).astype(int),
                                                        scores["predicted_churn_at_0.40"].values)))
    c.check("segment sums (7043 customers / 1869 churned)",
            int(scores["risk_segment"].value_counts().sum()) == 7043
            and int(scores.groupby("risk_segment")["Churn"].sum().sum()) == 1869)

    # --- Power BI ---
    log(c.section("POWER BI"))
    pbi = pd.read_csv(rs.PBI_DATASET_PATH)
    c.check("powerbi_dataset.csv exists", rs.PBI_DATASET_PATH.exists(), f"{pbi.shape[0]} x {pbi.shape[1]}")
    c.check("7043 rows", len(pbi) == 7043)
    c.check("one row per customer (unique customerID)", bool(pbi["customerID"].is_unique))
    c.check("required fields present", all(f in pbi.columns for f in PBI_REQUIRED_FIELDS),
            f"{len(PBI_REQUIRED_FIELDS)} fields")
    c.check("no missing risk scores", int(pbi["churn_probability"].isna().sum()) == 0)
    c.check("probability in [0,1]", bool(pbi["churn_probability"].between(0, 1).all()))
    c.check("all four risk segments present", set(pbi["risk_segment"]) == set(rs.SEGMENT_LABELS))
    hc_pbi = int(pbi["risk_segment"].isin(["High", "Critical"]).sum())
    hc_scores = int(scores["risk_segment"].isin(["High", "Critical"]).sum())
    c.check("High + Critical count matches scores", hc_pbi == hc_scores, f"{hc_pbi} == {hc_scores}")
    merged = pbi.merge(scores[rs.SCORE_COLS], on="customerID", suffixes=("", "_s"), validate="one_to_one")
    c.check("probability/segment/flag match risk-score dataset",
            bool(np.allclose(merged["churn_probability"], merged["churn_probability_s"], atol=1e-12)
                 and (merged["risk_segment"] == merged["risk_segment_s"]).all()
                 and (merged["predicted_churn_at_0.40"] == merged["predicted_churn_at_0.40_s"]).all()))
    for f in ["powerbi/data_dictionary.csv", "powerbi/dax_measures.md", "powerbi/dashboard_documentation.md"]:
        c.check(f"{f} exists", (prep.PROJECT_ROOT / f).exists())

    # --- Reports & notebooks ---
    log(c.section("REPORTS & NOTEBOOKS"))
    for r in REPORTS:
        c.check(f"report exists: {r}", (prep.PROJECT_ROOT / r).exists())
    for f in FIGURES:
        c.check(f"figure exists: {f}", (prep.PROJECT_ROOT / f).exists())
    c.check("run_pipeline.ps1 exists", (prep.PROJECT_ROOT / "run_pipeline.ps1").exists())
    c.check("README.md exists", (prep.PROJECT_ROOT / "README.md").exists())
    c.check("docs/PROJECT_DOCUMENTATION.md exists", (prep.PROJECT_ROOT / "docs/PROJECT_DOCUMENTATION.md").exists())
    c.check("docs/INTERVIEW_PREPARATION.md exists", (prep.PROJECT_ROOT / "docs/INTERVIEW_PREPARATION.md").exists())
    log("Notebook execution status:")
    for nb in NOTEBOOKS:
        status = _notebook_status(prep.PROJECT_ROOT / nb)
        c.check(f"notebook executed without errors: {nb}", status.startswith("OK"), status)
        log(f"  {status}  {nb}")

    # --- Project tree ---
    log(c.section("PROJECT TREE (.venv, .git, __pycache__ excluded)"))
    tree = build_tree(prep.PROJECT_ROOT)
    log(tree)

    log("")
    log("--- SUMMARY ---")
    log(f"Total checks passed : {len(c.results)}")
    log("Final status        : PASS")
    log("")
    log("PHASE 8 FINAL PROJECT VALIDATION: SUCCESS")

    FINAL_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if verbose:
        print(f"\nFinal validation report written: {FINAL_REPORT_PATH}")
    return {"checks": c.results, "report": FINAL_REPORT_PATH, "metadata": json.loads(METADATA_PATH.read_text(encoding="utf-8"))}


if __name__ == "__main__":
    build_model_metadata()
    run_validation()
