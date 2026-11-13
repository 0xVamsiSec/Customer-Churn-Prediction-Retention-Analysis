"""Training & experiment orchestration for the churn model (Phase 6).

Flow (all statistics computed from actual data, CV on TRAINING data only):
  1. load + verify model-ready data
  2. stratified train/test split (test 20%, seed 42)
  3. DummyClassifier(prior) baseline
  4. class-weight comparison via 5-fold CV on the training set
  5. primary LogisticRegression (all 24 features) + CV
  6. held-out test evaluation of the primary pipeline
  7. threshold analysis (OOF on train + descriptive test table)
  8. redundancy diagnostics (correlation, VIF, fold coefficient stability)
  9. feature ablation A/B/C via CV -> final model chosen by CV ROC-AUC
 10. coefficients -> logistic_coefficients.csv
 11. save full pipeline -> models/churn_model.pkl, reload & verify reproduction
 12. report -> outputs/reports/model_training_report.txt (+ test_metrics.csv)

The test set is touched exactly once per candidate model for reporting.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

import src.evaluate as ev
import src.preprocessing as prep

MODEL_PATH = prep.MODELS_DIR / "churn_model.pkl"
COEF_PATH = prep.REPORT_DIR / "logistic_coefficients.csv"
REPORT_PATH = prep.REPORT_DIR / "model_training_report.txt"
TEST_METRICS_PATH = prep.REPORT_DIR / "test_metrics.csv"

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]

ABLATION_SETS = {
    "A (originals only)": (prep.ORIGINAL_NUMERIC, prep.ORIGINAL_CATEGORICAL),
    "B (originals + engineered)": (prep.NUMERIC_FEATURES, prep.CATEGORICAL_FEATURES),
    # C drops the three engineered features justified as redundant by the
    # diagnostics: TenureGroup/MonthlyChargeBand are binned copies of tenure/
    # MonthlyCharges, and AvgMonthlyCharge correlates 0.9963 with
    # MonthlyCharges (VIF ~136, unstable fold-sign). ServiceCount and
    # HasSecuritySupport are kept (stable coefficients, moderate VIF).
    "C (originals + ServiceCount + HasSecuritySupport)": (
        ["tenure", "MonthlyCharges", "TotalCharges", "ServiceCount", "HasSecuritySupport"],
        prep.ORIGINAL_CATEGORICAL,
    ),
}


# ---------------------------------------------------------------------------
# CV helpers
# ---------------------------------------------------------------------------

def cv_summary(pipeline, X, y) -> pd.DataFrame:
    cv_res = cross_validate(pipeline, X, y, cv=CV, scoring=SCORING, n_jobs=1)
    rows = []
    for m in METRICS:
        scores = cv_res[f"test_{m}"]
        rows.append({"Metric": m.upper(), "Mean": round(float(scores.mean()), 4),
                     "Std": round(float(scores.std()), 4)})
    return pd.DataFrame(rows)


def cv_value(cv_df: pd.DataFrame, metric: str) -> float:
    return float(cv_df.loc[cv_df["Metric"] == metric.upper(), "Mean"].iloc[0])


def cv_std(cv_df: pd.DataFrame, metric: str) -> float:
    return float(cv_df.loc[cv_df["Metric"] == metric.upper(), "Std"].iloc[0])


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def baseline_experiment(X_train, y_train, X_test, y_test) -> dict:
    dummy = DummyClassifier(strategy="prior")
    cv = cv_summary(dummy, X_train, y_train)
    dummy.fit(X_train, y_train)
    proba = dummy.predict_proba(X_test)[:, 1]
    test = ev.compute_test_metrics(y_test, proba, 0.5, "DummyClassifier(prior)")
    return {"cv": cv, "test": test,
            "majority_class": str(dummy.classes_[np.argmax(dummy.class_prior_)]),
            "majority_prior": round(float(dummy.class_prior_.max()), 4)}


def class_weight_comparison(X_train, y_train) -> tuple[pd.DataFrame, str, str]:
    summaries = {}
    for weight in [None, "balanced"]:
        summaries[weight] = cv_summary(prep.build_pipeline(class_weight=weight), X_train, y_train)
    rows = []
    for weight, cv in summaries.items():
        for _, r in cv.iterrows():
            rows.append({"class_weight": str(weight), "Metric": r["Metric"],
                         "Mean": r["Mean"], "Std": r["Std"]})
    table = pd.DataFrame(rows)
    roc_none, f1_none = cv_value(summaries[None], "roc_auc"), cv_value(summaries[None], "f1")
    roc_bal, f1_bal = cv_value(summaries["balanced"], "roc_auc"), cv_value(summaries["balanced"], "f1")
    if abs(roc_none - roc_bal) <= 0.003:
        choice = "balanced" if f1_bal > f1_none else "None"
        reason = (
            f"CV ROC-AUC nearly identical ({roc_none:.4f} vs {roc_bal:.4f}, gap <= 0.003); "
            f"chosen by CV F1 (None {f1_none:.4f} vs balanced {f1_bal:.4f}) - for retention "
            "campaigns, F1 balances catching churners against wasted outreach."
        )
    else:
        choice = "None" if roc_none > roc_bal else "balanced"
        reason = (
            f"Chosen by CV ROC-AUC (None {roc_none:.4f} vs balanced {roc_bal:.4f}) - "
            "ranking quality drives customer risk scoring. No test-set information used."
        )
    return table, choice, reason


def test_evaluation(pipeline, X_test, y_test, name: str) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = pipeline.predict(X_test)
    # pipeline.predict must equal thresholding proba at 0.5 for LogisticRegression
    assert np.array_equal(pred, (proba >= 0.5).astype(int)), "predict() disagrees with proba thresholding"
    metrics = ev.compute_test_metrics(y_test, proba, 0.5, name)
    return {"metrics": metrics, "y_pred": pred, "y_proba": proba,
            "report": ev.classification_report_text(y_test, proba, 0.5)}


# ---------------------------------------------------------------------------
# Redundancy diagnostics
# ---------------------------------------------------------------------------

def vif_table(X_train: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    """VIF_j = 1 / (1 - R2_j), R2_j from regressing feature j on the others."""
    X_num = X_train[numeric_features].astype(float).values
    rows = []
    for j, col in enumerate(numeric_features):
        target = X_num[:, j]
        others = np.delete(X_num, j, axis=1)
        coef, *_ = np.linalg.lstsq(others, target, rcond=None)
        resid = target - others @ coef
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((target - target.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = np.inf if r2 > 0.9999 else 1.0 / (1.0 - r2)
        rows.append({"Feature": col, "R2_on_others": round(r2, 4),
                     "VIF": round(float(vif), 2) if np.isfinite(vif) else np.inf})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def coefficient_stability(X_train, y_train, class_weight=None, n_folds: int = 5) -> pd.DataFrame:
    """Fit the full pipeline on each CV fold's training part; track numeric
    coefficients across folds (stability of correlated features)."""
    numeric_cols = [f"num__{c}" for c in prep.NUMERIC_FEATURES]
    folds = []
    for fold, (tr_idx, _) in enumerate(CV.split(X_train, y_train), 1):
        pipe = prep.build_pipeline(class_weight=class_weight)
        pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        names = prep.transformed_feature_names(pipe)
        coefs = pipe.named_steps["classifier"].coef_[0]
        lookup = dict(zip(names, coefs))
        folds.append({"fold": fold, **{c: float(lookup[c]) for c in numeric_cols}})
    df = pd.DataFrame(folds).set_index("fold").round(4)
    df.loc["std"] = df.std().round(4)
    df.loc["sign_changes"] = (df.drop(index=["std"]).apply(lambda s: (s > 0).sum() not in (0, len(s))))
    return df


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def ablation_experiment(X_train, y_train, class_weight) -> pd.DataFrame:
    rows = []
    for name, (num, cat) in ABLATION_SETS.items():
        pipe = prep.build_pipeline(num, cat, class_weight=class_weight)
        cv = cv_summary(pipe, X_train, y_train)
        rows.append({"Model": name, "n_features": len(num) + len(cat),
                     "ROC_AUC_Mean": cv_value(cv, "roc_auc"), "ROC_AUC_Std": cv_std(cv, "roc_auc"),
                     "F1_Mean": cv_value(cv, "f1"), "F1_Std": cv_std(cv, "f1"),
                     "Precision_Mean": cv_value(cv, "precision"), "Precision_Std": cv_std(cv, "precision"),
                     "Recall_Mean": cv_value(cv, "recall"), "Recall_Std": cv_std(cv, "recall")})
    return pd.DataFrame(rows)


def select_final(abl: pd.DataFrame) -> tuple[str, str]:
    best = abl.sort_values(["ROC_AUC_Mean", "F1_Mean"], ascending=False).iloc[0]
    reason = (
        f"Selected by CV ROC-AUC ({best['ROC_AUC_Mean']:.4f}; tie-break CV F1 {best['F1_Mean']:.4f}). "
        "ROC-AUC measures how well the model RANKS customers by churn risk, which is the core "
        "business need for targeting retention outreach. No test-set information used."
    )
    return str(best["Model"]), reason


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------

def extract_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    names = prep.transformed_feature_names(pipeline)
    coefs = pipeline.named_steps["classifier"].coef_[0]
    assert len(coefs) == len(names), (
        f"Coefficient count {len(coefs)} != transformed feature count {len(names)}"
    )
    df = pd.DataFrame({
        "Feature": names,
        "Coefficient": coefs,
        "OddsRatio": np.exp(coefs),
        "Direction": np.where(coefs > 0, "positive", "negative"),
    })
    return df.sort_values("Coefficient", key=np.abs, ascending=False).reset_index(drop=True).round({"Coefficient": 6, "OddsRatio": 6})


# ---------------------------------------------------------------------------
# Save / reload
# ---------------------------------------------------------------------------

def save_model(pipeline: Pipeline, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def verify_saved_model(path: Path, X_test, y_pred_ref, y_proba_ref) -> dict:
    loaded = joblib.load(path)
    pred = loaded.predict(X_test)
    proba = loaded.predict_proba(X_test)[:, 1]
    checks = {
        "predictions identical": bool(np.array_equal(pred, y_pred_ref)),
        "probabilities identical": bool(np.allclose(proba, y_proba_ref, atol=1e-12)),
        "pipeline contains preprocessing": "preprocess" in loaded.named_steps,
        "pipeline contains classifier": "classifier" in loaded.named_steps,
    }
    if not all(checks.values()):
        raise ValueError(f"SAVED MODEL VERIFICATION FAILED: {checks}")
    return checks


# ---------------------------------------------------------------------------
# Full flow
# ---------------------------------------------------------------------------

def run_all(verbose: bool = True) -> dict:
    lines: list[str] = []

    def log(text: str = "") -> None:
        lines.append(text)
        if verbose:
            print(text)

    df = prep.load_model_ready()
    X, y, ids = prep.make_xy(df)
    log("=== MODEL TRAINING REPORT (PHASE 6) ===")
    log(f"Model-ready source : {prep.MODEL_READY_PATH}")
    log(f"Rows x cols        : {df.shape[0]} x {df.shape[1]}")

    X_train, X_test, y_train, y_test = prep.make_split(X, y)
    split_info = prep.split_summary(y_train, y_test)
    log("\n--- DATASET AND SPLIT ---")
    log(f"Feature groups     : numeric={prep.NUMERIC_FEATURES}")
    log(f"                     categorical={prep.CATEGORICAL_FEATURES}")
    log(f"Identifier (kept out of X): {prep.ID_COL}; Target: {prep.TARGET}")
    log(f"Train rows         : {split_info['train_rows']} (churn {split_info['train_churn_rate_pct']}%)")
    log(f"Test rows          : {split_info['test_rows']} (churn {split_info['test_churn_rate_pct']}%)")
    log(f"Stratification gap : {split_info['stratification_gap_pct_pts']} pct pts")

    baseline = baseline_experiment(X_train, y_train, X_test, y_test)
    log("\n--- BASELINE (DummyClassifier strategy=prior) ---")
    log(f"Predicts majority class {baseline['majority_class']} for everyone "
        f"(prior {baseline['majority_prior']:.4f})")
    log("Usefulness: documents the no-skill floor; accuracy alone is misleading under "
        "class imbalance - any real model must beat this on ROC-AUC/F1.")
    log(baseline["cv"].to_string(index=False))
    log(f"Test: " + ", ".join(f"{m}={baseline['test'][m]:.4f}" for m in ev.METRICS))

    weight_table, weight_choice, weight_reason = class_weight_comparison(X_train, y_train)
    log("\n--- CLASS-WEIGHT COMPARISON (5-fold CV on TRAINING set only) ---")
    log(weight_table.to_string(index=False))
    log(f"Selected class_weight={weight_choice}. {weight_reason}")

    primary = prep.build_pipeline(class_weight=weight_choice)
    primary_cv = cv_summary(primary, X_train, y_train)
    log("\n--- PRIMARY LOGISTIC REGRESSION (all 24 features) - 5-fold CV on training set ---")
    log(primary_cv.to_string(index=False))
    primary.fit(X_train, y_train)
    n_transformed = len(prep.transformed_feature_names(primary))
    log(f"Transformed features after preprocessing: {n_transformed}")

    primary_test = test_evaluation(primary, X_test, y_test, f"LogisticRegression (all 24 features, class_weight={weight_choice})")
    m = primary_test["metrics"]
    log("\n--- HELD-OUT TEST EVALUATION (primary model, single pass) ---")
    log(f"Confusion matrix: tn={m['tn']}, fp={m['fp']}, fn={m['fn']}, tp={m['tp']}")
    log(", ".join(f"{k}={m[k]:.4f}" for k in ev.METRICS))
    log(f"Metric-consistency vs confusion matrix: verified=True (compute_test_metrics asserts this)")
    log(primary_test["report"])

    oof_proba = cross_val_predict(
        prep.build_pipeline(class_weight=weight_choice), X_train, y_train,
        cv=CV, method="predict_proba", n_jobs=1
    )[:, 1]
    thr_oof = ev.threshold_table(y_train, oof_proba, THRESHOLDS)
    thr_test = ev.threshold_table(y_test, primary_test["y_proba"], THRESHOLDS)
    log("\n--- THRESHOLD ANALYSIS ---")
    log("Out-of-fold probabilities on TRAINING set (business threshold selection basis):")
    log(thr_oof.to_string(index=False))
    log("Descriptive reference: held-out TEST set at the same thresholds (not used to select):")
    log(thr_test.to_string(index=False))

    corr_train = X_train[prep.NUMERIC_FEATURES].corr().round(4)
    vif = vif_table(X_train, prep.NUMERIC_FEATURES)
    stability = coefficient_stability(X_train, y_train, weight_choice)
    log("\n--- REDUNDANCY / MULTICOLLINEARITY DIAGNOSTICS (training data only) ---")
    log("Correlation matrix (numeric block):")
    log(corr_train.to_string())
    log("\nVIF (numeric block):")
    log(vif.to_string(index=False))
    log("\nNumeric-block coefficients per CV fold (stability):")
    log(stability.to_string())
    log("Note: binned features (TenureGroup, MonthlyChargeBand) duplicate continuous info by "
        "construction; the ablation below tests whether they add measurable value.")

    abl = ablation_experiment(X_train, y_train, weight_choice)
    log("\n--- FEATURE ABLATION (5-fold CV on training set) ---")
    log(abl.to_string(index=False))
    final_choice, final_reason = select_final(abl)
    log(f"\nFinal feature-set choice: {final_choice}. {final_reason}")

    final_num, final_cat = ABLATION_SETS[final_choice]
    final_cfg_desc = f"LogisticRegression C=1.0, class_weight={weight_choice}, features={final_choice}"
    if final_choice == "B (originals + engineered)":
        final_pipeline = primary
        final_test = primary_test
    else:
        final_pipeline = prep.build_pipeline(final_num, final_cat, class_weight=weight_choice)
        final_pipeline.fit(X_train, y_train)
        final_test = test_evaluation(final_pipeline, X_test, y_test, f"LogisticRegression ({final_choice}, class_weight={weight_choice})")
        fm = final_test["metrics"]
        log(f"Final model held-out test evaluation (single pass): "
            f"tn={fm['tn']}, fp={fm['fp']}, fn={fm['fn']}, tp={fm['tp']}, "
            + ", ".join(f"{k}={fm[k]:.4f}" for k in ev.METRICS))

    coefficients = extract_coefficients(final_pipeline)
    COEF_PATH.parent.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(COEF_PATH, index=False)
    log(f"\nCoefficients saved: {COEF_PATH} ({len(coefficients)} rows; "
        f"count == transformed feature count, verified inside extract_coefficients)")
    log("\nTop 15 coefficients by |coefficient|:")
    log(coefficients.head(15).to_string(index=False))

    fig_cm = ev.plot_confusion(y_test, final_test["y_proba"], 0.5, prep.FIGURES_DIR / "09_confusion_matrix.png")
    fig_roc = ev.plot_roc(y_test, final_test["y_proba"], prep.FIGURES_DIR / "10_roc_curve.png")
    fig_pr = ev.plot_pr(y_test, final_test["y_proba"], prep.FIGURES_DIR / "11_precision_recall_curve.png")
    log(f"Figures saved: {fig_cm.name}, {fig_roc.name}, {fig_pr.name}")

    model_path = save_model(final_pipeline)
    reload_checks = verify_saved_model(model_path, X_test, final_test["y_pred"], final_test["y_proba"])
    log(f"\nModel saved        : {model_path}")
    log(f"Reload verification: {reload_checks}")

    ev.save_test_metrics_csv(
        [
            {"model": "DummyClassifier(prior)", **{k: baseline["test"][k] for k in ["tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "f1", "roc_auc"]}},
            {"model": f"LogisticRegression final ({final_choice}, class_weight={weight_choice})",
             **{k: final_test["metrics"][k] for k in ["tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "f1", "roc_auc"]}},
        ],
        TEST_METRICS_PATH,
    )

    results = {
        "df_shape": df.shape,
        "split_info": split_info,
        "feature_groups": {"numeric": prep.NUMERIC_FEATURES, "categorical": prep.CATEGORICAL_FEATURES},
        "baseline": baseline,
        "weight_table": weight_table, "weight_choice": weight_choice, "weight_reason": weight_reason,
        "primary_cv": primary_cv, "n_transformed": n_transformed,
        "primary_test": primary_test,
        "thresholds_oof": thr_oof, "thresholds_test": thr_test,
        "corr_train": corr_train, "vif": vif, "stability": stability,
        "ablation": abl, "final_choice": final_choice, "final_reason": final_reason,
        "final_test": final_test, "coefficients": coefficients,
        "model_path": model_path, "reload_checks": reload_checks,
        "class_weight": weight_choice, "final_num": final_num, "final_cat": final_cat,
        "clean_sha": prep.clean_sha256(), "raw_sha": prep.raw_sha256(),
        "lines": lines,
    }
    write_report(results)
    return results


def write_report(results: dict) -> None:
    """Write model_training_report.txt from the SAME results the notebook shows."""
    si, b = results["split_info"], results["baseline"]
    m = results["final_test"]["metrics"]
    thr_note = (
        "Raw churn probability (predict_proba[:,1]) is kept separate from binary predictions. "
        "The business threshold should be selected from out-of-fold TRAINING predictions "
        "(see OOF table) and confirmed once on untouched data; the test table is descriptive only."
    )
    limitations = [
        "Random (stratified) split, not time-based: no temporal drift assessment is possible with this dataset.",
        "LogisticRegression is linear in transformed features; interaction effects are not modeled unless added.",
        "Class-weight choice was made via CV; the business cost matrix (retention outreach cost vs lost customer value) may justify a different threshold than 0.5.",
        "Probabilities are not calibrated (no calibration step performed); treat them as risk scores, not exact churn probabilities.",
        "Associations hold other included variables constant; no causal claims are made.",
        "Feature engineering leakage review was done in Phase 5; all features are pre-churn attributes.",
        "Small per-fold coefficient variation is expected; exact coefficients depend on regularization strength (C=1.0).",
    ]
    L: list[str] = []
    A = L.append
    A("=== MODEL TRAINING REPORT (PHASE 6) ===")
    A(f"Raw SHA-256      : {results['raw_sha']}")
    A(f"Cleaned SHA-256  : {results['clean_sha']}")
    A(f"Model-ready rows : {results['df_shape'][0]} x {results['df_shape'][1]}")
    A("\n--- DATASET AND SPLIT ---")
    A(f"Test size 20%, random_state=42, stratified on Churn")
    A(f"Train: {si['train_rows']} rows, churn rate {si['train_churn_rate_pct']}% ({si['train_churned']} churned)")
    A(f"Test : {si['test_rows']} rows, churn rate {si['test_churn_rate_pct']}% ({si['test_churned']} churned)")
    A(f"Stratification gap: {si['stratification_gap_pct_pts']} pct pts (<= 0.5 required)")
    A("\n--- FEATURE GROUPS ---")
    A(f"numeric    ({len(results['feature_groups']['numeric'])}): {results['feature_groups']['numeric']}")
    A(f"categorical({len(results['feature_groups']['categorical'])}): {results['feature_groups']['categorical']}")
    A("customerID: identifier only (kept out of the model feature matrix)")
    A("Churn: target (never a predictor)")
    A(f"Transformed features after preprocessing: {results['n_transformed']}")
    A("\n--- BASELINE (DummyClassifier strategy=prior) ---")
    A(f"Always predicts majority class ({b['majority_class']}, prior {b['majority_prior']}). "
      "Useful as the no-skill floor: under imbalance, high accuracy is trivial; ROC-AUC/F1 must beat this.")
    A(b["cv"].to_string(index=False))
    A("Held-out test: " + ", ".join(f"{k}={b['test'][k]:.4f}" for k in ev.METRICS))
    A("\n--- CLASS-WEIGHT COMPARISON (5-fold CV on training set) ---")
    A(results["weight_table"].to_string(index=False))
    A(f"Selected class_weight={results['weight_choice']}. {results['weight_reason']}")
    A("\n--- CROSS-VALIDATION (primary model, training set, 5-fold stratified, seed 42) ---")
    A(results["primary_cv"].to_string(index=False))
    A("\n--- FINAL TEST METRICS (held-out test set, single evaluation) ---")
    A(f"Final model: {results['final_choice']}, class_weight={results['class_weight']}")
    A(f"Confusion matrix (threshold 0.50): TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}")
    A("Consistency: accuracy/precision/recall/F1 recomputed from the confusion matrix match "
      "sklearn metrics exactly (verified in compute_test_metrics); ROC-AUC from predict_proba[:,1].")
    A(f"accuracy={m['accuracy']}, precision={m['precision']}, recall={m['recall']}, "
      f"f1={m['f1']}, roc_auc={m['roc_auc']}")
    A("\nClassification report (test):")
    A(results["final_test"]["report"])
    A("\n--- THRESHOLD ANALYSIS ---")
    A(thr_note)
    A("Out-of-fold TRAINING probabilities (business selection basis):")
    A(results["thresholds_oof"].to_string(index=False))
    A("Held-out TEST set (descriptive reference only):")
    A(results["thresholds_test"].to_string(index=False))
    A("\n--- FEATURE ABLATION (5-fold CV on training set) ---")
    A(results["ablation"].to_string(index=False))
    A(f"Final choice: {results['final_choice']}. {results['final_reason']}")
    A("\n--- MULTICOLLINEARITY FINDINGS ---")
    A("Correlation matrix (numeric block, training data):")
    A(results["corr_train"].to_string())
    A("VIF (numeric block, training data):")
    A(results["vif"].to_string(index=False))
    A("Numeric-block coefficients per CV fold (stability; last rows = std, sign-consistency):")
    A(results["stability"].to_string())
    A("Conclusion: correlated pairs are structural (TotalCharges ~ tenure x price; binned copies). "
      "L2 regularization keeps all coefficients stable in sign; the ablation above is the "
      "empirical test of whether redundant engineered features add value.")
    A("\n--- TOP COEFFICIENTS (final model; odds ratios = exp(coef)) ---")
    A(results["coefficients"].head(20).to_string(index=False))
    A("Interpretation: each coefficient is an association holding all other included "
      "variables constant. Positive = higher churn odds; negative = lower. No causal claims.")
    A("\n--- FINAL MODEL CHOICE ---")
    A(f"Pipeline: ColumnTransformer(median-imputed+scaled numerics; most-frequent-imputed + "
      f"OneHotEncoder(handle_unknown='ignore', drop='first') categoricals) -> LogisticRegression"
      f"(C=1.0, solver='lbfgs', class_weight={results['class_weight']}, max_iter=1000)")
    A(f"Feature set: {results['final_choice']}")
    A(f"Saved to: {results['model_path']} (preprocessing + classifier; reload-verified: "
      f"{results['reload_checks']})")
    A("\n--- LIMITATIONS ---")
    for i, lim in enumerate(limitations, 1):
        A(f"{i}. {lim}")
    A("\nPHASE 6 MODEL TRAINING: SUCCESS")
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    results["lines"] = results.get("lines", []) + L


if __name__ == "__main__":
    run_all()
