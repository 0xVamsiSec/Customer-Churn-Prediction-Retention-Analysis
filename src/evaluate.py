"""Evaluation utilities for the churn model (Phase 6).

All test metrics come from ONE prediction array and ONE probability array.
compute_test_metrics() mathematically verifies that accuracy/precision/recall/F1
agree with the confusion matrix before results are returned (fails loudly).
ROC-AUC always uses predict_proba[:, 1], never binary predictions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    average_precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import src.preprocessing as prep

FIGURES_DIR = prep.FIGURES_DIR
DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _counts_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    assert tp + fp + fn + tn == len(y_true), "Confusion counts do not sum to n"
    return tn, fp, fn, tp


def compute_test_metrics(
    y_true, y_proba, threshold: float = 0.5, name: str = "model"
) -> dict:
    """Single-source metric computation with consistency verification."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = _counts_from_predictions(y_true, y_pred)
    n = tp + fp + fn + tn

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)  # from probabilities, NOT binary preds

    # --- Mathematical consistency: metrics must agree with the confusion matrix
    acc_m = (tp + tn) / n
    prec_m = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_m = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_m = 2 * prec_m * rec_m / (prec_m + rec_m) if (prec_m + rec_m) > 0 else 0.0
    consistency = {
        "accuracy": bool(np.isclose(accuracy, acc_m, atol=1e-12)),
        "precision": bool(np.isclose(precision, prec_m, atol=1e-12)),
        "recall": bool(np.isclose(recall, rec_m, atol=1e-12)),
        "f1": bool(np.isclose(f1, f1_m, atol=1e-12)),
    }
    if not all(consistency.values()):
        raise ValueError(
            f"METRIC CONSISTENCY FAILED for {name}: sklearn metrics disagree with "
            f"confusion-matrix math: {consistency} (tn={tn}, fp={fp}, fn={fn}, tp={tp})"
        )

    return {
        "name": name,
        "threshold": threshold,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "n": n,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "flagged": int(y_pred.sum()),
        "flagged_pct": round(float(y_pred.mean() * 100), 2),
        "consistency_verified": True,
    }


def classification_report_text(y_true, y_proba, threshold: float = 0.5) -> str:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return classification_report(
        y_true, y_pred, target_names=["Retained (0)", "Churned (1)"], digits=4
    )


def threshold_table(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """Descriptive threshold table. Business threshold selection must use
    training/CV predictions; this table documents the held-out behaviour."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "n_flagged": int(y_pred.sum()),
            "pct_flagged": round(float(y_pred.mean() * 100), 2),
        })
    return pd.DataFrame(rows)


def plot_confusion(y_true, y_proba, threshold: float, out_path: Path) -> Path:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int), y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], labels=["Pred retained", "Pred churned"])
    ax.set_yticks([0, 1], labels=["Actual retained", "Actual churned"])
    ax.set_title(f"Confusion Matrix - Test Set (threshold={threshold:.2f})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc(y_true, y_proba, out_path: Path) -> Path:
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_proba))
    auc = roc_auc_score(np.asarray(y_true).astype(int), np.asarray(y_proba))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.4f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Held-out Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pr(y_true, y_proba, out_path: Path) -> Path:
    y_true = np.asarray(y_true).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, np.asarray(y_proba))
    ap = average_precision_score(y_true, np.asarray(y_proba))
    prevalence = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"Logistic Regression (AP = {ap:.4f})", lw=2)
    ax.axhline(prevalence, color="gray", linestyle="--", lw=1,
               label=f"Baseline = prevalence ({prevalence:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Held-out Test Set")
    ax.legend(loc="upper right")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_test_metrics_csv(rows: list[dict], out_path: Path) -> Path:
    """rows: dicts with keys model, tn, fp, fn, tp, accuracy, precision, recall, f1, roc_auc"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path
