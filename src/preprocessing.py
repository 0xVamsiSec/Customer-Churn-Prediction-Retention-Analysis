"""Model preprocessing for the Telco churn project (Phase 6).

Loads data/processed/model_ready_data.csv (read-only), verifies it, defines
the explicit feature groups, and builds the sklearn preprocessing pipeline.

The preprocessing is always fitted inside a Pipeline on TRAINING data only -
never on the full dataset.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_READY_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_data.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_customer_churn.csv"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

ID_COL = "customerID"
TARGET = "Churn"

# --- Explicit feature groups (24 candidates) --------------------------------
# Numeric: continuous or count-valued candidates.
NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "ServiceCount", "HasSecuritySupport", "AvgMonthlyCharge",
]
# Categorical: nominal categories (SeniorCitizen is 0/1 - conceptually a
# category label, not a continuous quantity).
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "TenureGroup", "MonthlyChargeBand",
]
ALL_CANDIDATE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Ablation feature sets (used by train.py)
ORIGINAL_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
ORIGINAL_CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


def verify_model_ready(df: pd.DataFrame) -> list[str]:
    """Verify the model-ready dataset; raise ValueError on any violation."""
    from src.feature_engineering import MODEL_READY_ORDER

    results = []
    checks = {
        "row count == 7043": len(df) == 7043,
        "columns match design": list(df.columns) == MODEL_READY_ORDER,
        f"{ID_COL} unique": bool(df[ID_COL].is_unique),
        f"{TARGET} in {{0,1}}": set(df[TARGET].unique()) == {0, 1},
        "no missing values": int(df.isna().sum().sum()) == 0,
        "all 24 candidate predictors present": all(c in df.columns for c in ALL_CANDIDATE_FEATURES),
        "predictor count == 24": len(ALL_CANDIDATE_FEATURES) == 24,
    }
    for name, ok in checks.items():
        if not ok:
            raise ValueError(f"MODEL-READY VERIFICATION FAILED: {name}")
        results.append(f"PASS  {name}")
    return results


def load_model_ready(path: Path = MODEL_READY_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    verify_model_ready(df)
    return df


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, customerID). The identifier is kept OUT of X."""
    X = df[ALL_CANDIDATE_FEATURES].copy()
    y = df[TARGET].astype("int64").copy()
    ids = df[ID_COL].copy()
    assert ID_COL not in X.columns and TARGET not in X.columns
    return X, y, ids


def make_split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def split_summary(y_train: pd.Series, y_test: pd.Series) -> dict:
    info = {
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_churn_rate_pct": round(float(y_train.mean() * 100), 4),
        "test_churn_rate_pct": round(float(y_test.mean() * 100), 4),
        "train_churned": int(y_train.sum()),
        "test_churned": int(y_test.sum()),
    }
    info["stratification_gap_pct_pts"] = round(
        abs(info["train_churn_rate_pct"] - info["test_churn_rate_pct"]), 4
    )
    if info["stratification_gap_pct_pts"] > 0.5:
        raise ValueError("Stratification check failed: train/test churn rates differ by >0.5 pts")
    return info


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])


def build_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    class_weight=None,
    C: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline:
    """Full pipeline: preprocessing fitted on train folds only + LogisticRegression."""
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_features, categorical_features)),
        (
            "classifier",
            LogisticRegression(
                C=C, class_weight=class_weight, max_iter=max_iter, solver="lbfgs"
            ),
        ),
    ])


def transformed_feature_names(pipeline: Pipeline) -> np.ndarray:
    return pipeline.named_steps["preprocess"].get_feature_names_out()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_sha256() -> str:
    return sha256_of(RAW_PATH)


def clean_sha256() -> str:
    return sha256_of(CLEAN_PATH)
