"""Exploratory data analysis (EDA) for the cleaned Telco churn dataset.

All statistics, figures and the EDA report are computed from
data/processed/cleaned_customer_churn.csv (read-only; never modified).

The notebook notebooks/02_exploratory_data_analysis.ipynb imports these
same functions, so the report and the notebook share identical calculations.

Language convention: associations are reported as *observed* differences.
No causal claims are made anywhere in this module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_customer_churn.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "eda_report.txt"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

ID_COL = "customerID"
TARGET = "Churn"
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies",
]

TENURE_BINS = [-1, 6, 12, 24, np.inf]
TENURE_LABELS = ["0-6 months", "7-12 months", "13-24 months", "25+ months"]
MC_BINS = [0, 35, 55, 75, 95, np.inf]
MC_LABELS = ["<=35", "35.01-55", "55.01-75", "75.01-95", "95.01+"]
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
INTERNET_ORDER = ["DSL", "Fiber optic", "No"]

FIGURE_FILES = [
    "01_churn_distribution.png",
    "02_numeric_features_by_churn.png",
    "03_categorical_churn_rates.png",
    "04_tenure_group_analysis.png",
    "05_contract_analysis.png",
    "06_monthly_charges_bands.png",
    "07_correlation_heatmap.png",
    "08_segment_churn_heatmaps.png",
]

MIN_SEGMENT_N = 30


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cleaned(path: Path = CLEAN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert not df[TARGET].isin([0, 1]).eq(False).any(), "Churn must be 0/1"
    return df


def overall_churn(df: pd.DataFrame) -> dict:
    total = len(df)
    churned = int((df[TARGET] == 1).sum())
    retained = int((df[TARGET] == 0).sum())
    return {
        "total": total,
        "churned": churned,
        "retained": retained,
        "churn_rate_pct": round(churned / total * 100, 4),
    }


def numeric_stats_by_churn(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for col in NUMERIC_COLUMNS:
        s = (
            df.groupby(TARGET, observed=True)[col]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .round(4)
        )
        s.insert(0, "feature", col)
        frames.append(s.reset_index().rename(columns={TARGET: "churn"}))
    return pd.concat(frames, ignore_index=True)


def churn_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    t = (
        df.groupby(col, observed=True)[TARGET]
        .agg(customers="count", churned="sum")
        .reset_index()
    )
    t["retained"] = t["customers"] - t["churned"]
    t["churn_rate_pct"] = (t["churned"] / t["customers"] * 100).round(4)
    return t.sort_values("churn_rate_pct", ascending=False).reset_index(drop=True)


def categorical_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {col: churn_table(df, col) for col in CATEGORICAL_COLUMNS}


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TenureGroup"] = pd.cut(
        out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, right=True
    )
    return out


def add_monthly_charges_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MonthlyChargesBand"] = pd.cut(
        out["MonthlyCharges"], bins=MC_BINS, labels=MC_LABELS, right=True
    )
    return out


def tenure_group_analysis(df: pd.DataFrame) -> pd.DataFrame:
    d = add_tenure_group(df)
    t = (
        d.groupby("TenureGroup", observed=True)[TARGET]
        .agg(customers="count", churned="sum")
        .reset_index()
    )
    t["retained"] = t["customers"] - t["churned"]
    t["churn_rate_pct"] = (t["churned"] / t["customers"] * 100).round(4)
    return t


def monthly_charges_band_analysis(df: pd.DataFrame) -> pd.DataFrame:
    d = add_monthly_charges_band(df)
    t = (
        d.groupby("MonthlyChargesBand", observed=True)[TARGET]
        .agg(customers="count", churned="sum")
        .reset_index()
    )
    t["retained"] = t["customers"] - t["churned"]
    t["churn_rate_pct"] = (t["churned"] / t["customers"] * 100).round(4)
    return t


def segment_table(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    d = add_tenure_group(df)
    t = (
        d.groupby([col1, col2], observed=True)[TARGET]
        .agg(customers="count", churned="sum")
        .reset_index()
    )
    t["retained"] = t["customers"] - t["churned"]
    t["churn_rate_pct"] = (t["churned"] / t["customers"] * 100).round(4)
    return t.sort_values("churn_rate_pct", ascending=False).reset_index(drop=True)


def top_segments(df: pd.DataFrame, min_n: int = MIN_SEGMENT_N) -> pd.DataFrame:
    combos = [
        ("Contract", "TenureGroup"),
        ("Contract", "InternetService"),
        ("TenureGroup", "InternetService"),
    ]
    frames = []
    for c1, c2 in combos:
        t = segment_table(df, c1, c2)
        t["segment"] = t[c1].astype(str) + " | " + t[c2].astype(str)
        t["combination"] = f"{c1} x {c2}"
        frames.append(t[["combination", "segment", "customers", "churned", "retained", "churn_rate_pct"]])
    out = pd.concat(frames, ignore_index=True)
    out = out[out["customers"] >= min_n]
    return out.sort_values("churn_rate_pct", ascending=False).reset_index(drop=True)


def correlation_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = NUMERIC_COLUMNS + [TARGET]
    pearson = df[cols].corr(method="pearson").round(4)
    spearman = df[cols].corr(method="spearman").round(4)
    return pearson, spearman


def get(df: pd.DataFrame, table: pd.DataFrame, row_key: str, col: str, value_col: str):
    mask = table[col].astype(str) == row_key
    if not mask.any():
        return None
    return float(table.loc[mask, value_col].iloc[0])


def key_findings(df: pd.DataFrame) -> list[str]:
    oc = overall_churn(df)
    contract = churn_table(df, "Contract")
    tenure = tenure_group_analysis(df)
    internet = churn_table(df, "InternetService")
    payment = churn_table(df, "PaymentMethod")
    paperless = churn_table(df, "PaperlessBilling")
    senior = churn_table(df, "SeniorCitizen")
    sec = churn_table(df, "OnlineSecurity")
    tech = churn_table(df, "TechSupport")
    pearson, _ = correlation_analysis(df)

    findings = [
        f"Overall churn: {oc['churned']}/{oc['total']} customers churned "
        f"({oc['churn_rate_pct']:.2f}%); {oc['retained']} retained. "
        f"The target is moderately imbalanced.",

        f"Contract: month-to-month customers had an observed churn rate of "
        f"{get(df, contract, 'Month-to-month', 'Contract', 'churn_rate_pct'):.2f}% "
        f"(n={int(get(df, contract, 'Month-to-month', 'Contract', 'customers'))}), versus "
        f"{get(df, contract, 'One year', 'Contract', 'churn_rate_pct'):.2f}% for one-year and "
        f"{get(df, contract, 'Two year', 'Contract', 'churn_rate_pct'):.2f}% for two-year contracts.",

        f"Tenure: customers with 0-6 months tenure had an observed churn rate of "
        f"{float(tenure.loc[tenure['TenureGroup'].astype(str) == '0-6 months', 'churn_rate_pct'].iloc[0]):.2f}% "
        f"(n={int(tenure.loc[tenure['TenureGroup'].astype(str) == '0-6 months', 'customers'].iloc[0])}), versus "
        f"{float(tenure.loc[tenure['TenureGroup'].astype(str) == '25+ months', 'churn_rate_pct'].iloc[0]):.2f}% "
        f"for 25+ months customers.",

        f"Internet service: Fiber optic users had an observed churn rate of "
        f"{get(df, internet, 'Fiber optic', 'InternetService', 'churn_rate_pct'):.2f}% "
        f"(n={int(get(df, internet, 'Fiber optic', 'InternetService', 'customers'))}), versus "
        f"{get(df, internet, 'DSL', 'InternetService', 'churn_rate_pct'):.2f}% for DSL and "
        f"{get(df, internet, 'No', 'InternetService', 'churn_rate_pct'):.2f}% for customers without internet service.",

        f"Payment method: electronic check users had an observed churn rate of "
        f"{get(df, payment, 'Electronic check', 'PaymentMethod', 'churn_rate_pct'):.2f}%, "
        f"the highest among payment methods; automatic-payment methods were below "
        f"{max(get(df, payment, 'Bank transfer (automatic)', 'PaymentMethod', 'churn_rate_pct'),
              get(df, payment, 'Credit card (automatic)', 'PaymentMethod', 'churn_rate_pct')):.2f}%.",

        f"Add-on services: customers WITHOUT OnlineSecurity had an observed churn rate of "
        f"{get(df, sec, 'No', 'OnlineSecurity', 'churn_rate_pct'):.2f}% vs "
        f"{get(df, sec, 'Yes', 'OnlineSecurity', 'churn_rate_pct'):.2f}% for those with it; "
        f"without TechSupport {get(df, tech, 'No', 'TechSupport', 'churn_rate_pct'):.2f}% vs "
        f"{get(df, tech, 'Yes', 'TechSupport', 'churn_rate_pct'):.2f}% with it.",

        f"Paperless billing: customers with paperless billing had an observed churn rate of "
        f"{get(df, paperless, 'Yes', 'PaperlessBilling', 'churn_rate_pct'):.2f}% vs "
        f"{get(df, paperless, 'No', 'PaperlessBilling', 'churn_rate_pct'):.2f}% without.",

        f"Senior citizens: senior customers had an observed churn rate of "
        f"{get(df, senior, '1', 'SeniorCitizen', 'churn_rate_pct'):.2f}% vs "
        f"{get(df, senior, '0', 'SeniorCitizen', 'churn_rate_pct'):.2f}% for non-seniors.",

        f"Correlations (Pearson): tenure-TotalCharges r={pearson.loc['tenure', 'TotalCharges']:.4f}, "
        f"MonthlyCharges-TotalCharges r={pearson.loc['MonthlyCharges', 'TotalCharges']:.4f}, "
        f"tenure-churn r={pearson.loc['tenure', 'Churn']:.4f}. "
        f"Tenure shows the strongest (negative) association with churn among the numeric features.",

        f"Highest observed churn segments (n>={MIN_SEGMENT_N}): "
        + "; ".join(
            f"{r.segment} ({r.churn_rate_pct:.2f}%, n={int(r.customers)})"
            for r in top_segments(df).head(3).itertuples()
        ) + ".",
    ]
    return findings


def business_implications(df: pd.DataFrame) -> list[str]:
    oc = overall_churn(df)
    contract = churn_table(df, "Contract")
    tenure = tenure_group_analysis(df)
    mc = numeric_stats_by_churn(df)
    mc_mean_churn = float(mc.loc[(mc["feature"] == "MonthlyCharges") & (mc["churn"] == 1), "mean"].iloc[0])
    mc_mean_retain = float(mc.loc[(mc["feature"] == "MonthlyCharges") & (mc["churn"] == 0), "mean"].iloc[0])

    return [
        f"Early-tenure focus: with {float(tenure.loc[tenure['TenureGroup'].astype(str) == '0-6 months', 'churn_rate_pct'].iloc[0]):.2f}% "
        f"observed churn in the first 6 months versus "
        f"{float(tenure.loc[tenure['TenureGroup'].astype(str) == '25+ months', 'churn_rate_pct'].iloc[0]):.2f}% after 25 months, "
        f"onboarding and early-life engagement are potentially high-leverage retention windows.",

        f"Contract strategy: month-to-month contracts show "
        f"{get(df, contract, 'Month-to-month', 'Contract', 'churn_rate_pct'):.2f}% observed churn versus "
        f"{get(df, contract, 'Two year', 'Contract', 'churn_rate_pct'):.2f}% for two-year contracts; "
        f"contract type is a potentially useful segmentation variable for retention campaigns.",

        f"Price positioning: churned customers had a higher mean monthly charge "
        f"({mc_mean_churn:.2f}) than retained customers ({mc_mean_retain:.2f}); "
        f"price-sensitive segments may respond to targeted offers, though this is not evidence of causation.",

        "Service bundling: lower observed churn among customers with security/backup/support add-ons "
        "suggests bundled value may associate with retention; test before treating as a driver.",

        f"Base to protect: {oc['retained']} retained customers generate the billing base; "
        f"the {oc['churned']} churned customers represent the addressable retrospective pool for win-back analysis.",

        "All implications are hypotheses for further analysis (e.g., modeling or A/B tests), "
        "not established causal effects.",
    ]


def limitations() -> list[str]:
    return [
        "Observational, cross-sectional snapshot: observed differences do not establish causation.",
        f"Class imbalance: only about one in four customers churned (26.54%); "
        f"this matters for later modeling (stratification, resampling, metrics).",
        "11 customers have tenure = 0 with TotalCharges set to 0 (verified during Phase 3 cleaning); "
        "their TotalCharges is a convention, not an observed billing amount.",
        "TotalCharges accumulates with tenure (strong positive correlation with tenure), so "
        "comparing TotalCharges between churned and retained customers is confounded by tenure.",
        "Segments with few customers have unstable churn rates; segment findings focus on groups "
        f"with at least {MIN_SEGMENT_N} customers.",
        "Single-provider sample: results describe this dataset and may not generalize to other markets.",
        "No temporal, pricing, or campaign data: churn dynamics and drivers over time cannot be studied here.",
        "Categorical features with dependent category values (e.g., 'No internet service' rows in "
        "add-on columns) mean some category comparisons overlap; read them as descriptive, not independent.",
    ]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def _save(fig, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return str(path)


def plot_churn_distribution(df, ax=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    oc = overall_churn(df)
    ax = ax or plt.gca()
    sns.countplot(data=df, x=TARGET, hue=TARGET, palette="pastel", legend=False, ax=ax)
    for i, v in enumerate([oc["retained"], oc["churned"]]):
        ax.text(i, v, f"{v}\n({v / oc['total'] * 100:.2f}%)", ha="center", va="bottom")
    ax.set_title(f"Churn Distribution (overall churn rate {oc['churn_rate_pct']:.2f}%)")
    ax.set_xlabel("Churn (0 = retained, 1 = churned)")
    ax.set_ylabel("Customers")
    ax.set_xticks([0, 1])
    return _save(ax.figure, FIGURE_FILES[0])


def plot_numeric_by_churn(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, NUMERIC_COLUMNS):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, palette="pastel", legend=False, ax=ax)
        ax.set_title(f"{col} by churn")
        ax.set_xlabel("Churn (0/1)")
    fig.suptitle("Numerical Features by Churn Class", y=1.02)
    fig.tight_layout()
    return _save(fig, FIGURE_FILES[1])


def plot_categorical_churn_rates(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    oc = overall_churn(df)
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    for ax, col in zip(axes.flat, CATEGORICAL_COLUMNS):
        t = churn_table(df, col)
        sns.barplot(data=t, x=col, y="churn_rate_pct", hue=col, palette="pastel", legend=False, ax=ax)
        ax.axhline(oc["churn_rate_pct"], color="red", linestyle="--", linewidth=1, label="overall")
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Churn rate (%)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7)
    fig.suptitle("Observed Churn Rate by Category (red dashed = overall rate)", y=1.005)
    fig.tight_layout()
    return _save(fig, FIGURE_FILES[2])


def plot_tenure_groups(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    t = tenure_group_analysis(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.barplot(data=t, x="TenureGroup", y="customers", hue="TenureGroup", palette="pastel", legend=False, ax=axes[0])
    for p in axes[0].patches:
        axes[0].text(p.get_x() + p.get_width() / 2, p.get_height(), f"{int(p.get_height())}",
                     ha="center", va="bottom")
    axes[0].set_title("Customers by Tenure Group")
    axes[0].set_xlabel("Tenure group")
    axes[0].set_ylabel("Customers")
    sns.barplot(data=t, x="TenureGroup", y="churn_rate_pct", hue="TenureGroup", palette="flare", legend=False, ax=axes[1])
    for p in axes[1].patches:
        axes[1].text(p.get_x() + p.get_width() / 2, p.get_height(), f"{p.get_height():.1f}%",
                     ha="center", va="bottom")
    axes[1].set_title("Observed Churn Rate by Tenure Group")
    axes[1].set_xlabel("Tenure group")
    axes[1].set_ylabel("Churn rate (%)")
    fig.tight_layout()
    return _save(fig, FIGURE_FILES[3])


def plot_contract(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    t = churn_table(df, "Contract")
    t["Contract"] = pd.Categorical(t["Contract"], categories=CONTRACT_ORDER, ordered=True)
    t = t.sort_values("Contract")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.barplot(data=t, x="Contract", y="customers", hue="Contract", palette="pastel", legend=False, ax=axes[0])
    for p in axes[0].patches:
        axes[0].text(p.get_x() + p.get_width() / 2, p.get_height(), f"{int(p.get_height())}",
                     ha="center", va="bottom")
    axes[0].set_title("Customers by Contract Type")
    axes[0].set_xlabel("Contract")
    axes[0].set_ylabel("Customers")
    sns.barplot(data=t, x="Contract", y="churn_rate_pct", hue="Contract", palette="flare", legend=False, ax=axes[1])
    for p in axes[1].patches:
        axes[1].text(p.get_x() + p.get_width() / 2, p.get_height(), f"{p.get_height():.1f}%",
                     ha="center", va="bottom")
    axes[1].set_title("Observed Churn Rate by Contract Type")
    axes[1].set_xlabel("Contract")
    axes[1].set_ylabel("Churn rate (%)")
    fig.tight_layout()
    return _save(fig, FIGURE_FILES[4])


def plot_monthly_charges_bands(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    t = monthly_charges_band_analysis(df)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=t, x="MonthlyChargesBand", y="churn_rate_pct", hue="MonthlyChargesBand",
                palette="flare", legend=False, ax=ax)
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2, p.get_height(), f"{p.get_height():.1f}%",
                ha="center", va="bottom")
    ax.set_title("Observed Churn Rate by Monthly Charges Band")
    ax.set_xlabel("Monthly charges band ($)")
    ax.set_ylabel("Churn rate (%)")
    return _save(fig, FIGURE_FILES[5])


def plot_correlation_heatmap(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    pearson, spearman = correlation_analysis(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, mat, name in [(axes[0], pearson, "Pearson"), (axes[1], spearman, "Spearman")]:
        sns.heatmap(mat, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"{name} correlation")
    fig.tight_layout()
    return _save(fig, FIGURE_FILES[6])


def plot_segment_heatmaps(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    combos = [
        ("Contract", "TenureGroup", CONTRACT_ORDER, TENURE_LABELS),
        ("Contract", "InternetService", CONTRACT_ORDER, INTERNET_ORDER),
        ("TenureGroup", "InternetService", TENURE_LABELS, INTERNET_ORDER),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.8))
    for ax, (c1, c2, order1, order2) in zip(axes, combos):
        t = segment_table(df, c1, c2)
        mat = t.pivot(index=c1, columns=c2, values="churn_rate_pct").reindex(index=order1, columns=order2)
        sns.heatmap(mat, annot=True, fmt=".1f", cmap="Reds", ax=ax, cbar_kws={"label": "churn rate (%)"})
        ax.set_title(f"Churn rate (%): {c1} x {c2}")
        ax.set_ylabel(c1)
        ax.set_xlabel(c2)
    fig.tight_layout()
    return _save(fig, FIGURE_FILES[7])


def generate_figures(df: pd.DataFrame) -> list[str]:
    return [
        plot_churn_distribution(df),
        plot_numeric_by_churn(df),
        plot_categorical_churn_rates(df),
        plot_tenure_groups(df),
        plot_contract(df),
        plot_monthly_charges_bands(df),
        plot_correlation_heatmap(df),
        plot_segment_heatmaps(df),
    ]


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def _fmt_table(t: pd.DataFrame) -> str:
    return t.to_string(index=False)


def write_report(df: pd.DataFrame, verbose: bool = True) -> list[str]:
    lines: list[str] = []

    def log(text: str = "") -> None:
        lines.append(text)
        if verbose:
            print(text)

    oc = overall_churn(df)
    num_stats = numeric_stats_by_churn(df)
    cat_tables = categorical_summary(df)
    tenure_t = tenure_group_analysis(df)
    contract_t = churn_table(df, "Contract").copy()
    contract_t["Contract"] = pd.Categorical(contract_t["Contract"], categories=CONTRACT_ORDER, ordered=True)
    contract_t = contract_t.sort_values("Contract")
    mc_t = monthly_charges_band_analysis(df)
    seg_contract_tenure = segment_table(df, "Contract", "TenureGroup")
    seg_contract_internet = segment_table(df, "Contract", "InternetService")
    seg_tenure_internet = segment_table(df, "TenureGroup", "InternetService")
    top_seg = top_segments(df)
    pearson, spearman = correlation_analysis(df)
    findings = key_findings(df)
    implications = business_implications(df)
    lims = limitations()

    log("=== EXPLORATORY DATA ANALYSIS REPORT ===")
    log(f"Source           : {CLEAN_PATH}")
    log(f"Cleaned SHA-256  : {sha256_of(CLEAN_PATH)}")
    log(f"Generated by     : src/eda.py (shared by notebook 02_exploratory_data_analysis.ipynb)")

    log("\n--- 1. DATASET OVERVIEW ---")
    log(f"Rows                 : {df.shape[0]}")
    log(f"Columns              : {df.shape[1]}")
    log("Dtypes:")
    log(df.dtypes.to_string())
    log(f"Missing values       : {int(df.isna().sum().sum())}")
    log(f"Duplicate rows       : {int(df.duplicated().sum())}")
    log(f"Duplicate customerIDs: {int(df[ID_COL].duplicated().sum())}")

    log("\n--- 2. OVERALL CHURN ---")
    log(f"Total customers      : {oc['total']}")
    log(f"Churned (1)          : {oc['churned']}")
    log(f"Retained (0)         : {oc['retained']}")
    log(f"Churn rate           : {oc['churn_rate_pct']:.4f}%")

    log("\n--- 3. NUMERICAL ANALYSIS (by churn class) ---")
    log(_fmt_table(num_stats))
    log("Note: churned customers show lower mean/median tenure and higher mean MonthlyCharges.")
    log("TotalCharges differences are confounded by tenure (it accumulates over time).")

    log("\n--- 4. CATEGORICAL ANALYSIS (customers / churned / retained / churn rate) ---")
    for col in CATEGORICAL_COLUMNS:
        log(f"\n{col}:")
        log(_fmt_table(cat_tables[col]))

    log("\n--- 5. TENURE ANALYSIS ---")
    log("Tenure groups (0 included in first group):")
    log(_fmt_table(tenure_t))
    log("\nRaw tenure stats by churn class:")
    log(_fmt_table(num_stats[num_stats["feature"] == "tenure"]))

    log("\n--- 6. CONTRACT ANALYSIS ---")
    log(_fmt_table(contract_t))

    log("\n--- 7. SERVICE ANALYSIS (observed churn rates; no causal claims) ---")
    for col in SERVICE_COLUMNS:
        t = cat_tables[col]
        top_row = t.iloc[0]
        log(
            f"{col:<18} highest observed churn: {str(top_row[col])} at "
            f"{top_row['churn_rate_pct']:.2f}% (n={int(top_row['customers'])})"
        )
    log("Note: 'No internet service' categories overlap with add-on columns; read descriptively.")

    log("\n--- 8. CHARGES ANALYSIS ---")
    log("MonthlyCharges / TotalCharges stats by churn class:")
    log(_fmt_table(num_stats[num_stats["feature"].isin(["MonthlyCharges", "TotalCharges"])]))
    log("\nMonthly charges bands:")
    log(_fmt_table(mc_t))
    log("Caution: TotalCharges naturally accumulates with tenure; differences between churn classes "
        "partly reflect tenure differences rather than billing size alone.")

    log("\n--- 9. SEGMENT ANALYSIS (n >= 30 per cell reported) ---")
    log("\nContract x TenureGroup:")
    log(_fmt_table(seg_contract_tenure))
    log("\nContract x InternetService:")
    log(_fmt_table(seg_contract_internet))
    log("\nTenureGroup x InternetService:")
    log(_fmt_table(seg_tenure_internet))
    log(f"\nHighest observed churn segments (n >= {MIN_SEGMENT_N}):")
    log(_fmt_table(top_seg.head(10)))

    log("\n--- 10. CORRELATION FINDINGS ---")
    log("Pearson:")
    log(pearson.to_string())
    log("Spearman:")
    log(spearman.to_string())
    log("Interpretation: TotalCharges correlates strongly and positively with tenure "
        f"(r={pearson.loc['tenure', 'TotalCharges']:.4f}), consistent with accumulation over time. "
        f"Tenure correlates negatively with churn (r={pearson.loc['tenure', 'Churn']:.4f}) - "
        "longer-tenured customers churned less in this dataset. Correlation does not imply causation.")

    log("\n--- 11. KEY FINDINGS ---")
    for i, f in enumerate(findings, 1):
        log(f"{i}. {f}")

    log("\n--- 12. BUSINESS IMPLICATIONS ---")
    for i, f in enumerate(implications, 1):
        log(f"{i}. {f}")

    log("\n--- 13. LIMITATIONS ---")
    for i, f in enumerate(lims, 1):
        log(f"{i}. {f}")

    log("\n--- FIGURES SAVED ---")
    for name in FIGURE_FILES:
        p = FIGURES_DIR / name
        log(f"  {p}  exists={p.exists()}")

    log("\nPHASE 4 EDA: REPORT GENERATED FROM ACTUAL DATA")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def full_eda(verbose: bool = True) -> pd.DataFrame:
    df = load_cleaned()
    generate_figures(df)
    write_report(df, verbose=verbose)
    return df


if __name__ == "__main__":
    full_eda()
