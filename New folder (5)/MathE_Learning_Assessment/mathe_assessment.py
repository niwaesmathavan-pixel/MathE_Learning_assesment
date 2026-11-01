"""
mathe_assessment.py
===================
Mathematics Learning Assessment Analysis and Correctness Prediction
Using the UCI MathE Dataset (ID 1031).

Dataset citation:
    Flamia Azevedo, B., Pacheco, M. F., Fernandes, F. P., & Pereira, A. I. (2024).
    Assessing Mathematics Learning in Higher Education [Dataset].
    UCI Machine Learning Repository. https://doi.org/10.34620/dadosipb/PW3OWY

License: Creative Commons Attribution 4.0 International (CC BY 4.0).

Author: Generated for undergraduate engineering assignment.
Python: 3.9+
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
from typing import Optional
import sys
import time
import warnings
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on all platforms)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
CLF_REPORTS_DIR = OUT_DIR / "classification_reports"
CM_DIR = OUT_DIR / "confusion_matrices"
VIZ_DIR = OUT_DIR / "visualizations"
WHATIF_DIR = OUT_DIR / "what_if_experiment"

# Expected column keywords (lower-case, stripped) for fuzzy matching
EXPECTED_COLUMNS = {
    "student_id": ["student id", "studentid", "student_id"],
    "student_country": ["student country", "studentcountry", "country"],
    "question_id": ["question id", "questionid", "question_id"],
    "type_of_answer": ["type of answer", "typeofanswer", "type_of_answer", "answer type", "correct"],
    "question_level": ["question level", "questionlevel", "level", "difficulty"],
    "topic": ["topic"],
    "subtopic": ["subtopic", "sub topic", "sub_topic"],
    "keywords": ["keywords", "keyword"],
}

RANDOM_STATE = 42


# ===========================================================================
# 1. Setup
# ===========================================================================

def setup_output_dirs():
    """Create all required output directories if they do not already exist."""
    for d in [OUT_DIR, CLF_REPORTS_DIR, CM_DIR, VIZ_DIR, WHATIF_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("[INFO] Output directories ready.")


# ===========================================================================
# 2. Dataset discovery
# ===========================================================================

def find_dataset(data_dir: Path) -> Path:
    """
    Search the data/ directory for a CSV file.
    If exactly one CSV is found, return its path.
    If multiple are found, prefer one that contains 'mathe' in the name.
    Raise a clear FileNotFoundError with instructions if none is found.
    """
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        sep = "=" * 70
        msg = (
            f"\n{sep}\n"
            f"  DATASET NOT FOUND\n"
            f"{sep}\n"
            f"  No CSV file was found in:  {data_dir}\n\n"
            f"  Please do the following:\n"
            f"    1. Visit https://archive.ics.uci.edu/dataset/1031\n"
            f"    2. Click 'Download' to get the dataset.\n"
            f"    3. Extract the ZIP and locate the main CSV file\n"
            f"       (e.g., 'MathE dataset (4).csv').\n"
            f"    4. Copy that CSV into:  {data_dir}\n"
            f"    5. Re-run:  python mathe_assessment.py\n"
            f"{sep}"
        )
        raise FileNotFoundError(msg)


    if len(csv_files) == 1:
        return csv_files[0]

    # Prefer files with 'mathe' in name
    mathe_files = [f for f in csv_files if "mathe" in f.name.lower()]
    if len(mathe_files) == 1:
        return mathe_files[0]

    # Fall back to first found and warn
    print(f"[WARN] Multiple CSV files found in data/. Using: {csv_files[0].name}")
    print(f"       Others ignored: {[f.name for f in csv_files[1:]]}")
    return csv_files[0]


# ===========================================================================
# 3. Dataset loading
# ===========================================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the CSV with separator and encoding detection.
    The MathE dataset uses semicolons (;) as field separators.
    The Keywords column contains comma-separated values inside each field,
    so we must use sep=';' to avoid splitting keywords into extra columns.
    Tries UTF-8, then latin-1 (common for European-origin files).
    Returns a DataFrame.
    """
    print(f"\n[INFO] Loading dataset from: {path}")

    # Detect separator by reading the first line
    separators_to_try = [";", ",", "\t"]
    detected_sep = ","
    for enc_detect in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, encoding=enc_detect, errors="replace") as f:
                first_line = f.readline()
            for sep in separators_to_try:
                if first_line.count(sep) >= 3:
                    detected_sep = sep
                    break
            break
        except Exception:
            continue

    print(f"[INFO] Detected separator: {detected_sep!r}")

    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                sep=detected_sep,
                on_bad_lines="warn",   # skip/warn on malformed lines
                quoting=3,             # QUOTE_NONE -- avoid misinterpretation
                engine="python",
            )
            print(f"[INFO] Successfully loaded with encoding '{enc}', sep={detected_sep!r}.")
            print(f"[INFO] Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV '{path}': {e}") from e

    raise RuntimeError(
        f"Could not decode '{path.name}' with utf-8, latin-1, or cp1252. "
        "Please inspect the file encoding and set it manually."
    )



# ===========================================================================
# 4. Column inspection and normalization
# ===========================================================================

def inspect_columns(df: pd.DataFrame) -> None:
    """Print actual column names, dtypes, shape, and sample rows."""
    print("\n" + "=" * 60)
    print("  DATASET INSPECTION")
    print("=" * 60)
    print(f"  Shape   : {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    print(f"\n  Columns ({df.shape[1]}):")
    for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
        n_unique = df[col].nunique()
        n_null = df[col].isnull().sum()
        print(f"    {i:2}. {col!r:<35} dtype={dtype!s:<10} unique={n_unique:>6,}  nulls={n_null:>5,}")
    print("\n  Sample rows (first 3):")
    print(df.head(3).to_string(index=False))
    print("=" * 60)


def normalize_col_name(name: str) -> str:
    """Lowercase, strip whitespace, collapse internal spaces to underscore."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_columns(df: pd.DataFrame):
    """
    Return (df_normalized, col_mapping) where df_normalized uses clean internal
    column names and col_mapping maps internal names -> original names.
    The original DataFrame is not modified.
    """
    col_mapping = {normalize_col_name(c): c for c in df.columns}
    df_norm = df.copy()
    df_norm.columns = [normalize_col_name(c) for c in df.columns]
    return df_norm, col_mapping


# ===========================================================================
# 5. Column validation
# ===========================================================================

def match_column(normalized_cols: list, candidates: list) -> Optional[str]:
    """Return the first normalized column that matches any candidate string."""
    for candidate in candidates:
        norm_candidate = normalize_col_name(candidate)
        if norm_candidate in normalized_cols:
            return norm_candidate
    return None


def validate_columns(df_norm: pd.DataFrame, col_mapping: dict) -> dict:
    """
    Attempt to match each expected field to an actual column.
    Returns a dict: internal_field_key -> actual_normalized_col_name.
    Raises ValueError if a required field cannot be found.
    """
    print("\n[INFO] Validating columns against expected MathE schema...")
    norm_cols = list(df_norm.columns)
    field_map = {}
    missing = []

    for field_key, candidates in EXPECTED_COLUMNS.items():
        matched = match_column(norm_cols, candidates)
        if matched:
            field_map[field_key] = matched
            orig = col_mapping.get(matched, matched)
            print(f"  [OK]  {field_key:<20} -> '{orig}'  (normalized: '{matched}')")
        else:
            missing.append(field_key)
            print(f"  [!!]  {field_key:<20} -> NOT FOUND  (searched: {candidates})")

    if missing:
        print("\n[ERROR] The following expected fields were not matched:")
        for m in missing:
            print(f"         - {m}  (candidates tried: {EXPECTED_COLUMNS[m]})")
        print(
            "\n  The actual column names in your file are:\n"
            f"  {list(df_norm.columns)}\n\n"
            "  Please update the EXPECTED_COLUMNS dictionary at the top of\n"
            "  mathe_assessment.py to map to the correct column names in your file.\n"
            "  Do NOT rename or modify the original dataset."
        )
        raise ValueError(f"Could not match required columns: {missing}")

    return field_map


# ===========================================================================
# 6. Target validation
# ===========================================================================

def validate_target(df_norm: pd.DataFrame, target_col: str) -> tuple:
    """
    Inspect the target column.
    Confirm it is binary, document the encoding, and return
    (positive_label, negative_label, label_description).

    The UCI paper describes: 0 = correct, 1 = incorrect.
    We verify this from the actual file and document any difference.
    """
    print(f"\n[INFO] Validating target column: '{target_col}'")
    unique_vals = sorted(df_norm[target_col].dropna().unique())
    print(f"  Unique target values: {unique_vals}")
    val_counts = df_norm[target_col].value_counts()
    print(f"  Value counts:\n{val_counts.to_string()}")

    if len(unique_vals) != 2:
        raise ValueError(
            f"Target column '{target_col}' has {len(unique_vals)} unique values: {unique_vals}.\n"
            "Expected exactly 2 (binary classification). "
            "Inspect the CSV and update the target logic if needed."
        )

    # Determine encoding
    # Convention from UCI paper: 0 = correct, 1 = incorrect
    # Also handle string labels like 'correct'/'incorrect'
    v0, v1 = unique_vals[0], unique_vals[1]

    # String-based detection
    str_v0 = str(v0).strip().lower()
    str_v1 = str(v1).strip().lower()

    if str_v0 in ("correct", "right", "0") and str_v1 in ("incorrect", "wrong", "1"):
        positive_label = v1   # incorrect = positive (what we want to detect)
        negative_label = v0
        description = (
            f"  Label encoding confirmed:\n"
            f"    {v0!r} -> correct answer (negative class)\n"
            f"    {v1!r} -> incorrect answer (positive class, what we predict)\n"
            f"  This matches the UCI paper convention (0=correct, 1=incorrect)."
        )
    elif str_v0 in ("incorrect", "wrong", "1") and str_v1 in ("correct", "right", "0"):
        positive_label = v0
        negative_label = v1
        description = (
            f"  Label encoding (reversed from expected):\n"
            f"    {v0!r} -> incorrect answer (positive class)\n"
            f"    {v1!r} -> correct answer (negative class)\n"
            f"  NOTE: Order is reversed from UCI paper; positive class is still 'incorrect'."
        )
    else:
        # Numeric -- assume lower value = correct (paper: 0=correct, 1=incorrect)
        # but verify and document
        positive_label = v1   # higher value = incorrect (follows paper)
        negative_label = v0
        description = (
            f"  Numeric label encoding detected:\n"
            f"    {v0!r} -> assumed correct answer (negative class, lower value)\n"
            f"    {v1!r} -> assumed incorrect answer (positive class, higher value)\n"
            f"  This follows the UCI paper convention where 0=correct, 1=incorrect.\n"
            f"  IMPORTANT: Verify this against the dataset documentation."
        )

    print(description)
    print(f"\n  Positive class (incorrect answer): {positive_label!r}")
    print(f"  Negative class (correct answer)  : {negative_label!r}")
    return positive_label, negative_label, description


# ===========================================================================
# 7. Data quality report
# ===========================================================================

def data_quality_report(df_norm: pd.DataFrame, field_map: dict,
                         positive_label, negative_label,
                         label_description: str) -> pd.DataFrame:
    """
    Generate and save a full data-quality report.
    Returns the cleaned DataFrame (duplicates handled, no rows deleted silently).
    """
    print("\n" + "=" * 60)
    print("  DATA QUALITY REPORT")
    print("=" * 60)

    target_col = field_map["type_of_answer"]
    student_col = field_map["student_id"]
    question_col = field_map["question_id"]

    total_rows = len(df_norm)
    n_exact_dupes = df_norm.duplicated().sum()
    n_missing_per_col = df_norm.isnull().sum()
    n_unique_students = df_norm[student_col].nunique()
    n_unique_questions = df_norm[question_col].nunique()

    class_counts = df_norm[target_col].value_counts()
    n_correct = class_counts.get(negative_label, 0)
    n_incorrect = class_counts.get(positive_label, 0)
    pct_correct = 100 * n_correct / total_rows if total_rows else 0
    pct_incorrect = 100 * n_incorrect / total_rows if total_rows else 0

    lines = [
        "DATA QUALITY REPORT",
        "=" * 60,
        f"Total rows            : {total_rows:,}",
        f"Total columns         : {df_norm.shape[1]}",
        f"",
        f"Exact duplicate rows  : {n_exact_dupes:,}",
        f"  (Duplicates may represent repeated attempts by same student on same question.)",
        f"  (No rows are automatically deleted. All are retained for analysis.)",
        f"",
        f"Missing values per column:",
    ]
    for col, cnt in n_missing_per_col.items():
        lines.append(f"  {col:<35}: {cnt:,}  ({100*cnt/total_rows:.2f}%)")

    lines += [
        f"",
        f"Unique students       : {n_unique_students:,}",
        f"Unique questions      : {n_unique_questions:,}",
        f"",
        f"Class distribution ({target_col}):",
        f"  Correct   ({negative_label!r}) : {n_correct:>6,}  ({pct_correct:.1f}%)",
        f"  Incorrect ({positive_label!r}) : {n_incorrect:>6,}  ({pct_incorrect:.1f}%)",
        f"",
        f"Label encoding:",
    ]
    lines += [f"  {l}" for l in label_description.strip().splitlines()]

    lines += [
        f"",
        f"NOTE: The UCI documentation reports 9,546 records and 8 features.",
        f"      Actual file has {total_rows:,} records and {df_norm.shape[1]} columns.",
        f"      Differences (if any) are reported above rather than forced to match.",
        f"",
        f"Leakage and Identifier Notes:",
        f"  - Student ID is an identifier; excluded from primary model predictors.",
        f"  - Question ID is an identifier; excluded from primary model predictors.",
        f"  - Repeated Student IDs are expected (one student answers many questions).",
        f"  - Repeated Question IDs are expected (many students answer same question).",
        f"  - Exact duplicates are retained for now; none are silently removed.",
    ]

    report_text = "\n".join(lines)
    print(report_text)

    report_path = OUT_DIR / "data_quality_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[INFO] Data quality report saved: {report_path}")

    # Class distribution CSV
    class_dist = pd.DataFrame({
        "label": [str(negative_label), str(positive_label)],
        "meaning": ["correct answer", "incorrect answer"],
        "count": [n_correct, n_incorrect],
        "percentage": [round(pct_correct, 2), round(pct_incorrect, 2)],
    })
    class_dist.to_csv(OUT_DIR / "dataset_class_distribution.csv", index=False)

    # Dataset summary
    summary_lines = [
        "DATASET SUMMARY",
        "=" * 60,
        f"Source : UCI Machine Learning Repository",
        f"ID     : 1031",
        f"DOI    : https://doi.org/10.34620/dadosipb/PW3OWY",
        f"License: CC BY 4.0",
        f"",
        f"Rows   : {total_rows:,}",
        f"Cols   : {df_norm.shape[1]}",
        f"",
        f"Column summary:",
    ]
    for col, dtype in df_norm.dtypes.items():
        summary_lines.append(f"  {col:<35}: {dtype}")
    summary_lines += ["", "Citation:", "  Flamia Azevedo, B., Pacheco, M. F., Fernandes, F. P., & Pereira, A. I. (2024)."]
    summary_lines += ["  Assessing Mathematics Learning in Higher Education [Dataset]."]
    summary_lines += ["  UCI Machine Learning Repository. https://doi.org/10.34620/dadosipb/PW3OWY"]

    (OUT_DIR / "dataset_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[INFO] Dataset summary saved: {OUT_DIR / 'dataset_summary.txt'}")

    return df_norm   # no rows removed; caller can decide


# ===========================================================================
# 8. Exploratory Data Analysis
# ===========================================================================

def _save_fig(fig: plt.Figure, path: Path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[INFO] Plot saved: {path}")


def exploratory_analysis(df_norm: pd.DataFrame, field_map: dict,
                          positive_label, negative_label):
    """
    Perform EDA: descriptive stats, correctness breakdowns, and visualizations.
    All plots are saved to outputs/visualizations/.
    """
    print("\n" + "=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    target_col   = field_map["type_of_answer"]
    level_col    = field_map["question_level"]
    topic_col    = field_map["topic"]
    subtopic_col = field_map["subtopic"]
    country_col  = field_map["student_country"]

    total = len(df_norm)

    # --- 1. Overall class distribution bar chart ---
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df_norm[target_col].value_counts()
    colors = ["#4CAF50" if v == negative_label else "#E74C3C" for v in counts.index]
    bars = ax.bar([str(v) for v in counts.index], counts.values, color=colors, edgecolor="black")
    ax.set_title("Class Distribution: Correct vs Incorrect Answers", fontsize=13, fontweight="bold")
    ax.set_xlabel("Answer Type")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:,}\n({100*val/total:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.2)
    _save_fig(fig, VIZ_DIR / "01_class_distribution.png")

    # --- 2. Correctness rate by question level ---
    level_group = df_norm.groupby(level_col)[target_col].value_counts(normalize=True).unstack(fill_value=0)
    if positive_label in level_group.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        incorrect_rate = level_group[positive_label].sort_values(ascending=False)
        bars = ax.bar(incorrect_rate.index.astype(str), incorrect_rate.values * 100,
                      color="#E74C3C", edgecolor="black")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_title("Incorrect Answer Rate by Question Level", fontsize=13, fontweight="bold")
        ax.set_xlabel("Question Level")
        ax.set_ylabel("Incorrect Answer Rate (%)")
        ax.set_ylim(0, 110)
        ax.axhline(y=100 * (df_norm[target_col] == positive_label).mean(),
                   color="navy", linestyle="--", label="Overall incorrect rate")
        ax.legend()
        plt.xticks(rotation=30, ha="right")
        _save_fig(fig, VIZ_DIR / "02_incorrect_rate_by_level.png")

    # --- 3. Correctness rate by topic ---
    topic_counts = df_norm[topic_col].value_counts()
    MIN_TOPIC_COUNT = 30
    valid_topics = topic_counts[topic_counts >= MIN_TOPIC_COUNT].index
    df_topic = df_norm[df_norm[topic_col].isin(valid_topics)]

    if not df_topic.empty:
        topic_group = df_topic.groupby(topic_col)[target_col].value_counts(normalize=True).unstack(fill_value=0)
        if positive_label in topic_group.columns:
            inc_rate = topic_group[positive_label].sort_values(ascending=False)
            n_topics = len(inc_rate)
            fig_h = max(5, 0.4 * n_topics)
            fig, ax = plt.subplots(figsize=(10, fig_h))
            bars = ax.barh(inc_rate.index.astype(str), inc_rate.values * 100,
                           color="#E74C3C", edgecolor="black")
            for bar in bars:
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{bar.get_width():.1f}%", va="center", fontsize=8)
            ax.set_title(f"Incorrect Answer Rate by Topic\n(topics with >={MIN_TOPIC_COUNT} answers)",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("Incorrect Answer Rate (%)")
            ax.set_xlim(0, 115)
            ax.axvline(x=100 * (df_norm[target_col] == positive_label).mean(),
                       color="navy", linestyle="--", label="Overall incorrect rate")
            ax.legend()
            plt.tight_layout()
            _save_fig(fig, VIZ_DIR / "03_incorrect_rate_by_topic.png")

    # --- 4. Top topics by answer count ---
    top_n = min(15, len(topic_counts))
    fig, ax = plt.subplots(figsize=(10, 6))
    top_topics = topic_counts.head(top_n)
    ax.barh(top_topics.index.astype(str)[::-1], top_topics.values[::-1],
            color="#3498DB", edgecolor="black")
    ax.set_title(f"Top {top_n} Topics by Answer Count", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Answers")
    for i, val in enumerate(top_topics.values[::-1]):
        ax.text(val + 10, i, str(val), va="center", fontsize=8)
    plt.tight_layout()
    _save_fig(fig, VIZ_DIR / "04_top_topics_by_count.png")

    # --- 5. Correctness by country (only countries with >= 50 answers) ---
    country_counts = df_norm[country_col].value_counts()
    MIN_COUNTRY_COUNT = 50
    valid_countries = country_counts[country_counts >= MIN_COUNTRY_COUNT].index
    df_country = df_norm[df_norm[country_col].isin(valid_countries)]

    if not df_country.empty:
        cg = df_country.groupby(country_col)[target_col].value_counts(normalize=True).unstack(fill_value=0)
        if positive_label in cg.columns:
            inc_country = cg[positive_label].sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(inc_country))))
            bars = ax.barh(inc_country.index.astype(str), inc_country.values * 100,
                           color="#9B59B6", edgecolor="black")
            for bar in bars:
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{bar.get_width():.1f}%", va="center", fontsize=8)
            ax.set_title(
                f"Incorrect Answer Rate by Country\n"
                f"(countries with >={MIN_COUNTRY_COUNT} answers)\n"
                f"WARNING: Small groups may give unstable percentages; observational data only.",
                fontsize=11, fontweight="bold"
            )
            ax.set_xlabel("Incorrect Answer Rate (%)")
            ax.set_xlim(0, 120)
            ax.axvline(x=100 * (df_norm[target_col] == positive_label).mean(),
                       color="navy", linestyle="--", label="Overall incorrect rate")
            ax.legend()
            plt.tight_layout()
            _save_fig(fig, VIZ_DIR / "05_incorrect_rate_by_country.png")

    # --- 6. Subtopic answer count (top 15) ---
    subtopic_counts = df_norm[subtopic_col].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(subtopic_counts.index.astype(str)[::-1], subtopic_counts.values[::-1],
            color="#F39C12", edgecolor="black")
    ax.set_title("Top 15 Subtopics by Answer Count", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Answers")
    plt.tight_layout()
    _save_fig(fig, VIZ_DIR / "06_top_subtopics_by_count.png")

    print("\n[INFO] EDA visualizations saved to outputs/visualizations/")
    print(
        "\n  DISCLAIMER: All EDA findings are observational. Small group sizes may yield\n"
        "  unstable percentages. No causal conclusions can be drawn about student ability\n"
        "  or country-level differences from this dataset."
    )


# ===========================================================================
# 9. Feature selection and preprocessing
# ===========================================================================

FEATURE_KEYS = [
    "student_country",
    "question_level",
    "topic",
    "subtopic",
    "keywords",
]
# student_id and question_id are intentionally excluded from the primary model.
EXCLUDED_KEYS = ["student_id", "question_id"]


def get_feature_cols(field_map: dict) -> list:
    """Return list of actual (normalized) column names to use as predictors."""
    return [field_map[k] for k in FEATURE_KEYS if k in field_map]


def build_preprocessor(feature_cols: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies OneHotEncoder to all categorical features.
    All selected features are categorical; no numeric scaling needed for the primary model.

    Why OneHotEncoder?
      Machine learning algorithms require numeric input. Categorical columns like
      'Topic' or 'Country' are nominal -- they have no inherent order. OneHotEncoder
      converts each category into a separate binary (0/1) column, preserving this
      lack of ordering.

    Why handle_unknown='ignore'?
      Categories seen in test data but not in training data would otherwise cause an
      error. 'ignore' maps unknown categories to a zero vector, which is a safe default.

    We do NOT use TF-IDF because the Keywords column contains short categorical tags,
    not full natural-language documents.
    """
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, feature_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def save_feature_summary(preprocessor, feature_cols: list,
                          X_train, X_test, field_map: dict):
    """Save a readable feature summary to outputs/feature_summary.txt."""
    try:
        # Try to get feature names (scikit-learn >= 1.0)
        feature_names = preprocessor.get_feature_names_out()
        n_transformed = len(feature_names)
        sample_names = list(feature_names[:20])
    except Exception:
        n_transformed = "unknown (scikit-learn version does not support get_feature_names_out)"
        sample_names = []

    lines = [
        "FEATURE SUMMARY",
        "=" * 60,
        "",
        "Primary predictor columns (before encoding):",
    ]
    for col in feature_cols:
        orig_key = next((k for k, v in field_map.items() if v == col), col)
        lines.append(f"  - {col}  (field: {orig_key})")

    lines += [
        "",
        "Excluded from primary model (identifiers):",
        f"  - {field_map.get('student_id', 'student_id')}  -> identifier, not a predictor",
        f"  - {field_map.get('question_id', 'question_id')}  -> identifier, excluded to prevent question memorization",
        "",
        "Preprocessing applied:",
        "  - SimpleImputer (strategy='constant', fill_value='MISSING') for missing categoricals",
        "  - OneHotEncoder (handle_unknown='ignore') for all categorical columns",
        "  - Preprocessing fitted on TRAINING data only; applied to test data",
        "",
        f"Training set size  : {X_train.shape[0]:,} records",
        f"Test set size      : {X_test.shape[0]:,} records",
        f"Transformed features: {n_transformed}",
        "",
        "Sample transformed feature names (first 20):",
    ]
    for fn in sample_names:
        lines.append(f"  {fn}")
    if not sample_names:
        lines.append("  (Feature names not extractable in this scikit-learn version)")

    text = "\n".join(lines)
    (OUT_DIR / "feature_summary.txt").write_text(text, encoding="utf-8")
    print(f"[INFO] Feature summary saved: {OUT_DIR / 'feature_summary.txt'}")
    print(text)


def save_preprocessing_examples(df_norm: pd.DataFrame, feature_cols: list,
                                  target_col: str, n_rows: int = 5):
    """Save a small CSV comparing original feature values to transformed representation."""
    sample = df_norm[feature_cols + [target_col]].head(n_rows).copy()
    sample.to_csv(OUT_DIR / "preprocessing_examples.csv", index=False)
    print(f"[INFO] Preprocessing examples saved: {OUT_DIR / 'preprocessing_examples.csv'}")


# ===========================================================================
# 10. Train / Test split
# ===========================================================================

def group_split(df_norm: pd.DataFrame, feature_cols: list,
                 target_col: str, student_col: str):
    """
    Split dataset so that all records for a given student appear in either
    train or test -- not both. This prevents the model from learning individual
    student patterns and then being tested on the same student.

    Method: GroupShuffleSplit on Student ID.
    Fallback: StratifiedShuffleSplit if group split fails class-representation check.
    """
    X = df_norm[feature_cols]
    y = df_norm[target_col]
    groups = df_norm[student_col]

    print("\n[INFO] Splitting data using GroupShuffleSplit on Student ID...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Verify both classes present in train and test
    train_classes = set(y_train.unique())
    test_classes = set(y_test.unique())
    unique_classes = set(y.unique())

    if not unique_classes.issubset(train_classes) or not unique_classes.issubset(test_classes):
        print(
            "\n[WARN] Group split resulted in a missing class in train or test set.\n"
            "       Falling back to StratifiedShuffleSplit (row-wise, not student-aware).\n"
            "       This is a documented fallback. Results may overestimate generalization."
        )
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_idx, test_idx = next(sss.split(X, y))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        split_method = "StratifiedShuffleSplit (fallback)"
    else:
        split_method = "GroupShuffleSplit on Student ID (primary)"

    # Check for student overlap between train and test
    train_students = set(groups.iloc[train_idx].unique())
    test_students = set(groups.iloc[test_idx].unique())
    overlap = train_students & test_students
    if overlap:
        print(f"[WARN] {len(overlap)} students appear in both train and test sets.")
    else:
        print("[INFO] No student overlap between train and test sets. [OK]")

    print(f"\n  Split method     : {split_method}")
    print(f"  Train set size   : {len(X_train):,}  ({len(train_students):,} unique students)")
    print(f"  Test set size    : {len(X_test):,}  ({len(test_students):,} unique students)")
    print(f"  Train class dist :\n{y_train.value_counts().to_string()}")
    print(f"  Test class dist  :\n{y_test.value_counts().to_string()}")

    return X_train, X_test, y_train, y_test, split_method


# ===========================================================================
# 11. Model definitions
# ===========================================================================

def get_models() -> dict:
    """
    Return a dict of model_name -> sklearn estimator.

    DecisionTreeClassifier:
      Learns a series of if/else rules on the features. Interpretable but can overfit.

    KNeighborsClassifier:
      Classifies based on majority vote of K nearest training samples.
      Sensitive to feature scaling and high-dimensional sparse data.

    LinearSVC (SVM):
      Finds a linear hyperplane maximising margin between classes.
      Efficient for high-dimensional sparse features (like OHE outputs).
      Note: LinearSVC does not output probabilities by default.

    RandomForestClassifier:
      Ensemble of Decision Trees trained on random feature/sample subsets.
      Reduces overfitting by averaging many trees. Usually robust.
    """
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, metric="euclidean"
        ),
        "SVM (Linear)": LinearSVC(
            C=1.0, max_iter=2000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


# ===========================================================================
# 12. Training and evaluation
# ===========================================================================

def build_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """Wrap preprocessor + model in a single sklearn Pipeline."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def evaluate_model(pipeline: Pipeline, X_test, y_test,
                    model_name: str, positive_label,
                    negative_label) -> dict:
    """
    Evaluate a fitted pipeline on the test set.
    Returns a metrics dict and saves classification report + confusion matrix.
    """
    t_pred_start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    pred_time = time.perf_counter() - t_pred_start

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=positive_label, zero_division=0)

    clf_report = classification_report(y_test, y_pred, zero_division=0)

    # Save classification report
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    report_path = CLF_REPORTS_DIR / f"{safe_name}_classification_report.txt"
    report_content = (
        f"Classification Report -- {model_name}\n"
        f"{'=' * 50}\n"
        f"Positive class (incorrect answer): {positive_label!r}\n"
        f"Negative class (correct answer)  : {negative_label!r}\n\n"
        f"{clf_report}\n"
        f"Prediction time: {pred_time:.4f} seconds\n"
        f"\nConfusion Matrix Legend:\n"
        f"  True Positive  (TP): Incorrect answer correctly identified as incorrect\n"
        f"  True Negative  (TN): Correct answer correctly identified as correct\n"
        f"  False Positive (FP): Correct answer incorrectly flagged as incorrect\n"
        f"  False Negative (FN): Incorrect answer incorrectly predicted as correct\n"
    )
    report_path.write_text(report_content, encoding="utf-8")

    # Confusion matrix image
    labels_ordered = [negative_label, positive_label]
    display_labels = [f"Correct\n({negative_label})", f"Incorrect\n({positive_label})"]
    cm = confusion_matrix(y_test, y_pred, labels=labels_ordered)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix -- {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    cm_path = CM_DIR / f"{safe_name}_confusion_matrix.png"
    _save_fig(fig, cm_path)

    print(f"\n  [{model_name}]")
    print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-score  : {f1:.4f}")
    print(f"    Pred time : {pred_time:.4f}s")

    return {
        "model": model_name,
        "accuracy": acc,
        "precision_incorrect": prec,
        "recall_incorrect": rec,
        "f1_incorrect": f1,
        "pred_time_s": pred_time,
    }


def train_and_evaluate_all(preprocessor, X_train, X_test, y_train, y_test,
                            positive_label, negative_label) -> list:
    """
    Train all four models on the training set, evaluate on the test set.
    Returns list of metrics dicts.
    """
    print("\n" + "=" * 60)
    print("  MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    models = get_models()
    results = []

    for model_name, model in models.items():
        print(f"\n  Training: {model_name} ...")
        pipeline = build_pipeline(preprocessor, model)

        t_train_start = time.perf_counter()
        try:
            pipeline.fit(X_train, y_train)
        except Exception as e:
            print(f"  [ERROR] Training failed for {model_name}: {e}")
            continue
        train_time = time.perf_counter() - t_train_start
        print(f"    Training time: {train_time:.4f}s")

        metrics = evaluate_model(pipeline, X_test, y_test,
                                  model_name, positive_label, negative_label)
        metrics["train_time_s"] = train_time
        results.append(metrics)

    return results


# ===========================================================================
# 13. Model comparison
# ===========================================================================

def compare_and_visualize(results: list):
    """
    Save a comparison CSV and produce a bar chart comparing all four models
    across Accuracy, Precision, Recall, and F1-score for the incorrect class.
    """
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("f1_incorrect", ascending=False)
    print(df_res.to_string(index=False))
    df_res.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print(f"[INFO] Comparison CSV saved: {OUT_DIR / 'model_comparison.csv'}")

    # Bar chart
    metrics_to_plot = {
        "Accuracy": "accuracy",
        "Precision\n(Incorrect)": "precision_incorrect",
        "Recall\n(Incorrect)": "recall_incorrect",
        "F1-score\n(Incorrect)": "f1_incorrect",
    }
    n_metrics = len(metrics_to_plot)
    n_models = len(df_res)
    x = np.arange(n_metrics)
    bar_width = 0.8 / n_models

    colors = ["#3498DB", "#E67E22", "#2ECC71", "#9B59B6"]
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (_, row) in enumerate(df_res.iterrows()):
        vals = [row[col] for col in metrics_to_plot.values()]
        positions = x + i * bar_width - (n_models - 1) * bar_width / 2
        bars = ax.bar(positions, vals, bar_width * 0.9, label=row["model"],
                      color=colors[i % len(colors)], edgecolor="black")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison: All Four Classifiers", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot.keys(), fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.7)
    plt.tight_layout()
    _save_fig(fig, VIZ_DIR / "07_model_comparison.png")

    return df_res


# ===========================================================================
# 14. What-If Experiment: KNN with varying K
# ===========================================================================

def what_if_knn(preprocessor: ColumnTransformer,
                X_train, X_test, y_train, y_test,
                positive_label, negative_label):
    """
    Experiment: vary KNN's n_neighbors in {3, 5, 7, 9} and measure the effect
    on accuracy, precision, recall, and F1-score for the incorrect class.

    The same train/test split and preprocessor are reused.
    The preprocessor is re-fitted on the same X_train for each run to keep
    the pipeline self-contained (equivalent outcome since data doesn't change).
    """
    print("\n" + "=" * 60)
    print("  WHAT-IF EXPERIMENT: KNN with Varying n_neighbors")
    print("=" * 60)

    k_values = [3, 5, 7, 9]
    exp_results = []

    for k in k_values:
        print(f"\n  Testing KNN with n_neighbors={k} ...")
        if k < 1:
            print(f"  [SKIP] n_neighbors={k} is invalid (must be >= 1). Skipping.")
            continue

        model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        pipeline = build_pipeline(preprocessor, model)

        t_start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_t = time.perf_counter() - t_start

        t_pred = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        pred_t = time.perf_counter() - t_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=positive_label, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=positive_label, zero_division=0)

        print(f"    K={k}: Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
        exp_results.append({
            "n_neighbors": k,
            "accuracy": acc,
            "precision_incorrect": prec,
            "recall_incorrect": rec,
            "f1_incorrect": f1,
            "train_time_s": train_t,
            "pred_time_s": pred_t,
        })

    df_exp = pd.DataFrame(exp_results)
    df_exp.to_csv(WHATIF_DIR / "knn_what_if_results.csv", index=False)
    print(f"\n[INFO] What-if results saved: {WHATIF_DIR / 'knn_what_if_results.csv'}")

    # Chart
    if not df_exp.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        metrics = {
            "Accuracy": "accuracy",
            "Precision (Incorrect)": "precision_incorrect",
            "Recall (Incorrect)": "recall_incorrect",
            "F1-score (Incorrect)": "f1_incorrect",
        }
        line_colors = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6"]
        for (label, col), color in zip(metrics.items(), line_colors):
            ax.plot(df_exp["n_neighbors"], df_exp[col], marker="o",
                    label=label, color=color, linewidth=2)
            for _, row in df_exp.iterrows():
                ax.annotate(f"{row[col]:.3f}",
                            xy=(row["n_neighbors"], row[col]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)

        ax.set_xlabel("n_neighbors (K)", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("KNN What-If Experiment: Effect of K on Performance",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(df_exp["n_neighbors"])
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        _save_fig(fig, WHATIF_DIR / "knn_what_if_chart.png")

    return df_exp


# ===========================================================================
# 15. Final observations
# ===========================================================================

def save_final_observations(df_norm: pd.DataFrame, field_map: dict,
                              positive_label, negative_label,
                              label_description: str,
                              split_method: str,
                              X_train, X_test,
                              results_df: pd.DataFrame,
                              whatif_df: pd.DataFrame):
    """Write the final_observations.txt file."""

    target_col = field_map["type_of_answer"]
    student_col = field_map["student_id"]
    question_col = field_map["question_id"]

    total = len(df_norm)
    n_correct = (df_norm[target_col] == negative_label).sum()
    n_incorrect = (df_norm[target_col] == positive_label).sum()
    n_students = df_norm[student_col].nunique()
    n_questions = df_norm[question_col].nunique()
    n_dupes = df_norm.duplicated().sum()

    lines = [
        "FINAL OBSERVATIONS AND RECOMMENDATION",
        "=" * 70,
        "",
        "1. DATASET",
        f"   Total records    : {total:,}",
        f"   Unique students  : {n_students:,}",
        f"   Unique questions : {n_questions:,}",
        f"   Correct answers  : {n_correct:,}  ({100*n_correct/total:.1f}%)",
        f"   Incorrect answers: {n_incorrect:,}  ({100*n_incorrect/total:.1f}%)",
        f"   Exact duplicates : {n_dupes:,} rows (retained; represent repeated attempts)",
        "",
        "2. DATA QUALITY",
        f"   Missing values: see data_quality_report.txt",
        f"   No rows were silently deleted.",
        f"   All duplicate and repeated-ID rows were retained for analysis.",
        "",
        "3. PREPROCESSING",
        f"   Predictors used: {', '.join([field_map[k] for k in FEATURE_KEYS if k in field_map])}",
        f"   Excluded: {field_map.get('student_id','student_id')} (identifier), "
        f"{field_map.get('question_id','question_id')} (identifier, reduces memorization)",
        f"   Encoding : OneHotEncoder(handle_unknown='ignore') for all categorical columns",
        f"   Imputation: SimpleImputer(strategy='constant', fill_value='MISSING')",
        f"   Preprocessing fitted on TRAINING data only.",
        "",
        "4. TRAIN/TEST SPLIT",
        f"   Method          : {split_method}",
        f"   Train set size  : {len(X_train):,}",
        f"   Test set size   : {len(X_test):,}",
        f"   Random seed     : {RANDOM_STATE}",
        "",
        "5. MODEL RESULTS",
        f"   Positive class (what we predict) = incorrect answer: {positive_label!r}",
        "",
    ]

    if results_df is not None and not results_df.empty:
        for _, row in results_df.iterrows():
            lines.append(
                f"   {row['model']:<20}: Acc={row['accuracy']:.4f}  "
                f"Prec={row['precision_incorrect']:.4f}  "
                f"Rec={row['recall_incorrect']:.4f}  "
                f"F1={row['f1_incorrect']:.4f}"
            )

        best_f1_row = results_df.loc[results_df["f1_incorrect"].idxmax()]
        best_acc_row = results_df.loc[results_df["accuracy"].idxmax()]
        lines += [
            "",
            f"   Best by incorrect-answer F1-score : {best_f1_row['model']}  (F1={best_f1_row['f1_incorrect']:.4f})",
            f"   Best by accuracy                  : {best_acc_row['model']}  (Acc={best_acc_row['accuracy']:.4f})",
        ]
    else:
        lines.append("   Results not available (models did not complete).")

    lines += [
        "",
        "6. KNN WHAT-IF EXPERIMENT",
    ]
    if whatif_df is not None and not whatif_df.empty:
        for _, row in whatif_df.iterrows():
            lines.append(
                f"   K={int(row['n_neighbors'])}: Acc={row['accuracy']:.4f}  "
                f"Prec={row['precision_incorrect']:.4f}  "
                f"Rec={row['recall_incorrect']:.4f}  "
                f"F1={row['f1_incorrect']:.4f}"
            )
    else:
        lines.append("   Experiment not completed.")

    lines += [
        "",
        "7. MAIN FINDINGS AND LIMITATIONS",
        "   - This is an observational, metadata-only dataset.",
        "   - The model predicts answer correctness from categorical descriptors,",
        "     not from the content of questions or student responses.",
        "   - Group-aware splitting attempts to test on unseen students,",
        "     which gives a more realistic estimate of generalization.",
        "   - High accuracy can be misleading if the class distribution is imbalanced.",
        "   - Failing to identify incorrect answers (low recall) is costly in an",
        "     educational context: students in difficulty may go undetected.",
        "   - Falsely flagging correct answers (low precision) also has a cost:",
        "     unnecessary interventions waste educational resources.",
        "   - Predictions are probabilistic estimates, not diagnoses of student ability.",
        "",
        "8. RECOMMENDATION",
        "   (Based on actual measured results above.)",
        "   - Prefer the model with the best balance of incorrect-answer F1-score",
        "     and recall. In an educational context, recall for the positive class",
        "     (detecting actual incorrect answers) is often more important than accuracy.",
        "   - Random Forest is frequently robust on tabular categorical data,",
        "     but the choice should be confirmed from the actual output above.",
        "   - Do not use accuracy alone to select the best model.",
        "",
        "DISCLAIMER:",
        "   These predictions are statistical estimates based on limited metadata.",
        "   They do not diagnose, measure, or predict any individual student's",
        "   intelligence, learning ability, or true knowledge.",
        "",
        "Dataset: UCI MathE (ID 1031), CC BY 4.0",
        "Citation: Flamia Azevedo, B., Pacheco, M. F., Fernandes, F. P., & Pereira, A. I. (2024).",
        "          Assessing Mathematics Learning in Higher Education [Dataset].",
        "          UCI Machine Learning Repository. https://doi.org/10.34620/dadosipb/PW3OWY",
    ]

    text = "\n".join(lines)
    path = OUT_DIR / "final_observations.txt"
    path.write_text(text, encoding="utf-8")
    print(f"\n[INFO] Final observations saved: {path}")
    print("\n" + "=" * 60)
    print("  FINAL OBSERVATIONS SUMMARY")
    print("=" * 60)
    print(text)


# ===========================================================================
# 16. MAIN
# ===========================================================================

def main():
    print("\n" + "#" * 70)
    print("  MathE Learning Assessment -- Correctness Prediction")
    print("  UCI Dataset ID 1031 | CC BY 4.0")
    print("#" * 70)

    # --- Setup ---
    setup_output_dirs()

    # --- Find & load data ---
    csv_path = find_dataset(DATA_DIR)
    df_raw = load_dataset(csv_path)

    # --- Inspect ---
    inspect_columns(df_raw)

    # --- Normalize column names internally ---
    df_norm, col_mapping = normalize_columns(df_raw)

    # --- Validate columns ---
    field_map = validate_columns(df_norm, col_mapping)

    # --- Validate target ---
    target_col = field_map["type_of_answer"]
    positive_label, negative_label, label_description = validate_target(df_norm, target_col)

    # --- Data quality report ---
    df_norm = data_quality_report(df_norm, field_map, positive_label, negative_label, label_description)

    # --- EDA ---
    exploratory_analysis(df_norm, field_map, positive_label, negative_label)

    # --- Feature selection ---
    feature_cols = get_feature_cols(field_map)
    print(f"\n[INFO] Predictor columns selected: {feature_cols}")
    print(f"[INFO] Excluded from primary model: "
          f"{[field_map.get('student_id'), field_map.get('question_id')]}")

    # Save preprocessing examples before splitting
    save_preprocessing_examples(df_norm, feature_cols, target_col)

    # --- Build preprocessor ---
    preprocessor = build_preprocessor(feature_cols)

    # --- Train/test split ---
    student_col = field_map["student_id"]
    X_train, X_test, y_train, y_test, split_method = group_split(
        df_norm, feature_cols, target_col, student_col
    )

    # --- Fit preprocessor on train only, save feature summary ---
    # We fit once on training data for the summary; model pipelines re-fit internally
    preprocessor_for_summary = build_preprocessor(feature_cols)
    preprocessor_for_summary.fit(X_train, y_train)
    save_feature_summary(preprocessor_for_summary, feature_cols, X_train, X_test, field_map)

    # --- Train and evaluate all models ---
    results = train_and_evaluate_all(
        preprocessor, X_train, X_test, y_train, y_test,
        positive_label, negative_label
    )

    # --- Model comparison ---
    results_df = None
    if results:
        results_df = compare_and_visualize(results)
    else:
        print("[ERROR] No model results to compare.")

    # --- What-if experiment ---
    whatif_df = what_if_knn(
        preprocessor, X_train, X_test, y_train, y_test,
        positive_label, negative_label
    )

    # --- Final observations ---
    save_final_observations(
        df_norm, field_map, positive_label, negative_label,
        label_description, split_method,
        X_train, X_test,
        results_df, whatif_df
    )

    print("\n" + "#" * 70)
    print("  COMPLETED SUCCESSFULLY")
    print(f"  All outputs saved to: {OUT_DIR}")
    print("#" * 70)


if __name__ == "__main__":
    main()
