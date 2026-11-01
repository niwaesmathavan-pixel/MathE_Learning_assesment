# MathE Learning Assessment — Correctness Prediction

A complete undergraduate machine learning project that analyses student mathematics
answer patterns and predicts whether an answer is correct using the UCI MathE dataset.

---

## Problem Statement

Given structured metadata about student answers on the **MathE platform** (a mathematics
e-learning system used across European universities), predict whether an answer is
**correct** or **incorrect** using available categorical descriptors such as question level,
topic, subtopic, keywords, and student country.

This is a binary classification problem solved with four traditional machine learning algorithms.

---

## Objectives

1. Load, validate, and explore the UCI MathE dataset.
2. Analyse student answer patterns across topics, levels, and countries.
3. Build and compare four classifiers: Decision Tree, KNN, SVM, Random Forest.
4. Evaluate models rigorously using group-aware train/test splits.
5. Perform a what-if experiment (KNN with varying K).
6. Provide evidence-based recommendations for an educational context.

---

## Dataset

| Property | Details |
|---|---|
| **Name** | Assessing Mathematics Learning in Higher Education |
| **Platform** | MathE |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/1031) |
| **Dataset ID** | 1031 |
| **DOI** | https://doi.org/10.34620/dadosipb/PW3OWY |
| **License** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Reported size** | 9,546 answer records |
| **Collection period** | February 2019 – December 2023 |

### Dataset Citation

> Flamia Azevedo, B., Pacheco, M. F., Fernandes, F. P., & Pereira, A. I. (2024).
> *Assessing Mathematics Learning in Higher Education* [Dataset].
> UCI Machine Learning Repository. https://doi.org/10.34620/dadosipb/PW3OWY

**License notice:** The dataset is released under CC BY 4.0. You must give appropriate
credit when using or redistributing it.

### Eight Documented Fields

| # | Field Name | Description |
|---|---|---|
| 1 | Student ID | Unique student identifier (excluded from model) |
| 2 | Student Country | Country of the student |
| 3 | Question ID | Unique question identifier (excluded from primary model) |
| 4 | Type of Answer | **Target** — correct or incorrect |
| 5 | Question Level | Difficulty level (e.g., basic, advanced) |
| 6 | Topic | Mathematics topic area |
| 7 | Subtopic | Sub-area within the topic |
| 8 | Keywords | Categorical tags describing the question |

### What This Dataset Can and Cannot Do

✅ **Supported use cases:**
- Analysis of answer correctness patterns by topic, level, country.
- Binary classification: predict correct vs. incorrect from metadata.
- Evaluation of educational patterns across student groups.

❌ **Not supported:**
- The dataset does **not** contain full question text, student written answers,
  worked solutions, or teaching explanations.
- It cannot be used to generate worksheets, question content, or answer keys.
- It cannot prove causal relationships (e.g., that a topic *causes* failure).
- Predictions are probabilistic estimates, not diagnoses of student ability.

---

## Project Structure

```
MathE_Learning_Assessment/
│
├── data/
│   └── MathE dataset (4).csv          ← Place downloaded CSV here
│
├── outputs/
│   ├── dataset_summary.txt
│   ├── dataset_class_distribution.csv
│   ├── data_quality_report.txt
│   ├── preprocessing_examples.csv
│   ├── model_comparison.csv
│   ├── feature_summary.txt
│   ├── final_observations.txt
│   ├── classification_reports/
│   ├── confusion_matrices/
│   ├── visualizations/
│   └── what_if_experiment/
│
├── mathe_assessment.py                ← Main script
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 1.5.0 | Data loading, cleaning, analysis |
| `numpy` | ≥ 1.23.0 | Numerical operations |
| `scikit-learn` | ≥ 1.2.0 | ML models, preprocessing, evaluation |
| `matplotlib` | ≥ 3.6.0 | Visualizations, confusion matrices |
| `seaborn` | ≥ 0.12.0 | Statistical plots (EDA) |

---

## Installation and Setup (Windows)

### 1. Download the Dataset

1. Visit: https://archive.ics.uci.edu/dataset/1031
2. Click **Download** to get the ZIP file.
3. Extract the ZIP and locate the main CSV file (e.g., `MathE dataset (4).csv`).
4. Copy the CSV into the `data/` folder of this project.

Do **not** rename or modify the original file.

### 2. Create a Virtual Environment

Open **PowerShell** in the `MathE_Learning_Assessment/` folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Or using **Command Prompt**:

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

> **Note on PowerShell execution policy:** If you see a security error when activating,
> run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once.

### 3. Run the Project

```powershell
python mathe_assessment.py
```

---

## How It Works

### Exploratory Data Analysis

The script performs full EDA before training:
- Overall class distribution (correct vs. incorrect).
- Correctness rate by question level, topic, subtopic, and country.
- Top topics and subtopics by answer volume.
- Missing value and duplicate analysis.
- All charts are saved to `outputs/visualizations/`.

### Preprocessing and One-Hot Encoding

The selected predictors (`Student Country`, `Question Level`, `Topic`,
`Subtopic`, `Keywords`) are all **categorical** — they have no inherent numeric order.

Machine learning algorithms require numbers, so we use **One-Hot Encoding (OHE)**:
each unique category value becomes its own binary (0/1) column.

Example: If `Topic` has values `{Algebra, Calculus, Statistics}`, OHE produces:

| Topic_Algebra | Topic_Calculus | Topic_Statistics |
|---|---|---|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

`handle_unknown='ignore'` safely handles categories in the test set that weren't
seen during training (maps them to all-zero rows).

> ⚠️ **TF-IDF is not used.** TF-IDF is designed for full natural-language text documents.
> The Keywords column contains short categorical tags, not prose text.

**`Student ID` and `Question ID` are excluded from the primary model:**
- They are identifiers, not meaningful predictors.
- Including them would allow the model to memorize individual students or questions
  instead of learning generalizable patterns.

### Four Classifiers

| Model | How It Works | Strengths | Weaknesses |
|---|---|---|---|
| **Decision Tree** | Learns a hierarchy of if/else rules | Interpretable, fast | Prone to overfitting |
| **KNN** | Votes using K nearest training points | Simple, no training phase | Slow on large datasets, sensitive to high dimensions |
| **SVM (Linear)** | Finds a hyperplane maximising margin | Effective in high dimensions | Requires careful C tuning |
| **Random Forest** | Ensemble of Decision Trees | Robust, less overfitting | Less interpretable |

### Train/Test Split Strategy

**Primary method: `GroupShuffleSplit` on Student ID**

Multiple answer records belong to the same students. A simple random row split would
place some records from the same student in both training and test data — this inflates
performance because the model has seen the same student before.

Group-aware splitting ensures that **all records for a given student** appear in either
training or test, not both. This gives a more realistic estimate of how well the model
generalises to entirely new students.

**Fallback:** If group splitting creates a severe class imbalance, `StratifiedShuffleSplit`
is used and clearly documented.

### Evaluation Metrics

The **positive class** is the **incorrect answer** — what we most want to detect.

| Metric | Definition |
|---|---|
| **Accuracy** | Overall proportion of correctly predicted answers |
| **Precision** | Of answers predicted incorrect, how many were actually incorrect |
| **Recall** | Of all actually incorrect answers, how many were detected |
| **F1-score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Visual breakdown of TP, TN, FP, FN predictions |

In an educational context, **recall** is especially important: failing to detect
incorrect answers means students in difficulty may not receive timely support.

### What-If Experiment

KNN is run with `n_neighbors ∈ {3, 5, 7, 9}` on the same split and features.
Results show how the choice of K affects all four metrics. Charts and CSV are
saved to `outputs/what_if_experiment/`.

---

## Output Files

| File | Description |
|---|---|
| `outputs/dataset_summary.txt` | Dataset shape, columns, citation |
| `outputs/dataset_class_distribution.csv` | Correct/incorrect counts |
| `outputs/data_quality_report.txt` | Full quality analysis |
| `outputs/preprocessing_examples.csv` | Sample original feature values |
| `outputs/feature_summary.txt` | Encoded feature counts and names |
| `outputs/model_comparison.csv` | All four models' metrics |
| `outputs/final_observations.txt` | Summary and recommendation |
| `outputs/classification_reports/*.txt` | Per-model classification reports |
| `outputs/confusion_matrices/*.png` | Per-model confusion matrix images |
| `outputs/visualizations/*.png` | EDA and comparison charts |
| `outputs/what_if_experiment/knn_what_if_results.csv` | KNN K-sweep results |
| `outputs/what_if_experiment/knn_what_if_chart.png` | KNN K-sweep chart |

---

## How to Interpret Results

- **A high accuracy but low recall for incorrect answers** means the model rarely
  flags incorrect answers — it mostly predicts "correct" and happens to be right
  because there may be more correct answers overall.
- **Random Forest** often performs well on tabular categorical data but should
  be compared against actual measured results from your run.
- The **final_observations.txt** file contains a recommendation based on your
  actual experimental output.

---

## Troubleshooting Common Errors

| Error | Cause | Fix |
|---|---|---|
| `DATASET NOT FOUND` | No CSV in `data/` | Download from UCI, place CSV in `data/` |
| `UnicodeDecodeError` | CSV encoding issue | The script tries utf-8, latin-1, cp1252 automatically |
| `Could not match required columns` | Column names differ in your CSV | Edit `EXPECTED_COLUMNS` in `mathe_assessment.py` |
| `ModuleNotFoundError` | Missing dependency | Run `pip install -r requirements.txt` |
| `MemoryError` | Dataset too large for RAM | Reduce `n_estimators` in RandomForest or use a smaller subset |
| LinearSVC convergence warning | `max_iter` too low | Increase `max_iter` in `get_models()` |

---

## Limitations

- The model is trained on historical assessment metadata only.
- It cannot access question content, student explanations, or teacher judgements.
- Predictions are statistical estimates — they do not measure a student's true ability.
- Country-level and topic-level patterns are observational; causation cannot be inferred.
- Group-aware splitting reduces (but does not eliminate) all sources of data leakage.

---

## Conclusion

*(To be completed after running the script and reviewing `outputs/final_observations.txt`.)*

After running `python mathe_assessment.py`, open the files in `outputs/` to see actual
measured accuracy, precision, recall, and F1-scores for all four classifiers, plus the
KNN what-if experiment results. Base your written conclusion on those figures.

---

## Academic Integrity Note

All experimental results in this project are computed from the real dataset at runtime.
No results have been hard-coded or fabricated. Charts, confusion matrices, and metrics
reflect actual model behaviour on the held-out test set.
