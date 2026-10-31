# Credit Risk Assessment in MSME Lending: How Banks Can Improve Without Raising NPAs

## Project overview

This repository contains the Python implementation for an MSc research project investigating the use of machine learning to predict loan default and support credit risk assessment.

The project develops and evaluates an analytical pipeline using the Kaggle Loan Default Dataset. The dataset contains general loan-default records; it is used as a proxy for investigating credit-risk modelling and is **not an MSME-specific lending dataset**. The findings should therefore not be interpreted as direct evidence of model performance on actual MSME loan portfolios.

The pipeline covers:
- Data inspection and exploratory data analysis (EDA).
- Data preprocessing and categorical encoding.
- Feature engineering, including `LoanToIncomeRatio`.
- Comparison of classification models.
- Class-imbalance experiments using class weighting and SMOTE.
- Validation-based decision-threshold selection.
- Probability calibration.
- Model evaluation using classification and probability-based metrics.
- Model interpretation using SHAP.
- Subgroup performance analysis.

## Dataset

**File:** `Loan_default.csv`

**Source:** [Kaggle – Loan Default Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default)

The analysed dataset contains 255,347 records and 18 columns, including the binary target variable `Default`.

Target distribution:
- Non-default (`Default = 0`): 225,694 records (88.39%).
- Default (`Default = 1`): 29,653 records (11.61%).

## Repository contents

```text
credit-risk-assessment/
├── Final_Ajay_Sheffield_final_.ipynb
├── Loan_default.csv
├── README.md
└── requirements.txt
```

- `Final_Ajay_Sheffield_final_.ipynb` — main notebook containing data analysis, model development and evaluation.
- `Loan_default.csv` — dataset used by the notebook.
- `README.md` — project overview and instructions.
- `requirements.txt` — Python package dependencies.

## Models and methods

The notebook compares:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. LightGBM

The experiments also investigate class weighting, SMOTE, threshold optimisation and sigmoid probability calibration. SHAP is used to examine the contribution of input features to predictions.

## Reported final-model results

The calibrated Logistic Regression pipeline achieved the following results on the test split at the selected decision threshold of 0.18:

| Metric | Result |
|---|---:|
| Accuracy | 0.8069 |
| Precision | 0.3009 |
| Recall | 0.5011 |
| F1-score | 0.3760 |
| ROC-AUC | 0.7616 |
| PR-AUC | 0.3362 |
| Brier score | 0.0903 |
| Log loss | 0.3096 |

These results are specific to the dataset, split and experimental procedure used in the notebook. The test set was also used during earlier model comparisons; it should therefore not be described as a completely untouched independent final evaluation set.

The model is an analytical decision-support prototype. It does not make lending decisions, establish causation, or demonstrate a reduction in non-performing assets (NPAs).

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Ajaykanna-131/credit-risk-assessment.git
cd credit-risk-assessment
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the notebook

Open `Final_Ajay_Sheffield_final_.ipynb` in Jupyter Notebook, JupyterLab or Google Colab.

The notebook expects `Loan_default.csv` to be available in its working directory. If using Google Colab, upload the CSV or mount the storage location and update the dataset path in the notebook if necessary.

Run the notebook cells in order. Model training and SHAP analysis may take longer depending on the available hardware.

## Reproducibility

The implementation uses fixed random-state settings in the documented data-splitting and modelling steps where specified. For reproducibility:
- Use the same dataset version.
- Install the dependencies listed in `requirements.txt`.
- Run the notebook cells in sequence.
- Retain the documented random-state values and data-splitting procedure.

Package versions are not pinned in the requirements file. Small differences may therefore occur across software environments.

## Limitations

- The dataset is a general loan-default dataset and is not specifically identified as an MSME lending dataset.
- Predictive performance is moderate, and precision indicates that some borrowers predicted as potential defaulters are not actual defaulters.
- SHAP explanations describe the behaviour of the fitted model and do not establish causal relationships.
- Subgroup results require careful interpretation and further fairness assessment.
- The notebook presents an experimental research pipeline, not a validated production banking system.

## Author and academic context

This repository accompanies an MSc research project on explainable machine learning for credit risk assessment.

**Repository:** [https://github.com/Ajaykanna-131/credit-risk-assessment](https://github.com/Ajaykanna-131/credit-risk-assessment)
