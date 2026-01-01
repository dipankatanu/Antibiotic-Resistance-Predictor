# Antibiotic Resistance Predictor

This repository contains a machine learning pipeline for identifying candidate genetic markers associated with ciprofloxacin resistance in bacterial isolates. The workflow combines predictive modeling, feature attribution, and biological association metrics to prioritize genes for mechanistic follow-up.

---

## Objectives

1. Train and evaluate baseline machine learning models to classify isolates as resistant or sensitive.
2. Use SHAP values to identify genes that contribute most to model predictions.
3. Supplement model findings with Odds Ratio analysis to assess biological association strength.
4. Provide a reproducible analysis pipeline that serves as an entry point for AMR gene discovery.

---

## Method Overview

The analytical workflow includes:

- Data loading and preprocessing  
  Accessory gene presence/absence (binary matrix) and phenotype metadata.

- Baseline model training  
  Random Forest, Logistic Regression, and Gradient Boosting Classifier.

- Hyperparameter optimization  
  GridSearchCV search on Random Forest.

- Machine learning explainability  
  SHAP summary plots for global and per-feature interpretability.

- Biological association metrics  
  Odds ratio calculations on top SHAP-ranked genes.

This creates a two-dimensional evidence score:
- Model-based contribution (SHAP)
- Biological enrichment (Odds Ratio)

Both are used to determine candidate markers.

---

## Data directory note:
`AccessoryGene.csv` is compressed due to size and can be unzipped before running the pipeline.

---

## Usage

Run the main script:
python antibiotic_resistance_predictor.py

