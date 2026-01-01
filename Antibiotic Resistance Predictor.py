#!/usr/bin/env python
# coding: utf-8

# # Project Overview & Data Description
# #### This project focuses on identifying genomic markers of antibiotic resistance using machine learning. By analyzing the Pangenome (the total set of genes within a species), we aim to predict whether a specific bacterial isolate is Resistant (R) or Susceptible (S) to various antibiotics, specifically focusing on Ciprofloxacin (CIP).
# 
# # Data Description
# #### The analysis utilizes two primary datasets:
# 
# #### AccessoryGene.csv: A binary matrix representing the Genotype. Columns represent accessory genes (extra DNA picked up via horizontal gene transfer), and rows represent individual bacterial isolates. A value of 1 indicates gene presence; 0 indicates absence.
# 
# #### Metadata.csv: The Phenotype data. It contains lab-verified results (Antimicrobial Susceptibility Testing) for several antibiotics including Ciprofloxacin (CIP), Ampicillin (AMP), and Gentamicin (GEN).
# 
# ### They are taken from https://github.com/Lucy-Moctezuma/ML-Tutorial-for-Antibiotic-Resistance-Predictions-for-E.-Coli
# 
# # Methodology
# #### We employ an Explainable AI (XAI) approach:
# #### Data Integration: Merging genomic features with phenotypic labels.
# #### Supervised Learning: Training a Random Forest Classifier to learn patterns between gene presence and drug resistance.
# #### Optimization: Using GridSearchCV and K-Fold Cross-Validation to find the most robust model parameters.
# #### Biological Interpretation: Using SHAP (SHapley Additive exPlanations) and Odds Ratio (OR) analysis to validate the biological significance of the genes identified by the AI.

# In[1]:


# Environment Setup & Data Loading

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "data"  # put Metadata.csv + AccessoryGene.csv into a data/ folder

metadata = pd.read_csv(data_path / "Metadata.csv")
gene_data = pd.read_csv(data_path / "AccessoryGene.csv")

# Merging Genotype and Phenotype
merged_df = pd.merge(gene_data, metadata[['Isolate', 'CIP']], left_on='Unnamed: 0', right_on='Isolate')
merged_df['target'] = merged_df['CIP'].map({'R': 1, 'S': 0})
merged_df = merged_df.dropna(subset=['target'])

print(f"Merge Successful! Total bacterial isolates: {len(merged_df)}")


# In[2]:


# Initial Model Training & Feature Importance
# Define Features and Target
X = merged_df.drop(columns=['Unnamed: 0', 'Isolate', 'CIP', 'target'])
y = merged_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Visualization of Top 10 Genes
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Gene': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Gene', data=feature_importance_df, hue='Gene', legend=False, palette='viridis')
plt.title('Top 10 Genes Predicting Ciprofloxacin Resistance')
plt.savefig("Top10_Ciprofloxacin_Resistance_Genes.png", dpi=300, bbox_inches='tight')
plt.show()


# In[3]:


# Model Benchmarking & Hyperparameter Tuning

# Model Benchmarking
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, m in models.items():
    start = time.time()
    cv_res = cross_val_score(m, X, y, cv=kf)
    print(f"{name} | Accuracy: {cv_res.mean():.4f} | Time: {time.time()-start:.2f}s")

# Hyperparameter Optimization
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [10, 30, None],
    'criterion': ['gini', 'entropy']
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best Params: {grid_search.best_params_}")


# In[10]:


# --- 1. Use the BEST model found by GridSearchCV ---
best_model = grid_search.best_estimator_

# --- 2. Calculate SHAP Values ---
explainer = shap.TreeExplainer(best_model)
# We use X_test to see how the best model interprets unseen data
shap_values = explainer.shap_values(X_test, check_additivity=False)

# --- 3. Robust Shape Handling ---
# This ensures shap_data is always a 2D matrix (samples x features)
if isinstance(shap_values, list):
    # Standard format for older SHAP/Sklearn: List of arrays
    shap_data = shap_values[1] 
elif len(shap_values.shape) == 3:
    # Newer SHAP format: (samples, features, 2)
    shap_data = shap_values[:, :, 1]
else:
    shap_data = shap_values

# --- 4. Plotting ---
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_data, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance: Optimized Best Model")
plt.show()

# --- 5. Fixed Biological Odds Ratio Calculation ---
# We force top_indices to be a flat 1D array to avoid the Indexing Error
top_indices = np.argsort(np.abs(shap_data).mean(0))[-10:]
top_indices = np.array(top_indices).flatten() # Ensure 1D

top_genes = X_test.columns[top_indices]

print("\n--- Biological Odds Ratio Analysis ---")
for gene in top_genes:
    # Use the original merged_df to get the full population stats
    table = pd.crosstab(merged_df[gene], merged_df['target'])
    try:
        # a: Present & Resistant, b: Present & Susceptible
        # c: Absent & Resistant, d: Absent & Susceptible
        a, b = table.iloc[1, 1], table.iloc[1, 0]
        c, d = table.iloc[0, 1], table.iloc[0, 0]
        
        # OR = (ad) / (bc)
        or_val = (a * d) / (b * c) if (b * c) != 0 else float('inf')
        print(f"Gene: {gene:12} | Odds Ratio: {or_val:.2f}")
    except: 
        continue


# In[ ]:




