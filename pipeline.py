#!/usr/bin/env python3
"""
Instructor Effectiveness Modeling Pipeline (EdTech Context)
End-to-End Data Science & Machine Learning Implementation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
)
from sklearn.inspection import permutation_importance

# Configuration & Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "instructor_effectiveness_data.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("1. LOADING AND EXPLORING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Unique Instructors: {df['instructor_id'].nunique()}")
print(f"Unique Courses: {df['course_id'].nunique()}")
print(f"Unique Batches: {df['batch_id'].nunique()}")
print(f"Missing Values: {df.isnull().sum().sum()}")

# -------------------------------------------------------------
# STEP 1: EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("2. GENERATING EDA PLOTS")
print("=" * 70)

metric_cols = [
    'completion_rate', 'dropout_rate', 'avg_score_improvement', 'avg_quiz_score',
    'avg_watch_time', 'assignment_submission_rate', 'forum_activity_rate',
    'avg_feedback_score', 'feedback_response_rate'
]

# Plot 1: Feature Distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, col in enumerate(metric_cols):
    sns.histplot(df[col], kde=True, ax=axes[idx], color='#1f77b4', bins=25, stat="density", alpha=0.6)
    axes[idx].set_title(f"Distribution of {col}", fontsize=11, fontweight='bold', pad=8)
    axes[idx].set_xlabel("")
    axes[idx].set_ylabel("Density", fontsize=9)

plt.tight_layout()
fig_dist_path = os.path.join(FIG_DIR, "01_eda_distributions.png")
plt.savefig(fig_dist_path, dpi=300)
plt.close()
print(f"[+] Saved distribution plot: {fig_dist_path}")

# Plot 2: Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df[metric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix of Batch-Level Metrics", fontsize=14, fontweight='bold', pad=14)
plt.tight_layout()
fig_corr_path = os.path.join(FIG_DIR, "02_correlation_heatmap.png")
plt.savefig(fig_corr_path, dpi=300)
plt.close()
print(f"[+] Saved correlation heatmap: {fig_corr_path}")

# -------------------------------------------------------------
# STEP 2 & 3: BATCH-TO-INSTRUCTOR AGGREGATION & IES FORMULATION
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("3. AGGREGATING BATCH DATA & FORMULATING INSTRUCTOR EFFECTIVENESS SCORE (IES)")
print("=" * 70)

# Aggregation dictionary capturing Central Tendency, Consistency (Std), and Volume
agg_dict = {
    'batch_id': 'count',
    'course_id': 'nunique',
    'completion_rate': ['mean', 'std', 'median'],
    'avg_score_improvement': ['mean', 'std'],
    'avg_quiz_score': ['mean', 'std'],
    'dropout_rate': ['mean', 'std'],
    'avg_watch_time': ['mean', 'std'],
    'assignment_submission_rate': ['mean', 'std'],
    'forum_activity_rate': ['mean', 'std'],
    'avg_feedback_score': ['mean', 'std'],
    'feedback_response_rate': ['mean', 'std']
}

inst_agg = df.groupby('instructor_id').agg(agg_dict)
inst_agg.columns = ['_'.join(c).strip('_') for c in inst_agg.columns]
inst_agg.rename(columns={'batch_id_count': 'total_batches', 'course_id_nunique': 'courses_taught'}, inplace=True)
inst_agg.fillna(0, inplace=True)  # In case std is NaN for single batch

print(f"Aggregated Instructor DataFrame Shape: {inst_agg.shape}")
print("Batches per instructor stats:\n", inst_agg['total_batches'].describe())

# Normalization helper (MinMax to 0-1)
scaler = MinMaxScaler()
def min_max_norm(series):
    return (series - series.min()) / (series.max() - series.min())

# Compute Dimension Pillars
# 1. Outcomes Pillar (40%): Balanced combination of course completion & learning gain (score improvement)
norm_completion = min_max_norm(inst_agg['completion_rate_mean'])
norm_improvement = min_max_norm(inst_agg['avg_score_improvement_mean'])
outcomes_pillar = 0.5 * norm_completion + 0.5 * norm_improvement

# 2. Engagement Pillar (30%): Watch time (35%), assignments (35%), forum interaction (30%)
norm_watch = min_max_norm(inst_agg['avg_watch_time_mean'])
norm_assign = min_max_norm(inst_agg['assignment_submission_rate_mean'])
norm_forum = min_max_norm(inst_agg['forum_activity_rate_mean'])
engagement_pillar = 0.35 * norm_watch + 0.35 * norm_assign + 0.30 * norm_forum

# 3. Satisfaction Pillar (30%): Feedback rating weighted by response rate to penalize low-sample bias
norm_feedback = min_max_norm(inst_agg['avg_feedback_score_mean'])
response_weight = 0.5 + 0.5 * inst_agg['feedback_response_rate_mean']
satisfaction_pillar = norm_feedback * response_weight
# Normalize satisfaction pillar to 0-1
satisfaction_pillar = min_max_norm(satisfaction_pillar)

# Holistic Instructor Effectiveness Score (IES)
inst_agg['outcomes_pillar'] = outcomes_pillar
inst_agg['engagement_pillar'] = engagement_pillar
inst_agg['satisfaction_pillar'] = satisfaction_pillar

# Raw IES
raw_ies = 0.40 * outcomes_pillar + 0.30 * engagement_pillar + 0.30 * satisfaction_pillar

# Empirical Bayes / Sample Size Credibility Shrinkage:
# Instructors with few batches shrink slightly toward the population mean (M = 15 batches credibility weight)
pop_mean_ies = raw_ies.mean()
credibility_k = 5.0  # Shrinkage prior weight
shrinkage_weight = inst_agg['total_batches'] / (inst_agg['total_batches'] + credibility_k)
inst_agg['ies_score'] = shrinkage_weight * raw_ies + (1 - shrinkage_weight) * pop_mean_ies

# Effectiveness Tiers: Discretize into 3 Tiers (Low: 25%, Medium: 50%, High: 25%)
q25 = inst_agg['ies_score'].quantile(0.25)
q75 = inst_agg['ies_score'].quantile(0.75)

def assign_tier(score):
    if score >= q75:
        return 'High'
    elif score >= q25:
        return 'Medium'
    else:
        return 'Low'

inst_agg['effectiveness_tier'] = inst_agg['ies_score'].apply(assign_tier)
tier_order = ['Low', 'Medium', 'High']
inst_agg['tier_code'] = inst_agg['effectiveness_tier'].map({'Low': 0, 'Medium': 1, 'High': 2})

print("\nEffectiveness Score Distribution Summary:")
print(inst_agg['ies_score'].describe())
print(f"\nTier Cutoffs: Q25 (Low/Med) = {q25:.4f}, Q75 (Med/High) = {q75:.4f}")
print("\nTier Counts:\n", inst_agg['effectiveness_tier'].value_counts())

# Plot 3: IES Distribution & Tiers
plt.figure(figsize=(9, 5))
sns.histplot(inst_agg['ies_score'], kde=True, bins=20, color='#2b5c8f', edgecolor='black')
plt.axvline(q25, color='#e65100', linestyle='--', linewidth=2, label=f'Low/Med Cutoff ({q25:.3f})')
plt.axvline(q75, color='#2e7d32', linestyle='--', linewidth=2, label=f'Med/High Cutoff ({q75:.3f})')
plt.title("Distribution of Instructor Effectiveness Score (IES) with Tier Boundaries", fontsize=12, fontweight='bold')
plt.xlabel("Instructor Effectiveness Score (IES)")
plt.ylabel("Instructor Count")
plt.legend(frameon=True)
plt.tight_layout()
fig_ies_path = os.path.join(FIG_DIR, "03_ies_distribution.png")
plt.savefig(fig_ies_path, dpi=300)
plt.close()
print(f"[+] Saved IES score distribution plot: {fig_ies_path}")

# -------------------------------------------------------------
# STEP 4: MACHINE LEARNING MODELING
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("4. MACHINE LEARNING MODEL TRAINING & CROSS-VALIDATION")
print("=" * 70)

# Feature selection: Select observable operational batch summary features (excluding raw target pillars to avoid circular leakage)
feature_cols = [
    'completion_rate_mean', 'completion_rate_std',
    'avg_score_improvement_mean', 'avg_score_improvement_std',
    'avg_quiz_score_mean', 'avg_quiz_score_std',
    'dropout_rate_mean', 'dropout_rate_std',
    'avg_watch_time_mean', 'avg_watch_time_std',
    'assignment_submission_rate_mean', 'assignment_submission_rate_std',
    'forum_activity_rate_mean', 'forum_activity_rate_std',
    'avg_feedback_score_mean', 'avg_feedback_score_std',
    'feedback_response_rate_mean', 'feedback_response_rate_std',
    'total_batches', 'courses_taught'
]

X = inst_agg[feature_cols].copy()
y = inst_agg['tier_code'].copy()

# Stratified 5-Fold Cross Validation Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models to evaluate
models = {
    'Dummy (Baseline)': DummyClassifier(strategy='stratified', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42)
}

# Cross-Validation Evaluation Loop
cv_results = {}
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

for name, model in models.items():
    if name == 'Logistic Regression':
        # Scale features for logistic regression pipeline
        std_scaler = StandardScaler()
        X_scaled = std_scaler.fit_transform(X)
        scores = cross_validate(model, X_scaled, y, cv=skf, scoring=scoring)
    else:
        scores = cross_validate(model, X, y, cv=skf, scoring=scoring)
        
    cv_results[name] = {
        'Accuracy': (scores['test_accuracy'].mean(), scores['test_accuracy'].std()),
        'Macro F1': (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std()),
        'Precision': (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std()),
        'Recall': (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std())
    }

print("\nCross-Validation Performance Benchmarks (Mean ± Std over 5 Folds):")
bench_df = pd.DataFrame({
    model_name: {metric: f"{vals[0]:.3f} ± {vals[1]:.3f}" for metric, vals in metrics.items()}
    for model_name, metrics in cv_results.items()
}).T
print(bench_df.to_string())

# Plot 4: Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
model_names = list(models.keys())
metrics_to_plot = ['Accuracy', 'Macro F1', 'Precision', 'Recall']
x = np.arange(len(model_names))
width = 0.18

colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
for idx, metric in enumerate(metrics_to_plot):
    means = [cv_results[m][metric][0] for m in model_names]
    stds = [cv_results[m][metric][1] for m in model_names]
    ax.bar(x + idx * width, means, width, yerr=stds, capsize=4, label=metric, color=colors[idx], alpha=0.85)

ax.set_ylabel('Score (0 - 1.0)', fontsize=11)
ax.set_title('5-Fold Cross-Validation Model Benchmark Across Evaluation Metrics', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, fontsize=10)
ax.legend(loc='lower right', frameon=True)
ax.set_ylim(0, 1.05)
plt.tight_layout()
fig_comp_path = os.path.join(FIG_DIR, "04_model_comparison.png")
plt.savefig(fig_comp_path, dpi=300)
plt.close()
print(f"[+] Saved model comparison plot: {fig_comp_path}")

# -------------------------------------------------------------
# STEP 5: DETAILED EVALUATION, CONFUSION MATRIX & ROC-AUC
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("5. TRAIN-TEST EVALUATION & CONFUSION MATRICES")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit models on Train Split
fitted_models = {}
# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
fitted_models['Logistic Regression'] = (lr, X_test_scaled)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
fitted_models['Random Forest'] = (rf, X_test)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42)
gb.fit(X_train, y_train)
fitted_models['Gradient Boosting'] = (gb, X_test)

# Plot 5: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
class_labels = ['Low', 'Medium', 'High']

for idx, (m_name, (m_obj, test_feat)) in enumerate(fitted_models.items()):
    preds = m_obj.predict(test_feat)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=class_labels, yticklabels=class_labels, cbar=False)
    axes[idx].set_title(f"{m_name}\nAcc: {accuracy_score(y_test, preds):.2f} | Macro F1: {f1_score(y_test, preds, average='macro'):.2f}", fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("Predicted Tier")
    axes[idx].set_ylabel("True Tier")

plt.tight_layout()
fig_cm_path = os.path.join(FIG_DIR, "05_confusion_matrices.png")
plt.savefig(fig_cm_path, dpi=300)
plt.close()
print(f"[+] Saved confusion matrices: {fig_cm_path}")

# Plot 6: Multiclass One-vs-Rest ROC Curves for Random Forest & Gradient Boosting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax_idx, (m_name, (m_obj, test_feat)) in enumerate([('Random Forest', (rf, X_test)), ('Gradient Boosting', (gb, X_test))]):
    probs = m_obj.predict_proba(test_feat)
    y_test_bin = pd.get_dummies(y_test).values
    
    for c_idx, c_label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, c_idx], probs[:, c_idx])
        roc_val = auc(fpr, tpr)
        axes[ax_idx].plot(fpr, tpr, lw=2, label=f"Class {c_label} (AUC = {roc_val:.2f})")
        
    axes[ax_idx].plot([0, 1], [0, 1], 'k--', lw=1.2)
    axes[ax_idx].set_title(f"ROC Curves - {m_name}", fontsize=12, fontweight='bold')
    axes[ax_idx].set_xlabel("False Positive Rate")
    axes[ax_idx].set_ylabel("True Positive Rate")
    axes[ax_idx].legend(loc="lower right")

plt.tight_layout()
fig_roc_path = os.path.join(FIG_DIR, "06_roc_curves.png")
plt.savefig(fig_roc_path, dpi=300)
plt.close()
print(f"[+] Saved ROC curves: {fig_roc_path}")

# -------------------------------------------------------------
# STEP 6: FEATURE IMPORTANCE & INTERPRETABILITY
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("6. FEATURE IMPORTANCE & INTERPRETABILITY")
print("=" * 70)

# Random Forest Gini Importance
rf_importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

# Permutation Importance on Test Set
perm_res = permutation_importance(rf, X_test, y_test, n_repeats=15, random_state=42)
perm_importances = pd.Series(perm_res.importances_mean, index=feature_cols).sort_values(ascending=False)

# Plot 7: Top Feature Importances (MDI vs Permutation)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

rf_importances.head(10).plot(kind='barh', ax=axes[0], color='#2b5c8f', edgecolor='black')
axes[0].set_title("Top 10 Features (Random Forest MDI Importance)", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Mean Decrease in Impurity")
axes[0].invert_yaxis()

perm_importances.head(10).plot(kind='barh', ax=axes[1], color='#e65100', edgecolor='black')
axes[1].set_title("Top 10 Features (Permutation Importance on Test Set)", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Mean Accuracy Drop when Shuffled")
axes[1].invert_yaxis()

plt.tight_layout()
fig_imp_path = os.path.join(FIG_DIR, "07_feature_importance.png")
plt.savefig(fig_imp_path, dpi=300)
plt.close()
print(f"[+] Saved feature importance plot: {fig_imp_path}")

# Save detailed model results and metrics for reporting
results_summary = {
    "num_instructors": int(inst_agg.shape[0]),
    "num_batches": int(df.shape[0]),
    "tier_thresholds": {"Q25": float(q25), "Q75": float(q75)},
    "tier_distribution": inst_agg['effectiveness_tier'].value_counts().to_dict(),
    "top_5_features_rf": rf_importances.head(5).to_dict(),
    "top_5_features_perm": perm_importances.head(5).to_dict(),
    "test_classification_report_rf": classification_report(y_test, rf.predict(X_test), target_names=class_labels, output_dict=True)
}

with open(os.path.join(BASE_DIR, "pipeline_results.json"), "w") as f:
    json.dump(results_summary, f, indent=2)

print("\n" + "=" * 70)
print("PIPELINE EXECUTION COMPLETE! ALL ARTIFACTS AND PLOTS GENERATED SUCCESSFULLY.")
print("=" * 70)
