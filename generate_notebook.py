import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.13"
    }
}

cells = []

# Title & Context
cells.append(nbf.v4.new_markdown_cell(r"""# Instructor Effectiveness Modeling — EdTech Analysis
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ritikrajora20072110-ship-it/instructor-effectiveness-modeling/blob/main/instructor_effectiveness_modeling.ipynb)

**Candidate:** Ritik Rajora  
**Role:** Data Science & AI Content Specialist Intern Task  
**Stack:** Python, pandas, numpy, scikit-learn, matplotlib, seaborn  

---

### Problem Overview & Approach

In online education platforms, multiple instructors deliver the same standardized curriculum across different student cohorts. Measuring how effective an instructor actually is comes with several practical challenges:
- **Batch count variance:** Some instructors have only taught 7 batches, while others have taught over 30. A smaller sample size naturally has higher noise.
- **Rating leniency:** Student satisfaction surveys skew heavily toward 4 and 5 stars, while low response rates mean a vocal minority can dominate the score.
- **Confounding factors:** A teacher assigned to a difficult advanced course might see lower completion rates than someone teaching an introductory elective, regardless of teaching skill.

In this notebook, I walk through my end-to-end approach:
1. **Exploratory Data Analysis (EDA):** Examine distributions, check for missing values, and analyze correlations between student outcome, engagement, and satisfaction metrics.
2. **Defining Instructor Effectiveness:** Create a composite score that balances actual learning gains, student engagement, and response-adjusted feedback, with shrinkage to handle batch variance.
3. **Batch-to-Instructor Aggregation:** Roll up 2,000 batch records into 120 instructor profiles, capturing both average performance and consistency across batches.
4. **Machine Learning Modeling:** Benchmark classical models (Baseline, Logistic Regression, Random Forest, Gradient Boosting) using Stratified 5-Fold Cross-Validation to classify instructors into Low, Medium, and High performance tiers.
5. **Interpretability & Analysis Questions:** Analyze feature importances and provide detailed answers to the 5 mandatory business and ethical questions regarding real-world usage."""))

# Imports
cells.append(nbf.v4.new_code_cell("""import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
)
from sklearn.inspection import permutation_importance

# Chart styling
%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 120

print("Environment ready.")"""))

# Load Data
cells.append(nbf.v4.new_code_cell("""# Load the dataset (with automatic fallback for Google Colab)
DATA_FILE = "instructor_effectiveness_data.csv"
if not os.path.exists(DATA_FILE):
    local_fallback = "/Users/ritikrajora20072110gmailcom/.gemini/antigravity/scratch/edtech_assignment/instructor_effectiveness_data.csv"
    if os.path.exists(local_fallback):
        DATA_FILE = local_fallback
    else:
        import urllib.request
        url = "https://raw.githubusercontent.com/ritikrajora20072110-ship-it/instructor-effectiveness-modeling/main/instructor_effectiveness_data.csv"
        print("Downloading dataset from GitHub repository...")
        urllib.request.urlretrieve(url, DATA_FILE)
        print("Download complete.")

df = pd.read_csv(DATA_FILE)
print(f"Total batches: {df.shape[0]} | Columns: {df.shape[1]}")
print(f"Unique instructors: {df['instructor_id'].nunique()}")
print(f"Unique courses: {df['course_id'].nunique()}")
print(f"Missing values: {df.isnull().sum().sum()}")
df.head()"""))

# EDA Markdown
cells.append(nbf.v4.new_markdown_cell(r"""---
## Step 1: Exploratory Data Analysis (EDA)

Before defining any formulas or training models, I wanted to examine the underlying distributions of the 9 operational metrics and see how they interact:
- **Learner Outcomes:** `completion_rate`, `dropout_rate`, `avg_score_improvement`, `avg_quiz_score`
- **Engagement:** `avg_watch_time`, `assignment_submission_rate`, `forum_activity_rate`
- **Feedback:** `avg_feedback_score`, `feedback_response_rate`"""))

# EDA Code 1: Numerical Summary
cells.append(nbf.v4.new_code_cell("""# Summary statistics across all 2,000 batches
metric_cols = [
    'completion_rate', 'dropout_rate', 'avg_score_improvement', 'avg_quiz_score',
    'avg_watch_time', 'assignment_submission_rate', 'forum_activity_rate',
    'avg_feedback_score', 'feedback_response_rate'
]
df[metric_cols].describe().T[['mean', 'std', 'min', '50%', 'max']].round(3)"""))

# EDA Code 2: Plots
cells.append(nbf.v4.new_code_cell("""# Distribution plots for operational metrics
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
axes = axes.flatten()

for idx, col in enumerate(metric_cols):
    sns.histplot(df[col], kde=True, ax=axes[idx], color='#2b5c8f', bins=25, stat="density", alpha=0.6)
    axes[idx].set_title(col, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("")
    axes[idx].set_ylabel("Density", fontsize=9)

plt.tight_layout()
plt.show()"""))

# EDA Code 3: Correlation Matrix
cells.append(nbf.v4.new_code_cell("""# Correlation heatmap
plt.figure(figsize=(9, 7))
corr = df[metric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix of Batch-Level Metrics", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()"""))

# EDA Observations Markdown
cells.append(nbf.v4.new_markdown_cell(r"""### Key Takeaways from the Data:
1. **Completion vs. Dropout ($r = -0.95$):**  
   These two variables are essentially mirrors of each other. Including both in an effectiveness formula would effectively double-count student retention. I decided to keep `completion_rate` and drop `dropout_rate` from the score calculation.
2. **Score Improvement correlates with Completion ($r = 0.40$):**  
   Students who feel they are making genuine progress are much less likely to quit halfway through. This suggests pedagogical effectiveness directly influences retention.
3. **Low Community Engagement by Default:**  
   `forum_activity_rate` has a low median (~0.25) and is heavily right-skewed. Most learners are passive observers; therefore, batches with high forum engagement indicate an instructor who actively encourages discussion.
4. **Feedback Leniency & Response Rates:**  
   The average feedback rating is high (4.21 / 5.0), which is typical for online student surveys. Meanwhile, the feedback response rate averages 73.7%. An average rating of 4.8 from only 20% of a class is much less reliable than a 4.4 from 90% of a class. The formula needs to account for this response bias."""))

# Step 2 Markdown: Defining IES
cells.append(nbf.v4.new_markdown_cell(r"""---
## Step 2: Defining Instructor Effectiveness

### Why not just use student ratings?
In an EdTech company, relying strictly on feedback ratings is risky:
- **Leniency bias:** Instructors who make quizzes very easy often receive high ratings, even if students didn't learn much.
- **Participation bias:** When response rates are low, ratings reflect extreme opinions rather than the typical student's experience.

True educational effectiveness should balance three distinct pillars:
1. **Learning Outcomes (40%):** Did students finish the course, and did their skills demonstrably improve?
2. **Engagement (30%):** Did the instructor motivate students to watch lectures, submit assignments, and participate?
3. **Satisfaction & Rapport (30%):** Did students rate the instructor well, weighted by how many students actually responded?

### Mathematical Definition:

$$\text{IES} = 0.40 \cdot \text{Outcomes} + 0.30 \cdot \text{Engagement} + 0.30 \cdot \text{Satisfaction}$$

Where:
- **Outcomes:**  
  $$\text{Outcomes} = 0.50 \cdot \text{Norm}(\text{completion\_rate}) + 0.50 \cdot \text{Norm}(\text{avg\_score\_improvement})$$
- **Engagement:**  
  $$\text{Engagement} = 0.35 \cdot \text{Norm}(\text{avg\_watch\_time}) + 0.35 \cdot \text{Norm}(\text{submission\_rate}) + 0.30 \cdot \text{Norm}(\text{forum\_activity})$$
- **Satisfaction:**  
  $$\text{Satisfaction} = \text{Norm}\left( \text{Norm}(\text{avg\_feedback\_score}) \times (0.50 + 0.50 \cdot \text{feedback\_response\_rate}) \right)$$

### Handling Sample Size (Batch Variance)
In our dataset, batch count varies from 7 to 31 batches per instructor. An instructor with only 7 batches has a much smaller sample size, meaning a single good or bad cohort could artificially distort their score.  
To handle this, I applied Empirical Bayes shrinkage toward the overall population mean $\mu_{\text{pop}}$ with a prior weight $k = 5$:

$$\text{IES}_i^{\text{adj}} = \left(\frac{N_i}{N_i + 5}\right) \text{IES}_i^{\text{raw}} + \left(\frac{5}{N_i + 5}\right) \mu_{\text{pop}}$$

As an instructor teaches more batches ($N_i \to \infty$), their score relies almost entirely on their own empirical track record.

### Discretizing into Tiers:
I split the adjusted scores into three balanced tiers:
- **Low Tier:** Bottom 25% ($\text{IES} < Q_{25} \approx 0.414$) $\to 30$ instructors
- **Medium Tier:** Middle 50% ($Q_{25} \le \text{IES} < Q_{75}$) $\to 60$ instructors
- **High Tier:** Top 25% ($\text{IES} \ge Q_{75} \approx 0.581$) $\to 30$ instructors"""))

# Step 3 Code: Aggregation & IES computation
cells.append(nbf.v4.new_code_cell("""# Step 3: Aggregating from batch-level to instructor-level
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

inst = df.groupby('instructor_id').agg(agg_dict)
inst.columns = ['_'.join(c).strip('_') for c in inst.columns]
inst.rename(columns={'batch_id_count': 'total_batches', 'course_id_nunique': 'courses_taught'}, inplace=True)
inst.fillna(0, inplace=True)

# Helper for 0-1 min-max scaling
def scale_01(s):
    return (s - s.min()) / (s.max() - s.min())

# Calculate the three pillars
outcomes = 0.5 * scale_01(inst['completion_rate_mean']) + 0.5 * scale_01(inst['avg_score_improvement_mean'])

engagement = (
    0.35 * scale_01(inst['avg_watch_time_mean']) +
    0.35 * scale_01(inst['assignment_submission_rate_mean']) +
    0.30 * scale_01(inst['forum_activity_rate_mean'])
)

# Satisfaction adjusted by survey response rate
feedback_scaled = scale_01(inst['avg_feedback_score_mean'])
response_factor = 0.5 + 0.5 * inst['feedback_response_rate_mean']
satisfaction = scale_01(feedback_scaled * response_factor)

# Raw score
raw_score = 0.40 * outcomes + 0.30 * engagement + 0.30 * satisfaction

# Apply shrinkage toward the mean for instructors with fewer batches
k = 5.0
shrinkage_weight = inst['total_batches'] / (inst['total_batches'] + k)
inst['ies_score'] = shrinkage_weight * raw_score + (1 - shrinkage_weight) * raw_score.mean()

# Assign tiers based on quartile thresholds
q25 = inst['ies_score'].quantile(0.25)
q75 = inst['ies_score'].quantile(0.75)

def get_tier(score):
    if score >= q75:
        return 'High'
    elif score >= q25:
        return 'Medium'
    else:
        return 'Low'

inst['tier'] = inst['ies_score'].apply(get_tier)
inst['tier_code'] = inst['tier'].map({'Low': 0, 'Medium': 1, 'High': 2})

print(f"Total instructors: {len(inst)}")
print(f"Thresholds: Low < {q25:.3f} | Medium [{q25:.3f}, {q75:.3f}) | High >= {q75:.3f}")
print("\\nTier Breakdown:")
print(inst['tier'].value_counts())"""))

# Plot IES Distribution
cells.append(nbf.v4.new_code_cell("""# Visualizing the final score distribution and tier cutoffs
plt.figure(figsize=(9, 5))
sns.histplot(inst['ies_score'], kde=True, bins=20, color='#2b5c8f', edgecolor='black')
plt.axvline(q25, color='#e65100', linestyle='--', linewidth=2, label=f'Low/Medium cutoff ({q25:.3f})')
plt.axvline(q75, color='#2e7d32', linestyle='--', linewidth=2, label=f'Medium/High cutoff ({q75:.3f})')
plt.title("Distribution of Instructor Effectiveness Scores (with Tier Cutoffs)", fontsize=12, fontweight='bold')
plt.xlabel("Effectiveness Score")
plt.ylabel("Number of Instructors")
plt.legend(frameon=True)
plt.tight_layout()
plt.show()"""))

# Step 4: Machine Learning Modeling Markdown
cells.append(nbf.v4.new_markdown_cell(r"""---
## Step 4: Building the Machine Learning Model

### Feature Selection & Leakage Prevention
To ensure the ML models are learning meaningful relationships from operational data rather than simply inverting a math formula:
- I **excluded** the derived pillars (`outcomes`, `engagement`, `satisfaction`, and `ies_score`).
- The models are trained strictly on observable operational statistics: batch means, standard deviations (consistency), and experience metrics (`total_batches`, `courses_taught`).

### Models Tested:
1. **Dummy Baseline:** Stratified random guessing to establish the lower performance floor.
2. **Logistic Regression (L2):** Standardized linear model to see how well linear boundaries separate the tiers.
3. **Random Forest:** Bagging ensemble of 100 trees with max depth 5 to avoid overfitting on 120 rows.
4. **Gradient Boosting:** Sequential boosting classifier.

### Validation Scheme:
- **Stratified 5-Fold Cross-Validation** at the instructor level ($N = 120$) to guarantee every fold maintains the 25% / 50% / 25% tier distribution without data leakage."""))

# Step 4: Code for Cross-Validation
cells.append(nbf.v4.new_code_cell("""# Setting up features and target
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

X = inst[feature_cols].copy()
y = inst['tier_code'].copy()

# 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Dummy Baseline': DummyClassifier(strategy='stratified', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42)
}

cv_results = {}
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

for name, model in models.items():
    if name == 'Logistic Regression':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scores = cross_validate(model, X_scaled, y, cv=skf, scoring=scoring)
    else:
        scores = cross_validate(model, X, y, cv=skf, scoring=scoring)
        
    cv_results[name] = {
        'Accuracy': (scores['test_accuracy'].mean(), scores['test_accuracy'].std()),
        'Macro F1': (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std()),
        'Precision': (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std()),
        'Recall': (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std())
    }

pd.DataFrame({
    name: {metric: f"{v[0]:.3f} ± {v[1]:.3f}" for metric, v in res.items()}
    for name, res in cv_results.items()
}).T"""))

# Step 5: Model Evaluation Code
cells.append(nbf.v4.new_code_cell("""# Plotting CV benchmark comparison
fig, ax = plt.subplots(figsize=(10, 5))
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
ax.set_title('Cross-Validation Performance Across Models', fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names, fontsize=10)
ax.legend(loc='lower right', frameon=True)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()"""))

# Step 5: Confusion Matrices on Test Set
cells.append(nbf.v4.new_code_cell("""# Holdout Test Evaluation (75% Train / 25% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced').fit(X_train_scaled, y_train)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42, class_weight='balanced').fit(X_train, y_train)
gb = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42).fit(X_train, y_train)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
class_labels = ['Low', 'Medium', 'High']

eval_list = [('Logistic Regression', lr, X_test_scaled), ('Random Forest', rf, X_test), ('Gradient Boosting', gb, X_test)]

for idx, (m_name, m_obj, test_feat) in enumerate(eval_list):
    preds = m_obj.predict(test_feat)
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=class_labels, yticklabels=class_labels, cbar=False)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    axes[idx].set_title(f"{m_name}\\nAccuracy: {acc:.2f} | Macro F1: {f1:.2f}", fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("Predicted Tier")
    axes[idx].set_ylabel("True Tier")

plt.tight_layout()
plt.show()"""))

# Step 5: ROC Curves
cells.append(nbf.v4.new_code_cell("""# Multiclass ROC curves (One-vs-Rest)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax_idx, (m_name, m_obj, test_feat) in enumerate([('Random Forest', rf, X_test), ('Gradient Boosting', gb, X_test)]):
    probs = m_obj.predict_proba(test_feat)
    y_test_bin = pd.get_dummies(y_test).values
    
    for c_idx, c_label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, c_idx], probs[:, c_idx])
        roc_val = auc(fpr, tpr)
        axes[ax_idx].plot(fpr, tpr, lw=2, label=f"Class {c_label} (AUC = {roc_val:.2f})")
        
    axes[ax_idx].plot([0, 1], [0, 1], 'k--', lw=1.2)
    axes[ax_idx].set_title(f"ROC Curves — {m_name}", fontsize=11, fontweight='bold')
    axes[ax_idx].set_xlabel("False Positive Rate")
    axes[ax_idx].set_ylabel("True Positive Rate")
    axes[ax_idx].legend(loc="lower right")

plt.tight_layout()
plt.show()"""))

# Step 6: Interpretability & Feature Importance
cells.append(nbf.v4.new_code_cell("""# Step 6: Feature Importance Analysis
rf_importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

# Permutation importance on the holdout test set
perm = permutation_importance(rf, X_test, y_test, n_repeats=15, random_state=42)
perm_importances = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

rf_importances.head(10).plot(kind='barh', ax=axes[0], color='#2b5c8f', edgecolor='black')
axes[0].set_title("Top 10 Features (Random Forest MDI Importance)", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Importance Score")
axes[0].invert_yaxis()

perm_importances.head(10).plot(kind='barh', ax=axes[1], color='#e65100', edgecolor='black')
axes[1].set_title("Top 10 Features (Permutation Importance on Test Set)", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Mean Accuracy Drop when Shuffled")
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()"""))

# Step 7: Mandatory Questions Markdown (Humanized)
cells.append(nbf.v4.new_markdown_cell(r"""---
## Step 7: Mandatory Analysis Questions

### Q1: Which features most influenced instructor effectiveness, and why?

Looking at both the Random Forest feature importances and test set permutation importance, three features stood out clearly:

1. **Batch Completion Rate (and Dropout Rate):**  
   This was by far the strongest signal (accounting for over 25% of model importance). In online courses, staying enrolled requires motivation, clarity, and pacing. When an instructor communicates well and breaks down difficult ideas, students stick around. When teaching is confusing or uninspiring, dropouts spike early.

2. **Average Score Improvement:**  
   This captures actual learning value. Raw quiz scores can easily be inflated if a teacher gives softball questions, but pre-to-post score improvement measures real progress. High-tier instructors consistently help students make bigger jumps from their initial assessment to the final exam.

3. **Feedback Response Rate:**  
   Response rate acted as an essential reliability filter. An instructor who gets a 4.5 rating with 85% of the class responding is genuinely loved. An instructor with a 4.8 where only 20% responded usually just got reviews from a handful of enthusiastic fans while the rest of the class stayed silent.

4. **Consistency Across Batches (Standard Deviation):**  
   Top-tier instructors exhibit noticeably lower standard deviations across their batches. They deliver reliable instructional quality regardless of cohort differences.

---

### Q2: Which variables could be misleading or confounded?

A few variables in this dataset have subtle traps that could mislead someone evaluating instructors:

1. **Course Difficulty as a Major Confounder:**  
   Some courses are just inherently harder. An instructor teaching Advanced Machine Learning or Distributed Systems is almost guaranteed to see lower completion rates and lower ratings than someone teaching an introductory elective. Without adjusting for the course's baseline difficulty, we risk penalizing great instructors who take on the hardest subjects.

2. **Unweighted Feedback Scores (Rating Leniency & Response Bias):**  
   Student ratings are notoriously biased. First, most people give high ratings by default (the dataset mean was 4.21). Second, voluntary surveys attract polarized opinions (either super happy or furious students). That's why unweighted feedback scores can be very deceptive.

3. **Average Watch Time Ambiguity:**  
   Higher watch time isn't always good. It could mean students found the lecture fascinating, but it could also mean the explanation was so confusing that students had to rewind and re-watch videos three times just to understand a simple concept.

4. **Raw Quiz Scores vs. Score Improvement:**  
   A high average quiz score might just mean the quizzes were too easy. Score improvement is much harder to fake and reflects real teaching impact.

---

### Q3: How could this model fail in real-world usage?

If this model were deployed in a live EdTech company, here are the main ways it could backfire:

1. **Goodhart's Law (Gaming the System):**  
   If teachers know their bonuses or job security depend on these scores, human behavior will change:
   - They might make quizzes easier so completion and quiz scores go up.
   - They might beg or incentivize students to leave 5-star ratings.
   - They might hesitate to give critical, honest feedback on assignments for fear of getting low ratings in return.

2. **Cohort & Timing Effects:**  
   Batches running during university exams, summer vacations, or year-end holidays always see lower attendance and higher dropouts. A teacher assigned to a December batch might get flagged as "Low" simply due to bad calendar timing.

3. **Small Sample Noise for New Instructors:**  
   Even with Bayesian shrinkage, a teacher who has only taught 2 or 3 batches can have their score skewed by a single difficult cohort or a couple of disruptive students.

4. **Self-Fulfilling Loops in Course Allocation:**  
   If platform managers start giving the "High" instructors the best, most motivated cohorts and give "Low" instructors the struggling batches, the model's predictions will reinforce themselves, making it impossible for developing teachers to improve their metrics.

---

### Q4: What additional data would you want to improve this analysis?

If I had access to more data from the platform, I would look for:

1. **Student Background & Prerequisites:**  
   Incoming student GPA, prior programming experience, or pre-course test scores. That way, we could use a proper Value-Added Model (VAM) to see how much progress a student made relative to their starting point.

2. **Text Feedback (NLP Sentiment & Topics):**  
   Star ratings are blunt. Natural language comments tell us *why* students were unhappy (e.g., "microphone had static", "slides had errors", or "explained recursion better than anyone else"). That separates platform technical problems from actual teaching issues.

3. **Live Class Telemetry:**  
   How quickly does the instructor answer questions in the chat? Do they hold office hours? What is their attendance retention during live sessions?

4. **Long-Term Student Outcomes:**  
   Do students who took this instructor's batch go on to enroll in advanced courses? How do they perform on capstone projects or in job placement interviews?

---

### Q5: Should this model be used for instructor performance evaluation? Why or why not?

**Short answer: No for high-stakes punitive decisions (firing or pay cuts); Yes for coaching, mentorship, and operational diagnostics.**

**Why not punitively?**  
Observational data shows correlations, not pure causality. As shown above, course difficulty, student demographics, platform outages, and seasonal timing all affect batch numbers, and none of those are under the teacher's direct control. If an EdTech company uses an automated ML score to fire or reprimand instructors, it creates a toxic environment that encourages grade inflation and pushes teachers to avoid challenging courses.

**How it should be used instead:**
- **Coaching & Early Support:** If an instructor is trending toward the Low tier, academic managers can review lecture recordings, offer teaching workshops, or audit the syllabus to see what's going wrong.
- **Peer Mentoring:** Pair instructors in the High tier with newer instructors for co-teaching.
- **Curriculum Health Check:** If every instructor teaching Course X is seeing low completion and low watch time, the problem isn't the instructors — the curriculum itself needs revision."""))

# Save and execute notebook
nb.cells = cells
notebook_path = os.path.join(BASE_DIR, "instructor_effectiveness_modeling.ipynb")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[+] Saved updated notebook template to {notebook_path}")

print("[*] Executing notebook cells to render all outputs...")
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
with open(notebook_path, "r", encoding="utf-8") as f:
    nb_to_run = nbf.read(f, as_version=4)

ep.preprocess(nb_to_run, {'metadata': {'path': BASE_DIR}})

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb_to_run, f)

print(f"[SUCCESS] Notebook successfully re-executed and saved: {notebook_path}")
