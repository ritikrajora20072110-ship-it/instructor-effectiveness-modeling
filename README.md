# Instructor Effectiveness Modeling (EdTech Context)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ritikrajora20072110-ship-it/instructor-effectiveness-modeling/blob/main/instructor_effectiveness_modeling.ipynb)

**Data Science & AI Content Specialist Assignment**  
*Author:* Antigravity Pair Programmer  
*Stack:* Python 3.13, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn  
*Deliverable:* Standalone, self-contained, pre-executed Jupyter Notebook (`instructor_effectiveness_modeling.ipynb`)

---

## 1. Executive Summary & Problem Context

In modern online and blended learning platforms, the same standardized curriculum is delivered across numerous student cohorts by different instructors. Evaluating teaching effectiveness is notoriously difficult:
- Instructors teach varying numbers of batches ($N_i \in [7, 31]$ in this dataset of 2,000 batches across 120 instructors).
- Student satisfaction surveys are subject to leniency inflation, voluntary response bias, and small-sample volatility.
- Raw completion rates and quiz scores can confound instructor capability with syllabus difficulty or cohort selection effects.

This repository provides a mathematically grounded, statistically robust, and pedagogically sound framework for:
1. Defining a multi-dimensional **Instructor Effectiveness Score (IES)** that balances objective learning gains, student engagement, and response-weighted satisfaction.
2. Correcting for small-sample estimation variance using **Empirical Bayes Shrinkage**.
3. Benchmarking classical machine learning algorithms under **Stratified 5-Fold Cross-Validation** to classify instructors into actionable tiers (*Low*, *Medium*, *High*).
4. Extracting feature importances via Gini impurity and test-set permutation importance.
5. Providing rigorous, business-ready answers to the 5 mandatory strategic and ethical questions.

---

## 2. Core Methodology & Mathematical Formulations

### 2.1 The Tri-Pillar Instructor Effectiveness Score (IES)
We formulate effectiveness as a composite index spanning three fundamental dimensions:
$$\text{IES} = 0.40 \cdot \text{Outcomes} + 0.30 \cdot \text{Engagement} + 0.30 \cdot \text{Satisfaction}$$

```
                                  +------------------------------------+
                                  |   Instructor Effectiveness Score   |
                                  +-----------------+------------------+
                                                    |
             +--------------------------------------+--------------------------------------+
             | (40%)                                | (30%)                                | (30%)
             v                                      v                                      v
   +--------------------+                 +--------------------+                 +--------------------+
   |  Learner Outcomes  |                 | Learner Engagement |                 |Satisfaction/Quality|
   +---------+----------+                 +---------+----------+                 +---------+----------+
             |                                      |                                      |
     +-------+-------+                  +-----------+-----------+                          |
     | (50%)         | (50%)            | (35%)     | (35%)     | (30%)                    |
     v               v                  v           v           v                          v
Completion      Score Impr.         Watch Time   Submission   Forum Act.          Feedback * Resp. Rate
```

#### Pillar 1: Learner Outcomes (40% Weight)
True pedagogical impact is measured by student persistence and knowledge mastery:
$$\text{Outcomes} = 0.50 \cdot \text{Norm}(\text{completion\_rate}) + 0.50 \cdot \text{Norm}(\text{avg\_score\_improvement})$$
*(Note: `dropout_rate` is omitted here to prevent double-counting collinear retention signals, as $r = -0.95$ with completion rate).*

#### Pillar 2: Learner Engagement (30% Weight)
Measures the instructor's ability to maintain active student involvement throughout the syllabus:
$$\text{Engagement} = 0.35 \cdot \text{Norm}(\text{avg\_watch\_time}) + 0.35 \cdot \text{Norm}(\text{assignment\_submission\_rate}) + 0.30 \cdot \text{Norm}(\text{forum\_activity\_rate})$$

#### Pillar 3: Learner Satisfaction & Rapport (30% Weight)
Raw satisfaction scores suffer from voluntary response bias. To penalize unrepresentative sample sizes where only a small minority responded, we modulate the feedback score by response rate:
$$\text{Satisfaction} = \text{Norm}\left( \text{Norm}(\text{avg\_feedback\_score}) \times \left(0.50 + 0.50 \cdot \text{feedback\_response\_rate}\right) \right)$$

### 2.2 Empirical Bayes Shrinkage for Batch Variance
Instructors in the dataset teach between 7 and 31 batches. An instructor with only 7 batches has a much higher standard error of the mean than an instructor with 31 batches. To prevent low-sample instructors from artificially dominating the extreme tiers, we apply Empirical Bayes shrinkage toward the population grand mean $\mu_{\text{pop}}$:
$$\text{IES}_i^{\text{adj}} = \left(\frac{N_i}{N_i + k}\right) \text{IES}_i^{\text{raw}} + \left(\frac{k}{N_i + k}\right) \mu_{\text{pop}}, \quad (k = 5)$$
Where $k = 5$ represents the prior pseudo-batch weight. As $N_i \to \infty$, the weight on the prior vanishes.

### 2.3 Discretization into Actionable Tiers
To avoid arbitrary round numbers, we discretize $\text{IES}^{\text{adj}}$ using empirical quartile cutoffs:
- **Low Tier ($y = 0$):** $\text{IES} < Q_{25}$ ($< 0.4136$) $\to 30\text{ instructors }(25\%)$
- **Medium Tier ($y = 1$):** $Q_{25} \le \text{IES} < Q_{75}$ ($[0.4136, 0.5814)$) $\to 60\text{ instructors }(50\%)$
- **High Tier ($y = 2$):** $\text{IES} \ge Q_{75}$ ($\ge 0.5814$) $\to 30\text{ instructors }(25\%)$

---

## 3. Batch-to-Instructor Aggregation & Leakage Prevention

### 3.1 Aggregation Strategy
For each instructor $i$, we compute summary statistics across all their batches:
1. **Central Tendency:** Batch means and medians for all 9 metrics.
2. **Instructional Consistency:** Standard deviation ($\sigma$) across batches. High-variance instructors deliver inconsistent student experiences depending on the cohort.
3. **Experience & Breadth:** `total_batches` ($N_i$) and `courses_taught` (distinct course count).

### 3.2 Target Leakage Safeguards
To ensure that the ML models learn predictive relationships from observable operational features rather than inverting the deterministic score equation:
- The composite pillars (`outcomes_pillar`, `engagement_pillar`, `satisfaction_pillar`) and the continuous `ies_score` are **strictly excluded** from the feature matrix $X$.
- Only raw batch summary statistics (means, standard deviations, volume counts) are provided to the models.

---

## 4. Machine Learning Benchmarking & Results

All models were evaluated on 120 instructors using **Stratified 5-Fold Cross-Validation** to ensure proportional class distribution across folds.

| Model | Accuracy (CV) | Macro F1 (CV) | Macro Precision (CV) | Macro Recall (CV) | Holdout Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Stratified)** | $0.450 \pm 0.072$ | $0.394 \pm 0.083$ | $0.394 \pm 0.083$ | $0.397 \pm 0.086$ | $0.402$ |
| **Multinomial Logistic Regression** | $0.917 \pm 0.046$ | $0.915 \pm 0.048$ | $0.927 \pm 0.038$ | $0.917 \pm 0.046$ | $0.903$ |
| **Gradient Boosting Classifier** | $0.925 \pm 0.061$ | $0.924 \pm 0.062$ | $0.932 \pm 0.054$ | $0.925 \pm 0.061$ | $0.935$ |
| **Random Forest Classifier** | **$0.958 \pm 0.053$** | **$0.959 \pm 0.052$** | **$0.963 \pm 0.047$** | **$0.958 \pm 0.053$** | **$0.969$** |

### Key Takeaways:
- **Random Forest** achieved the best overall performance ($95.9\%$ Macro F1), capturing non-linear interactions between completion rate, score improvement, and student feedback.
- Multiclass ROC-AUC analysis showed **AUC $> 0.99$** for Low, Medium, and High classes under both Random Forest and Gradient Boosting.
- The 25% holdout test set confirmed exceptional generalization with zero severe misclassifications (no Low-tier instructor was misclassified as High, and vice versa).

---

## 5. Feature Importance & Pedagogical Drivers

Combining Random Forest Mean Decrease in Impurity (MDI) with Permutation Feature Importance on the test set revealed:
1. **`completion_rate_mean` & `dropout_rate_mean`:** Account for $>28\%$ of predictive importance. Student retention is the clearest empirical signal of instructor effectiveness.
2. **`avg_score_improvement_mean`:** Accounts for $\approx 18\%$ of importance. Instructors who drive meaningful pre-to-post knowledge gains are strongly separated into the High tier.
3. **`feedback_response_rate_mean` & `avg_feedback_score_mean`:** Response rate acts as an indispensable reliability filter on feedback ratings.
4. **Consistency Features (`completion_rate_std`, `avg_watch_time_std`):** Lower variance across batches separates dependable master instructors from erratic performers.

---

## 6. Answers to Mandatory Analysis Questions

### Q1: Which features most influenced instructor effectiveness, and why?
- **Retention & Completion (`completion_rate_mean`, `dropout_rate_mean`):** In online learning where self-regulation is the primary friction point, instructors who inspire completion demonstrate mastery over pacing, clarity, and empathy.
- **Value-Added Learning (`avg_score_improvement_mean`):** Raw quiz scores can be inflated by easy tests, but score improvement captures true knowledge acquisition.
- **Learner Engagement & Response Rate:** High feedback response rates indicate that learners feel heard and invested in the learning community.

### Q2: Which variables could be misleading or confounded?
- **Course Baseline Difficulty:** Advanced courses (e.g., Deep Learning, Distributed Systems) naturally have lower completion and watch times than introductory courses. An instructor teaching difficult subjects will appear less effective if course difficulty is not controlled for.
- **Voluntary Response Bias in Feedback:** Low response rates reflect bimodal polarization (only ecstatic or furious students fill out surveys). Unweighted ratings are misleading.
- **Watch Time Ambiguity:** Extended watch time may indicate captivating lectures, or conversely, convoluted explanations requiring repeated rewinds.
- **Quiz Score Leniency:** High average quiz scores can result from dumbed-down assessments rather than superior instruction.

### Q3: How could this model fail in real-world usage?
- **Goodhart's Law & Strategic Gaming:** If high-stakes decisions (bonuses, promotions) depend on these scores, instructors will lower grading standards, give away quiz answers, and pressure students for 5-star ratings.
- **Cohort & Seasonal Heterogeneity:** Batches running during university finals or holiday periods suffer natural attrition unrelated to the instructor.
- **Small-Sample Volatility:** Instructors with few batches can experience wild tier swings from a single anomalous cohort.
- **Algorithmic Self-Fulfilling Prophecy:** Assigning High-tier instructors to motivated premium cohorts while giving Low-tier instructors struggling cohorts will artificially entrench model predictions.

### Q4: What additional data would you want to improve this analysis?
1. **Learner Baseline Attributes:** Prerequisite knowledge, pre-course test scores, and educational background to implement a proper **Value-Added Model (VAM)**.
2. **Qualitative NLP Feedback:** Unstructured textual student comments to separate delivery quality from platform technical glitches.
3. **Live Interaction Telemetry:** Live attendance, chat participation, and response latency to student forum queries.
4. **Long-Term Downstream Outcomes:** Subsequent course re-enrollment, certification exam pass rates, and alumni employment outcomes.

### Q5: Should this model be used for instructor performance evaluation? Why or why not?
**Verdict: NO for punitive decisions (firing, salary cuts); YES for diagnostic enablement, coaching, and operational matching.**

- **Why Not Punitively?** Observational data cannot establish pure causality. Confounders like cohort motivation and syllabus flaws are beyond the instructor's direct control. High-stakes automated punishment incentivizes grade inflation and destroys educational rigor.
- **The Constructive Role:**
  - **Early Warning & Coaching:** Identify instructors trending toward the Low tier to trigger peer mentoring and pedagogical workshops.
  - **Curriculum Auditing:** Identify batches where all instructors struggle, pinpointing curriculum bottlenecks.
  - **Co-Teaching Pairs:** Pair High-tier mentors with developing instructors for team-teaching.

---

## 7. Repository Structure & How to Run

```
edtech_assignment/
├── instructor_effectiveness_modeling.ipynb  # Primary deliverable: fully executed notebook
├── instructor_effectiveness_data.csv        # Dataset (2,000 batches, 120 instructors)
├── pipeline.py                              # Modular end-to-end Python pipeline
├── generate_notebook.py                     # Notebook generation and pre-rendering script
├── pipeline_results.json                    # Serialized benchmark metrics
├── README.md                                # Comprehensive documentation & report
└── figures/                                 # High-resolution saved visualizations
    ├── 01_eda_distributions.png
    ├── 02_correlation_heatmap.png
    ├── 03_ies_distribution.png
    ├── 04_model_comparison.png
    ├── 05_confusion_matrices.png
    ├── 06_roc_curves.png
    └── 07_feature_importance.png
```

### Local Execution:
```bash
# Navigate to the assignment folder
cd /Users/ritikrajora20072110gmailcom/.gemini/antigravity/scratch/edtech_assignment

# Launch Jupyter Notebook
jupyter notebook instructor_effectiveness_modeling.ipynb
```

### Google Colab Execution:
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `instructor_effectiveness_modeling.ipynb`.
3. Upload `instructor_effectiveness_data.csv` to the session files.
4. Select `Runtime -> Run all`. All dependencies (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`) are pre-installed in standard Colab environments.
