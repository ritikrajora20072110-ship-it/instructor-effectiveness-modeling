# Instructor Effectiveness Modeling — EdTech Analysis
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ritikrajora20072110-ship-it/instructor-effectiveness-modeling/blob/main/instructor_effectiveness_modeling.ipynb)

**Candidate:** Ritik Rajora  
**Role:** Data Science / AI Content Specialist Intern Task  
**Environment:** Python 3.13, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn  

---

## 1. Problem Overview & Thought Process

In an EdTech company running the same courses across multiple batches with different instructors, evaluating teaching quality is tricky:
- Some instructors teach only 7 batches while others teach over 30, meaning variance and sample size differences are substantial.
- Feedback surveys suffer from severe rating inflation (the dataset average is 4.21 / 5.0) and response bias (a vocal minority of unhappy or super enthusiastic students).
- Completion rates and quiz scores can be confounded by course difficulty or easy grading rather than actual teaching effectiveness.

In this project, I built an end-to-end framework to evaluate and predict instructor effectiveness:
1. **Exploratory Data Analysis (EDA):** Discovered that `completion_rate` and `dropout_rate` have a -0.95 correlation (mirror images), and identified response bias in feedback.
2. **Defining Instructor Effectiveness:** Built a balanced **Instructor Effectiveness Score (IES)** combining learning outcomes (40%), student engagement (30%), and satisfaction weighted by response rate (30%). Applied Empirical Bayes shrinkage to account for instructors with fewer batches.
3. **Aggregation:** Rolled up batch data to instructor level, computing central tendencies, consistency (standard deviation across batches), and volume indicators.
4. **Machine Learning:** Trained classical classifiers (Baseline, Regularized Logistic Regression, Random Forest, Gradient Boosting) using Stratified 5-Fold Cross-Validation, achieving **95.9% Macro F1** with Random Forest.
5. **Practical Answers:** Addressed the 5 key questions around feature drivers, confounders, failure modes, data needs, and ethical usage.

---

## 2. Defining the Effectiveness Score

### The Formula:
$$\text{IES} = 0.40 \cdot \text{Outcomes} + 0.30 \cdot \text{Engagement} + 0.30 \cdot \text{Satisfaction}$$

- **Outcomes (40%):**  
  Did students finish the course, and did their scores actually improve?
  $$\text{Outcomes} = 0.50 \cdot \text{Norm}(\text{completion\_rate}) + 0.50 \cdot \text{Norm}(\text{avg\_score\_improvement})$$
  *(Note: `dropout_rate` was left out because of its -0.95 correlation with completion).*
- **Engagement (30%):**  
  Did the instructor keep students actively involved?
  $$\text{Engagement} = 0.35 \cdot \text{Norm}(\text{avg\_watch\_time}) + 0.35 \cdot \text{Norm}(\text{submission\_rate}) + 0.30 \cdot \text{Norm}(\text{forum\_activity})$$
- **Satisfaction (30%):**  
  Did students rate the instructor well, penalized if only a tiny fraction of the class filled out the survey?
  $$\text{Satisfaction} = \text{Norm}\left(\text{Norm}(\text{avg\_feedback\_score}) \times (0.50 + 0.50 \cdot \text{feedback\_response\_rate})\right)$$

### Handling Sample Size (Batch Variance)
Because instructors taught between 7 and 31 batches, an instructor with only 7 batches has much higher noise. I applied Empirical Bayes shrinkage toward the overall grand mean $\mu_{\text{pop}}$ with prior weight $k = 5$:
$$\text{IES}_i^{\text{adj}} = \left(\frac{N_i}{N_i + 5}\right) \text{IES}_i^{\text{raw}} + \left(\frac{5}{N_i + 5}\right) \mu_{\text{pop}}$$

### Discretizing into Balanced Tiers:
- **Low Tier:** Bottom 25% (IES < 0.414) $\to 30$ instructors
- **Medium Tier:** Middle 50% (0.414 $\le$ IES < 0.581) $\to 60$ instructors
- **High Tier:** Top 25% (IES $\ge$ 0.581) $\to 30$ instructors

---

## 3. Batch-to-Instructor Aggregation & Leakage Prevention

For each instructor, I computed:
- **Mean & Median:** Overall level of performance across batches.
- **Standard Deviation ($\sigma$):** Consistency. Top educators deliver reliable quality across cohorts; volatile teachers have high variance.
- **Experience:** Total batches taught and unique courses taught.

**Target Leakage Safeguard:** The composite pillars (`outcomes`, `engagement`, `satisfaction`) and `ies_score` were **excluded** from the feature matrix $X$. The machine learning models are trained purely on observable operational metrics.

---

## 4. Model Benchmarking & Results

All models were evaluated across the 120 instructors using **Stratified 5-Fold Cross-Validation**:

| Model | Accuracy (CV) | Macro F1 (CV) | Precision (CV) | Recall (CV) | Holdout Test F1 (25%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Stratified)** | $0.450 \pm 0.072$ | $0.394 \pm 0.083$ | $0.394 \pm 0.083$ | $0.397 \pm 0.086$ | $0.402$ |
| **Logistic Regression (L2, Scaled)** | $0.917 \pm 0.046$ | $0.915 \pm 0.048$ | $0.927 \pm 0.038$ | $0.917 \pm 0.046$ | $0.903$ |
| **Gradient Boosting** | $0.925 \pm 0.061$ | $0.924 \pm 0.062$ | $0.932 \pm 0.054$ | $0.925 \pm 0.061$ | $0.935$ |
| **Random Forest** | **$0.958 \pm 0.053$** | **$0.959 \pm 0.052$** | **$0.963 \pm 0.047$** | **$0.958 \pm 0.053$** | **$0.969$** |

### Key Observations:
- **Random Forest** achieved the strongest overall performance ($95.9\%$ Macro F1), capturing non-linear interactions between completion rates and score improvement.
- **Multiclass ROC-AUC** was $>0.99$ for all three classes under Random Forest on the holdout test set.
- On the holdout test set ($N=30$), there were zero cross-tier blunders (no Low-tier instructor was misclassified as High, and vice versa).

---

## 5. Answers to the 5 Mandatory Analysis Questions

### Q1: Which features most influenced instructor effectiveness, and why?
- **Completion Rate & Dropout Rate:** Accounts for over 25% of importance. In online learning, keeping students motivated to finish is the clearest indicator of clear explanations, good pacing, and strong empathy.
- **Average Score Improvement:** Captures true value-added learning. An instructor who takes a cohort from low pre-assessment scores to high mastery demonstrates genuine teaching effectiveness.
- **Feedback Response Rate:** Serves as a great rapport filter. When students feel personally invested in the course, they take time to respond to surveys.

### Q2: Which variables could be misleading or confounded?
- **Course Difficulty:** Advanced topics (e.g., Distributed Systems) have naturally lower completion rates than introductory courses. Without adjusting for course baseline difficulty, great instructors teaching tough subjects get unfairly penalized.
- **Unweighted Feedback Scores:** Student surveys suffer from leniency bias (mean rating was 4.21) and voluntary response bias (mostly extremes respond).
- **Watch Time Ambiguity:** High watch time can mean great lectures, or it can mean convoluted explanations that students had to rewind multiple times.
- **Quiz Score Leniency:** High average quiz scores can simply mean tests were watered down. Score improvement is a much safer metric.

### Q3: How could this model fail in real-world usage?
- **Goodhart's Law & Gaming:** If bonuses or contracts depend on these tiers, instructors will make tests easier, inflate grades, and pressure students for 5-star reviews.
- **Seasonal & Calendar Effects:** Batches during college exam periods or holidays see natural dips in completion unrelated to the instructor.
- **Small Sample Volatility:** New instructors with only 2–3 batches can be pushed into the Low tier by one unlucky cohort.
- **Self-Fulfilling Allocation Loops:** If High-tier instructors are given the most motivated cohorts while Low-tier instructors get struggling batches, the model's predictions become self-fulfilling.

### Q4: What additional data would you want to improve this analysis?
- **Student Baseline Covariates:** Prior GPA, coding experience, or prerequisite test scores to fit a proper Value-Added Model (VAM).
- **NLP Text Reviews:** Student comments help separate platform technical issues (e.g., broken audio) from teaching ability.
- **Live Class Telemetry:** Attendance during live sessions, live chat participation, and response latency to forum questions.
- **Downstream Outcomes:** Subsequent course enrollment, capstone project quality, and post-graduation job placement.

### Q5: Should this model be used for instructor performance evaluation? Why or why not?
**Short answer: No for high-stakes punitive decisions (firing or pay cuts); Yes for coaching, mentorship, and operational diagnostics.**

Observational data shows correlations, not pure causality. Factors like student motivation, platform outages, and course difficulty are outside the instructor's direct control. Using automated ML scores punitively harms morale, encourages grade inflation, and damages educational standards.

Instead, the model should be used constructively:
- **Early-Warning & Coaching:** Identify instructors trending toward the Low tier to offer peer mentoring and pedagogy workshops.
- **Curriculum Health Checks:** If every instructor teaching Course X sees poor completion, the curriculum itself is the problem.
- **Co-Teaching Pairs:** Pair High-tier mentors with developing instructors for co-teaching.

---

## 6. How to Run

### Google Colab (1-Click):
Click the badge above or visit:  
[Open in Google Colab](https://colab.research.google.com/github/ritikrajora20072110-ship-it/instructor-effectiveness-modeling/blob/main/instructor_effectiveness_modeling.ipynb)  
*(The notebook automatically fetches the dataset from GitHub if running in Colab).*

### Local Execution:
```bash
# Clone the repository
git clone https://github.com/ritikrajora20072110-ship-it/instructor-effectiveness-modeling.git
cd instructor-effectiveness-modeling

# Run Jupyter Notebook
jupyter notebook instructor_effectiveness_modeling.ipynb
```
