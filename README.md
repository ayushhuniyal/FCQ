# Predicting Instructor Ratings Across University of Colorado Campuses

### What makes a good professor a great one? A multi-campus ML analysis of FCQ evaluation data.

## The Problem

University course evaluations capture thousands of data points each semester
but rarely surface the factors that actually drive instructor ratings.
Administrators rely on summary statistics. Instructors receive generic feedback.
The underlying predictors of high and low ratings — and the structural
differences between campuses — remain unexamined.

This project analyzes FCQ (Faculty Course Questionnaire) data across three
University of Colorado campuses — Denver, Anschutz, and Boulder — to identify
the key drivers of instructor ratings, compare predictive model performance,
and uncover natural course groupings through unsupervised clustering.

## Team

Completed in collaboration with Ethan Leap and Pender Bauhan as a
Data Science capstone at the University of Colorado Boulder.

## Solution

**Denver \& Anschutz — Supervised Learning:**
Stepwise regression was used for feature selection, identifying the most
significant predictors of instructor ratings from 30+ candidate variables.
Four models were trained and cross-validated (10-fold) on each campus dataset:
SVM (radial kernel), Random Forest, Bagging, and Decision Trees.

**Boulder — Unsupervised Clustering:**
K-Means and Random Forest proximity matrix clustering were applied to identify
natural groupings in course evaluation patterns across departments and colleges.
Enrollment and Interact variables were transformed (log and reverse square root)
to address skewness before clustering.

## Results

### Denver Campus — Model Performance

| Model | RMSE | R² | MAE |
|-------|------|-----|-----|
| **Random Forest** | **0.2533** | **0.8462** | **0.1731** |
| Bagging | 0.2578 | 0.8517 | 0.1931 |
| SVM | 0.2695 | 0.8270 | 0.1823 |
| Decision Tree | 0.4044 | 0.6323 | 0.3299 |

### Anschutz Campus — Model Performance

| Model | RMSE | R² | MAE |
|-------|------|-----|-----|
| **Random Forest** | **0.3517** | **0.7724** | **0.2029** |
| Bagging | 0.3496 | 0.7702 | 0.2052 |
| SVM | 0.4347 | 0.6437 | 0.2730 |
| Decision Tree | 0.4157 | 0.6729 | 0.3267 |

### Top Predictors (Both Campuses)

Course Effectiveness and Course Learned ranked as the top two predictors
across all models on both campuses. The Decision Tree root node confirmed
this — CrseEffect was the primary split in both campus trees, with 77–80\%
of observations falling into the high-rating branch when CrseEffect was
above threshold.

### Boulder Campus — Clustering

| Method | Clusters | Silhouette Score | Interpretation |
|--------|----------|-----------------|----------------|
| K-Means | 2 | 0.346 | Separates high-engagement vs. lower-engagement courses |
| RF Proximity Matrix | 2 | 0.023 | Weak separation — feature interactions blur boundaries |

K-Means Cluster 1 (n=964): Higher feedback, reflection, and collaboration
scores — consistent with advanced or specialized courses with smaller enrollment.

K-Means Cluster 2 (n=463): Lower engagement metrics, higher enrollment —
consistent with large introductory or general education courses.

ARSC (Arts \& Sciences) dominated both clusters due to its outsized presence
in the dataset. BUSN courses consistently separated more cleanly, likely
reflecting standardized teaching structures.

## Key Decisions and Tradeoffs

**1. Stepwise regression for feature selection over manual selection**
With 30+ candidate predictors across campuses, manual selection risked
confirmation bias. Bidirectional stepwise AIC minimized information loss
while pruning irrelevant variables — resulting in 14 final predictors
for Denver and 11 for Anschutz.

**2. Random Forest over SVM as the primary model**
Random Forest outperformed SVM on both campuses across all three metrics.
Its ensemble structure reduces overfitting risk on the noisy,
high-dimensional FCQ data. SVM's advantage on non-linear boundaries was
not significant enough to offset Random Forest's stability across the
full feature set.

**3. K-Means over RF proximity clustering for Boulder**
RF proximity matrix clustering produced a near-zero silhouette score (0.023),
indicating that proximity-based similarity was too diffuse for meaningful
separation in this dataset. K-Means, despite its simplicity, produced
interpretable clusters (silhouette: 0.346) aligned with meaningful
course-level distinctions. Complexity was not rewarded here.

**4. Downsampling Denver to 5\% of original size**
The Denver dataset was substantially larger than Anschutz and Boulder.
Downsampling to 5\% (n$\approx$1,473) managed computational load while
preserving sufficient observations for reliable 10-fold cross-validation.
A 70/30 train-test split was applied consistently across both supervised
datasets.

**5. Log and reverse square root transformations for Boulder clustering**
Enrollment was strongly right-skewed (skewness = 4.92) and Interact was
strongly left-skewed (skewness = $-$2.61). Untransformed variables would
have allowed enrollment scale to dominate Euclidean distance calculations
in K-Means. Transformations normalized both distributions before scaling.

## What I'd Do Differently

The Boulder clustering would benefit from separating ARSC into
sub-departments before clustering — its sheer volume dominates both clusters
and masks finer distinctions among smaller colleges like Law and Music.
A hierarchical approach that first clusters within college, then across,
would produce more interpretable groupings. Incorporating temporal features
(semester, year) could also reveal whether instructor rating patterns have
shifted over time — a question the current static analysis cannot answer.

## Technical Stack

- **Language:** R
- **Key Packages:** caret, randomForest, rpart, e1071, cluster, ggplot2,
  tidyverse, MASS (stepAIC), forcats, tictoc
- **Validation:** 10-fold cross-validation
- **Clustering Evaluation:** Silhouette scores, PCA visualization

## Data

FCQ evaluation data from three CU campuses: Denver, Anschutz, and Boulder.
Data is institutional and not included in this repository.
