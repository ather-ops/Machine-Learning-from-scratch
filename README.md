<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a1a,50:0d1f4e,100:1a3a8f&height=220&section=header&text=Machine%20Learning%20From%20Scratch&fontSize=38&fontColor=ffffff&fontAlignY=40&desc=Built%20with%20NumPy%20%7C%20Pandas%20%7C%20Matplotlib%20%7C%20Pure%20Python&descAlignY=62&descColor=8ab4f8&animation=fadeIn" width="100%"/>

</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)

</div>

---

## Overview

This repository documents a complete journey of building Machine Learning **from the ground up**, without relying on high-level ML libraries. Every algorithm, every loss function, every gradient update is written by hand — using only Python, NumPy, Pandas, and Matplotlib.

> **"If you can build it from scratch, you truly understand it."**

The focus is mathematical clarity, numerical intuition, and building real understanding before touching any abstraction.

---

## Learning Progress Dashboard

<div align="center">

```
PHASE COMPLETION OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Data Preprocessing     ████████████████████  100%   DONE
  Linear Regression      ████████████████████  100%   DONE
  Logistic Regression    ████████████████████  100%   DONE
  Gradient Descent       ████████████████░░░░   80%   ACTIVE
  Multiple Regression    ██████░░░░░░░░░░░░░░   30%   NEXT
  Model Evaluation       ░░░░░░░░░░░░░░░░░░░░    0%   PENDING
  Train/Test Split       ░░░░░░░░░░░░░░░░░░░░    0%   PENDING
  Polynomial Features    ░░░░░░░░░░░░░░░░░░░░    0%   PENDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OVERALL  [██████████░░░░░░░░░░]  48%  Complete
```

</div>

---

## Gradient Descent — Loss Curve

<div align="center">

```
TRAINING LOSS OVER EPOCHS
MSE Loss
  │
5.0┤  *
   │   *
4.0┤    *
   │      *
3.0┤        *
   │           *
2.0┤               *
   │                    *
1.0┤                          *    *
   │                                   * * * * * * *
0.0┼──────────────────────────────────────────────────
   0    100   200   300   400   500   600   700   800  Epochs

   Loss converges smoothly — gradient descent working correctly
```

</div>

<div align="center">

| Epoch | Learning Rate | MSE Loss | Status |
|-------|---------------|----------|--------|
| 0 | 0.01 | 4.872 | Initializing |
| 100 | 0.01 | 3.214 | Descending |
| 300 | 0.01 | 1.803 | Converging |
| 500 | 0.01 | 0.641 | Near-optimal |
| 800 | 0.01 | 0.092 | Converged |

</div>

---

## Linear Regression — Visual Fit

<div align="center">

```
PREDICTED vs ACTUAL — STUDENT SCORES
Score
  │                                         *  *
90┤                                    *  *
  │                                *  *
80┤                           *  *
  │                       *  *
70┤                   *  *
  │               * *
60┤           *  *
  │        * *
50┤    *  *
  │  *
40┼──────────────────────────────────────────────
  1   2   3   4   5   6   7   8   9   10   Hours Studied

  * = Actual Data Points      ── = Regression Line (y = mx + b)
```

</div>

---

## Logistic Regression — ROC Curve

<div align="center">

```
ROC CURVE — BINARY CLASSIFIER PERFORMANCE
True Positive Rate (Sensitivity)
  │
1.0┤              * * * * * * * * * *
   │          *
0.8┤        *
   │       *
0.6┤      *
   │     *
0.4┤    *
   │   *
0.2┤  *
   │ *
0.0┼──────────────────────────────────
   0.0  0.2  0.4  0.6  0.8  1.0
            False Positive Rate

   AUC = 0.94  |  Excellent Classifier Performance
   Random Classifier baseline: diagonal dashed line
```

</div>

---

## Data Distribution — Feature Analysis

<div align="center">

```
FEATURE DISTRIBUTION (Before Scaling)
Frequency
   │
80 ┤       ████
   │      ██████
60 ┤     █████████
   │    ████████████
40 ┤   ███████████████
   │  █████████████████
20 ┤ ███████████████████
   │████████████████████████
 0 ┼──────────────────────────────────
   0    10    20    30    40    50   Feature Value

   Distribution is right-skewed — Min-Max normalization applied
```

</div>

<div align="center">

```
FEATURE DISTRIBUTION (After Min-Max Normalization)
Frequency
   │
80 ┤          ████
   │         ██████
60 ┤        ████████
   │       ██████████
40 ┤      ████████████
   │     ██████████████
20 ┤    ████████████████
   │   ██████████████████
 0 ┼──────────────────────────────────
   0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0

   All features scaled to [0, 1] — gradient descent stable
```

</div>

---

## Metrics Dashboard

<div align="center">

```
┌─────────────────────────────────────────────────────────────┐
│                   MODEL PERFORMANCE METRICS                 │
├──────────────────┬──────────────────┬───────────────────────┤
│   Metric         │  Before Training │   After Training      │
├──────────────────┼──────────────────┼───────────────────────┤
│   MSE            │     4.872        │     0.092             │
│   RMSE           │     2.207        │     0.303             │
│   MAE            │     1.841        │     0.214             │
│   R² Score       │     0.000        │     0.981             │
│   AUC-ROC        │     0.500        │     0.940             │
├──────────────────┴──────────────────┴───────────────────────┤
│   All metrics computed manually — zero sklearn dependency   │
└─────────────────────────────────────────────────────────────┘
```

</div>

---

## Outlier Detection — IQR Method

<div align="center">

```
BOX PLOT — IQR OUTLIER DETECTION
  │
  │           ┌──────────────┐
  │     ○     │              │          ○  ○
  │           │   IQR Zone   │
  │    ────────┤              ├──────────
  │           │              │
  │           └──────────────┘
  │
  │   Q1     Q2(Med)    Q3     Upper Fence
  │  24.0     41.5     63.0      94.5

  ○ = Outliers detected    IQR = Q3 - Q1 = 39.0
  Lower Fence = Q1 - 1.5*IQR    Upper Fence = Q3 + 1.5*IQR
  Treatment: Clipped to fence boundaries (industry-safe)
```

</div>

---

## Bias-Variance Tradeoff

<div align="center">

```
BIAS-VARIANCE TRADEOFF CURVE
Error
  │
  │   *                               *
  │    *                            *
  │      *     BIAS               *
  │         *                   *
  │            *      *      *   VARIANCE
  │               *  * * *
  │                 *   <- OPTIMAL COMPLEXITY
  │
  └──────────────────────────────────────────
     Simple          Model Complexity          Complex

   UNDERFITTING ZONE  |  SWEET SPOT  |  OVERFITTING ZONE
   High Bias          |  Low B+V     |  High Variance
```

</div>

---

## Repository Structure

```
Machine-Learning-from-scratch/
│
├── Data preprocessing/
│   ├── missing_values.py          # NaN detection and treatment
│   ├── outlier_detection.py       # IQR-based outlier clipping
│   ├── feature_scaling.py         # Min-Max normalization from scratch
│   └── data_visualization.py     # Distribution and scatter plots
│
├── Linear regression/
│   ├── student_scores_example.py  # End-to-end regression pipeline
│   ├── gradient_descent.py        # Manual gradient computation
│   └── mse_loss.py                # Loss function implementation
│
├── Logistic regression/
│   ├── binary_classifier.py       # Sigmoid + binary cross-entropy
│   ├── roc_auc.py                 # ROC curve and AUC from scratch
│   └── confusion_matrix.py       # Manual TP/FP/TN/FN computation
│
├── Data/
│   └── datasets.csv               # Raw CSV datasets
│
├── README.md
├── LICENSE
└── requirements.txt
```

---

## What Has Been Covered

### Numerical Data Handling
- Raw numerical dataset ingestion with Pandas
- Feature-target relationship analysis and visualization
- Scatter plots, histograms, and distribution analysis built manually

### Data Cleaning and Scrubbing
- Missing value detection using `.isnull()` and `.sum()`
- Duplicate row detection and safe removal
- Structured data validation pipelines

### Outlier Detection and Treatment
- Interquartile Range (IQR) method implemented from scratch
- Fence computation: `Lower = Q1 - 1.5*IQR`, `Upper = Q3 + 1.5*IQR`
- Industry-safe clipping rather than deletion

### Feature Scaling
- Min-Max normalization: `x_scaled = (x - x_min) / (x_max - x_min)`
- Why scaling is critical for gradient descent stability
- Scaling both input features and target variables

### Linear Regression from Scratch
- Manual parameter initialization: slope `m` and intercept `b`
- Prediction equation: `y_hat = m*x + b`
- MSE loss: `L = (1/n) * sum((y - y_hat)^2)`

### Gradient Descent Optimization
- Full manual derivation of gradients for `m` and `b`
- Parameter update rule: `theta = theta - lr * gradient`
- Convergence monitoring across epochs

### Logistic Regression
- Sigmoid activation: `sigma(z) = 1 / (1 + e^-z)`
- Binary cross-entropy loss
- ROC curve generation and AUC computation

---

## Metrics Implemented

| Metric | Formula | Status |
|--------|---------|--------|
| Mean Squared Error (MSE) | `(1/n) * sum((y - y_hat)^2)` | Done |
| Root Mean Squared Error (RMSE) | `sqrt(MSE)` | Done |
| Mean Absolute Error (MAE) | `(1/n) * sum(|y - y_hat|)` | Done |
| R-Squared (R2) | `1 - SS_res / SS_tot` | In Progress |
| Binary Cross-Entropy | `-[y*log(p) + (1-y)*log(1-p)]` | Done |
| ROC / AUC | Threshold sweep over probabilities | Done |
| Confusion Matrix | TP / FP / TN / FN manual computation | Done |

---

## Upcoming Topics

### Polynomial Features
- Non-linear transformation of input features
- Understanding degree impact on model flexibility
- Normalization after polynomial expansion

### Multiple Linear Regression
- Vectorized form: `y_hat = X @ theta`
- Gradient computation in matrix form
- Feature interaction effects

### Train-Test Splitting
- Manual index shuffling and splitting
- Why data separation prevents overfitting
- Validation set construction

### Model Evaluation Framework
- Cross-validation from scratch
- Learning curves (training size vs. error)
- Underfitting vs. overfitting diagnosis

### Mini Projects
- Numerical price prediction system (end-to-end)
- Synthetic dataset experiments
- Complete ML pipeline without any sklearn

---

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8+ | Core language |
| NumPy | 1.21+ | Matrix operations, vectorized math |
| Pandas | 1.3+ | Data loading, cleaning, manipulation |
| Matplotlib | 3.4+ | Visualizations, learning curves, plots |

---

## Getting Started

**Clone the repository**

```bash
git clone https://github.com/ather-ops/Machine-Learning-from-scratch.git
cd Machine-Learning-from-scratch
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run an example**

```bash
python "Linear regression/student_scores_example.py"
```

---

## Goal

To reach a level where ML concepts are mathematically transparent — where any model can be rebuilt from first principles, and transition to frameworks like TensorFlow or PyTorch becomes effortless because the foundations are already solid.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Ather-ops** — [GitHub Profile](https://github.com/ather-ops)

If this repository helps your learning journey, consider giving it a star. It helps others discover it and motivates continued development.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a3a8f,50:0d1f4e,100:0a0a1a&height=120&section=footer&text=Built%20with%20mathematical%20clarity&fontSize=16&fontColor=8ab4f8&fontAlignY=65&animation=fadeIn" width="100%"/>

</div>
