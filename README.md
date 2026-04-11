# Stock Market Direction Prediction using Machine Learning

## Overview

This project investigates whether **historical price behavior and statistical features** can be used to predict the **next trading day's market direction**.

Rather than forecasting exact prices, the task is framed as a **binary classification problem**:

> **Will the stock price go UP or DOWN tomorrow?**

The project follows a complete **end-to-end machine learning pipeline**, including:

* Data Understanding
* Data Preparation
* Feature Engineering
* Preliminary Modeling
* Model Evaluation

The goal is **not algorithmic trading**, but to study how machine learning models behave when applied to noisy financial time-series data.

---

## Dataset

### Source

Historical daily stock market data containing **505 publicly traded companies**. [https://www.kaggle.com/datasets/camnugent/sandp500]

### Time Range

**Feb 2013 — Feb 2018**

### Size

* ~619,000 trading records
* 505 unique stock symbols

### Original Features

| Feature | Description         |
| ------- | ------------------- |
| date    | Trading day         |
| open    | Opening price       |
| high    | Daily highest price |
| low     | Daily lowest price  |
| close   | Closing price       |
| volume  | Shares traded       |
| Name    | Stock ticker        |

---

## Problem Formulation

Financial prices are highly stochastic and difficult to predict precisely.

Instead of regression, we define a **classification target**:

### Target Variable

```
next_day_direction
```

| Value | Meaning                            |
| ----- | ---------------------------------- |
| 1     | Next day's closing price increased |
| 0     | Next day's closing price decreased |

### Target Creation

The next closing price is generated using a **shift operation**:

```
next_close = close.shift(-1)
```

This aligns today's features with tomorrow’s outcome.

---

## Data Preparation

### 1. Data Cleaning

* Removed rows containing missing values in:

  * open
  * high
  * low
* Removed 11 corrupted records
* Verified:

  * No duplicate rows
  * Continuous trading timeline

Final dataset size:

```
615,999 observations
```

---

### 2. Temporal Processing

The `date` column was converted from **object → datetime** to enable:

* chronological sorting
* time-series analysis
* leakage prevention during train/test split

---

### 3. Outlier Analysis

Outliers detected using **IQR method**:

```
Lower = Q1 − 1.5×IQR
Upper = Q3 + 1.5×IQR
```

Observation:

Large numbers of extreme values exist due to real market volatility.

**Decision:**
Outliers were retained because they represent genuine market movements rather than data errors.

---

## Feature Engineering

Machine learning models require historical context.
We created features capturing **recent market behavior**.

---

### Price Behavior Features

| Feature         | Meaning                          |
| --------------- | -------------------------------- |
| daily_return    | Percent price change             |
| price_range     | Intraday volatility (high − low) |
| open_close_diff | Market sentiment during day      |

---

### Lag Features

Capture momentum effects.

| Feature      |
| ------------ |
| return_lag_1 |

Represents previous day's return.

---

### Rolling Statistical Features

Computed per stock symbol:

| Feature       | Description             |
| ------------- | ----------------------- |
| close_mean_3  | 3-day moving average    |
| close_mean_5  | 5-day moving average    |
| close_max_5   | Recent resistance level |
| close_min_5   | Recent support level    |
| return_std_5  | Short-term volatility   |
| volume_mean_5 | Trading activity trend  |

Rows lacking sufficient history were removed.

---

## Feature Selection

Correlation analysis performed between features and target.

Selected **11 predictive features**:

```
['volume',
 'daily_return',
 'price_range',
 'open_close_diff',
 'return_lag_1',
 'close_mean_3',
 'close_mean_5',
 'close_max_5',
 'close_min_5',
 'return_std_5',
 'volume_mean_5']
```

---

## Train/Test Strategy

Time-series problems require chronological splitting.

```
Train Period: 2013 → 2017
Test Period: 2017 → 2018
```

This prevents **future information leakage**.

---

## Data Normalization

Features scaled using:

```
StandardScaler
```

Necessary for distance-based algorithms such as KNN.

---

## Modeling

Models required by coursework were implemented.

### Algorithms Tested

* Decision Tree Classifier
* Naïve Bayes Classifier
* K-Nearest Neighbors (k = 3,5,7,9)

---

## Evaluation Metrics

Models evaluated using:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Precision-Recall Curve

---

## Baseline Comparison

A **Dummy Classifier** predicting the majority class was used as benchmark.

Key Insight:

Financial direction prediction is extremely difficult; performance near 50% accuracy is expected.

---

## Results

| Model          | Accuracy    | ROC-AUC |
| -------------- | ----------- | ------- |
| Dummy Baseline | ~52.4%      | —       |
| Decision Tree  | ~51.7%      | ~0.51   |
| Naïve Bayes    | ~50.2%      | ~0.50   |
| KNN            | ~50.3–50.5% | ~0.50   |

---

## Key Findings

### 1. Financial Markets Are Highly Noisy

Traditional ML models struggle to outperform a naive baseline.

### 2. Weak Linear Relationships

Correlation analysis showed extremely small relationships between engineered features and future direction.

### 3. Feature Engineering Matters More Than Model Complexity

Improving representations of historical behavior is more impactful than changing algorithms.

### 4. Market Efficiency Evidence

Results align with the **Efficient Market Hypothesis**, suggesting short-term price direction is difficult to predict using only historical prices.

---

## Limitations

* No macroeconomic or news sentiment data
* No sector-specific modeling
* Models limited to coursework algorithms
* No deep learning or ensemble methods used

---

## Future Improvements

Potential extensions outside course scope:

* Momentum indicators (RSI, MACD)
* Cross-asset correlations
* Sector-based modeling
* Ensemble learning
* Walk-forward validation
* Alternative targets (volatility prediction)

---

## Tech Stack

**Python, Pandas, NumPy, Scikit-learn, Time Series Analysis, Statistical Modeling, Feature Engineering, Financial Data Analysis**

---

## Repository Structure

```
├── data/
├── notebook/
│   ├── EDA_FeatEng_Modeling.ipynb
├── figures/
├── README.md
```

---

## Educational Objective

This project demonstrates the complete workflow of applying machine learning to financial time-series data while highlighting practical challenges faced in quantitative research.

---

## Authors

Group #2 Project — Data Mining Course Daniels, Brian, Hector

---
