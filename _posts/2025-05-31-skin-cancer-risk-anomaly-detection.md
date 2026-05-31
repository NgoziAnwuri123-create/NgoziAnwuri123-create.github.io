---
layout: post
title: Detecting Skin Cancer Risk Using Anomaly Detection, SQL Analytics & Machine Learning
image: "img/posts/andect.jpeg"
tags: [Healthcare Analytics, SQL, Python, Anomaly Detection, Machine Learning]
---

# Table of Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results & Discussion](#overview-results)
- [01. Business Problem](#business-problem)
- [02. Dataset Overview](#dataset-overview)
- [03. SQL Analysis](#sql-analysis)
- [04. Exploratory Data Analysis](#eda)
- [05. Anomaly Detection Model](#anomaly-detection)
- [06. Risk Scoring Framework](#risk-scoring)
- [07. Key Insights](#insights)
- [08. Business & Healthcare Impact](#impact)
- [09. Conclusions](#conclusions)

___

# Project Overview <a name="overview-main"></a>

## Context <a name="overview-context"></a>

Skin cancer remains one of the most preventable yet increasingly common diseases when detected early. Healthcare providers collect large volumes of demographic, environmental, and clinical data, but identifying high-risk patients manually can be challenging.

This project combines SQL analytics, anomaly detection techniques, and machine learning workflows to identify unusual patient profiles associated with elevated skin cancer risk.

The goal was to uncover hidden patterns across environmental exposure, demographics, and lesion characteristics to support proactive screening and early intervention strategies.

<br>

## Actions <a name="overview-actions"></a>

The project followed four key stages:

### Stage 1: SQL Analytics

- Patient segmentation
- Environmental exposure analysis
- Demographic risk profiling
- Lesion characteristic exploration

### Stage 2: Exploratory Data Analysis

- Data cleaning
- Feature engineering
- Risk factor visualisation
- Correlation analysis

### Stage 3: Anomaly Detection

- Isolation Forest modelling
- Outlier identification
- High-risk patient flagging

### Stage 4: Risk Scoring

- Composite risk score creation
- Patient prioritisation framework
- Clinical decision support recommendations

### Tools Used

- SQL
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

<br>

## Results & Discussion <a name="overview-results"></a>

The analysis revealed that skin cancer risk is driven by combinations of:

- Age progression
- Smoking behaviour
- Alcohol consumption
- Pesticide exposure
- Lesion growth
- Persistent itching
- Bleeding symptoms

Patients exhibiting multiple risk indicators were consistently flagged as anomalous and categorised as high-priority screening candidates.

___

# Business Problem <a name="business-problem"></a>

Healthcare organisations face several challenges:

- Identifying high-risk patients early
- Distinguishing between benign and malignant indicators
- Allocating screening resources effectively
- Detecting unusual clinical profiles before diagnosis

### Key Question

How can healthcare providers use data analytics and anomaly detection to identify patients most at risk of skin cancer and improve early intervention outcomes?

___

# Dataset Overview <a name="dataset-overview"></a>

The analysis combines multiple healthcare datasets containing:

### Patient Demographics

- Age
- Gender
- Family history
- Ancestry

### Environmental Exposure

- Smoking
- Alcohol consumption
- Pesticide exposure
- Water infrastructure
- Sewage systems

### Clinical Lesion Information

- Lesion size
- Lesion location
- Lesion characteristics
- Skin cancer history

### Symptom Indicators

- Itching
- Growth
- Pain
- Bleeding
- Elevation
- Morphological change

___

# SQL Analysis <a name="sql-analysis"></a>

## Environmental Risk Analysis

The first stage explored environmental factors associated with skin cancer prevalence.

```sql
SELECT
    pesticide,
    COUNT(*) AS skin_cancer_cases
FROM patient_data
WHERE skin_cancer_history = TRUE
GROUP BY pesticide
ORDER BY skin_cancer_cases DESC;
```

### Key Finding

Pesticide exposure was associated with the highest volume of recorded skin cancer cases.

---

## Age Risk Segmentation

```sql
SELECT
CASE
    WHEN age BETWEEN 18 AND 44 THEN '18-44'
    WHEN age BETWEEN 45 AND 59 THEN '45-59'
    WHEN age BETWEEN 60 AND 74 THEN '60-74'
    ELSE '75+'
END AS age_group,
COUNT(*) AS total_cases
FROM patient_data
WHERE skin_cancer_history = TRUE
GROUP BY age_group
ORDER BY total_cases DESC;
```

### Key Finding

Patients aged 45+ represented the highest-risk segment.

---

## Lesion Region Analysis

```sql
SELECT
    lesion_region,
    COUNT(*) AS total_cases
FROM lesion_data
WHERE skin_cancer_history = TRUE
GROUP BY lesion_region
ORDER BY total_cases DESC;
```

___

# Exploratory Data Analysis <a name="eda"></a>

## Data Preparation

```python
import pandas as pd
import numpy as np

df = pd.read_csv("skin_cancer_data.csv")

df.info()
df.describe()
```

---

## Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

categorical_columns = [
    "smoke",
    "drink",
    "pesticide",
    "gender",
    "background_father"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])
```

---

# Visual 1: Skin Cancer Cases by Age Group

```python
import matplotlib.pyplot as plt

age_counts = (
    df.groupby("age_group")
      ["skin_cancer_history"]
      .sum()
      .sort_values()
)

plt.figure(figsize=(8,5))
age_counts.plot(kind="barh")
plt.title("Skin Cancer Cases by Age Group")
plt.xlabel("Cases")
plt.tight_layout()
plt.show()
```

![Age Group Analysis](/posts/skin-cancer-age-risk.png)

---

# Visual 2: Environmental Risk Factors

```python
risk_factors = [
    "smoke",
    "drink",
    "pesticide"
]

risk_rates = df.groupby(risk_factors)["skin_cancer_history"].mean()

risk_rates.sort_values().tail(10).plot(kind="barh")

plt.title("Highest Risk Environmental Profiles")
plt.xlabel("Cancer Prevalence")
plt.tight_layout()
plt.show()
```

![Environmental Risk Factors](/posts/environmental-risk-factors.png)

---

# Visual 3: Symptom Indicators

```python
symptoms = [
    "itch",
    "grew",
    "bleed",
    "changed",
    "elevation"
]

symptom_rates = (
    df.groupby("skin_cancer_history")[symptoms]
      .mean()
      .T
)

symptom_rates.plot(kind="bar")

plt.title("Symptom Frequency by Diagnosis")
plt.ylabel("Average Frequency")
plt.tight_layout()
plt.show()
```

img/posts/an1.PNG

___

# Anomaly Detection Model <a name="anomaly-detection"></a>

To identify unusual patient profiles, an Isolation Forest model was implemented.

```python
from sklearn.ensemble import IsolationForest

features = [
    "age",
    "smoke",
    "drink",
    "pesticide",
    "itch",
    "grew",
    "bleed",
    "changed"
]

X = df[features]

iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

df["anomaly"] = iso.fit_predict(X)

df["high_risk_flag"] = np.where(
    df["anomaly"] == -1,
    1,
    0
)
```

---

## Visual 4: Anomaly Distribution

```python
df["high_risk_flag"].value_counts().plot(
    kind="bar"
)

plt.title("High-Risk Patient Distribution")
plt.xlabel("Risk Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```

![Anomaly Detection Results](/posts/anomaly-distribution.png)

___

# Risk Scoring Framework <a name="risk-scoring"></a>

A composite risk score was developed.

```python
df["risk_score"] = (
      df["age"] * 0.20
    + df["smoke"] * 0.15
    + df["drink"] * 0.10
    + df["pesticide"] * 0.20
    + df["itch"] * 0.10
    + df["grew"] * 0.10
    + df["bleed"] * 0.10
    + df["changed"] * 0.05
)
```

### Risk Categories

```python
df["risk_band"] = pd.cut(
    df["risk_score"],
    bins=[0,20,40,60,100],
    labels=[
        "Low",
        "Moderate",
        "High",
        "Critical"
    ]
)
```

---

## Visual 5: Risk Score Distribution

```python
df["risk_score"].plot(
    kind="hist",
    bins=30
)

plt.title("Patient Risk Score Distribution")
plt.xlabel("Risk Score")
plt.tight_layout()
plt.show()
```

![Risk Score Distribution](/posts/risk-score-distribution.png)

___

# Key Insights <a name="insights"></a>

### Environmental Factors

- Pesticide exposure emerged as the strongest environmental predictor.
- Smoking and alcohol consumption showed elevated prevalence rates.

### Demographic Factors

- Risk increased substantially after age 45.
- Family background influenced risk patterns.

### Clinical Indicators

The strongest behavioural predictors were:

- Itching
- Lesion growth
- Bleeding
- Morphological change

### Anomaly Detection

Patients displaying multiple concurrent risk factors were consistently identified as high-risk anomalies.

___

# Business & Healthcare Impact <a name="impact"></a>

This solution enables healthcare providers to:

### Early Intervention

Identify high-risk patients before diagnosis.

### Screening Optimisation

Prioritise limited screening resources.

### Resource Allocation

Target high-risk populations more effectively.

### Predictive Healthcare

Create a foundation for future AI-powered diagnostic systems.

### Public Health Planning

Support evidence-based prevention strategies.

___

# Conclusions <a name="conclusions"></a>

This project demonstrates how SQL analytics, anomaly detection techniques, and machine learning can be combined to support proactive healthcare decision-making.

The findings show that skin cancer risk is multifactorial and driven by:

- Environmental exposure
- Lifestyle behaviours
- Age progression
- Family background
- Lesion evolution patterns

Most importantly, the analysis highlights that combinations of risk factors are substantially more predictive than individual indicators in isolation.

By identifying anomalous patient profiles early, healthcare organisations can improve screening effectiveness, accelerate diagnosis, and enhance patient outcomes.
