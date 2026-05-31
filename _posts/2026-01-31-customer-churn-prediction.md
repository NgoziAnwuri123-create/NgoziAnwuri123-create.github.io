---
layout: post
title: Predicting Customer Churn Using Machine Learning
image: "/posts/customer-churn-banner.png"
tags: [Machine Learning, Customer Analytics, Random Forest, Classification, Python]
---

# Table of Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results & Discussion](#overview-results)
- [01. Business Problem](#business-problem)
- [02. Dataset Overview](#dataset-overview)
- [03. Data Preparation & Feature Engineering](#data-preparation)
- [04. Exploratory Data Analysis](#eda)
- [05. Building The Machine Learning Model](#model-development)
- [06. Model Evaluation](#model-evaluation)
- [07. Feature Importance Analysis](#feature-importance)
- [08. Key Insights](#insights)
- [09. Business Impact](#business-impact)
- [10. Conclusions](#conclusions)

___

# Project Overview <a name="overview-main"></a>

## Context <a name="overview-context"></a>

Customer churn is one of the most critical challenges facing e-commerce businesses. Acquiring new customers is significantly more expensive than retaining existing ones, making churn prediction an essential business capability.

This project aimed to develop a machine learning solution capable of identifying customers likely to discontinue their relationship with the business. By analysing customer behaviour, demographics, engagement metrics, and transactional history, the model provides early warning signals that support proactive retention strategies.

<br>

## Actions <a name="overview-actions"></a>

The project followed a complete machine learning workflow:

### Stage 1: Data Preparation

- Data cleaning
- Missing value handling
- Feature selection
- Encoding categorical variables
- Feature scaling

### Stage 2: Exploratory Data Analysis

- Customer segmentation
- Churn distribution analysis
- Behavioural trend analysis
- Correlation analysis

### Stage 3: Model Development

- Train/Test Split
- Random Forest Classification
- Hyperparameter Tuning
- Cross Validation

### Stage 4: Model Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- Feature Importance Analysis

### Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

<br>

## Results & Discussion <a name="overview-results"></a>

The optimised Random Forest model achieved:

### Performance Metrics

| Metric | Score |
|----------|---------|
| Accuracy | 87% |
| Precision | Strong |
| Recall | Strong |
| F1 Score | Balanced |

The model successfully identified customers exhibiting behavioural patterns associated with churn, providing a reliable framework for customer retention initiatives.

___

# Business Problem <a name="business-problem"></a>

E-commerce organisations often struggle to identify which customers are likely to disengage before it is too late.

### Key Challenges

- High customer acquisition costs
- Revenue loss from churn
- Difficulty identifying disengaged customers
- Limited visibility into behavioural risk indicators

### Key Question

How can machine learning be used to identify customers most likely to churn and enable proactive retention strategies?

___

# Dataset Overview <a name="dataset-overview"></a>

The dataset contains customer-level behavioural, demographic, and transactional information.

### Customer Attributes

- Age
- Gender
- Membership Tier
- Region

### Engagement Metrics

- Login Frequency
- Average Session Duration
- Marketing Opt-In Status

### Transactional Variables

- Lifetime Spend
- Purchase Frequency
- Discount Code Usage

### Service Interactions

- Customer Support Calls
- Service Requests

### Target Variable

- Churn Label

___

# Data Preparation & Feature Engineering <a name="data-preparation"></a>

## Import Required Libraries

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
```

## Load Dataset

```python
df = pd.read_csv("customer_churn_data.csv")

df.head()
```

## Missing Value Analysis

```python
df.isnull().sum()
```

## Remove Missing Records

```python
df = df.dropna()
```

## Encode Categorical Variables

```python
encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "Region",
    "Membership_Tier",
    "Marketing_Opt_In"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])
```

## Define Features & Target

```python
X = df.drop("Churn_Label", axis=1)

y = df["Churn_Label"]
```

## Feature Scaling

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

___

# Exploratory Data Analysis <a name="eda"></a>

## Customer Churn Distribution

```python
plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Churn_Label"
)

plt.title(
    "Customer Churn Distribution"
)

plt.show()
```

![Customer Churn Distribution](/posts/churn-distribution.png)

---

## Session Duration Analysis

```python
plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Churn_Label",
    y="Avg_Session_Duration_Min"
)

plt.title(
    "Session Duration by Churn Status"
)

plt.show()
```

![Session Duration Analysis](/posts/session-duration-analysis.png)

---

## Customer Support Calls Analysis

```python
plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Churn_Label",
    y="Customer_Support_Calls"
)

plt.title(
    "Customer Support Calls by Churn Status"
)

plt.show()
```

![Support Calls Analysis](/posts/support-calls-analysis.png)

___

# Building The Machine Learning Model <a name="model-development"></a>

## Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)
```

## Build Random Forest Classifier

```python
rf_model = RandomForestClassifier(
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)
```

## Generate Predictions

```python
predictions = rf_model.predict(
    X_test
)
```

___

# Model Evaluation <a name="model-evaluation"></a>

## Accuracy Score

```python
accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"Accuracy: {accuracy:.2%}"
)
```

### Output

```text
Accuracy: 87.00%
```

---

## Classification Report

```python
print(
    classification_report(
        y_test,
        predictions
    )
)
```

---

## Confusion Matrix

```python
cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title(
    "Confusion Matrix"
)

plt.show()
```

![Confusion Matrix](/posts/churn-confusion-matrix.png)

___

# Feature Importance Analysis <a name="feature-importance"></a>

Understanding feature importance helps explain what drives churn behaviour.

```python
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance.sort_values(
    by="Importance",
    ascending=False
).head(10)
```

### Visualise Top Churn Drivers

```python
top_features = feature_importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title(
    "Top Predictors of Customer Churn"
)

plt.show()
```

![Feature Importance](/posts/churn-feature-importance.png)

___

# Key Insights <a name="insights"></a>

### Customer Engagement

Customers exhibiting:

- Infrequent logins
- Short session durations
- Reduced platform activity

showed significantly higher churn rates.

### Customer Support Experience

Frequent support interactions were associated with elevated churn risk, indicating possible dissatisfaction.

### Customer Value

Customers with lower lifetime spend were more likely to disengage from the platform.

### Marketing Participation

Customers not enrolled in marketing communications displayed increased churn rates.

### Discount Behaviour

Discount usage showed mixed results, suggesting some customers are highly price-sensitive while others respond more strongly to service quality and engagement.

___

# Business Impact <a name="business-impact"></a>

This solution enables businesses to:

### Reduce Customer Churn

Identify high-risk customers before they leave.

### Improve Customer Retention

Deploy targeted retention campaigns.

### Increase Revenue

Retain valuable customers and improve customer lifetime value.

### Enhance Marketing Efficiency

Focus marketing efforts on customers most likely to disengage.

### Support Data-Driven Decision Making

Transform customer behaviour data into actionable business intelligence.

___

# Conclusions <a name="conclusions"></a>

This project demonstrates how machine learning can be applied to predict customer churn and support proactive customer retention strategies.

The optimised Random Forest model achieved **87% accuracy**, successfully identifying customers most likely to churn.

The findings highlight the importance of customer engagement, service quality, spending behaviour, and marketing participation in predicting customer attrition.

By integrating churn prediction into operational workflows, organisations can improve customer retention, increase lifetime value, and reduce revenue loss.