# Interpretable-churn-prediction-ml
Interpretable Machine Learning project on Customer Churn Prediction using SHAP analysis for global and local interpretability.  
#  Customer Churn Prediction with SHAP Interpretability

This project builds a machine learning model to predict customer churn using a telecommunications dataset. It emphasizes **model interpretability** using SHAP (SHapley Additive exPlanations), providing both global and local insights into why customers are predicted to churn.

---

##  Objectives

- Predict customer churn using XGBoost
- Evaluate model performance using AUC and F1-score
- Apply SHAP to explain global feature importance
- Generate SHAP force plots for individual customer predictions
- Identify counter-intuitive feature interactions
```

```
**Repository Structure**
```
interpretable-churn-prediction-project/
│
├── project.ipynb
├── report.md
├── requirements.txt
├── readme.md
|__ summary.md
-
└── plots/
    |__shap summary plot.png
    ├── customer1.png
    ├── customer2.png
    ├── customer3.png
    ├── customer4.png
    └── customer5.png
    |__ dependence.png
```
---

##  Introduction
- **Problem Statement:** Customer churn leads to significant business losses.  
- **Objective:** Predict churn and understand key drivers using interpretable ML.  
- **Importance:** Helps design targeted retention strategies and improve customer satisfaction.

---
#  Dataset Overview: Customer Churn Prediction

This project uses a structured customer dataset from a telecom provider to predict churn behavior using machine learning and SHAP interpretability.

---

## Source
- **Dataset Name**: Telco Customer Churn
- **Format**: CSV
- **Rows**: ~7,000 customers
- **Columns**: 21 features + 1 target (`Churn`)

---

This dataset enables:
- Binary classification (Churn vs. No Churn)
- Feature importance analysis via SHAP
- Local and global interpretability
---

##  Preprocessing Notes
- `TotalCharges` is converted to numeric and cleaned for missing values.
- Categorical features are encoded using one-hot encoding.
- Binary features are mapped to `0/1`.
- `customerID` is dropped as it is non-predictive.

---

##  Methodology
### Models Used
- Logistic Regression  
- Random Forest  
- XGBoost  

### Evaluation Strategy
- Train-test split  
- Cross-validation  
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC  

```
```
#  Model Interpretability with SHAP

Interpretability is critical in customer churn prediction, as it allows us to understand **why** the model makes certain decisions. SHAP (SHapley Additive exPlanations) provides two complementary perspectives:

---

##  Global Interpretability

Global interpretability explains the **overall behavior of the model** across the entire dataset.

- **Definition**: Identifies which features are most important in driving churn predictions for the population as a whole.
- **Method**: SHAP summary plots aggregate feature contributions across all customers.
- **Insights**:
  - Short **tenure** strongly increases churn risk.
  - **Month-to-month contracts** are highly associated with churn, while long-term contracts reduce risk.
  - Lack of **OnlineSecurity** and **TechSupport** services consistently push predictions toward churn.
  - **High monthly charges** combined with low total charges (new customers) are linked to churn.
- **Use Case**: Helps businesses understand the main drivers of churn and design global retention strategies.

---

##  Local Interpretability

Local interpretability explains the **prediction for an individual customer**.

- **Definition**: Breaks down how each feature contributed to a single customer’s churn prediction.
- **Method**: SHAP force plots and dependence plots show feature-level impacts for one customer at a time.
- **Insights**:
  - For a churner with **low tenure** and **no security services**, SHAP highlights these as strong positive contributors to churn.
  - For a loyal customer with a **two-year contract** and multiple add-on services, SHAP shows negative contributions that reduce churn probability.
- **Use Case**: Enables personalized retention actions, such as offering discounts or service upgrades to at-risk customers.

---


##  How to Run the Project

### Option 1 — Google Colab (Recommended)
1. Upload the dataset and notebook  
2. Install packages using `requirements.txt`  
3. Run cells from top to bottom  

### Option 2 — Local Machine
```
pip install -r requirements.txt
jupyter notebook project.ipynb
```

---

### Key Insights
- Customers with **Month-to-Month contracts** show higher churn rates.
- **High MonthlyCharges** are positively correlated with churn.
- **Longer Tenure** reduces churn likelihood.

### Visualizations
- SHAP Summary Plot
- SHAP Force Plot
- Dependence Plot
